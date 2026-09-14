import os
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def set_random_seeds(seed: int = 42):
    """Set random seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_bboxes(annotation_path: Path):
    bboxes = {"boxes": [], "labels": []}
    for line in annotation_path.read_text().strip().splitlines():
        label, *box = line.split()
        bboxes["boxes"].append([int(coord) for coord in box])
        bboxes["labels"].append(int(label))
    return bboxes


def _forward_pass(
    model: torch.nn.Module,
    image_size: tuple[int, int] = (96, 96),
):
    x = torch.randn(
        1,
        3,
        *image_size,
        device=next(model.parameters()).device,
    )
    with torch.no_grad():
        return model(x)


def get_out_shape(
    model: torch.nn.Module,
    image_size: tuple[int, int] = (96, 96),
) -> tuple[int, ...]:
    training = model.training
    try:
        model.eval()
        return tuple(_forward_pass(model, image_size).shape)
    finally:
        model.train(training)


@torch.no_grad()
def _init_linear(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, val=0)


class ProjectionHead(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        norm_last_layer: bool = False,
    ):
        super().__init__()
        num_layers = max(num_layers, 1)
        if num_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            ]
            for _ in range(num_layers - 2):
                layers.extend(
                    [
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.GELU(),
                    ]
                )
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(_init_linear)
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        mag_weights = self.last_layer.parametrizations.weight.original0
        mag_weights.data.fill_(1)
        if norm_last_layer:
            mag_weights.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class ProjectionNetwork(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        projection_dim: int,
        head_num_layers: int = 3,
        head_hidden_dim: int = 2048,
        head_bottleneck_dim: int = 256,
        head_norm_last_layer: bool = False,
    ):
        super().__init__()
        backbone_out_shp = get_out_shape(backbone)
        if len(backbone_out_shp) != 2:
            raise ValueError(
                "Backbone output must have shape "
                f"(batch, N), got {backbone_out_shp}"
            )
        self.backbone = backbone
        self.head = ProjectionHead(
            in_dim=backbone_out_shp[-1],
            out_dim=projection_dim,
            num_layers=head_num_layers,
            hidden_dim=head_hidden_dim,
            bottleneck_dim=head_bottleneck_dim,
            norm_last_layer=head_norm_last_layer,
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


class SSLLoss(torch.nn.Module):
    def __init__(
        self,
        projection_dim: int,
        num_local_views: int,
        teacher_temp: float,
        teacher_temp_start: float,
        teacher_temp_warmup: int,
        num_epochs: int,
        student_temp: float,
        center_momentum: float,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.num_local_crops = num_local_views
        self.register_buffer("center", torch.zeros(1, projection_dim))
        self.teacher_temp_schedule = np.concatenate(
            [
                np.linspace(
                    teacher_temp_start,
                    teacher_temp,
                    teacher_temp_warmup,
                ),
                np.full(
                    num_epochs - teacher_temp_warmup,
                    fill_value=teacher_temp,
                ),
            ]
        )

    def forward(
        self,
        teacher_out,
        student_global_out,
        student_local_out,
        epoch,
    ):
        # Student sharpening
        student_global_out = student_global_out / self.student_temp
        student_global_out = student_global_out.chunk(2)
        student_local_out = student_local_out / self.student_temp
        student_local_out = student_local_out.chunk(self.num_local_crops)
        # Teacher centering and sharpening
        teacher_out = teacher_out.detach()
        temp = self.teacher_temp_schedule[epoch]
        centered = F.softmax((teacher_out - self.center) / temp, dim=-1)
        centered = centered.chunk(2)
        # Loss computation
        total_loss = 0.0
        ## Teacher global x Student global
        total_loss += torch.sum(
            -centered[0] * F.log_softmax(student_global_out[1], dim=-1),
            dim=-1,
        ).mean()
        total_loss += torch.sum(
            -centered[1] * F.log_softmax(student_global_out[0], dim=-1),
            dim=-1,
        ).mean()
        ## Teacher global x Student local
        for student_i_local in student_local_out:
            total_loss += torch.sum(
                -centered[0] * F.log_softmax(student_i_local, dim=-1),
                dim=-1,
            ).mean()
            total_loss += torch.sum(
                -centered[1] * F.log_softmax(student_i_local, dim=-1),
                dim=-1,
            ).mean()
        total_loss /= 2 * self.num_local_crops + 2
        # Update center
        self.update_center(teacher_out)
        return total_loss

    def update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


def get_params_groups(model: nn.Module):
    regularize = []
    dont_regularize = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases and Norm parameters are not regularized
        if name.endswith(".bias") or len(param.shape) == 1:
            dont_regularize.append(param)
        else:
            regularize.append(param)
    return [
        {"params": regularize},
        {"params": dont_regularize, "weight_decay": 0.0},
    ]


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    niter_per_ep: int,
    warmup_epochs: int = 0,
    start_warmup_value: float = 0.0,
):
    warmup_epochs = max(0, warmup_epochs)
    warmup_iters = warmup_epochs * niter_per_ep
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = 1 + np.cos(np.pi * iters / len(iters))
    schedule = final_value + 0.5 * (base_value - final_value) * schedule
    if not warmup_epochs:
        return schedule
    warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    return np.concatenate((warmup_schedule, schedule), axis=0)


def save_states(
    *,
    save_file: Path,
    states_dict: dict[str, Any],
    temp_suffix: str = ".tmp",
):
    tmp_file = save_file.with_suffix(f"{save_file.suffix}{temp_suffix}")
    try:
        with tmp_file.open("wb") as tmp:
            torch.save(obj=states_dict, f=tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
    except Exception as e:
        tmp_file.unlink(missing_ok=True)
        raise RuntimeError(f"Error while saving state to {save_file}") from e
    else:
        # If an error occurs here, the tmp_file is already safe
        try:
            os.replace(tmp_file, save_file)
        except Exception as e:
            raise RuntimeError(
                f"Error when replacing {save_file} with tmp file {tmp_file}."
                f"It is safe to manually move those files to recover the state."
            ) from e


def restart_from_checkpoint(
    ckp_path: Path,
    run_variables: dict[str, Any] | None = None,
    **kwargs,
):
    if not ckp_path.is_file():
        return
    checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=False)
    # reload torch states
    for key, value in kwargs.items():
        value.load_state_dict(checkpoint[key])
    # reload variables
    if run_variables is None:
        return
    for var_name in run_variables:
        run_variables[var_name] = checkpoint[var_name]


class SmoothedValue:
    def __init__(
        self,
        window_size: int | None = 20,
        fmt: str = "{value:.6f} ({local_avg:.6f})",
    ):
        self.max_value = 0.0
        self.min_value = np.inf
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value: Real):
        self.max_value = max(self.max_value, value)
        self.min_value = min(self.min_value, value)
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def local_avg(self):
        return sum(self.deque) / len(self.deque)

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def local_max(self):
        return max(self.deque)

    @property
    def global_max(self):
        return self.max_value

    @property
    def local_min(self):
        return min(self.deque)

    @property
    def global_min(self):
        return self.min_value

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            local_avg=self.local_avg,
            global_avg=self.global_avg,
            local_max=self.local_max,
            global_max=self.global_max,
            local_min=self.local_min,
            global_min=self.global_min,
            value=self.value,
        )


def get_time_string(seconds: int):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    # days, hours = divmod(hours, 24)
    time_string = " ".join(
        f"{v:02d}{k}"
        for k, v in (("h", hours), ("m", minutes), ("s", seconds))
        # in (("d", days), ("h", hours), ("m", minutes), ("s", seconds))
    )
    return time_string


class MetricLogger:
    MB = 1024.0 * 1024.0

    def __init__(self, delimiter: str = "\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            try:
                v = v.item()
            except AttributeError:
                pass
            self.meters[k].update(v)

    def add_meter(self, name: str, meter: SmoothedValue):
        self.meters[name] = meter

    def __str__(self):
        logger_str = [f"{name}: {meter}" for name, meter in self.meters.items()]
        return self.delimiter.join(logger_str)

    def log_every(
        self,
        iterable: Iterable,
        print_freq: int,
        header: str = "",
        cpu_training: bool = False,
    ):
        start_time = time.monotonic()
        end_time = start_time
        iter_duration = SmoothedValue(fmt="{value:7.3f}")
        data_duration = SmoothedValue(fmt="{value:7.3f}")
        log_msg = [
            header,
            f"[{{0:{len(str(len(iterable)))}d}}/{{1}}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        cuda_training = not cpu_training and torch.cuda.is_available()
        if cuda_training:
            log_msg.append("mem: {memory:.0f} MB")
        log_msg = self.delimiter.join(log_msg)
        for i, item in enumerate(iterable):
            data_duration.update(time.monotonic() - end_time)
            yield item
            iter_duration.update(time.monotonic() - end_time)
            if not (i % print_freq) or i == (len(iterable) - 1):
                eta_seconds = iter_duration.global_avg * (len(iterable) - i - 1)
                eta_string = get_time_string(int(eta_seconds))
                print(
                    log_msg.format(
                        i + 1,
                        len(iterable),
                        eta=f"({eta_string})",
                        meters=str(self),
                        time=str(iter_duration),
                        data=str(data_duration),
                        memory=(
                            torch.cuda.max_memory_allocated() / self.MB
                            if cuda_training
                            else None
                        ),
                    )
                )
            end_time = time.monotonic()
        total_time = time.monotonic() - start_time
        total_time_str = get_time_string(int(total_time))
        print(
            "{} Total time: {} ({:.6f} s/it)".format(
                header,
                total_time_str,
                total_time / len(iterable),
            )
        )

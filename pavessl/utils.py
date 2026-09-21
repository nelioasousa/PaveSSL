import os
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_random_seeds(seed: int = 42):
    """Set random seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_bboxes(annotation_path: str | Path):
    bboxes = {"boxes": [], "labels": []}
    with open(annotation_path, mode="r") as ann_file:
        for line in ann_file.read().strip().splitlines():
            label, *box = line.split()
            bboxes["boxes"].append([int(coord) for coord in box])
            bboxes["labels"].append(int(label))
    return bboxes


def get_params_groups(model: torch.nn.Module):
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


def get_device():
    return torch.device(
        torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    )


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

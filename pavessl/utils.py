from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


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


def _backbone_forward_pass(
    backbone: torch.nn.Module,
    image_size: tuple[int, int] = (96, 96),
):
    x = torch.randn(
        1, 3, *image_size,
        device=next(backbone.parameters()).device,
    )
    with torch.no_grad():
        out = backbone(x)
    return out


def get_backbone_out_dim(backbone: torch.nn.Module) -> int:
    training = backbone.training
    try:
        backbone.eval()
        shp = _backbone_forward_pass(backbone).shape
        if len(shp) != 2:
            raise ValueError(
                f"Backbone output must have shape (batch, N), got {shp}"
            )
        return shp[-1]
    finally:
        backbone.train(training)


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
            for _ in range(num_layers-2):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                ])
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        # self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

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
        backbone_out_dim = get_backbone_out_dim(backbone)
        self.backbone = backbone
        self.head = ProjectionHead(
            in_dim=backbone_out_dim,
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
        teacher_temp_warmup_epochs: int,
        num_epochs: int,
        student_temp: float,
        center_momentum: float,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.num_local_crops = num_local_views
        self.register_buffer("center", torch.zeros(1, projection_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(teacher_temp_start, teacher_temp, teacher_temp_warmup_epochs),
            np.full(num_epochs - teacher_temp_warmup_epochs, fill_value=teacher_temp),
        ))

    def forward(self, teacher_out, student_global_out, student_local_out, epoch):
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
                -centered[0] * F.log_softmax(student_i_local, dim=-1), dim=-1,
            ).mean()
            total_loss += torch.sum(
                -centered[1] * F.log_softmax(student_i_local, dim=-1), dim=-1,
            ).mean()
        total_loss /= (2 * self.num_local_crops + 2)
        # Update center
        self.update_center(teacher_out)
        return total_loss

    def update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


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
    return [{'params': regularize}, {'params': dont_regularize, 'weight_decay': 0.0}]


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    niter_per_ep: int,
    warmup_epochs: int = 0,
    start_warmup_value: float = 0.0
):
    warmup_epochs = max(0, warmup_epochs)
    warmup_iters = warmup_epochs * niter_per_ep
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    if not warmup_epochs:
        return schedule
    warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    return np.concatenate((warmup_schedule, schedule), axis=0)

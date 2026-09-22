import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@torch.inference_mode()
def _forward_pass(
    model: nn.Module,
    image_size: tuple[int, int] = (96, 96),
):
    x = torch.randn(
        1,
        3,
        *image_size,
        device=next(model.parameters()).device,
    )
    return model(x)


def get_out_shape(
    model: nn.Module,
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


class ProjectionHead(nn.Module):
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
        last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        _init_linear(last_layer)
        self.last_layer = nn.utils.parametrizations.weight_norm(last_layer)
        mag_weights = self.last_layer.parametrizations.weight.original0
        nn.init.constant_(mag_weights, 1.0)
        self.norm_last_layer = norm_last_layer
        if norm_last_layer:
            mag_weights.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

    def freeze_last_layer(self, freeze: bool):
        if freeze:
            self.last_layer.parametrizations.weight.original0.requires_grad = False
            self.last_layer.parametrizations.weight.original1.requires_grad = False
        else:
            self.last_layer.parametrizations.weight.original0.requires_grad = not self.norm_last_layer
            self.last_layer.parametrizations.weight.original1.requires_grad = True


class SingleHeadProjectionNetwork(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
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
        self.backbone_out_dim = backbone_out_shp[-1]
        self.global_head = ProjectionHead(
            in_dim=self.backbone_out_dim,
            out_dim=projection_dim,
            num_layers=head_num_layers,
            hidden_dim=head_hidden_dim,
            bottleneck_dim=head_bottleneck_dim,
            norm_last_layer=head_norm_last_layer,
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_head(x)
        return x


class DualHeadProjectionNetwork(SingleHeadProjectionNetwork):
    def __init__(
        self,
        backbone: nn.Module,
        projection_dim: int,
        heads_num_layers: tuple[int, int] = (3, 4),
        heads_hidden_dim: tuple[int, int] = (2048, 2048),
        heads_bottleneck_dim: tuple[int, int] = (256, 512),
        heads_norm_last_layer: tuple[bool, bool] = (False, False),
    ):
        super().__init__(
            backbone=backbone,
            projection_dim=projection_dim,
            head_num_layers=heads_num_layers[0],
            head_hidden_dim=heads_hidden_dim[0],
            head_bottleneck_dim=heads_bottleneck_dim[0],
            head_norm_last_layer=heads_norm_last_layer[0],
        )
        self.local_head = ProjectionHead(
            in_dim=self.backbone_out_dim,
            out_dim=projection_dim,
            num_layers=heads_num_layers[1],
            hidden_dim=heads_hidden_dim[1],
            bottleneck_dim=heads_bottleneck_dim[1],
            norm_last_layer=heads_norm_last_layer[1],
        )

    def forward(self, x_global, x_local):
        x_global = self.backbone(x_global)
        x_global = self.global_head(x_global)
        x_local = self.backbone(x_local)
        x_local = self.local_head(x_local)
        return x_global, x_local


class SSLLoss(nn.Module):
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
        weights,
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
        ## Global vs Global
        total_loss += (
            weights
            * torch.sum(
                -centered[0] * F.log_softmax(student_global_out[1], dim=-1),
                dim=-1,
            )
        ).sum()
        total_loss += (
            weights
            * torch.sum(
                -centered[1] * F.log_softmax(student_global_out[0], dim=-1),
                dim=-1,
            )
        ).sum()
        ## Global vs Local
        for student_i_local in student_local_out:
            total_loss += (
                weights
                * torch.sum(
                    -centered[0] * F.log_softmax(student_i_local, dim=-1),
                    dim=-1,
                )
            ).sum()
            total_loss += (
                weights
                * torch.sum(
                    -centered[1] * F.log_softmax(student_i_local, dim=-1),
                    dim=-1,
                )
            ).sum()
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

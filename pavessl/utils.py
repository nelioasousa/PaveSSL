from pathlib import Path

import numpy as np
import torch
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


def _backbone_forward_pass(backbone: torch.nn.Module, image_size: tuple[int, int] = (96, 96)):
    x = torch.randn(1, 3, *image_size)
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

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def pretrain(args):
    if args.dino_base:
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    else:
        model = models.resnet50()
        model.fc = nn.Identity()
    print(model)
    return


def get_args_parser():
    parser = argparse.ArgumentParser()
    # Architecture/weights
    # parser.add_argument("--archi", type=str, choices=["resnet50", "vits16"], default="resnet50")
    parser.add_argument("--dino-base", action="store_true")
    parser.add_argument("--projection-dim", type=int, default=16384)
    # Momentum coefficients
    parser.add_argument("--teacher-momentum", type=float, default=0.996)
    parser.add_argument("--center-momentum", type=float, default=0.9)
    # Training/optim parameters
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-end", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--weight-decay-end", type=float, default=0.4)
    parser.add_argument("--optim", type=str, choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--freeze-proj", type=int, default=1)
    # Multi-view
    parser.add_argument("--global-scale", type=float, nargs=2, default=(0.75, 1.0))
    parser.add_argument("--local-scale", type=float, nargs=2, default=(0.05, 0.4))
    parser.add_argument("--num-local-views", type=int, default=2)
    # Others
    parser.add_argument("--data", type=Path, default=Path("data/RDD2022_proc"))
    parser.add_argument("--save-dir", type=Path, default=Path("data/results/pretrain"))
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    try:
        pretrain(args)
        raise SystemExit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise SystemExit(1) from e

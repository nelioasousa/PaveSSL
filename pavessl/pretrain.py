import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from pavessl.utils import set_random_seeds, ProjectionNetwork
from pavessl.data import RDD2022, SSLMultiviewTransform


def pretrain(args):
    set_random_seeds(42)
    if args.dino_base:
        backbone = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    else:
        backbone = models.resnet50()
        backbone.fc = nn.Identity()
    student = ProjectionNetwork(backbone, args.projection_dim)
    teacher = ProjectionNetwork(
        backbone, args.projection_dim, head_norm_last_layer=True,
    )
    if torch.cuda.is_available():
        device = torch.device(torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    student = student.cuda(device=device)
    teacher = teacher.cuda(device=device)
    # teacher and student start with the same weights
    teacher.load_state_dict(student.state_dict())
    # there is no backpropagation through the teacher
    for p in teacher.parameters():
        p.requires_grad = False
    # Data
    train_ds, train_loader, valid_ds, valid_loader = get_train_valid_data()
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
    parser.add_argument("--global-size", type=int, default=224)
    parser.add_argument("--global-scale", type=float, nargs=2, default=(0.75, 1.0))
    parser.add_argument("--local-size", type=int, default=96)
    parser.add_argument("--local-scale", type=float, nargs=2, default=(0.05, 0.4))
    parser.add_argument("--num-local-views", type=int, default=2)
    # Others
    parser.add_argument("--data", type=Path, default=Path("data/RDD2022_proc"))
    parser.add_argument("--save-dir", type=Path, default=Path("data/results/pretrain"))
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


def get_train_valid_data():
    multiview_trans = SSLMultiviewTransform(
        global_size=args.global_size,
        global_scale=args.global_scale,
        local_size=args.local_size,
        local_scale=args.local_scale,
        num_local_views=args.num_local_views,
    )
    train_dataset = RDD2022(
        data_path=args.data,
        task="ssl",
        countries="all",
        split="train",
        transform=multiview_trans,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=SSLMultiviewTransform.collate_fn,
        pin_memory=False,
        drop_last=True,
    )
    valid_dataset = RDD2022(
        data_path=args.data,
        task="classification",
        countries="all",
        split="valid",
        transform=multiview_trans.teacher_global,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return train_dataset, train_loader, valid_dataset, valid_loader


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

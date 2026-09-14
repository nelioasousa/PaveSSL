import argparse
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader

from pavessl.data import get_rdd2022_countries
from pavessl.data import MultiBottomSplit, RDD2022
from pavessl.utils import MetricLogger


def load_resnet50_backbone(state_file: Path, checkpoint_key: str | None = None):
    backbone = torchvision.models.resnet50()
    backbone.fc = torch.nn.Identity()
    state_dict = torch.load(state_file, map_location="cpu", weights_only=False)
    if checkpoint_key is not None:
        state_dict = {
            key[9:]: state
            for key, state in state_dict[checkpoint_key].items()
            if key.startswith("backbone.")
        }
    backbone.load_state_dict(state_dict)
    return backbone


@torch.no_grad()
def compute_features(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = False,
):
    metric_logger = MetricLogger(delimiter="  ")
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    device = next(model.parameters()).device
    features = None
    multihot_labels = None
    for i, (images, labels) in enumerate(metric_logger.log_every(data_loader, 10)):
        images = [imgs.to(device, non_blocking=True) for imgs in images]
        feats = torch.stack([model(imgs) / len(images) for imgs in images])
        feats = feats.sum(dim=0, keepdim=False)
        if features is None:
            features = torch.empty(
                (len(dataset), feats.shape[-1]),
                dtype=feats.dtype,
                device=device,
            )
            multihot_labels = torch.empty(
                (len(dataset), labels.shape[1]),
                dtype=labels.dtype,
                device=device,
            )
        start = i * batch_size
        index = torch.arange(start, start + len(labels), dtype=torch.long, device=device)
        features.index_copy_(dim=0, index=index, source=feats)
        multihot_labels.index_copy_(dim=0, index=index, source=labels)
    return features.detach(), multihot_labels.detach()


def knn_eval(args):
    backbone = load_resnet50_backbone(args.backbone, args.checkpoint_key)
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device(torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    backbone.to(device=device)
    target_countries = get_rdd2022_countries(args.countries_subset)
    transform = MultiBottomSplit(
        size=args.crop_size,
        num_crops=args.num_crops,
        overlap=args.crops_overlap,
    )
    train_dataset = RDD2022(
        data_path=args.data,
        task="classification",
        countries=target_countries,
        split="train",
        transform=transform,
    )
    train_features = compute_features(
        model=backbone,
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    valid_dataset = RDD2022(
        data_path=args.data,
        task="classification",
        countries=target_countries,
        split="train",
        transform=transform,
    )
    valid_features = compute_features(
        model=backbone,
        dataset=valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    return


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Run k-NN evaluation on a pretrained backbone."
    )
    # Backbone location
    parser.add_argument("-bk", "--backbone", type=Path, required=True)
    parser.add_argument("-ck", "--checkpoint-key", type=str, default=None)
    # k-NN specific
    parser.add_argument(
        "-k",
        "--num-neighbours",
        type=int,
        nargs="+",
        default=[10, 25, 100],
    )
    parser.add_argument("-t", "--temperature", type=float, default=0.07)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--num-crops", type=int, default=3)
    parser.add_argument("--crops-overlap", type=float, default=1/3)
    # Data and save locations
    parser.add_argument("--data", type=Path, default=Path("data/RDD2022_proc"))
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--countries-subset", choices=[1, 2, 3], type=int, default=3)
    # Others
    parser.add_argument("--use-cuda", type=bool, action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


def run():
    try:
        parser = get_args_parser()
        args = parser.parse_args()
        knn_eval(args)
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(run())

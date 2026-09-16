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
def _compute_features(
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
    labels = None
    for i, (images, targets) in enumerate(
        metric_logger.log_every(data_loader, 10)
    ):
        images = [imgs.to(device, non_blocking=True) for imgs in images]
        feats = torch.stack([model(imgs) / len(images) for imgs in images])
        feats = feats.sum(dim=0, keepdim=False)
        if features is None:
            features = torch.empty(
                (len(dataset), feats.shape[-1]),
                dtype=feats.dtype,
                device=device,
            )
            labels = torch.empty(
                (len(dataset), targets.shape[-1]),
                dtype=targets.dtype,
                device=device,
            )
        start = i * batch_size
        end = start + len(targets)
        features[start:end].copy_(feats)
        labels[start:end].copy_(targets)
    return features.detach(), labels.detach()


def _get_features_and_labels(
    model: torch.nn.Module,
    data_path: Path,
    target_folder: str,
    image_size: int,
    num_split_crops: int,
    crops_overlap: float,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = False,
):
    transform = MultiBottomSplit(
        size=image_size,
        num_crops=num_split_crops,
        overlap=crops_overlap,
    )
    dataset = RDD2022(
        data_path=data_path,
        countries=[target_folder],
        split="train",
        task="classification",
        transform=transform,
        independent_labels=True,
    )
    train_features, train_labels = _compute_features(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dataset = RDD2022(
        data_path=data_path,
        countries=[target_folder],
        split="valid",
        task="classification",
        transform=transform,
        independent_labels=True,
    )
    valid_features, valid_labels = _compute_features(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_features, train_labels, valid_features, valid_labels


def eval(
    model: torch.nn.Module,
    data_path: Path,
    target_folder: str,
    image_size: int,
    num_split_crops: int,
    crops_overlap: float,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = False,
):
    classes = [
        line.split(maxsplit=1)[1]
        for line in (
            data_path / target_folder / "classes.txt"
        ).read_text().splitlines()
    ]
    feats_and_labels = _get_features_and_labels(
        model=model,
        data_path=data_path,
        target_folder=target_folder,
        image_size=image_size,
        num_split_crops=num_split_crops,
        crops_overlap=crops_overlap,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    train_feats, train_labels, valid_feats, valid_labels = feats_and_labels
    ...
    return


def knn_eval(args):
    backbone = load_resnet50_backbone(args.backbone, args.checkpoint_key)
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device(torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    backbone.to(device=device)
    target_countries = get_rdd2022_countries(args.countries_subset)
    ...
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
    parser.add_argument(
        "--countries-subset",
        choices=[1, 2, 3],
        type=int,
        default=3,
    )
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

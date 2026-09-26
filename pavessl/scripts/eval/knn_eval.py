import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from pavessl.data import RDD2022, MultiBottomSplit, get_rdd2022_sources
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


def _compute_features(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    header: str = "",
):
    metric_logger = MetricLogger(delimiter="  ")
    device = next(model.parameters()).device
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    features = None
    labels = None
    for i, (images, targets) in enumerate(
        metric_logger.log_every(
            iterable=data_loader,
            print_freq=10,
            header=header,
            cpu_training=(device.type != "cuda"),
        )
    ):
        targets = targets["target"]
        if isinstance(images, list):
            images = [imgs.to(device, non_blocking=True) for imgs in images]
            feats = model(images[0])
            for imgs in images[1:]:
                feats.add_(model(imgs))
            feats.div_(len(images))
        else:
            images = images.to(device, non_blocking=True)
            feats = model(images)
        feats.div_(torch.linalg.vector_norm(feats, dim=-1, keepdims=True))
        if features is None:
            features = torch.empty(
                size=(len(dataset), feats.shape[-1]),
                dtype=feats.dtype,
                device=device,
            )
            labels = torch.empty(
                size=(len(dataset), targets.shape[-1]),
                dtype=targets.dtype,
                device=device,
            )
        start = i * batch_size
        end = start + len(targets)
        features[start:end].copy_(feats)
        labels[start:end].copy_(targets)
    return features, labels


def _get_feature_similarities_and_labels(
    model: torch.nn.Module,
    data_path: Path,
    target_source: str,
    image_size: int,
    num_split_crops: int,
    crops_overlap: float,
    batch_size: int,
    num_workers: int,
):
    transform = MultiBottomSplit(
        size=image_size,
        num_crops=num_split_crops,
        overlap=crops_overlap,
    )
    dataset = RDD2022(
        data_path=data_path,
        sources=[target_source],
        split="train",
        task="classification",
        transform=transform,
    )
    train_features, train_labels = _compute_features(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        header=f"(Valid) [{target_source}/train]",
    )
    dataset = RDD2022(
        data_path=data_path,
        sources=[target_source],
        split="valid",
        task="classification",
        transform=transform,
    )
    valid_features, valid_labels = _compute_features(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        header=f"(Valid) [{target_source}/valid]",
    )
    similarities = torch.mm(valid_features, train_features.t())
    return similarities, train_labels, valid_labels


@torch.inference_mode()
def knn_eval_one(
    model: torch.nn.Module,
    ks: list[int],
    temperature: float,
    data_path: Path,
    target_source: str,
    image_size: int,
    num_split_crops: int,
    crops_overlap: float,
    batch_size: int,
    num_workers: int,
    threshold: float = 0.3,
):
    max_k = max(ks)
    similarities_and_labels = _get_feature_similarities_and_labels(
        model=model,
        data_path=data_path,
        target_source=target_source,
        image_size=image_size,
        num_split_crops=num_split_crops,
        crops_overlap=crops_overlap,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    similarities, train_labels, valid_labels = similarities_and_labels
    similarities, idxs = similarities.topk(
        k=max_k,
        dim=-1,
        largest=True,
        sorted=True,
    )
    closest_labels = train_labels.new_empty((*valid_labels.shape, max_k))
    for i in range(max_k):
        closest_labels[..., i].copy_(train_labels[idxs[:, i]])
    similarities.div_(temperature).exp_().unsqueeze_(dim=1)
    closest_labels.mul_(similarities)
    counts = valid_labels.sum(dim=0, keepdim=True)
    accuracy = np.empty((len(ks), valid_labels.shape[-1]), dtype=np.float32)
    results = {
        "accuracy": accuracy,
        "sensitivity": np.empty_like(accuracy),
        "specificity": np.empty_like(accuracy),
    }
    for i, k_value in enumerate(ks):
        probs = closest_labels[..., :k_value].sum(dim=-1)
        probs.div_(similarities[..., :k_value].sum(dim=-1))
        preds = torch.where(probs > threshold, 1.0, 0.0)
        tps = preds.mul(valid_labels).sum(dim=0)
        tns = (1.0 - preds).mul(1.0 - valid_labels).sum(dim=0)
        results["accuracy"][i] = ((tps + tns) / len(valid_labels)).cpu().numpy()
        results["sensitivity"][i] = (tps / counts).cpu().numpy()
        results["specificity"][i] = (
            (tns / (len(valid_labels) - counts)).cpu().numpy()
        )
    return results


def knn_eval(
    model: torch.nn.Module,
    ks: list[int],
    distance_temperature: float,
    pred_threshold: float,
    data_path: Path,
    target_sources: list[str],
    crop_size: int,
    num_crops: int,
    crops_overlap: float,
    batch_size: int,
    num_workers: int,
    simplified: bool = False,
):
    results = {}
    for source in target_sources:
        source_rsts = knn_eval_one(
            model=model,
            ks=ks,
            temperature=distance_temperature,
            data_path=data_path,
            target_source=source,
            image_size=crop_size,
            num_split_crops=num_crops,
            crops_overlap=crops_overlap,
            batch_size=batch_size,
            num_workers=num_workers,
            threshold=pred_threshold,
        )
        if simplified:
            results[source] = {
                metric: values.mean(axis=1)
                for metric, values in source_rsts.items()
            }
        else:
            results[source] = source_rsts
    return results


def _main(args):
    backbone = load_resnet50_backbone(args.backbone, args.checkpoint_key)
    device = torch.device(
        torch.cuda.current_device()
        if args.use_cuda and torch.cuda.is_available()
        else "cpu"
    )
    backbone = backbone.to(device=device)
    target_sources = get_rdd2022_sources(args.sources_subset)
    results = knn_eval(
        model=backbone,
        ks=args.num_neighbours,
        distance_temperature=args.temperature,
        pred_threshold=args.threshold,
        data_path=args.data,
        target_sources=target_sources,
        crop_size=args.crop_size,
        num_crops=args.num_crops,
        crops_overlap=args.crops_overlap,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    with open(args.save_file, mode="wb") as f:
        pickle.dump(obj=results, file=f, protocol=5)


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
    parser.add_argument("--crops-overlap", type=float, default=(1 / 3))
    parser.add_argument("--threshold", type=float, default=0.5)
    # Data and save locations
    parser.add_argument("--data", type=Path, default=Path("data/RDD2022_proc"))
    parser.add_argument("--save-file", type=Path, required=True)
    parser.add_argument(
        "--sources-subset",
        choices=[1, 2, 3],
        type=int,
        default=3,
    )
    # Others
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


def run():
    try:
        parser = get_args_parser()
        args = parser.parse_args()
        _main(args)
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(run())

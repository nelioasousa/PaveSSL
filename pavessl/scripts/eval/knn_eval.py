import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from pavessl import data, utils
from pavessl.scripts.eval import eval_utils


def _compute_features(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    header: str = "",
):
    metric_logger = utils.MetricLogger(delimiter="  ")
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
            # More or less 50 log lines per iteration
            print_freq=max(5, 5 * len(data_loader) // (5 * 50)),
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


def _get_feature_similarities_and_values(
    model: torch.nn.Module,
    data_path: Path,
    target_source: str,
    crops_target_size: int,
    crops_overlap: float,
    batch_size: int,
    num_workers: int,
    key_split: Literal["train", "valid", "test"] = "train",
    query_split: Literal["train", "valid", "test"] = "valid",
    header: str = "",
):
    transform = data.FourPointCrops(crops_target_size, crops_overlap)
    key_features, key_values = _compute_features(
        model=model,
        dataset=data.RDD2022(
            data_path=data_path,
            sources=[target_source],
            split=key_split,
            task="classification",
            transform=transform,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        header=(
            f"{header}{' ' if header else ''}"
            f"[{target_source}: {key_split} split]"
        ),
    )
    query_features, target_values = _compute_features(
        model=model,
        dataset=data.RDD2022(
            data_path=data_path,
            sources=[target_source],
            split=query_split,
            task="classification",
            transform=transform,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        header=(
            f"{header}{' ' if header else ''}"
            f"[{target_source}: {query_split} split]"
        ),
    )
    similarities = torch.mm(query_features, key_features.t())
    return similarities, key_values, target_values


@torch.inference_mode()
def knn_eval_one(
    model: torch.nn.Module,
    data_path: Path,
    target_source: str,
    ks: list[int],
    distance_temperature: float,
    pred_threshold: float,
    crops_target_size: int,
    crops_overlap: float,
    batch_size: int,
    num_workers: int,
    key_split: Literal["train", "valid", "test"] = "train",
    query_split: Literal["train", "valid", "test"] = "valid",
    logging_header: str = "",
):
    max_k = max(ks)
    similarities_and_values = _get_feature_similarities_and_values(
        model=model,
        data_path=data_path,
        target_source=target_source,
        crops_target_size=crops_target_size,
        crops_overlap=crops_overlap,
        batch_size=batch_size,
        num_workers=num_workers,
        key_split=key_split,
        query_split=query_split,
        header=logging_header,
    )
    similarities, key_values, target_values = similarities_and_values
    similarities, idxs = similarities.topk(
        k=max_k,
        dim=-1,
        largest=True,
        sorted=True,
    )
    closest_values = key_values.new_empty((*target_values.shape, max_k))
    for i in range(max_k):
        closest_values[..., i].copy_(key_values[idxs[:, i]])
    similarities.div_(distance_temperature).exp_().unsqueeze_(dim=1)
    closest_values.mul_(similarities)
    accuracy = np.empty((len(ks), target_values.shape[-1]), dtype=np.float32)
    macro_accuracy = np.empty(len(ks), dtype=np.float32)
    results = {
        "accuracy": accuracy,
        "sensitivity": np.empty_like(accuracy),
        "specificity": np.empty_like(accuracy),
        "macro_accuracy": macro_accuracy,
        "weighted_accuracy": np.empty_like(macro_accuracy),
        "balanced_accuracy": np.empty_like(macro_accuracy),
        "macro_sensitivity": np.empty_like(macro_accuracy),
        "weighted_sensitivity": np.empty_like(macro_accuracy),
        "macro_specificity": np.empty_like(macro_accuracy),
        "weighted_specificity": np.empty_like(macro_accuracy),
    }
    for i, k_value in enumerate(ks):
        probs = closest_values[..., :k_value].sum(dim=-1)
        probs.div_(similarities[..., :k_value].sum(dim=-1))
        preds = torch.where(probs > pred_threshold, 1.0, 0.0)
        metrics = eval_utils.multi_value_classification_metrics(
            preds=preds.cpu().numpy(),
            targets=target_values.cpu().numpy(),
        )
        for metric, values in metrics.items():
            results[metric][i] = values
    return results


def knn_eval_all(
    model: torch.nn.Module,
    data_path: Path,
    target_sources: list[str],
    ks: list[int],
    distance_temperature: float,
    pred_threshold: float,
    crops_target_size: int,
    crops_overlap: float,
    batch_size: int,
    num_workers: int,
    key_split: Literal["train", "valid", "test"] = "train",
    query_split: Literal["train", "valid", "test"] = "valid",
    logging_header: str = "",
):
    return {
        source: knn_eval_one(
            model=model,
            data_path=data_path,
            target_source=source,
            ks=ks,
            distance_temperature=distance_temperature,
            pred_threshold=pred_threshold,
            crops_target_size=crops_target_size,
            crops_overlap=crops_overlap,
            batch_size=batch_size,
            num_workers=num_workers,
            key_split=key_split,
            query_split=query_split,
            logging_header=logging_header,
        )
        for source in target_sources
    }


def _main(args):
    backbone = eval_utils.load_backbone(
        state_file=(args.experiment_dir / "checkpoint.pth"),
        checkpoint_key=args.checkpoint_key,
    )
    device = torch.device(
        "cpu"
        if args.use_cpu or not torch.cuda.is_available()
        else torch.cuda.current_device()
    )
    backbone = backbone.to(device=device, non_blocking=True)
    results = knn_eval_one(
        model=backbone,
        data_path=args.data,
        target_source=args.eval_target_source,
        ks=args.ks,
        distance_temperature=args.dist_temperature,
        pred_threshold=args.pred_threshold,
        crops_target_size=args.crops_target_size,
        crops_overlap=args.crops_overlap,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        key_split="train",
        query_split="test",
        logging_header="(Backbone kNN Eval)",
    )
    # Save results
    for metric, values in results.items():
        print(f"{metric=}:")
        print(values, end="\n\n")


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Run k-NN evaluation on a pretrained backbone.",
    )
    parser = utils.add_data_arg(parser)
    parser = eval_utils.add_eval_args(parser, "knn_eval.csv")
    parser = utils.add_knn_eval_args(parser)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
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

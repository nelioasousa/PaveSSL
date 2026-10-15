from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

from pavessl import models, utils


def add_eval_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("-d", "--experiment-dir", type=Path, required=True)
    parser.add_argument("-k", "--checkpoint-key", type=str, required=True)
    parser.add_argument(
        "--eval-target-source",
        choices=[
            "China_Drone",
            "China_MotorBike",
            "Czech",
            "India",
            "Japan",
            "Norway",
            "United_States",
        ],
        type=str,
        default="India",
    )
    parser.add_argument("--eval-save-folder", type=str, default="evals")
    return parser


def load_experiment_backbone(
    experiment_dir: Path,
    checkpoint_key: str,
) -> torchvision.models.ResNet:
    backbone = models.build_backbone(None)
    state_dict = torch.load(
        f=(experiment_dir / "checkpoint.pth"),
        map_location="cpu",
        weights_only=False,
    )
    state_dict = {
        key[9:]: state_tensor
        for key, state_tensor in state_dict[checkpoint_key].items()
        if key.startswith("backbone.")
    }
    backbone.load_state_dict(state_dict, strict=True)
    backbone.eval()
    return backbone


def multi_value_classification_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    classes: list[str] | None = None,
) -> dict[str, float | list[float]] | pd.DataFrame:
    tps = (preds * targets).sum(axis=0)
    fps = (preds * (1.0 - targets)).sum(axis=0)
    tns = ((1.0 - preds) * (1.0 - targets)).sum(axis=0)
    counts = targets.sum(axis=0)
    class_weights = counts / counts.sum()
    accuracy = (tps + tns) / len(targets)
    precision = tps / np.maximum(tps + fps, 1)
    sensitivity = tps / np.maximum(counts, 1)
    specificity = tns / np.maximum(len(targets) - counts, 1)
    f1_score = (
        2 * precision * sensitivity / np.maximum(precision + sensitivity, 1)
    )
    metrics: dict[str, float | list[float]] = {
        "accuracy": accuracy.tolist(),
        "precision": precision.tolist(),
        "sensitivity": sensitivity.tolist(),
        "specificity": specificity.tolist(),
        "f1_score": f1_score.tolist(),
        "macro_accuracy": accuracy.mean().item(),
        "weighted_accuracy": (accuracy * class_weights).sum().item(),
        "macro_precision": precision.mean().item(),
        "weighted_precision": (precision * class_weights).sum().item(),
        "macro_sensitivity": sensitivity.mean().item(),
        "weighted_sensitivity": (sensitivity * class_weights).sum().item(),
        "macro_specificity": specificity.mean().item(),
        "weighted_specificity": (specificity * class_weights).sum().item(),
        "macro_f1_score": f1_score.mean().item(),
        "weighted_f1_score": (f1_score * class_weights).sum().item(),
    }
    if classes is None:
        return metrics
    return metrics_dict_to_pandas(metrics, classes)


def metrics_dict_to_pandas(
    metrics: dict[str, float | list[float]],
    classes: list[str],
) -> pd.DataFrame:
    metric_names = [
        metric
        for metric, value in metrics.items()
        if isinstance(value, list)
    ]
    aggreg_prefixes = ["macro", "weighted"]
    df_columns = {}
    for metric_name in metric_names:
        metric_column = metrics[metric_name] + (
            [metrics[f"{agg}_{metric_name}"] for agg in aggreg_prefixes]
        )
        df_columns[metric_name] = metric_column
    df = pd.DataFrame(data=df_columns, index=(classes + aggreg_prefixes))
    df.index.name = "index"
    return df


@torch.inference_mode()
def compute_features_and_labels(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    target_label_keys: list[str] | None = None,
    header: str = "",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
    for i, (images, batch_labels) in enumerate(
        metric_logger.log_every(
            iterable=data_loader,
            print_freq=max(5, 5 * len(data_loader) // (5 * 10)),
            header=header,
            cpu_training=(device.type == "cpu"),
        )
    ):
        if isinstance(images, list):
            images = [imgs.to(device, non_blocking=True) for imgs in images]
            batch_features = model(images[0])
            for imgs in images[1:]:
                batch_features.add_(model(imgs))
            batch_features.div_(len(images))
        else:
            images = images.to(device, non_blocking=True)
            batch_features = model(images)
        if features is None:
            features = torch.empty(
                size=(len(dataset), *batch_features.shape[1:]),
                dtype=batch_features.dtype,
                device=device,
            )
            target_label_keys = (
                list(batch_labels)
                if target_label_keys is None
                else target_label_keys
            )
            labels = {
                key: torch.empty(
                    size=(len(dataset), *batch_labels[key].shape[1:]),
                    dtype=batch_labels[key].dtype,
                    device=device,
                )
                for key in target_label_keys
            }
        start = i * batch_size
        end = start + len(batch_features)
        features[start:end].copy_(batch_features)
        for key in target_label_keys:
            labels[key][start:end].copy_(batch_labels[key])
    features.div_(torch.linalg.vector_norm(features, dim=-1, keepdims=True))
    return features, labels

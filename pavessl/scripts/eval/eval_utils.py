from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
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
        "--eval-source",
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
    parser.add_argument("--eval-folder", type=str, default="evals")
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
) -> dict[str, float | list[float] | list[int]] | pd.DataFrame:
    tps = ((preds * targets) > 0.0).sum(axis=0)
    fps = ((preds * (1.0 - targets)) > 0.0).sum(axis=0)
    tns = (((1.0 - preds) * (1.0 - targets)) > 0.0).sum(axis=0)
    fns = (((1.0 - preds) * targets) > 0.0).sum(axis=0)
    assert ((tps + fps + tns + fns) == len(preds)).all().item()
    counts = targets.sum(axis=0)
    class_weights = counts / counts.sum()
    accuracy = (tps + tns) / len(targets)
    precision = tps / np.maximum(tps + fps, 1)
    sensitivity = tps / np.maximum(counts, 1)
    specificity = tns / np.maximum(len(targets) - counts, 1)
    f1_score = (
        2 * precision * sensitivity / np.maximum(precision + sensitivity, 1)
    )
    metrics: dict[str, float | list[float] | list[int]] = {
        "tp": tps.tolist(),
        "fp": fps.tolist(),
        "tn": tns.tolist(),
        "fn": fns.tolist(),
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
        metric for metric, value in metrics.items() if isinstance(value, list)
    ]
    aggreg_prefixes = ["macro", "weighted"]
    skip_aggreg = frozenset(["tp", "fp", "tn", "fn"])
    df_data = {"class": classes + aggreg_prefixes}
    for metric_name in metric_names:
        df_data[metric_name] = metrics[metric_name]
        if metric_name in skip_aggreg:
            df_data[metric_name].extend([np.nan] * len(aggreg_prefixes))
        else:
            df_data[metric_name].extend(
                [metrics[f"{agg}_{metric_name}"] for agg in aggreg_prefixes]
            )
    return pd.DataFrame(data=df_data)


def save_metrics_dataframe_to_csv(
    metrics_df: pd.DataFrame,
    save_file: str | Path,
):
    new_df = {}
    int_cols = frozenset(["tp", "fp", "tn", "fn"])
    for col_name in metrics_df.columns:
        if col_name == "class":
            to_pad = max(len(cls_name) for cls_name in metrics_df[col_name])
            to_pad = max(to_pad, len(col_name))
            str_column = metrics_df[col_name]
        elif col_name in int_cols:
            to_pad = max(len(str(val)) for val in metrics_df[col_name])
            to_pad = max(to_pad, len(col_name)) + 1
            str_column = metrics_df[col_name].map(str)
        else:
            to_pad = max(len(col_name), 7) + 1
            str_column = metrics_df[col_name].map(lambda x: f"{x:.5f}")
        new_df[col_name.rjust(to_pad, " ")] = str_column.str.pad(width=to_pad)
    pd.DataFrame(new_df).to_csv(
        save_file,
        header=True,
        index=False,
    )


def _plot_binary_confusion_matrix(
    confusion_matrix: np.ndarray,
    normalize: bool = False,
):
    confusion_matrix = confusion_matrix.reshape(2, 2)
    gt_sum = confusion_matrix.sum(axis=0, keepdims=True, dtype=np.int32)
    if normalize:
        confusion_matrix = confusion_matrix / gt_sum
    fig, ax = plt.subplots(figsize=(8, 8))
    ref_val = 1.0 if normalize else confusion_matrix.max()
    img = ax.matshow(
        confusion_matrix,
        cmap="Greens",
        vmin=0.0,
        vmax=ref_val,
    )
    ref_val /= 3
    for i in range(2):
        for j in range(2):
            val = confusion_matrix[i, j]
            color = "white" if val > ref_val else "green"
            ax.text(
                j,
                i,
                f"{val:.2f}" if normalize else str(int(val)),
                ha="center",
                va="center",
                color=color,
            )
    ax.tick_params(
        axis="x",
        top=False,
        labeltop=False,
        bottom=True,
        labelbottom=True,
    )
    ax.set_xlabel("Ground truth")
    ax.set_xticks([0, 1], [f"1 ({gt_sum[0, 0]})", f"0 ({gt_sum[0, 1]})"])
    ax.set_ylabel("Prediction")
    ax.set_yticks([0, 1], ["1", "0"])
    fig.colorbar(img, ax=ax)
    return fig, ax


def plot_metrics(
    metrics_dataframes: list[pd.DataFrame],
    thresholds: list[float],
    save_dir: Path,
):
    classes = None
    metric_names = None
    # Confusion matrices
    cm_dir = save_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    for thresh, metrics in zip(thresholds, metrics_dataframes):
        thresh *= 100
        cm_thres_dir = cm_dir / f"thresh_{int(thresh)}"
        cm_thres_dir.mkdir(exist_ok=True)
        if classes is None:
            classes = [
                val
                for val in metrics["class"]
                if val != "macro" and val != "weighted"
            ]
            metric_names = list(metrics.columns)
            metric_names.remove("class")
        metrics.set_index("class", inplace=True)
        for cls_name in classes:
            cm = metrics.loc[cls_name][["tp", "fp", "fn", "tn"]].to_numpy()
            fig, ax = _plot_binary_confusion_matrix(cm)
            ax.set_title(cls_name)
            fig.savefig(
                cm_thres_dir / f"{cls_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)
            fig, ax = _plot_binary_confusion_matrix(cm, normalize=True)
            ax.set_title(cls_name)
            fig.savefig(
                cm_thres_dir / f"{cls_name}_normalized.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)
    # Metrics versus threshold
    for metric_name in metric_names:
        fig, ax = plt.subplots(figsize=(10, 10))
        for cls_name in classes:
            metric_values = [
                metrics.at[cls_name, metric_name]
                for metrics in metrics_dataframes
            ]
            ax.plot(thresholds, metric_values, marker="o", label=cls_name)
        if metric_name not in ("tp", "fp", "tn", "fn"):
            for aggreg in ("macro", "weighted"):
                metric_values = [
                    metrics.at[aggreg, metric_name]
                    for metrics in metrics_dataframes
                ]
                ax.plot(thresholds, metric_values, marker="o", label=aggreg)
        ax.set_title(metric_name.upper())
        ax.set_xlabel("Prediction threshold (%)")
        ax.set_xticks(thresholds, [f"{int(100 * t)}%" for t in thresholds])
        ax.set_ylabel(f"{metric_name.upper()} (%)")
        ax.grid(axis="y", alpha=0.7, linestyle="--")
        ax.legend()
        fig.savefig(
            fname=(save_dir / f"{metric_name}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


@torch.inference_mode()
def compute_features_and_labels(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    target_label_keys: list[str] | None = None,
    pin_memory: bool = False,
    header: str = "",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    metric_logger = utils.MetricLogger(delimiter="  ")
    device = next(model.parameters()).device
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
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

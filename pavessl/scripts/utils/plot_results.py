import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pavessl.utils import load_exp_arguments


def plot_loss_history(
    loss_history_df: pd.DataFrame,
    metric_name: str,
    metric_unit: str,
    save_file: str | Path,
):
    loss_history_df = loss_history_df.copy()
    loss_history_df["epoch"] = np.arange(1, len(loss_history_df) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(
        x="epoch",
        y1="min",
        y2="max",
        color="tab:blue",
        data=loss_history_df,
        alpha=0.4,
        label="min-max",
    )
    ax.plot("epoch", "avg", data=loss_history_df, color="red", label="avg")
    epoch_avg_duration = loss_history_df["duration"].mean() / 60
    ax.set_xlabel(f"Epoch [{epoch_avg_duration:.2f} mins/epoch]")
    ax.set_ylabel(f"{metric_name} ({metric_unit})")
    ax.legend()
    ax.grid(axis="y", alpha=0.7, linestyle="--")
    fig.savefig(save_file, dpi=300, bbox_inches="tight")
    return fig, ax


def _plot_loss_history(
    experiment_dir: Path,
    metric_name: str,
    metric_unit: str,
    save_file: Path,
):
    try:
        loss_df = pd.read_csv(experiment_dir / "loss_history.csv", header=0)
    except FileNotFoundError:
        from torch import load

        ckpt = load(
            f=(experiment_dir / "checkpoint.pth"),
            map_location="cpu",
            weights_only=False,
        )
        loss_df = pd.DataFrame(
            data=ckpt["loss_history"][: ckpt["epoch"]],
            columns=["min", "max", "avg", "duration"],
        )
    return plot_loss_history(loss_df, metric_name, metric_unit, save_file)


def _plot_knn_valid_history(
    experiment_dir: Path,
    model_names: list[str],
    save_file: Path,
):
    exp_args = load_exp_arguments(experiment_dir)
    try:
        valid_df = pd.read_csv(experiment_dir / "valid_history.csv", header=0)
    except FileNotFoundError:
        from torch import load

        ckpt = load(
            f=(experiment_dir / "checkpoint.pth"),
            map_location="cpu",
            weights_only=False,
        )
        valid_df = pd.DataFrame(
            data=ckpt["valid_history"][: ckpt["epoch"]],
            columns=[
                f"{model}__{source}__{k}__{metric}"
                for model in model_names
                for source in exp_args.sources
                for k in exp_args.knn_ks
                for metric in ckpt["valid_metrics"]
            ],
        )
    eval_metrics = sorted(
        frozenset(col.split("__")[-1] for col in valid_df.columns)
    )
    valid_df = valid_df.copy()
    valid_df["epoch"] = np.arange(1, len(valid_df) + 1)
    fig, axs = plt.subplots(
        len(eval_metrics),
        len(exp_args.knn_ks),
        figsize=(len(exp_args.knn_ks) * 6, len(eval_metrics) * 4),
        sharex=True,
        sharey="row",
    )
    for (x, metric), (y, k_value), (color_index, source) in product(
        enumerate(eval_metrics),
        enumerate(exp_args.knn_ks),
        enumerate(exp_args.sources),
    ):
        ax = axs[x, y]
        for model_name, ls in zip(model_names, ("solid", "dotted")):
            label_suffix = "" if len(model_names) < 2 else f" ({model_name})"
            ax.plot(
                "epoch",
                f"{model_name}__{source}__{k_value}__{metric}",
                data=valid_df,
                label=f"{source}{label_suffix}",
                color=f"C{color_index}",
                linestyle=ls,
            )
        ax.grid(axis="y", alpha=0.6, linestyle="--")
        if not x:
            ax.set_title(f"K = {k_value}")
        if x + 1 == len(eval_metrics):
            ax.set_xlabel("Epoch")
        if not y:
            ax.set_ylabel(f"{metric.upper()} (%)")
            ax.legend(ncols=2)
        if y + 1 == len(exp_args.knn_ks):
            ax.tick_params(
                axis="y",
                left=False,
                right=True,
                labelleft=False,
                labelright=True,
            )
    fig.tight_layout()
    fig.savefig(save_file, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_valid_history(
    valid_history_df: pd.DataFrame,
    save_file: str | Path,
):
    valid_history_df = valid_history_df.copy()
    valid_history_df["epoch"] = np.arange(1, len(valid_history_df) + 1)
    macro_metrics = sorted(
        [c for c in valid_history_df.columns if c.startswith("macro_")]
    )
    weighted_metrics = sorted(
        [c for c in valid_history_df.columns if c.startswith("weighted_")]
    )
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True, sharey=True)
    for macro_metric in macro_metrics:
        axs[0].plot(
            "epoch",
            macro_metric,
            data=valid_history_df,
            label=macro_metric[6:],
        )
    for weighted_metric in weighted_metrics:
        axs[1].plot(
            "epoch",
            weighted_metric,
            data=valid_history_df,
            label=weighted_metric[9:],
        )
    axs[0].grid(axis="y", alpha=0.6, linestyle="--")
    axs[0].set_ylabel("MACRO METRICS (%)")
    axs[0].legend()
    axs[1].grid(axis="y", alpha=0.6, linestyle="--")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("WEIGHTED METRICS (%)")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(save_file, dpi=300, bbox_inches="tight")
    return fig, axs


def _get_args_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate results visualizations (plots) for the experiments."
        ),
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        default="plots",
        help="Folder name in which to save the plots.",
    )
    parser.add_argument(
        "--ssl-pretrain-dir",
        type=Path,
        default=Path("data/results/ssl_pretrain"),
        help="Directory in which the SSL pre-training experiments are stored.",
    )
    parser.add_argument(
        "--ssl-exps",
        type=str,
        nargs="*",
        default=[],
        help="Which experiments to generate plots to.",
    )
    parser.add_argument(
        "--sup-pretrain-dir",
        type=Path,
        default=Path("data/results/sup_pretrain"),
        help=(
            "Directory in which the supervised "
            "pre-training experiments are stored."
        ),
    )
    parser.add_argument(
        "--sup-exps",
        type=str,
        nargs="*",
        default=[],
        help="Which experiments to generate plots to.",
    )
    return parser


def _main(args):
    # Self-supervised pre-training
    for ssl_exp in args.ssl_exps:
        exp_dir = args.ssl_pretrain_dir / ssl_exp
        plots_dir = exp_dir / args.save_folder
        plots_dir.mkdir(exist_ok=True)
        _ = _plot_loss_history(
            experiment_dir=exp_dir,
            metric_name="Cross Entropy",
            metric_unit="nats",
            save_file=(plots_dir / "loss_history.png"),
        )
        _ = _plot_knn_valid_history(
            experiment_dir=exp_dir,
            model_names=("student", "teacher"),
            save_file=(plots_dir / "valid_history.png"),
        )
    # Supervised pre-training
    for sup_exp in args.sup_exps:
        exp_dir = args.sup_pretrain_dir / sup_exp
        plots_dir = exp_dir / args.save_folder
        plots_dir.mkdir(exist_ok=True)
        _ = _plot_loss_history(
            experiment_dir=exp_dir,
            metric_name="Binary Cross Entropy",
            metric_unit="1",
            save_file=(plots_dir / "loss_history.png"),
        )
        _ = _plot_knn_valid_history(
            experiment_dir=exp_dir,
            model_names=("classifier",),
            save_file=(plots_dir / "valid_history.png"),
        )


def _run():
    try:
        parser = _get_args_parser()
        args = parser.parse_args()
        _main(args)
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())

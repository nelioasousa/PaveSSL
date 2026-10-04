import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pavessl.utils import load_exp_arguments


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
            columns=["min", "max", "avg"],
        )
    loss_df["epoch"] = np.arange(1, len(loss_df) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(
        x="epoch",
        y1="min",
        y2="max",
        color="tab:blue",
        data=loss_df,
        alpha=0.4,
        label="min-max",
    )
    ax.plot("epoch", "avg", data=loss_df, color="red", label="avg")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{metric_name} ({metric_unit})")
    ax.legend()
    ax.grid(axis="y", alpha=0.7, linestyle="--")
    fig.savefig(save_file, dpi=300, bbox_inches="tight")
    return fig, ax


def _plot_valid_history(
    experiment_dir: Path,
    save_file: Path,
    model_names: list[str] | None = None,
):
    model_names = [""] if model_names is None else model_names
    model_prefixes = [(f"{name}__" if name else "") for name in model_names]
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
        index = ckpt["epoch"] // exp_args.validation_freq
        valid_df = pd.DataFrame(
            data=ckpt["valid_history"][:index],
            columns=[
                "epoch",
                *(
                    f"{prefix}{source}__{k}__{metric}"
                    for prefix in model_prefixes
                    for source in exp_args.sources
                    for k in exp_args.eval_ks
                    for metric in exp_args.eval_metrics
                ),
            ],
        )
    fig, axs = plt.subplots(
        len(exp_args.eval_metrics),
        len(exp_args.eval_ks),
        figsize=(len(exp_args.eval_ks) * 6, len(exp_args.eval_metrics) * 4),
        sharex=True,
        sharey="row",
    )
    for (x, metric), (y, k_value), (color_index, source) in product(
        enumerate(exp_args.eval_metrics),
        enumerate(exp_args.eval_ks),
        enumerate(exp_args.sources),
    ):
        ax = axs[x, y]
        for z, (prefix, ls) in enumerate(
            zip(model_prefixes, ("solid", "dotted"))
        ):
            label_suffix = f" ({model_names[z]})" if model_names[z] else ""
            ax.plot(
                "epoch",
                f"{prefix}{source}__{k_value}__{metric}",
                data=valid_df,
                label=f"{source}{label_suffix}",
                color=f"C{color_index}",
                linestyle=ls,
            )
        ax.grid(axis="y", alpha=0.6, linestyle="--")
        if not x:
            ax.set_title(f"K = {k_value}")
        if x == len(exp_args.eval_metrics) - 1:
            ax.set_xlabel("Epoch")
        if not y:
            ax.set_ylabel(f"{metric.upper()} (%)")
            ax.legend(ncols=2)
        if y == len(exp_args.eval_ks) - 1:
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


def get_args_parser():
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
        "--ssl-pretrain-exps",
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
        "--sup-pretrain-exps",
        type=str,
        nargs="*",
        default=[],
        help="Which experiments to generate plots to.",
    )
    return parser


def _main(args):
    # Self-supervised pre-training
    for ssl_exp in args.ssl_pretrain_exps:
        exp_dir = args.ssl_pretrain_dir / ssl_exp
        plots_dir = exp_dir / args.save_folder
        plots_dir.mkdir(exist_ok=True)
        _ = _plot_loss_history(
            experiment_dir=exp_dir,
            metric_name="Cross Entropy",
            metric_unit="nats",
            save_file=(plots_dir / "loss_history.png"),
        )
        _ = _plot_valid_history(
            experiment_dir=exp_dir,
            save_file=(plots_dir / "valid_history.png"),
            model_names=("student", "teacher"),
        )
    # Supervised pre-training
    for sup_exp in args.sup_pretrain_exps:
        exp_dir = args.sup_pretrain_dir / sup_exp
        plots_dir = exp_dir / args.save_folder
        plots_dir.mkdir(exist_ok=True)
        _ = _plot_loss_history(
            experiment_dir=exp_dir,
            metric_name="Binary Cross Entropy",
            metric_unit="1",
            save_file=(plots_dir / "loss_history.png"),
        )
        _ = _plot_valid_history(
            experiment_dir=exp_dir,
            save_file=(plots_dir / "valid_history.png"),
        )


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

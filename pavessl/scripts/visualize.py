import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_loss_history(
    loss_history_csv: Path,
    metric_name: str,
    metric_unit: str,
    save_file: Path,
):
    loss_df = pd.read_csv(loss_history_csv, header=0)
    loss_df["epochs"] = np.arange(1, len(loss_df) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(
        x="epochs",
        y1="min",
        y2="max",
        color="tab:blue",
        data=loss_df,
        alpha=0.4,
        label="min-max",
    )
    ax.plot("epochs", "avg", data=loss_df, color="red", label="avg")
    ax.set_xlabel("Epochs")
    ax.set_ylabel(f"{metric_name} ({metric_unit})")
    ax.legend()
    fig.savefig(save_file, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_valid_history(valid_history_csv: Path, save_file: Path):
    valid_df = pd.read_csv(valid_history_csv, header=0)
    sources, ks, metrics = set(), set(), set()
    for col in valid_df.columns:
        parts = col.split("_")
        metrics.add(parts[-1])
        ks.add(int(parts[-2]))
        sources.add("_".join(parts[:-2]))
    valid_df["epochs"] = np.arange(1, len(valid_df) + 1)
    fig, axs = plt.subplots(
        len(metrics),
        len(ks),
        figsize=(len(ks) * 5, len(metrics) * 3),
        sharex=True,
        sharey="row",
    )
    sources = sorted(sources)
    for i, metric in enumerate(sorted(metrics)):
        for j, k in enumerate(sorted(ks)):
            ax = axs[i, j]
            for source in sources:
                ax.plot(
                    "epochs",
                    f"{source}_{k}_{metric}",
                    data=valid_df,
                    label=source,
                )
            ax.grid(axis="y", alpha=0.7, linestyle="--")
            if not i:
                ax.set_title(f"K = {k}")
            if i == len(metrics) - 1:
                ax.set_xlabel("Epoch")
            if not j:
                ax.set_ylabel(f"{metric.upper()} (%)")
                ax.legend()
            if j == len(ks) - 1:
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
        description="Generate visualizations for the diverse results."
    )
    parser.add_argument(
        "--ssl-pretrain-dir",
        type=Path,
        default=Path("data/results/ssl_pretrain"),
    )
    parser.add_argument(
        "--ssl-exps",
        type=str,
        nargs="+",
        default=["test_10epochs"],
    )
    return parser


def main(args):
    # Self-supervisde pre-train
    ssl_dir = args.ssl_pretrain_dir
    for ssl_exp in args.ssl_exps:
        exp_dir = ssl_dir / ssl_exp
        plots_dir = exp_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        _ = plot_loss_history(
            loss_history_csv=(exp_dir / "loss_history.csv"),
            metric_name="Cross Entropy",
            metric_unit="nats",
            save_file=(plots_dir / "loss_history.png"),
        )
        _ = plot_valid_history(
            valid_history_csv=(exp_dir / "valid_history.csv"),
            save_file=(plots_dir / "valid_history.png"),
        )


def run():
    try:
        parser = get_args_parser()
        args = parser.parse_args()
        main(args)
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(run())

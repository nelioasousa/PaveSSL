import argparse
from pathlib import Path
from typing import Literal

import pandas as pd
import torch

from pavessl import data, utils
from pavessl.scripts.eval import eval_utils
from pavessl.scripts.pretrain import pretrain_utils


def _get_features_similarities_and_targets(
    model: torch.nn.Module,
    data_path: Path,
    target_source: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    ref_split: Literal["train", "valid", "test"] = "train",
    eval_split: Literal["train", "valid", "test"] = "valid",
    header: str = "",
):
    transform = data.FourCornersCrop(image_size)
    dataset = data.RDD2022(
        data_path=data_path,
        sources=target_source,
        split=ref_split,
        task="classification",
        transform=transform,
    )
    classes = dataset.classes
    ref_features, ref_labels = eval_utils.compute_features_and_labels(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        target_label_keys=["target"],
        header=(
            f"{header}{' ' if header else ''}"
            f"[{target_source}: {ref_split} split]"
        ),
    )
    eval_features, eval_labels = eval_utils.compute_features_and_labels(
        model=model,
        dataset=data.RDD2022(
            data_path=data_path,
            sources=target_source,
            split=eval_split,
            task="classification",
            transform=transform,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        target_label_keys=["target"],
        header=(
            f"{header}{' ' if header else ''}"
            f"[{target_source}: {eval_split} split]"
        ),
    )
    similarities = torch.mm(eval_features, ref_features.t())
    return classes, similarities, ref_labels["target"], eval_labels["target"]


@torch.inference_mode()
def knn_eval_one(
    model: torch.nn.Module,
    data_path: Path,
    target_source: str,
    ks: list[int],
    distance_temperature: float,
    pred_threshold: float,
    image_size: int,
    batch_size: int,
    num_workers: int,
    ref_split: Literal["train", "valid", "test"] = "train",
    eval_split: Literal["train", "valid", "test"] = "valid",
    as_pandas: bool = False,
    logging_header: str = "",
) -> list[pd.DataFrame] | list[dict[str, float | list[float]]]:
    max_k = max(ks)
    classes, *sim_and_targets = _get_features_similarities_and_targets(
        model=model,
        data_path=data_path,
        target_source=target_source,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        ref_split=ref_split,
        eval_split=eval_split,
        header=logging_header,
    )
    similarities, ref_targets, eval_targets = sim_and_targets
    similarities, idxs = similarities.topk(
        k=max_k,
        dim=-1,
        largest=True,
        sorted=True,
    )
    closest_targets = ref_targets.new_empty((*eval_targets.shape, max_k))
    for i in range(max_k):
        closest_targets[..., i].copy_(ref_targets[idxs[:, i]])
    similarities.div_(distance_temperature).exp_().unsqueeze_(dim=1)
    closest_targets.mul_(similarities)
    results = []
    for i, k_value in enumerate(ks):
        probs = closest_targets[..., :k_value].sum(dim=-1)
        probs.div_(similarities[..., :k_value].sum(dim=-1))
        preds = torch.where(probs > pred_threshold, 1.0, 0.0)
        results.append(
            eval_utils.multi_value_classification_metrics(
                preds=preds.cpu().numpy(),
                targets=eval_targets.cpu().numpy(),
                classes=(classes if as_pandas else None),
            )
        )
    return results


def knn_eval_all(
    model: torch.nn.Module,
    data_path: Path,
    target_sources: list[str],
    ks: list[int],
    distance_temperature: float,
    pred_threshold: float,
    image_size: int,
    batch_size: int,
    num_workers: int,
    ref_split: Literal["train", "valid", "test"] = "train",
    eval_split: Literal["train", "valid", "test"] = "valid",
    as_pandas: bool = False,
    logging_header: str = "",
) -> (
    dict[str, list[pd.DataFrame]]
    | dict[str, list[dict[str, float | list[float]]]]
):
    return {
        source: knn_eval_one(
            model=model,
            data_path=data_path,
            target_source=source,
            ks=ks,
            distance_temperature=distance_temperature,
            pred_threshold=pred_threshold,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            ref_split=ref_split,
            eval_split=eval_split,
            as_pandas=as_pandas,
            logging_header=logging_header,
        )
        for source in target_sources
    }


def _main(args):
    backbone = eval_utils.load_experiment_backbone(
        experiment_dir=args.experiment_dir,
        checkpoint_key=args.checkpoint_key,
    )
    device = utils.get_device()
    backbone = backbone.to(device=device, non_blocking=True)
    results = knn_eval_one(
        model=backbone,
        data_path=args.data,
        target_source=args.eval_target_source,
        ks=args.knn_ks,
        distance_temperature=args.knn_dist_temperature,
        pred_threshold=args.knn_pred_threshold,
        image_size=pretrain_utils.GLOBAL_SIZE,
        batch_size=args.knn_batch_size,
        num_workers=args.num_workers,
        ref_split="train",
        eval_split="test",
        as_pandas=True,
        logging_header="(kNN Eval)",
    )
    # Save results
    save_dir = args.experiment_dir / args.eval_save_folder
    save_dir.mkdir(exist_ok=True)
    k_padding = len(str(max(args.knn_ks)))
    for k_value, metrics in zip(args.knn_ks, results):
        k_value_str = str(k_value).rjust(k_padding, "0")
        index_name_pad = max(len(name) for name in metrics.index)
        metrics.index = metrics.index.str.pad(width=index_name_pad)
        metrics.to_csv(
            save_dir / f"knn_eval_{args.eval_target_source}_k{k_value_str}.csv",
            float_format="%.5f",
            header=True,
            index=True,
        )


def add_knn_eval_args(
    parser: argparse.ArgumentParser
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--knn-ks",
        type=int,
        nargs="+",
        default=[5, 10, 25, 50],
        help="Number of neighbours to use in kNN evalatuation.",
    )
    parser.add_argument(
        "--knn-dist-temperature",
        type=float,
        default=0.07,
        help=(
            "Sharpening constant for samples' distance weighting. "
            "Smaller values give more importance to closest samples."
        ),
    )
    parser.add_argument(
        "--knn-pred-threshold",
        type=float,
        default=0.3,
        help="Threshold for kNN prediction.",
    )
    parser.add_argument(
        "--knn-batch-size",
        type=int,
        default=32,
        help="Batch size for kNN evaluation.",
    )
    return parser


def _get_args_parser():
    parser = argparse.ArgumentParser(
        description="Run k-NN evaluation on a pretrained backbone.",
    )
    parser = utils.add_data_arg(parser)
    parser = eval_utils.add_eval_args(parser)
    parser = add_knn_eval_args(parser)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading.",
    )
    return parser


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

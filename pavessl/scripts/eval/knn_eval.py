import argparse
from pathlib import Path
from typing import Literal

import torch
import pandas as pd

from pavessl import data, utils
from pavessl.scripts.eval import eval_utils


def _get_features_similarities_and_targets(
    model: torch.nn.Module,
    data_path: Path,
    target_source: str,
    crops_target_size: int,
    crops_overlap: float,
    batch_size: int,
    num_workers: int,
    ref_split: Literal["train", "valid", "test"] = "train",
    eval_split: Literal["train", "valid", "test"] = "valid",
    header: str = "",
):
    transform = data.FourPointCrops(crops_target_size, crops_overlap)
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
    crops_target_size: int,
    crops_overlap: float,
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
        crops_target_size=crops_target_size,
        crops_overlap=crops_overlap,
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
    crops_target_size: int,
    crops_overlap: float,
    batch_size: int,
    num_workers: int,
    ref_split: Literal["train", "valid", "test"] = "train",
    eval_split: Literal["train", "valid", "test"] = "valid",
    as_pandas: bool = False,
    logging_header: str = "",
) -> dict[str, list[pd.DataFrame]] | dict[str, list[dict[str, float | list[float]]]]:
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
        crops_target_size=args.knn_crops_target_size,
        crops_overlap=args.knn_crops_overlap,
        batch_size=args.knn_batch_size,
        num_workers=args.num_workers,
        ref_split=args.ref_split,
        eval_split=args.eval_split,
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
            save_dir / f"knn_eval_k{k_value_str}.csv",
            float_format="%.5f",
            header=True,
            index=True,
        )


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Run k-NN evaluation on a pretrained backbone.",
    )
    parser = utils.add_data_arg(parser)
    parser = eval_utils.add_eval_args(parser)
    parser = utils.add_knn_eval_args(parser)
    parser.add_argument(
        "--ref-split",
        choices=["train", "valid", "test"],
        default="train",
        help="Which split to use as reference.",
    )
    parser.add_argument(
        "--eval-split",
        choices=["train", "valid", "test"],
        default="test",
        help="Which split to evaluate.",
    )
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

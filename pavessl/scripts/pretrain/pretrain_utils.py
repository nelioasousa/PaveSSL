import pickle
from argparse import Namespace
from pathlib import Path

from torch import nn

from pavessl.scripts.eval.knn_eval import knn_eval_all


def check_run(experiment_args: Namespace) -> Namespace:
    args_file_path = experiment_args.results_dir / "run.args"
    try:
        with args_file_path.open("rb") as args_file:
            return pickle.load(args_file)
    except FileNotFoundError:
        with args_file_path.open("wb") as args_file:
            pickle.dump(obj=experiment_args, file=args_file, protocol=5)
        return experiment_args


def backbone_knn_eval(
    backbone: nn.Module,
    data_path: Path,
    target_sources: list[str],
    ks: list[int],
    distance_temperature: float,
    pred_threshold: float,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = False,
    logging_header: str = "",
) -> dict[str, list[dict[str, float]]]:
    backbone.eval()
    results = knn_eval_all(
        model=backbone,
        data_path=data_path,
        target_sources=target_sources,
        ks=ks,
        distance_temperature=distance_temperature,
        pred_threshold=pred_threshold,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        ref_split="train",
        eval_split="valid",
        pin_memory=pin_memory,
        logging_header=logging_header,
    )
    return {
        source: [
            {
                metric: value
                for metric, value in k_metrics.items()
                if isinstance(value, float)
            }
            for k_metrics in metrics_per_k
        ]
        for source, metrics_per_k in results.items()
    }

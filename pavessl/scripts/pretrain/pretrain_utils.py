from pathlib import Path

from torch import nn

from pavessl.scripts.eval.knn_eval import knn_eval_all


def backbone_knn_eval(
    backbone: nn.Module,
    data_path: Path,
    target_sources: list[str],
    ks: list[int],
    distance_temperature: float,
    pred_threshold: float,
    crops_target_size: int,
    crops_overlap: float,
    batch_size: int,
    num_workers: int,
    logging_header: str = "",
):
    backbone.eval()
    results = knn_eval_all(
        model=backbone,
        data_path=data_path,
        target_sources=target_sources,
        ks=ks,
        distance_temperature=distance_temperature,
        pred_threshold=pred_threshold,
        crops_target_size=crops_target_size,
        crops_overlap=crops_overlap,
        batch_size=batch_size,
        num_workers=num_workers,
        key_split="train",
        query_split="valid",
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

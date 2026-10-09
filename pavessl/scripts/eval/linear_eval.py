import argparse
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader

from pavessl import data, models, utils
from pavessl.scripts.eval import eval_utils
from pavessl.scripts.pretrain import sup_pretrain


def _get_classifier(num_classes, device, args):
    classifier = models.build_classification_network(
        network_state_path=(args.start_points_dir / "linear_eval_net.pth"),
        num_classes=num_classes,
        head_num_layers=1,
    )
    classifier.backbone = eval_utils.load_experiment_backbone(
        exp_dir=args.experiment_dir,
        checkpoint_key=args.checkpoint_key,
    )
    # Static/frozen backbone
    for p in classifier.backbone.parameters():
        p.requires_grad = False
    return classifier.to(device=device, non_blocking=True)


@torch.inference_mode()
def _predict(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    pred_threshold: float,
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
    preds = None
    targets = None
    for i, (images, labels) in enumerate(
        metric_logger.log_every(
            iterable=data_loader,
            # More or less 20 log lines per iteration
            print_freq=max(5, 5 * len(data_loader) // (5 * 20)),
            header=header,
            cpu_training=(device.type != "cuda"),
        )
    ):
        batch_targets = labels["target"]
        images = images.to(device, non_blocking=True)
        batch_preds = model(images)
        if preds is None:
            preds = torch.empty(
                size=(len(dataset), *batch_preds.shape[1:]),
                dtype=batch_preds.dtype,
                device=device,
            )
            targets = torch.empty(
                size=(len(dataset), *batch_targets.shape[1:]),
                dtype=batch_targets.dtype,
                device="cpu",
            )
        start = i * batch_size
        end = start + len(batch_preds)
        preds[start:end].copy_(batch_preds)
        targets[start:end].copy_(batch_targets)
    preds = preds.sigmoid_()
    preds = torch.where(preds > pred_threshold, 1.0, 0.0)
    return preds.cpu().numpy(), targets.numpy()


@torch.inference_mode()
def linear_eval_one(
    model: torch.nn.Module,
    data_path: Path,
    target_source: str,
    target_split: Literal["train", "valid", "test"],
    pred_threshold: float,
    batch_size: int,
    num_workers: int,
    logging_header: str = "",
):
    dataset = data.RDD2022(
        data_path=data_path,
        sources=target_source,
        split=target_split,
        task="classification",
    )
    preds, targets = _predict(
        model=model,
        dataset=dataset,
        pred_threshold=pred_threshold,
        batch_size=batch_size,
        num_workers=num_workers,
        header=logging_header,
    )
    return eval_utils.multi_value_classification_metrics(
        preds=preds,
        targets=targets,
    )


def _main(args):
    model = sup_pretrain.pretrain(args=args, get_classifier=_get_classifier)
    results = linear_eval_one(
        model=model,
        data_path=args.data,
        target_source=args.eval_target_source,
        target_split="test",
        pred_threshold=args.eval_pred_threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        logging_header="(Backbone Linear Eval)",
    )
    # Save results
    for metric, values in results.items():
        print(f"{metric=}:")
        print(values, end="\n\n")


def check_args(args: argparse.Namespace) -> argparse.Namespace:
    args.save_dir = args.experiment_dir
    args.save_folder = "linear_eval_finetuing"
    args = sup_pretrain.check_args(args)
    return args


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Run linear evaluation on a pretrained backbone.",
        parents=[sup_pretrain.get_args_parser()],
        conflict_handler="resolve",
    )
    parser = utils.add_training_args(
        parser,
        save_dir_default="",
        save_folder_default="",
        sources_subset_default=0,
        epochs_default=100,
        lr_default=1e-3,
        lr_end_default=1e-6,
        weight_decay_default=0.01,
        weight_decay_end_default=0.1,
    )
    parser = eval_utils.add_eval_args(parser, "linear_eval.csv")
    parser.add_argument(
        "--eval-pred-threshold",
        type=float,
        default=0.5,
        help="Prediction threshold for linear evaluation.",
    )
    return parser


def run():
    try:
        parser = get_args_parser()
        args = parser.parse_args()
        _main(check_args(args))
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(run())

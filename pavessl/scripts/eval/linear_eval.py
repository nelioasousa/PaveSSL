import argparse
import math
import time
from typing import Literal

import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader

from pavessl import data, models, utils
from pavessl.scripts.eval import eval_utils


@torch.inference_mode()
def _predict(
    classifier: torch.nn.Module,
    dataset: Dataset,
    pred_threshold: float,
    batch_size: int,
):
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    preds = None
    targets = None
    for i, (images, labels) in enumerate(data_loader):
        batch_preds = classifier(images)
        if preds is None:
            preds = torch.empty(
                size=(len(dataset), *batch_preds.shape[1:]),
                dtype=batch_preds.dtype,
                device=images.device,
            )
            targets = torch.empty(
                size=(len(dataset), *labels.shape[1:]),
                dtype=labels.dtype,
                device=labels.device,
            )
        start = i * batch_size
        end = start + len(batch_preds)
        preds[start:end].copy_(batch_preds)
        targets[start:end].copy_(labels)
    preds = preds.sigmoid_()
    preds = torch.where(preds > pred_threshold, 1.0, 0.0)
    return preds.cpu().numpy(), targets.cpu().numpy()


def _eval(
    classifier: torch.nn.Module,
    dataset: Dataset,
    pred_threshold: float,
    batch_size: int,
):
    classifier.eval()
    preds, targets = _predict(
        classifier=classifier,
        dataset=dataset,
        pred_threshold=pred_threshold,
        batch_size=batch_size,
    )
    return eval_utils.multi_value_classification_metrics(preds, targets)


def _train_one_epoch(
    classifier,
    data_loader,
    loss_fn,
    optimizer,
    lr_schedule,
    wd_schedule,
    epoch,
    num_epochs,
):
    classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "loss",
        utils.SmoothedValue(fmt="({value:10.6f} / {local_avg:10.6f} avg)"),
    )
    metric_logger.add_meter("lr", utils.SmoothedValue(fmt="{value:.2e}"))
    metric_logger.add_meter("wd", utils.SmoothedValue(fmt="{value:.2e}"))
    epoch_str = str(epoch + 1).rjust(len(str(num_epochs)), "0")
    header = f"(Finetune) Epoch: [{epoch_str}/{num_epochs}]"
    for it, (features, targets, weights) in enumerate(
        metric_logger.log_every(
            data_loader,
            max(5, 5 * len(data_loader) // (5 * 10)),
            header,
        )
    ):
        it = len(data_loader) * epoch + it  # Global training iteration
        # Only the first group is regularized
        optimizer.param_groups[0]["weight_decay"] = wd_schedule[it]
        optimizer.param_groups[0]["lr"] = lr_schedule[it]
        optimizer.param_groups[1]["lr"] = lr_schedule[it]
        # Foward pass and loss computation
        logits = classifier(features)
        loss = loss_fn(logits, targets, weights.div_(weights.sum()))
        # Loss explosion, unstable training
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss exploded with value {loss_value}")
        # Classifier update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Logging
        metric_logger.update(
            loss=loss_value,
            lr=lr_schedule[it].item(),
            wd=wd_schedule[it].item(),
        )
    return metric_logger.meters["loss"]


def _finetune(
    classifier: torch.nn.Module,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    valid_features: torch.Tensor,
    valid_labels: torch.Tensor,
    lr: float,
    lr_end: float,
    lr_warmup_size: int,
    wd: float,
    wd_end: float,
    optim: Literal["adamw", "sgd"],
    batch_size: int,
    num_epochs: int,
    pred_threshold: float,
):
    pos_counts = train_labels.sum(dim=0, keepdim=True)
    pos_weights = len(train_labels) / (2 * pos_counts)
    neg_weights = len(train_labels) / (2 * (len(train_labels) - pos_counts))
    weights = torch.where(train_labels > 0, pos_weights, neg_weights).sum(dim=-1)
    valid_dataset = TensorDataset(valid_features, valid_labels)
    train_dataloader = DataLoader(
        dataset=TensorDataset(train_features, train_labels, weights),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    lr_schedule = utils.cosine_scheduler(
        base_value=(lr * batch_size / 256.0),
        final_value=(lr_end * batch_size / 256.0),
        epochs=num_epochs,
        niter_per_ep=len(train_dataloader),
        warmup_epochs=lr_warmup_size,
    )
    wd_schedule = utils.cosine_scheduler(
        base_value=wd,
        final_value=wd_end,
        epochs=num_epochs,
        niter_per_ep=len(train_dataloader),
    )
    loss_history = np.zeros((num_epochs, 3), dtype=np.float32)
    valid_history = None
    valid_metrics = None
    loss_fn = models.MultiValueClassificationLoss()
    loss_fn = loss_fn.to(
        device=next(classifier.parameters()).device,
        non_blocking=True,
    )
    optimizer = models.get_optimizer(target_model=classifier, which=optim)
    for epoch in range(num_epochs):
        # Finetune step
        loss_log = _train_one_epoch(
            classifier=classifier,
            data_loader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
            epoch=epoch,
            num_epochs=num_epochs,
        )
        loss_history[epoch] = [
            loss_log.min_value,
            loss_log.max_value,
            loss_log.global_avg,
        ]
        # Validation step
        valid_results = _eval(
            classifier=classifier,
            dataset=valid_dataset,
            pred_threshold=pred_threshold,
            batch_size=batch_size,
        )
        if valid_history is None:
            valid_metrics = [
                metric
                for metric, value in valid_results.items()
                if isinstance(value, float)
            ]
            valid_history = np.zeros(
                shape=(num_epochs, len(valid_metrics)),
                dtype=np.float32,
            )
        for i, metric in enumerate(valid_metrics):
            valid_history[epoch, i] = valid_results[metric]
    return classifier, loss_history, valid_metrics, valid_history


def _main(args):
    device = utils.get_device()
    backbone = eval_utils.load_experiment_backbone(
        exp_dir=args.experiment_dir,
        checkpoint_key=args.checkpoint_key,
    )
    backbone = backbone.to(device=device, non_blocking=True)
    start_time = time.monotonic()
    train_features, train_labels = eval_utils.compute_features_and_labels(
        model=backbone,
        dataset=data.RDD2022(
            data_path=args.data,
            sources=args.sources,
            split="train",
            task="classification",
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        header="(Feature extraction: train split)",
    )
    valid_features, valid_labels = eval_utils.compute_features_and_labels(
        model=backbone,
        dataset=data.RDD2022(
            data_path=args.data,
            sources=args.sources,
            split="valid",
            task="classification",
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        header="(Feature extraction: valid split)",
    )
    classifier = models.build_classification_head(
        state_path=(args.start_points_dir / "linear_eval_head.pth"),
        in_dim=train_features.shape[-1],
        out_dim=train_labels.shape[-1],
        num_layers=args.eval_num_layers,
        hidden_dim=args.eval_hidden_dim,
    )
    classifier = classifier.to(device=device, non_blocking=True)
    classifier, loss_history, valid_metrics, valid_history = _finetune(
        classifier=classifier,
        train_features=train_features,
        train_labels=train_labels,
        valid_features=valid_features,
        valid_labels=valid_labels,
        lr=args.lr,
        lr_end=args.lr_end,
        lr_warmup_size=args.lr_warmup_size,
        wd=args.weight_decay,
        wd_end=args.weight_decay_end,
        optim=args.optim,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        pred_threshold=args.eval_pred_threshold,
    )
    del train_features, train_labels
    del valid_features, valid_labels
    # Save finetuned classifier
    utils.save_states(
        save_file=(args.results_dir / "classification_head.pth"),
        states_dict=classifier.state_dict(),
    )
    # Save finetuning histories
    np.savetxt(
        fname=(args.results_dir / "loss_history.csv"),
        X=loss_history,
        delimiter=",",
        fmt="%20.10e",
        header="min,max,avg",
        comments="",
    )
    np.savetxt(
        fname=(args.results_dir / "valid_history.csv"),
        X=valid_history,
        delimiter=",",
        fmt=(["%.4f"] * len(valid_metrics)),
        header=(",".join(valid_metrics)),
        comments="",
    )
    # Linear evaluation
    test_features, test_labels = eval_utils.compute_features_and_labels(
        model=backbone,
        dataset=data.RDD2022(
            data_path=args.data,
            sources=args.eval_target_source,
            split="test",
            task="classification",
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        header=f"(Linear eval: {args.eval_target_source} test split)",
    )
    results = _eval(
        classifier=classifier,
        dataset=TensorDataset(test_features, test_labels),
        pred_threshold=args.eval_pred_threshold,
        batch_size=args.batch_size,
    )
    # Save results
    for metric, values in results.items():
        print(f"{metric=}:")
        print(values, end="\n\n")
    # Report duration
    duration = int(time.monotonic() - start_time)
    print(f"Linear evaluation ended in {utils.get_time_string(duration)}!")


def check_args(args: argparse.Namespace) -> argparse.Namespace:
    args.save_dir = args.experiment_dir
    args.save_folder = "linear_eval_finetuning"
    args = utils.check_training_args(args)
    return args


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Run linear evaluation on a pretrained backbone.",
    )
    parser = utils.add_data_arg(parser)
    parser = utils.add_training_args(
        parser,
        save_dir_default="",
        save_folder_default="",
        sources_subset_default=0,
        epochs_default=100,
        batch_size_default=16,
        lr_default=1e-4,
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
    parser.add_argument(
        "--eval-image-size",
        type=int,
        default=448,
        help="Input image dimension.",
    )
    parser.add_argument(
        "--eval-crop-scale",
        type=float,
        nargs=2,
        default=(0.85, 1.0),
        help="Min-max area range for image cropping.",
    )
    parser.add_argument(
        "--eval-num-layers",
        type=int,
        default=3,
        help="Number of linear layers in the classification head network.",
    )
    parser.add_argument(
        "--eval-hidden-dim",
        type=int,
        default=4096,
        help="Hidden layers dimension in classificaation head network.",
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

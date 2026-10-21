import argparse
import math
import time
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms.v2 import Resize

from pavessl import data, models, utils
from pavessl.scripts.eval import eval_utils
from pavessl.scripts.pretrain import pretrain_utils


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
    for i, (images, batch_targets) in enumerate(data_loader):
        batch_preds = classifier(images)
        if preds is None:
            preds = torch.empty(
                size=(len(dataset), *batch_preds.shape[1:]),
                dtype=batch_preds.dtype,
                device=batch_preds.device,
            )
            targets = torch.empty(
                size=(len(dataset), *batch_targets.shape[1:]),
                dtype=batch_targets.dtype,
                device=batch_targets.device,
            )
        start = i * batch_size
        end = start + len(batch_preds)
        preds[start:end].copy_(batch_preds)
        targets[start:end].copy_(batch_targets)
    preds = preds.sigmoid_()
    preds = torch.where(preds > pred_threshold, 1.0, 0.0)
    return preds.cpu().numpy(), targets.cpu().numpy()


def _eval(
    classifier: torch.nn.Module,
    dataset: Dataset,
    pred_threshold: float,
    batch_size: int,
    classes: list[str] | None = None,
):
    classifier.eval()
    preds, targets = _predict(
        classifier=classifier,
        dataset=dataset,
        pred_threshold=pred_threshold,
        batch_size=batch_size,
    )
    return eval_utils.multi_value_classification_metrics(
        preds,
        targets,
        classes,
    )


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
    for it, (features, targets, classes_weights, samples_weight) in enumerate(
        metric_logger.log_every(
            data_loader,
            max(5, 5 * len(data_loader) // (5 * 10)),
            header,
        )
    ):
        samples_weight.div_(samples_weight.sum())
        # Only the first group is regularized
        it = len(data_loader) * epoch + it
        optimizer.param_groups[0]["weight_decay"] = wd_schedule[it]
        optimizer.param_groups[0]["lr"] = lr_schedule[it]
        optimizer.param_groups[1]["lr"] = lr_schedule[it]
        # Foward pass and loss computation
        logits = classifier(features)
        loss = loss_fn(logits, targets, classes_weights, samples_weight)
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
    return metric_logger


def _finetune(
    classifier: torch.nn.Module,
    train_features: torch.Tensor,
    train_labels: dict[str, torch.Tensor],
    valid_features: torch.Tensor,
    valid_labels: dict[str, torch.Tensor],
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
    classes_weights = torch.where(
        train_labels["target"] > 0,
        train_labels["pos_weights"],
        train_labels["neg_weights"],
    )
    classes_weights.div_(classes_weights.sum(dim=-1, keepdim=True))
    train_dataloader = DataLoader(
        dataset=TensorDataset(
            train_features,
            train_labels["target"],
            classes_weights,
            train_labels["source_weight"],
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    valid_dataset = TensorDataset(valid_features, valid_labels["target"])
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
    loss_history = np.zeros((num_epochs, 4), dtype=np.float32)
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
        logger = _train_one_epoch(
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
            logger.meters["loss"].min_value,
            logger.meters["loss"].max_value,
            logger.meters["loss"].global_avg,
            logger.meters["iter"].total_accumulated,
        ]
        # Validation step
        valid_results = _eval(
            classifier=classifier,
            dataset=valid_dataset,
            pred_threshold=pred_threshold,
            batch_size=batch_size,
        )
        if valid_history is None:
            valid_metrics = sorted(
                metric
                for metric, value in valid_results.items()
                if isinstance(value, float)
            )
            valid_history = np.zeros(
                shape=(num_epochs, len(valid_metrics)),
                dtype=np.float32,
            )
        valid_history[epoch] = [
            valid_results[metric] for metric in valid_metrics
        ]
    return classifier, loss_history, valid_metrics, valid_history


def _main(args):
    utils.set_random_seeds(42)
    device = utils.get_device()
    backbone = eval_utils.load_experiment_backbone(
        experiment_dir=args.experiment_dir,
        checkpoint_key=args.checkpoint_key,
    )
    backbone = backbone.to(device=device, non_blocking=True)
    transform = Resize(size=pretrain_utils.GLOBAL_SIZE)
    start_time = time.monotonic()
    train_features, train_labels = eval_utils.compute_features_and_labels(
        model=backbone,
        dataset=data.RDD2022(
            data_path=args.data,
            sources=args.eval_target_source,
            split="train",
            task="classification",
            transform=transform,
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        header=f"(Features extraction: {args.eval_target_source} train split)",
    )
    valid_features, valid_labels = eval_utils.compute_features_and_labels(
        model=backbone,
        dataset=data.RDD2022(
            data_path=args.data,
            sources=args.eval_target_source,
            split="valid",
            task="classification",
            transform=transform,
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_label_keys=["target"],
        header=f"(Features extraction: {args.eval_target_source} valid split)",
    )
    state_filename = "{}_{}l_{}d_linear_eval_net.pth".format(
        args.eval_target_source,
        args.eval_num_layers,
        args.eval_hidden_dim,
    )
    classifier = models.build_classification_head(
        state_path=(args.start_points_dir / state_filename),
        in_dim=train_features.shape[-1],
        out_dim=train_labels["target"].shape[-1],
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
    checkpoint_filename = "{}_{}l_{}d_linear_eval_net.pth".format(
        args.eval_target_source,
        args.eval_num_layers,
        args.eval_hidden_dim,
    )
    utils.save_states(
        save_file=(args.results_dir / checkpoint_filename),
        states_dict=classifier.state_dict(),
    )
    # Save finetuning histories
    np.savetxt(
        fname=(args.results_dir / "loss_history.csv"),
        X=loss_history,
        delimiter=",",
        fmt="%20.10e",
        header="min,max,avg,duration",
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
    dataset = data.RDD2022(
        data_path=args.data,
        sources=args.eval_target_source,
        split="test",
        task="classification",
        transform=transform,
    )
    test_features, test_labels = eval_utils.compute_features_and_labels(
        model=backbone,
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_label_keys=["target"],
        header=f"(Linear eval: {args.eval_target_source} test split)",
    )
    metrics = _eval(
        classifier=classifier,
        dataset=TensorDataset(test_features, test_labels["target"]),
        pred_threshold=args.eval_pred_threshold,
        batch_size=args.batch_size,
        classes=dataset.classes,
    )
    # Save results
    metrics_filename = "linear_eval_{}_{}l_{}d.csv".format(
        args.eval_target_source,
        args.eval_num_layers,
        args.eval_hidden_dim,
    )
    save_dir = args.experiment_dir / args.eval_save_folder
    save_dir.mkdir(exist_ok=True)
    eval_utils.save_metrics_dataframe_to_csv(
        metrics_df=metrics,
        save_file=(save_dir / metrics_filename),
    )
    # Report duration
    duration = int(time.monotonic() - start_time)
    print(f"Linear evaluation ended in {utils.get_time_string(duration)}!")


def _check_args(args):
    args.save_dir = (
        args.experiment_dir if args.save_dir is None else args.save_dir
    )
    args.save_folder = (
        "linear_eval_{}_{}l_{}d".format(
            args.eval_target_source,
            args.eval_num_layers,
            args.eval_hidden_dim,
        )
        if args.save_folder is None
        else args.save_folder
    )
    args = utils.check_training_args(args)
    return args


def _get_args_parser():
    parser = argparse.ArgumentParser(
        description="Run linear evaluation on a pretrained backbone.",
    )
    parser = utils.add_data_arg(parser)
    parser = eval_utils.add_eval_args(parser)
    parser = utils.add_training_args(
        parser,
        sources_subset_default=0,
        batch_size_default=16,
    )
    parser.add_argument(
        "--eval-pred-threshold",
        type=float,
        default=0.5,
        help="Prediction threshold for linear evaluation.",
    )
    parser.add_argument(
        "--eval-num-layers",
        type=int,
        default=1,
        help="Number of linear layers in the classification head network.",
    )
    parser.add_argument(
        "--eval-hidden-dim",
        type=int,
        default=2048,
        help="Hidden layers dimension in classification head network.",
    )
    return parser


def _run():
    try:
        parser = _get_args_parser()
        args = parser.parse_args()
        _main(_check_args(args))
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())

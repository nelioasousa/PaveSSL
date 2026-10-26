import argparse
import time
from math import isfinite

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import Resize

from pavessl import GLOBAL_SCALE, GLOBAL_SIZE, models, utils
from pavessl.data import RDD2022, get_augmentation
from pavessl.scripts.eval import eval_utils
from pavessl.scripts.pretrain import pretrain_utils
from pavessl.scripts.utils.plot_results import (
    plot_loss_history,
    plot_valid_history,
)


@torch.inference_mode()
def _predict(
    classifier: torch.nn.Module,
    dataset: Dataset,
    thresholds: float | list[float],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: torch.device,
):
    data_loader = DataLoader(
        dataset=dataset,
        num_workers=num_workers,
        pin_memory=pin_memory,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    preds = None
    targets = None
    for i, (images, labels) in enumerate(data_loader):
        images = images.to(device=device, non_blocking=True)
        batch_targets = labels["target"].to(device=device, non_blocking=True)
        batch_preds = classifier(images)
        if preds is None:
            preds = torch.empty(
                size=(len(dataset), batch_preds.shape[-1]),
                dtype=batch_preds.dtype,
                device=device,
            )
            targets = torch.empty(
                size=(len(dataset), batch_targets.shape[-1]),
                dtype=batch_targets.dtype,
                device=device,
            )
        start = i * batch_size
        end = start + len(batch_preds)
        preds[start:end].copy_(batch_preds)
        targets[start:end].copy_(batch_targets)
    preds = preds.sigmoid_()
    if isinstance(thresholds, float):
        preds = torch.where(preds > thresholds, 1.0, 0.0)
    else:
        thresholds = torch.tensor(thresholds).reshape(-1, 1, 1)
        thresholds = thresholds.to(device=device, non_blocking=True)
        preds = preds.unsqueeze_(0).repeat(thresholds.numel(), 1, 1)
        preds = torch.where(preds > thresholds, 1.0, 0.0)
    return preds.cpu().numpy(), targets.cpu().numpy()


def _eval(
    classifier: torch.nn.Module,
    thresholds: float | list[float],
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: torch.device,
    classes: list[str] | None = None,
):
    classifier.eval()
    preds, targets = _predict(
        classifier=classifier,
        dataset=dataset,
        thresholds=thresholds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        device=device,
    )
    if isinstance(thresholds, float):
        metrics = eval_utils.multi_value_classification_metrics(
            preds=preds,
            targets=targets,
        )
        return {
            metric: value
            for metric, value in metrics.items()
            if isinstance(value, float)
        }
    return [
        eval_utils.multi_value_classification_metrics(
            preds=preds[i],
            targets=targets,
            classes=classes,
        )
        for i in range(len(thresholds))
    ]


def _get_data(args):
    train_loader = DataLoader(
        dataset=RDD2022(
            data_path=args.data,
            sources=args.eval_source,
            split="train",
            task="classification",
            transform=get_augmentation(GLOBAL_SIZE, GLOBAL_SCALE),
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin,
        shuffle=True,
        drop_last=True,
        persistent_workers=True,
    )
    valid_dataset = RDD2022(
        data_path=args.data,
        sources=args.eval_source,
        split="valid",
        task="classification",
        transform=Resize(GLOBAL_SIZE),
    )
    return train_loader, valid_dataset


def _get_classifier(num_classes, device, args):
    if args.experiment_dir is None:
        backbone = models.build_backbone(
            args.start_points_dir / "resnet50_backbone.pth"
        )
    else:
        backbone = eval_utils.load_experiment_backbone(
            experiment_dir=args.experiment_dir,
            checkpoint_key=args.checkpoint_key,
        )
    classifier = models.ClassificationNetwork(
        backbone=backbone,
        num_classes=num_classes,
        head_num_layers=args.eval_num_layers,
        head_hidden_dim=args.eval_hidden_dim,
    )
    head_state_path = args.start_points_dir / (
        "{}l_{}d_{}_linear_head.pth".format(
            args.eval_num_layers,
            args.eval_hidden_dim,
            args.eval_source,
        )
    )
    if head_state_path.is_file():
        classifier.head.load_state_dict(
            state_dict=torch.load(
                f=head_state_path,
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
    else:
        torch.save(classifier.head.state_dict(), head_state_path)
    # Freeze backbone during the whole training
    # Setting requires_grad here will prevent the...
    # ...parameters from being added to the optimizer
    if args.freeze == args.epochs:
        for p in classifier.backbone.parameters():
            p.requires_grad = False
    backbone_nparams = utils.get_num_parameters(classifier.backbone)
    classifier_nparams = utils.get_num_parameters(classifier)
    print(f"Backbone # params: {backbone_nparams:,d}")
    print(f"Network  # params: {classifier_nparams:,d}")
    print(f"Head     # params: {(classifier_nparams - backbone_nparams):,d}")
    return classifier.to(device=device, non_blocking=True)


def _get_loss_function(device):
    loss_fn = models.MultiValueClassificationLoss()
    return loss_fn.to(device=device, non_blocking=True)


def _train_one_epoch(
    classifier,
    loss_fn,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    epoch,
    device,
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
    header = f"(Train) Epoch: [{epoch_str}/{num_epochs}]"
    log_every = max(5, 5 * len(data_loader) // (5 * 20))
    for it, (images, labels) in enumerate(
        metric_logger.log_every(data_loader, log_every, header)
    ):
        images = images.to(device=device, non_blocking=True)
        targets = labels["target"].to(device=device, non_blocking=True)
        classes_weights = torch.where(
            labels["target"] > 0,
            labels["pos_weights"],
            labels["neg_weights"],
        )
        classes_weights = classes_weights.to(device=device, non_blocking=True)
        classes_weights.div_(classes_weights.sum(dim=-1, keepdim=True))
        # Only the first group is regularized
        it = len(data_loader) * epoch + it
        optimizer.param_groups[0]["weight_decay"] = wd_schedule[it]
        optimizer.param_groups[0]["lr"] = lr_schedule[it]
        optimizer.param_groups[1]["lr"] = lr_schedule[it]
        # Foward pass and loss computation
        logits = classifier(images)
        loss = loss_fn(
            logits,
            targets,
            classes_weights,
        )
        # Check for loss explosion
        loss_value = loss.item()
        if not isfinite(loss_value):
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


def _save_and_plot_histories(
    loss_history,
    valid_history,
    valid_metrics,
    save_dir,
):
    np.savetxt(
        fname=(save_dir / "loss_history.csv"),
        X=loss_history,
        delimiter=",",
        fmt="%20.10e",
        header="min,max,avg,duration",
        comments="",
    )
    plot_loss_history(
        loss_history_df=pd.DataFrame(
            loss_history,
            columns=["min", "max", "avg", "duration"],
        ),
        metric_name="Binary Cross Entropy",
        metric_unit="1",
        save_file=(save_dir / "loss_history.png"),
    )
    np.savetxt(
        fname=(save_dir / "valid_history.csv"),
        X=valid_history,
        delimiter=",",
        fmt=(["%.4f"] * len(valid_metrics)),
        header=(",".join(valid_metrics)),
        comments="",
    )
    plot_valid_history(
        valid_history_df=pd.DataFrame(
            valid_history,
            columns=valid_metrics,
        ),
        save_file=(save_dir / "valid_history.png"),
    )


def _main(args):
    device = utils.get_device()
    # Check run
    stored_args = pretrain_utils.check_run(args)
    if args is not stored_args:
        print(
            "Run arguments restored.\n"
            "Ignoring any argument changes.\n"
            "Continuing previous training run."
        )
        stored_args.pin = args.pin
        args = stored_args
    # Restore training
    train_loader, valid_dataset = _get_data(args)
    classifier = _get_classifier(
        len(train_loader.dataset.classes),
        device,
        args,
    )
    loss_fn = _get_loss_function(device)
    optimizer = models.get_optimizer(classifier, args.optim)
    if args.freeze:
        for p in classifier.backbone.parameters():
            p.requires_grad = False
    # Restore training vars if any
    run_variables = {
        "epoch": 0,
        "loss_history": None,
        "valid_history": None,
        "valid_metrics": None,
    }
    utils.restart_from_checkpoint(
        ckp_path=(args.results_dir / "checkpoint.pth"),
        run_variables=run_variables,
        classifier=classifier,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )
    # Run variables
    start_epoch = run_variables["epoch"]
    if start_epoch == args.epochs:
        print(f'Training at "{args.results_dir}" already finished.')
        return
    print(f'Saving results in "{args.results_dir}".')
    valid_history = run_variables["valid_history"]
    valid_metrics = run_variables["valid_metrics"]
    loss_history = (
        np.zeros((args.epochs, 4), dtype=np.float32)
        if run_variables["loss_history"] is None
        else run_variables["loss_history"]
    )
    # Schedulers
    lr_schedule = utils.cosine_scheduler(
        base_value=(args.lr * args.batch_size / 256.0),
        final_value=(args.lr_end * args.batch_size / 256.0),
        epochs=args.epochs,
        niter_per_ep=len(train_loader),
        warmup_epochs=args.lr_warmup_size,
    )
    wd_schedule = utils.cosine_scheduler(
        base_value=args.weight_decay,
        final_value=args.weight_decay_end,
        epochs=args.epochs,
        niter_per_ep=len(train_loader),
    )
    # Training
    training_start = time.monotonic()
    for epoch in range(start_epoch, args.epochs):
        utils.set_random_seeds(42 + epoch)
        if epoch == args.freeze:
            for p in classifier.backbone.parameters():
                p.requires_grad = True
        logger = _train_one_epoch(
            classifier=classifier,
            loss_fn=loss_fn,
            data_loader=train_loader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
            epoch=epoch,
            device=device,
            num_epochs=args.epochs,
        )
        loss_history[epoch] = [
            logger.meters["loss"].min_value,
            logger.meters["loss"].max_value,
            logger.meters["loss"].global_avg,
            logger.meters["iter"].total_accumulated,
        ]
        # Run validation
        eval_start = time.monotonic()
        eval_results = _eval(
            classifier=classifier,
            thresholds=0.5,
            dataset=valid_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin,
            device=device,
        )
        if valid_metrics is None:
            valid_metrics = sorted(eval_results)
            valid_history = np.zeros(
                shape=(args.epochs, len(valid_metrics)),
                dtype=np.float32,
            )
        valid_history[epoch] = [
            eval_results[metric] for metric in valid_metrics
        ]
        duration = int(time.monotonic() - eval_start)
        print(f"Validation took {utils.get_time_string(duration)}.")
        # Save checkpoints
        states_dict = {
            "classifier": classifier.state_dict(),
            "loss_fn": loss_fn.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "loss_history": loss_history,
            "valid_history": valid_history,
            "valid_metrics": valid_metrics,
        }
        utils.save_states(
            save_file=(args.results_dir / "checkpoint.pth"),
            states_dict=states_dict,
            epoch=(
                None
                if states_dict["epoch"] % args.checkpoint_freq
                else states_dict["epoch"]
            ),
        )
    # Save histories
    _save_and_plot_histories(
        loss_history,
        valid_history,
        valid_metrics,
        args.results_dir,
    )
    # Evaluate on test set
    test_dataset = RDD2022(
        data_path=args.data,
        sources=args.eval_source,
        split="test",
        task="classification",
        transform=Resize(GLOBAL_SIZE),
    )
    test_metrics = _eval(
        classifier=classifier,
        thresholds=args.eval_thresholds,
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin,
        device=device,
        classes=test_dataset.classes,
    )
    test_results_dir = args.results_dir / "eval"
    test_results_dir.mkdir(exist_ok=True)
    for i, thresh in enumerate(args.eval_thresholds):
        eval_utils.save_metrics_dataframe_to_csv(
            metrics_df=test_metrics[i],
            save_file=(test_results_dir / f"thresh{int(100 * thresh)}.csv"),
        )
    eval_utils.plot_metrics(
        metrics_dataframes=test_metrics,
        thresholds=args.eval_thresholds,
        save_dir=(test_results_dir / "plots"),
    )
    # Report duration
    duration = int(time.monotonic() - training_start)
    print(f"Training ended in {utils.get_time_string(duration)}!")


def _check_args(args):
    args.save_dir = args.experiment_dir / "classification"
    args.freeze = (
        args.epochs if args.freeze < 0 else min(args.freeze, args.epochs)
    )
    return utils.check_training_args(args)


def _get_args_parser():
    parser = argparse.ArgumentParser(
        description="Finetune backbone in a classification task.",
    )
    parser = utils.add_data_arg(parser)
    parser = eval_utils.add_eval_args(parser)
    parser = utils.add_training_args(
        parser=parser,
        save_folder_default="default",
        sources_subset_default=0,
        epochs_default=100,
        batch_size_default=16,
        weight_decay_default=0.05,
        weight_decay_end_default=0.4,
    )
    parser.add_argument(
        "--eval-thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9],
        help="Prediction threshold for linear evaluation.",
    )
    parser.add_argument(
        "--eval-num-layers",
        type=int,
        default=4,
        help="Number of linear layers in the classification head network.",
    )
    parser.add_argument(
        "--eval-hidden-dim",
        type=int,
        default=4096,
        help="Hidden layers dimension in classification head network.",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Freeze backbone parameters for N epochs.",
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

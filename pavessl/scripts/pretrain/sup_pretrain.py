import argparse
import math
import pickle
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from pavessl import models, utils
from pavessl.data import RDD2022, get_augmentation
from pavessl.scripts.eval.knn_eval import knn_eval


def _save_args(args):
    with (args.results_dir / "run.args").open("wb") as args_file:
        pickle.dump(obj=args, file=args_file, protocol=5)


def _check_run(args):
    try:
        with (args.results_dir / "run.args").open("rb") as args_file:
            return pickle.load(args_file)
    except FileNotFoundError:
        _save_args(args)
        return args


def _get_train_data_loader(args):
    train_dataset = RDD2022(
        data_path=args.data,
        sources=args.sources,
        split="train",
        task="classification",
        transform=get_augmentation(args.image_size, args.crop_scale),
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_dataset.get_balanced_sampler(args.batch_size),
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
    )
    return train_loader


def _get_classifier(num_classes, device, args):
    backbone = models.build_resnet50_backbone(
        args.start_points_dir / "resnet50_backbone.pth",
    )
    print(f"ResNet-50 # parameters: {utils.get_num_parameters(backbone):,d}")
    classifier = models.ClassificationNetwork(
        backbone=backbone,
        num_classes=num_classes,
    )
    print(f"Classifier # parameters: {utils.get_num_parameters(classifier):,d}")
    return classifier.to(device=device, non_blocking=True)


def _get_loss_function(device):
    loss_fn = models.MultiValueClassificationLoss()
    return loss_fn.to(device=device, non_blocking=True)


def _get_optimizer(model, args):
    params_groups = utils.get_params_groups(model)
    if args.optim == "adamw":
        return torch.optim.AdamW(params_groups)
    if args.optim == "sgd":
        return torch.optim.SGD(params_groups, momentum=0.9)
    raise ValueError(f'Optimizer "{args.optim}" not supported')


def _train_one_epoch(
    classifier,
    loss_fn,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    epoch,
    device,
    args,
):
    classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "loss",
        utils.SmoothedValue(fmt="({value:10.6f} / {local_avg:10.6f} avg)"),
    )
    metric_logger.add_meter("lr", utils.SmoothedValue(fmt="{value:.2e}"))
    metric_logger.add_meter("wd", utils.SmoothedValue(fmt="{value:.2e}"))
    epoch_str = str(epoch + 1).rjust(len(str(args.epochs)), "0")
    header = f"(Train) Epoch: [{epoch_str}/{args.epochs}]"
    # More or less 50 log lines per iteration
    log_every = max(5, 5 * len(data_loader) // (5 * 50))
    for it, (images, labels) in enumerate(
        metric_logger.log_every(data_loader, log_every, header)
    ):
        images = images.to(device=device, non_blocking=True)
        targets = labels["target"]
        targets = targets.to(device=device, non_blocking=True)
        weights = labels["weight"]
        weights = weights.to(device=device, non_blocking=True)
        weights.div_(weights.sum())
        mask = labels["mask"]
        mask = mask.to(device=device, non_blocking=True)
        it = len(data_loader) * epoch + it  # Global training iteration
        # Only the first group is regularized
        optimizer.param_groups[0]["weight_decay"] = wd_schedule[it]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[it]
        # Foward pass and loss computation
        logits = classifier(images)
        loss = loss_fn(
            logits,
            targets,
            weights,
            mask,
        )
        # Loss explosion, unstable training
        loss_value = loss.item()  # CPU-GPU synchronization
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
    # print("Averaged stats:", metric_logger)
    return metric_logger.meters["loss"]


def _eval(model, header, args):
    model.eval()
    eval_results = knn_eval(
        model=model.backbone,
        ks=args.eval_ks,
        distance_temperature=args.eval_dist_temperature,
        pred_threshold=args.eval_pred_threshold,
        data_path=args.data,
        target_sources=args.sources,
        crops_target_size=args.eval_crops_target_size,
        crops_overlap=args.eval_crops_overlap,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        simplified=True,
        logging_header=header,
    )
    return np.array(
        [
            eval_results[source][metric][i]
            for source in args.sources
            for i in range(len(args.eval_ks))
            for metric in args.eval_metrics
        ]
    )


def _pretrain(args):
    utils.set_random_seeds(42)
    stored_args = _check_run(args)
    device = utils.get_device()
    # Data
    train_loader = _get_train_data_loader(stored_args)
    # Classifier
    classifier = _get_classifier(
        len(train_loader.dataset.classes),
        device,
        stored_args,
    )
    # Loss function
    loss_fn = _get_loss_function(device)
    # Optimizer
    optimizer = _get_optimizer(classifier, stored_args)
    # Resume training
    vars_to_restore = {
        "epoch": 0,
        "loss_history": None,
        "valid_history": None,
    }
    utils.restart_from_checkpoint(
        ckp_path=(stored_args.results_dir / "checkpoint.pth"),
        run_variables=vars_to_restore,
        classifier=classifier,
        loss=loss_fn,
        optimizer=optimizer,
    )
    # Check run state
    if vars_to_restore["epoch"] == stored_args.epochs:
        if not args.increment:
            print(f'Training at "{stored_args.results_dir}" already finished.')
            print("Use --increment to increment upon this training run.")
            return
        # Increment finished training run
        args.save_folder = f"{args.save_folder}{args.increment_suffix}"
        args.results_dir = args.save_dir / args.save_folder
        args.results_dir.mkdir(exist_ok=True, parents=True)
        if (args.results_dir / "run.args").is_file():
            raise ValueError(f'Directory "{args.results_dir}" already in use.')
        _save_args(args)
        # Re-build with the new arguments
        train_loader = _get_train_data_loader(args)
        if classifier.num_classes != len(train_loader.dataset.classes):
            raise ValueError(
                f"Number of classes changed from {classifier.num_classes} "
                f"to {len(train_loader.dataset.classes)}"
            )
        loss_fn = _get_loss_function(device)
        optimizer = _get_optimizer(classifier, args)
        # Unset run variables
        vars_to_restore["epoch"] = 0
        vars_to_restore["loss_history"] = None
        vars_to_restore["valid_history"] = None
    else:
        if vars_to_restore["epoch"]:
            print("Run arguments restored. Continuing training...")
        args = stored_args
    print(f'Saving results in "{args.results_dir}".')
    # Restore loss and validation histories
    loss_history = (
        np.zeros((args.epochs, 3), dtype=np.float32)
        if vars_to_restore["loss_history"] is None
        else vars_to_restore["loss_history"]
    )
    num_metrics = len(args.sources) * len(args.eval_ks) * len(args.eval_metrics)
    valid_history = (
        np.zeros(
            (
                1 + math.ceil((args.epochs - 1) / args.validation_freq),
                num_metrics + 1,
            ),
            dtype=np.float32,
        )
        if vars_to_restore["valid_history"] is None
        else vars_to_restore["valid_history"]
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
    start_time = time.monotonic()
    for epoch in range(vars_to_restore["epoch"], args.epochs):
        loss_log = _train_one_epoch(
            classifier=classifier,
            loss_fn=loss_fn,
            data_loader=train_loader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
            epoch=epoch,
            device=device,
            args=args,
        )
        loss_history[epoch] = [
            loss_log.min_value,
            loss_log.max_value,
            loss_log.global_avg,
        ]
        # Run validation
        if not (epoch % args.validation_freq) or (epoch + 1) == args.epochs:
            row_index = math.ceil(epoch / args.validation_freq)
            valid_history[row_index, 0] = epoch + 1
            valid_history[row_index, 1:] = _eval(
                model=classifier,
                header="(Eval/Classifier)",
                args=args,
            )
        # Save checkpoints
        if not (epoch % args.checkpoint_freq) or (epoch + 1) == args.epochs:
            states_dict = {
                "classifier": classifier.state_dict(),
                "loss": loss_fn.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "loss_history": loss_history,
                "valid_history": valid_history,
            }
            utils.save_states(
                save_dir=args.results_dir,
                states_dict=states_dict,
                epoch=epoch,
            )
    # Save histories and report run duration
    np.savetxt(
        fname=(args.results_dir / "loss_history.csv"),
        X=loss_history,
        delimiter=",",
        fmt="%20.10e",
        header="min,max,avg",
        comments="",
    )
    valid_headers = ["epoch"]
    valid_headers.extend(
        [
            f"{source}__{k}__{metric}"
            for source in args.sources
            for k in args.eval_ks
            for metric in args.eval_metrics
        ]
    )
    np.savetxt(
        fname=(args.results_dir / "valid_history.csv"),
        X=valid_history,
        delimiter=",",
        fmt=(["%04d"] + ["%.4f"] * (len(valid_headers) - 1)),
        header=(",".join(valid_headers)),
        comments="",
    )
    train_duration = int(time.monotonic() - start_time)
    print(f"Training ended in {utils.get_time_string(train_duration)}!")


def check_args(args: argparse.Namespace) -> argparse.Namespace:
    args = utils.check_data_args(args)
    args = utils.check_training_args(args)
    return args


def get_args_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Supervised backbone pre-training using multi-value classification."
        ),
    )
    # Data arguments
    parser = utils.add_data_args(parser, "sup_pretrain")
    # Training arguments
    parser = utils.add_training_args(parser)
    # Input transformation
    parser.add_argument(
        "--image-size",
        type=int,
        default=448,
        help="Input image dimension.",
    )
    parser.add_argument(
        "--crop-scale",
        type=float,
        nargs=2,
        default=(0.85, 1.0),
        help="Min-max area range for image cropping.",
    )
    # kNN evaluation arguments
    parser = utils.add_knn_eval_args(parser, prefix="eval-")
    return parser


def run():
    try:
        parser = get_args_parser()
        args = parser.parse_args()
        _pretrain(check_args(args))
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(run())

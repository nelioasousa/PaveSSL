import argparse
import time
from itertools import product
from math import ceil, isfinite

import numpy as np
from torch.utils.data import DataLoader

from pavessl import models, utils
from pavessl.data import RDD2022, get_augmentation
from pavessl.scripts.pretrain import pretrain_utils


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
    classifier = models.build_classification_network(
        network_state_path=(args.start_points_dir / "sup_classifier_net.pth"),
        backbone_state_path=(args.start_points_dir / "resnet50_backbone.pth"),
        num_classes=num_classes,
    )
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
        optimizer.param_groups[0]["lr"] = lr_schedule[it]
        optimizer.param_groups[1]["lr"] = lr_schedule[it]
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
    # print("Averaged stats:", metric_logger)
    return metric_logger.meters["loss"]


def pretrain(args):
    utils.set_random_seeds(42)
    device = utils.get_device()
    # Check run
    stored_args = pretrain_utils.check_run(args)
    if args is not stored_args:
        print(
            "Run arguments restored.\n"
            "Ignoring any argument changes.\n"
            "Continuing previous training run."
        )
        args = stored_args
    # Restore training
    train_loader = _get_train_data_loader(args)
    classifier = _get_classifier(
        len(train_loader.dataset.classes),
        device,
        args,
    )
    loss_fn = _get_loss_function(device)
    optimizer = models.get_optimizer(classifier, args.optim)
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
        np.zeros((args.epochs, 3), dtype=np.float32)
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
        loss_log = _train_one_epoch(
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
            loss_log.min_value,
            loss_log.max_value,
            loss_log.global_avg,
        ]
        # Run validation
        if not (epoch % args.validation_freq) or (epoch + 1) == args.epochs:
            eval_start = time.monotonic()
            classifier_eval = pretrain_utils.backbone_knn_eval(
                backbone=classifier.backbone,
                data_path=args.data,
                target_sources=args.sources,
                ks=args.knn_ks,
                distance_temperature=args.knn_dist_temperature,
                pred_threshold=args.knn_pred_threshold,
                crops_target_size=args.knn_crops_target_size,
                crops_overlap=args.knn_crops_overlap,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                logging_header="(Eval/Classifier)",
            )
            if valid_metrics is None:
                valid_metrics = sorted(classifier_eval[args.sources[0]][0])
                num_rows = 1 + ceil((args.epochs - 1) / args.validation_freq)
                num_cols = 1 + len(valid_metrics) * len(args.sources) * len(
                    args.knn_ks
                )
                valid_history = np.zeros((num_rows, num_cols), dtype=np.float32)
            row_index = ceil(epoch / args.validation_freq)
            valid_history[row_index, 0] = epoch + 1
            valid_history[row_index, 1:] = [
                classifier_eval[source][k_idx][metric]
                for source, k_idx, metric in product(
                    args.sources, range(len(args.knn_ks)), valid_metrics
                )
            ]
            duration = int(time.monotonic() - eval_start)
            print(f"Validation ended in {utils.get_time_string(duration)}!")
        # Save checkpoints
        if not (epoch % args.checkpoint_freq) or (epoch + 1) == args.epochs:
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
            f"classifier__{source}__{k}__{metric}"
            for source, k, metric in product(
                args.sources, args.knn_ks, valid_metrics
            )
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
    duration = int(time.monotonic() - training_start)
    print(f"Training ended in {utils.get_time_string(duration)}!")
    return classifier


def check_args(args: argparse.Namespace) -> argparse.Namespace:
    args = utils.check_training_args(args)
    return args


def get_args_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Supervised backbone pre-training using multi-value classification."
        ),
    )
    # Data and training arguments
    parser = utils.add_data_arg(parser)
    parser = utils.add_training_args(
        parser=parser,
        save_dir_default="data/results/sup_pretrain",
        lr_default=1e-3,
        lr_end_default=1e-5,
    )
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
    parser = utils.add_knn_eval_args(parser)
    return parser


def run():
    try:
        parser = get_args_parser()
        args = parser.parse_args()
        pretrain(check_args(args))
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(run())

import argparse
import time
from itertools import product
from math import isfinite

import numpy as np
import torch
from torch.utils.data import DataLoader

from pavessl import models, utils, GLOBAL_SIZE, GLOBAL_SCALE
from pavessl.data import RDD2022, get_augmentation
from pavessl.scripts.eval.knn_eval import add_knn_eval_args
from pavessl.scripts.pretrain import pretrain_utils


def _get_train_data_loader(args):
    train_dataset = RDD2022(
        data_path=args.data,
        sources=args.sources,
        split="train",
        task="classification",
        transform=get_augmentation(GLOBAL_SIZE, GLOBAL_SCALE),
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_dataset.get_balanced_sampler(args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin,
        persistent_workers=True,
    )
    return train_loader


def _get_classifier(num_classes, device, args):
    previous_source = max(1, args.sources_subset - 1)
    state_filename = f"sup_classifier_src{previous_source}_net.pth"
    classifier = models.build_classification_network(
        network_state_path=(args.start_points_dir / state_filename),
        backbone_state_path=(args.start_points_dir / "resnet50_backbone.pth"),
        num_classes=num_classes,
    )
    state_filename = f"sup_classifier_src{args.sources_subset}_net.pth"
    torch.save(classifier.state_dict(), args.start_points_dir / state_filename)
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
        samples_weight = labels["source_weight"].to(
            device=device,
            non_blocking=True,
        )
        classes_weights = torch.where(
            labels["target"] > 0,
            labels["pos_weights"],
            labels["neg_weights"],
        )
        classes_weights = classes_weights.to(device=device, non_blocking=True)
        samples_weight.div_(samples_weight.sum())
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
            samples_weight,
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


def _pretrain(args):
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
        classifier_eval = pretrain_utils.backbone_knn_eval(
            backbone=classifier.backbone,
            data_path=args.data,
            target_sources=args.sources,
            ks=args.knn_ks,
            distance_temperature=args.knn_dist_temperature,
            pred_threshold=args.knn_pred_threshold,
            image_size=GLOBAL_SIZE,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin,
            logging_header="(Eval/Classifier)",
        )
        if valid_metrics is None:
            valid_metrics = sorted(classifier_eval[args.sources[0]][0])
            num_cols = len(valid_metrics) * len(args.sources) * len(args.knn_ks)
            valid_history = np.zeros((args.epochs, num_cols), dtype=np.float32)
        valid_history[epoch] = [
            classifier_eval[source][k_idx][metric]
            for source, k_idx, metric in product(
                args.sources,
                range(len(args.knn_ks)),
                valid_metrics,
            )
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
    # Save histories and report run duration
    np.savetxt(
        fname=(args.results_dir / "loss_history.csv"),
        X=loss_history,
        delimiter=",",
        fmt="%20.10e",
        header="min,max,avg,duration",
        comments="",
    )
    valid_headers = [
        f"classifier__{source}__{k}__{metric}"
        for source, k, metric in product(
            args.sources,
            args.knn_ks,
            valid_metrics,
        )
    ]
    np.savetxt(
        fname=(args.results_dir / "valid_history.csv"),
        X=valid_history,
        delimiter=",",
        fmt=(["%.4f"] * len(valid_headers)),
        header=(",".join(valid_headers)),
        comments="",
    )
    duration = int(time.monotonic() - training_start)
    print(f"Training ended in {utils.get_time_string(duration)}!")
    return classifier


def _check_args(args):
    args = utils.check_training_args(args)
    return args


def _get_args_parser():
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
        save_folder_default="test",
    )
    parser = add_knn_eval_args(parser)
    return parser


def _run():
    try:
        parser = _get_args_parser()
        args = parser.parse_args()
        _pretrain(_check_args(args))
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())

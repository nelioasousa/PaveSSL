import argparse
import math
import pickle
import time

import numpy as np
import torch
import torchvision.models
from torch import nn
from torch.utils.data import DataLoader

from pavessl import models, utils
from pavessl.data import RDD2022, SSLMultiviewTransform, get_rdd2022_sources
from pavessl.scripts.knn_eval import knn_eval


def _get_models(device, args):
    if args.dino_base:
        backbone = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
    else:
        backbone = torchvision.models.resnet50()
        backbone.fc = nn.Identity()
    student = models.DualHeadProjectionNetwork(
        backbone=backbone,
        projection_dim=args.projection_dim,
    )
    teacher = models.SingleHeadProjectionNetwork(
        backbone=backbone,
        projection_dim=args.projection_dim,
        head_norm_last_layer=False,
    )
    student = student.to(device=device, non_blocking=True)
    teacher = teacher.to(device=device, non_blocking=True)
    # Teacher and Student start with the same weights
    teacher.load_state_dict(student.state_dict(), strict=False)
    # There is no backpropagation through the teacher
    for p in teacher.parameters():
        p.requires_grad = False
    return student, teacher


def _get_loss_function(device, args):
    loss_fn = models.SSLLoss(
        projection_dim=args.projection_dim,
        num_local_views=args.num_local_views,
        teacher_temp=args.teacher_temp,
        teacher_temp_start=args.teacher_temp_warmup_start,
        teacher_temp_warmup=args.teacher_temp_warmup_size,
        num_epochs=args.epochs,
        student_temp=args.student_temp,
        center_momentum=args.center_momentum,
    )
    return loss_fn.to(device=device, non_blocking=True)


def _get_optimizer(model, args):
    params_groups = utils.get_params_groups(model)
    if args.optim == "adamw":
        return torch.optim.AdamW(params_groups)
    if args.optim == "sgd":
        return torch.optim.SGD(params_groups, momentum=0.9)
    raise ValueError(f'Optimizer "{args.optim}" not supported')


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


def pretrain(args):
    utils.set_random_seeds(42)
    args = _check_args(args)
    stored_args = _check_run(args)
    device = utils.get_device()
    # Models
    student, teacher = _get_models(device, stored_args)
    # Data
    train_loader = _get_train_data_loader(stored_args)
    # Loss function
    loss_fn = _get_loss_function(device, stored_args)
    # Optimizer
    optimizer = _get_optimizer(student, stored_args)
    # k-NN validation
    valid_ks = [10, 25, 50, 100]
    valid_metrics = ("accuracy", "sensitivity", "specificity")
    # Resume training
    vars_to_restore = {
        "epoch": 0,
        "loss_history": None,
        "valid_history": None,
    }
    utils.restart_from_checkpoint(
        ckp_path=(stored_args.results_dir / "checkpoint.pth"),
        run_variables=vars_to_restore,
        student=student,
        teacher=teacher,
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
        # Sync some arguments
        args.projection_dim = stored_args.projection_dim
        args.dino_base = False
        _save_args(args)
        # Re-build with the new arguments
        train_loader = _get_train_data_loader(args)
        loss_fn = _get_loss_function(device, args)
        optimizer = _get_optimizer(student, args)
        # Unset run variables
        vars_to_restore["epoch"] = 0
        vars_to_restore["loss_history"] = None
        vars_to_restore["valid_history"] = None
    else:
        if vars_to_restore["epoch"]:
            print(
                "Run arguments restored.",
                f"Continuing training from epoch {vars_to_restore['epoch']}.",
            )
        args = stored_args
    print(f'Saving results in "{args.results_dir}"')
    # Restore loss and validation histories
    loss_history = (
        np.zeros((args.epochs, 3), dtype=np.float32)
        if vars_to_restore["loss_history"] is None
        else vars_to_restore["loss_history"]
    )
    valid_history = (
        np.zeros(
            (
                1 + math.ceil((args.epochs - 1) / args.validation_freq),
                len(args.sources) * len(valid_ks) * len(valid_metrics),
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
        start_warmup_value=(args.lr_warmup_start * args.batch_size / 256.0),
    )
    wd_schedule = utils.cosine_scheduler(
        base_value=args.weight_decay,
        final_value=args.weight_decay_end,
        epochs=args.epochs,
        niter_per_ep=len(train_loader),
    )
    momentum_schedule = utils.cosine_scheduler(
        base_value=args.teacher_momentum,
        final_value=1.0,
        epochs=args.epochs,
        niter_per_ep=len(train_loader),
    )
    # Training
    start_time = time.monotonic()
    for epoch in range(vars_to_restore["epoch"], args.epochs):
        loss_log = _train_one_epoch(
            student=student,
            teacher=teacher,
            loss_fn=loss_fn,
            data_loader=train_loader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
            momentum_schedule=momentum_schedule,
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
            valid_history[math.ceil(epoch / args.validation_freq)] = _eval(
                model=teacher,
                ks=valid_ks,
                metrics=valid_metrics,
                args=args,
            )
        # Save checkpoints
        states_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss_fn.state_dict(),
            "epoch": epoch + 1,
            "loss_history": loss_history,
            "valid_history": valid_history,
        }
        utils.save_states(
            save_file=(args.results_dir / "checkpoint.pth"),
            states_dict=states_dict,
        )
        if not (epoch % args.checkpoint_freq):
            utils.save_states(
                save_file=(args.results_dir / f"checkpoint_{epoch:04d}.pth"),
                states_dict=states_dict,
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
    np.savetxt(
        fname=(args.results_dir / "valid_history.csv"),
        X=valid_history,
        delimiter=",",
        fmt="%.4f",
        header=",".join(
            f"{source}_{k}_{metric}"
            for source in args.sources
            for k in valid_ks
            for metric in valid_metrics
        ),
        comments="",
    )
    train_duration = int(time.monotonic() - start_time)
    print(f"Training ended in {utils.get_time_string(train_duration)}!")


def _train_one_epoch(
    student,
    teacher,
    loss_fn,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    epoch,
    device,
    args,
):
    student.train()
    teacher.train()  # Necessary for BatchNorm updates
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "loss",
        utils.SmoothedValue(fmt="({value:10.6f} / {local_avg:10.6f} avg)"),
    )
    metric_logger.add_meter("lr", utils.SmoothedValue(fmt="{value:.2e}"))
    metric_logger.add_meter("wd", utils.SmoothedValue(fmt="{value:.2e}"))
    epoch_str = str(epoch + 1).rjust(len(str(args.epochs)), "0")
    header = f"(Train) Epoch: [{epoch_str}/{args.epochs}]"
    for it, (global_views, local_views, weights) in enumerate(
        metric_logger.log_every(data_loader, 10, header)
    ):
        global_views = global_views.to(device=device, non_blocking=True)
        local_views = local_views.to(device=device, non_blocking=True)
        weights = weights.to(device=device, non_blocking=True)
        weights.div_(weights.sum())
        it = len(data_loader) * epoch + it  # Global training iteration
        # Only the first group is regularized
        optimizer.param_groups[0]["weight_decay"] = wd_schedule[it]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[it]
        # Teacher and student forward passes && compute loss
        teacher_out = teacher(global_views)
        student_g_out, student_l_out = student(global_views, local_views)
        loss = loss_fn(
            teacher_out,
            student_g_out,
            student_l_out,
            weights,
            epoch,
        )
        # Loss explosion, unstable training
        loss_value = loss.item()  # CPU-GPU synchronization
        if not math.isfinite(loss_value):
            raise SystemExit(f"Loss exploded with value {loss_value}")
        # Student update
        optimizer.zero_grad()
        loss.backward()
        if epoch < args.freeze_last_layer_for:
            # Freeze last layer of the projection head in this iteration
            student.global_head.last_layer.zero_grad(set_to_none=True)
            student.local_head.last_layer.zero_grad(set_to_none=True)
        optimizer.step()
        # EMA teacher update
        with torch.no_grad():
            m = momentum_schedule[it]
            for param_name, stdt_param in student.named_parameters():
                try:
                    tchr_param = teacher.get_parameter(param_name)
                    tchr_param.data.mul_(m).add_(
                        (1 - m) * stdt_param.detach().data
                    )
                except AttributeError:
                    continue
        # Logging
        metric_logger.update(
            loss=loss_value,
            lr=lr_schedule[it].item(),
            wd=wd_schedule[it].item(),
        )
    # print("Averaged stats:", metric_logger)
    return metric_logger.meters["loss"]


def _eval(model, ks, metrics, args):
    model.eval()
    eval_results = knn_eval(
        model=model.backbone,
        ks=ks,
        distance_temperature=0.07,
        pred_threshold=0.5,
        data_path=args.data,
        target_sources=args.sources,
        crop_size=args.global_size,
        num_crops=3,
        crops_overlap=0.3,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        simplified=True,
    )
    return np.array(
        [
            eval_results[source][metric][i]
            for source in args.sources
            for i in range(len(ks))
            for metric in metrics
        ]
    )


def _check_args(args):
    args.results_dir = args.save_dir / args.save_folder
    args.results_dir.mkdir(exist_ok=True, parents=True)
    args.sources = get_rdd2022_sources(args.sources_subset)
    args.epochs = max(1, args.epochs)
    args.batch_size = max(args.batch_size, len(args.sources))
    if args.disable_schedules:
        args.lr_end = args.lr
        args.weight_decay_end = args.weight_decay
    if args.disable_warmups:
        args.lr_warmup_size = 0
        args.teacher_temp_warmup_size = 0
    else:
        args.lr_warmup_size = max(0, min(args.lr_warmup_size, args.epochs))
        args.teacher_temp_warmup_size = max(
            0,
            min(args.teacher_temp_warmup_size, args.epochs),
        )
    args.freeze_last_layer_for = max(
        0,
        min(args.freeze_last_layer_for, args.epochs),
    )
    args.num_local_views = max(args.num_local_views, 1)
    args.checkpoint_freq = (
        args.checkpoint_freq if args.checkpoint_freq > 0 else args.epochs
    )
    return args


def get_args_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Self-supervised backbone pre-training "
            "using DINOv1 with multi-task heads."
        ),
        parents=[utils.get_base_args_parser()],
    )
    # Architecture
    parser.add_argument("--dino-base", action="store_true")
    parser.add_argument("--projection-dim", type=int, default=16384)
    # Momentum coefficients
    parser.add_argument("--teacher-momentum", type=float, default=0.996)
    parser.add_argument("--center-momentum", type=float, default=0.9)
    # Training stability
    parser.add_argument("--freeze-last-layer-for", type=int, default=3)
    # Sharpness temperature
    parser.add_argument("--student-temp", type=float, default=0.1)
    parser.add_argument("--teacher-temp", type=float, default=0.04)
    parser.add_argument("--teacher-temp-warmup-size", type=int, default=3)
    parser.add_argument("--teacher-temp-warmup-start", type=float, default=0.02)
    # Multi-view configuration
    parser.add_argument("--global-size", type=int, default=448)
    parser.add_argument(
        "--global-scale",
        type=float,
        nargs=2,
        default=(0.6, 0.85),
    )
    parser.add_argument("--local-size", type=int, default=224)
    parser.add_argument(
        "--local-scale",
        type=float,
        nargs=2,
        default=(0.125, 0.4),
    )
    parser.add_argument("--num-local-views", type=int, default=2)
    return parser


def _get_train_data_loader(args):
    multiview_trans = SSLMultiviewTransform(
        global_size=args.global_size,
        global_scale=args.global_scale,
        local_size=args.local_size,
        local_scale=args.local_scale,
        num_local_views=args.num_local_views,
    )
    train_dataset = RDD2022(
        data_path=args.data,
        sources=args.sources,
        split="train",
        task="ssl",
        transform=multiview_trans,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_dataset.get_balanced_sampler(args.batch_size),
        num_workers=args.num_workers,
        collate_fn=SSLMultiviewTransform.multiview_collate_fn,
        pin_memory=False,
        persistent_workers=True,
    )
    return train_loader


def run():
    try:
        parser = get_args_parser()
        args = parser.parse_args()
        pretrain(args)
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(run())

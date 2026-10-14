import argparse
import time
from itertools import product
from math import ceil, isfinite

import numpy as np
import torch
from torch.utils.data import DataLoader

from pavessl import models, utils
from pavessl.data import RDD2022, SSLMultiviewTransform
from pavessl.scripts.pretrain import pretrain_utils


def _get_models(device, args):
    # Student network
    student = models.build_projection_network(
        network_cls=models.DualHeadProjectionNetwork,
        network_state_path=(args.start_points_dir / "ssl_student_net.pth"),
        backbone_state_path=(args.start_points_dir / "resnet50_backbone.pth"),
        projection_dim=args.ssl_projection_dim,
    )
    student_nparams = utils.get_num_parameters(student)
    backbone_nparams = utils.get_num_parameters(student.backbone)
    print(f"Backbone    # params: {backbone_nparams:,d}")
    print(f"Student net # params: {student_nparams:,d}")
    # Teacher network
    teacher = models.build_projection_network(
        network_cls=models.SingleHeadProjectionNetwork,
        projection_dim=args.ssl_projection_dim,
    )
    teacher_nparams = utils.get_num_parameters(teacher)
    print(f"Teacher net # params: {teacher_nparams:,d}")
    print(f"Global head # params: {(teacher_nparams - backbone_nparams):,d}")
    print(f"Local head  # params: {(student_nparams - teacher_nparams):,d}")
    # Teacher and Student start with the same weights
    teacher.load_state_dict(student.state_dict(), strict=False)
    student = student.to(device=device, non_blocking=True)
    teacher = teacher.to(device=device, non_blocking=True)
    # There is no backpropagation through the Teacher
    for p in teacher.parameters():
        p.requires_grad = False
    return student, teacher


def _get_train_data_loader(args):
    multiview_trans = SSLMultiviewTransform(
        global_size=args.mv_global_size,
        global_scale=args.mv_global_scale,
        local_size=args.mv_local_size,
        local_scale=args.mv_local_scale,
        num_local_views=args.mv_num_local_views,
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


def _get_loss_function(device, args):
    loss_fn = models.SSLLoss(
        projection_dim=args.ssl_projection_dim,
        num_local_views=args.mv_num_local_views,
        teacher_temp=args.ssl_teacher_temp,
        teacher_temp_start=args.ssl_teacher_temp_warmup_start,
        teacher_temp_warmup=args.ssl_teacher_temp_warmup_size,
        num_epochs=args.epochs,
        student_temp=args.ssl_student_temp,
        center_momentum=args.ssl_center_momentum,
    )
    return loss_fn.to(device=device, non_blocking=True)


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
    freeze_last_layer = epoch < args.ssl_freeze_last_layer_for
    student.local_head.freeze_last_layer(freeze_last_layer)
    student.global_head.freeze_last_layer(freeze_last_layer)
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
    for it, (global_views, local_views, samples_weight) in enumerate(
        metric_logger.log_every(data_loader, log_every, header)
    ):
        global_views = global_views.to(device=device, non_blocking=True)
        local_views = local_views.to(device=device, non_blocking=True)
        samples_weight = samples_weight.to(device=device, non_blocking=True)
        samples_weight.div_(samples_weight.sum())
        it = len(data_loader) * epoch + it  # Global training iteration
        # Only the first group is regularized
        optimizer.param_groups[0]["weight_decay"] = wd_schedule[it]
        optimizer.param_groups[0]["lr"] = lr_schedule[it]
        optimizer.param_groups[1]["lr"] = lr_schedule[it]
        # Teacher and student forward passes && compute loss
        teacher_out = teacher(global_views)
        student_g_out, student_l_out = student(global_views, local_views)
        loss = loss_fn(
            teacher_out,
            student_g_out,
            student_l_out,
            samples_weight,
            epoch,
        )
        # Loss explosion, unstable training
        loss_value = loss.item()  # CPU-GPU synchronization
        if not isfinite(loss_value):
            raise RuntimeError(f"Loss exploded with value {loss_value}")
        # Student update
        optimizer.zero_grad()
        loss.backward()
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
        args = stored_args
    # Restore training
    student, teacher = _get_models(device, args)
    train_loader = _get_train_data_loader(args)
    loss_fn = _get_loss_function(device, args)
    optimizer = models.get_optimizer(student, args.optim)
    run_variables = {
        "epoch": 0,
        "loss_history": None,
        "valid_history": None,
        "valid_metrics": None,
    }
    utils.restart_from_checkpoint(
        ckp_path=(args.results_dir / "checkpoint.pth"),
        run_variables=run_variables,
        student=student,
        teacher=teacher,
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
    momentum_schedule = utils.cosine_scheduler(
        base_value=args.ssl_teacher_momentum,
        final_value=1.0,
        epochs=args.epochs,
        niter_per_ep=len(train_loader),
    )
    # Training
    training_start = time.monotonic()
    for epoch in range(start_epoch, args.epochs):
        utils.set_random_seeds(42 + epoch)
        logger = _train_one_epoch(
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
            logger.meters["loss"].min_value,
            logger.meters["loss"].max_value,
            logger.meters["loss"].global_avg,
            logger.meters["iter"].total_accumulated,
        ]
        # Run validation
        if not (epoch % args.validation_freq) or (epoch + 1) == args.epochs:
            eval_start = time.monotonic()
            student_eval = pretrain_utils.backbone_knn_eval(
                backbone=student.backbone,
                data_path=args.data,
                target_sources=args.sources,
                ks=args.knn_ks,
                distance_temperature=args.knn_dist_temperature,
                pred_threshold=args.knn_pred_threshold,
                crops_target_size=args.knn_crops_target_size,
                crops_overlap=args.knn_crops_overlap,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                logging_header="(Eval/Student)",
            )
            teacher_eval = pretrain_utils.backbone_knn_eval(
                backbone=teacher.backbone,
                data_path=args.data,
                target_sources=args.sources,
                ks=args.knn_ks,
                distance_temperature=args.knn_dist_temperature,
                pred_threshold=args.knn_pred_threshold,
                crops_target_size=args.knn_crops_target_size,
                crops_overlap=args.knn_crops_overlap,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                logging_header="(Eval/Teacher)",
            )
            if valid_metrics is None:
                valid_metrics = sorted(student_eval[args.sources[0]][0])
                num_rows = 1 + ceil((args.epochs - 1) / args.validation_freq)
                num_cols = 1 + 2 * len(valid_metrics) * len(args.sources) * len(
                    args.knn_ks
                )
                valid_history = np.zeros((num_rows, num_cols), dtype=np.float32)
            row_index = ceil(epoch / args.validation_freq)
            valid_history[row_index, 0] = epoch + 1
            valid_history[row_index, 1:] = [
                metrics[source][k_idx][metric]
                for metrics, source, k_idx, metric in product(
                    (student_eval, teacher_eval),
                    args.sources,
                    range(len(args.knn_ks)),
                    valid_metrics,
                )
            ]
            duration = int(time.monotonic() - eval_start)
            print(f"Validation ended in {utils.get_time_string(duration)}!")
        # Save checkpoints
        if not (epoch % args.checkpoint_freq) or (epoch + 1) == args.epochs:
            states_dict = {
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
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
        header="min,max,avg,duration",
        comments="",
    )
    valid_headers = ["epoch"]
    valid_headers.extend(
        [
            f"{model}__{source}__{k}__{metric}"
            for model, source, k, metric in product(
                ("student", "teacher"),
                args.sources,
                args.knn_ks,
                valid_metrics,
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


def check_args(args: argparse.Namespace) -> argparse.Namespace:
    args = utils.check_training_args(args)
    if args.disable_warmups:
        args.ssl_teacher_temp_warmup_size = 0
    else:
        args.ssl_teacher_temp_warmup_size = max(
            0,
            min(args.ssl_teacher_temp_warmup_size, args.epochs),
        )
    args.ssl_freeze_last_layer_for = max(
        0,
        min(args.ssl_freeze_last_layer_for, args.epochs),
    )
    args.mv_num_local_views = max(args.mv_num_local_views, 1)
    return args


def add_multiview_args(parser: argparse.ArgumentParser):
    # Multi-view configuration
    parser.add_argument(
        "--mv-global-size",
        type=int,
        default=448,
        help="Global-view crops target dimension.",
    )
    parser.add_argument(
        "--mv-global-scale",
        type=float,
        nargs=2,
        default=(0.7, 0.9),
        help="Global-view crops min-max range.",
    )
    parser.add_argument(
        "--mv-local-size",
        type=int,
        default=224,
        help="Local-view crops target dimension.",
    )
    parser.add_argument(
        "--mv-local-scale",
        type=float,
        nargs=2,
        default=(0.25, 0.5),
        help="Local-view crops min-max range.",
    )
    parser.add_argument(
        "--mv-num-local-views",
        type=int,
        default=2,
        help="Generate N >= 1 local views for each image.",
    )
    return parser


def get_args_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Self-supervised backbone pre-training "
            "using DINOv1 with multi-task heads."
        ),
    )
    # Data and training arguments
    parser = utils.add_data_arg(parser)
    parser = utils.add_training_args(
        parser=parser,
        save_dir_default="data/results/ssl_pretrain",
        lr_default=5e-4,
        lr_end_default=1e-6,
    )
    # Architecture
    parser.add_argument(
        "--ssl-projection-dim",
        type=int,
        default=8192,
        help="Projection heads output dimension.",
    )
    # Momentum coefficients
    parser.add_argument(
        "--ssl-teacher-momentum",
        type=float,
        default=0.996,
        help="EMA update constant for teacher network.",
    )
    parser.add_argument(
        "--ssl-center-momentum",
        type=float,
        default=0.9,
        help="EMA update constant for the teacher output center.",
    )
    # Training stability
    parser.add_argument(
        "--ssl-freeze-last-layer-for",
        type=int,
        default=5,
        help="Freeze last layer of student projection heads for N epochs.",
    )
    # Sharpness temperature
    parser.add_argument(
        "--ssl-student-temp",
        type=float,
        default=0.1,
        help="Student softmax sharpening constant.",
    )
    parser.add_argument(
        "--ssl-teacher-temp",
        type=float,
        default=0.04,
        help="Teacher softmax sharpening constant.",
    )
    parser.add_argument(
        "--ssl-teacher-temp-warmup-size",
        type=int,
        default=3,
        help="Warmup duration for the teacher softmax sharpening constant.",
    )
    parser.add_argument(
        "--ssl-teacher-temp-warmup-start",
        type=float,
        default=0.02,
        help="Start value for the teacher softmax sharpening constant warmup.",
    )
    # Multi-view arguments
    parser = add_multiview_args(parser)
    # kNN evaluation arguments
    parser = utils.add_knn_eval_args(parser)
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

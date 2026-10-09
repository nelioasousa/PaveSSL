import argparse
import math
import pickle
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from pavessl import models, utils
from pavessl.data import RDD2022, SSLMultiviewTransform
from pavessl.scripts.eval.knn_eval import knn_eval_all


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


def _get_models(device, args):
    # Student network
    student = models.build_projection_network(
        network_cls=models.DualHeadProjectionNetwork,
        network_state_path=(args.start_points_dir / "ssl_student_net.pth"),
        backbone_state_path=(args.start_points_dir / "resnet50_backbone.pth"),
        projection_dim=args.projection_dim,
    )
    student_nparams = utils.get_num_parameters(student)
    backbone_nparams = utils.get_num_parameters(student.backbone)
    print(f"Backbone    # params: {backbone_nparams:,d}")
    print(f"Student net # params: {student_nparams:,d}")
    # Teacher network
    teacher = models.build_projection_network(
        network_cls=models.SingleHeadProjectionNetwork,
        projection_dim=args.projection_dim,
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
    student.local_head.freeze_last_layer(epoch < args.freeze_last_layer_for)
    student.global_head.freeze_last_layer(epoch < args.freeze_last_layer_for)
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
    for it, (global_views, local_views, weights) in enumerate(
        metric_logger.log_every(data_loader, log_every, header)
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
    # print("Averaged stats:", metric_logger)
    return metric_logger.meters["loss"]


def _eval(model, header, args):
    model.eval()
    eval_results = knn_eval_all(
        model=model.backbone,
        data_path=args.data,
        target_sources=args.sources,
        ks=args.eval_ks,
        distance_temperature=args.eval_dist_temperature,
        pred_threshold=args.eval_pred_threshold,
        crops_target_size=args.eval_crops_target_size,
        crops_overlap=args.eval_crops_overlap,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        key_split="train",
        query_split="valid",
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
    # Models
    student, teacher = _get_models(device, stored_args)
    # Data
    train_loader = _get_train_data_loader(stored_args)
    # Loss function
    loss_fn = _get_loss_function(device, stored_args)
    # Optimizer
    optimizer = models.get_optimizer(student, stored_args.optim)
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
        # Check from projection dimension mismatch
        if args.projection_dim != stored_args.projection_dim:
            raise ValueError(
                "Projection dimension changed from "
                f"{stored_args.projection_dim} to {args.projection_dim}"
            )
        _save_args(args)
        # Re-build with the new arguments
        train_loader = _get_train_data_loader(args)
        loss_fn = _get_loss_function(device, args)
        optimizer = models.get_optimizer(student, args.optim)
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
                2 * num_metrics + 1,
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
            row_index = math.ceil(epoch / args.validation_freq)
            valid_history[row_index, 0] = epoch + 1
            valid_history[row_index, 1 : num_metrics + 1] = _eval(
                model=student,
                header="(Eval/Student)",
                args=args,
            )
            valid_history[row_index, num_metrics + 1 :] = _eval(
                model=teacher,
                header="(Eval/Teacher)",
                args=args,
            )
        # Save checkpoints
        if not (epoch % args.checkpoint_freq) or (epoch + 1) == args.epochs:
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
            f"{model}__{source}__{k}__{metric}"
            for model in ("student", "teacher")
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
    args = utils.check_training_args(args)
    if args.disable_warmups:
        args.teacher_temp_warmup_size = 0
    else:
        args.teacher_temp_warmup_size = max(
            0,
            min(args.teacher_temp_warmup_size, args.epochs),
        )
    args.freeze_last_layer_for = max(
        0,
        min(args.freeze_last_layer_for, args.epochs),
    )
    args.num_local_views = max(args.num_local_views, 1)
    return args


def add_multiview_args(parser: argparse.ArgumentParser):
    # Multi-view configuration
    parser.add_argument(
        "--global-size",
        type=int,
        default=448,
        help="Resize global crops to this dimension.",
    )
    parser.add_argument(
        "--global-scale",
        type=float,
        nargs=2,
        default=(0.7, 0.9),
        help="Min-max range of image area for global crops.",
    )
    parser.add_argument(
        "--local-size",
        type=int,
        default=224,
        help="Resize local crops to this dimension.",
    )
    parser.add_argument(
        "--local-scale",
        type=float,
        nargs=2,
        default=(0.25, 0.5),
        help="Min-max range of image area for local crops.",
    )
    parser.add_argument(
        "--num-local-views",
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
        "--projection-dim",
        type=int,
        default=8192,
        help="Output dimension of the feature projection heads.",
    )
    # Momentum coefficients
    parser.add_argument(
        "--teacher-momentum",
        type=float,
        default=0.996,
        help="EMA update constant for teacher network.",
    )
    parser.add_argument(
        "--center-momentum",
        type=float,
        default=0.9,
        help="EMA update constant for the teacher output center.",
    )
    # Training stability
    parser.add_argument(
        "--freeze-last-layer-for",
        type=int,
        default=5,
        help="Freeze last layer of student projection heads for N epochs.",
    )
    # Sharpness temperature
    parser.add_argument(
        "--student-temp",
        type=float,
        default=0.1,
        help="Student softmax sharpening constant.",
    )
    parser.add_argument(
        "--teacher-temp",
        type=float,
        default=0.04,
        help="Teacher softmax sharpening constant.",
    )
    parser.add_argument(
        "--teacher-temp-warmup-size",
        type=int,
        default=3,
        help="Warmup duration for the teacher softmax sharpening constant.",
    )
    parser.add_argument(
        "--teacher-temp-warmup-start",
        type=float,
        default=0.02,
        help="Start value for the teacher softmax sharpening constant warmup.",
    )
    # Multi-view arguments
    parser = add_multiview_args(parser)
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

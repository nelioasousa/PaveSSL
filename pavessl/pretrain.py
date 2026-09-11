import argparse
from pathlib import Path
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

import pavessl.utils as utils
from pavessl.data import RDD2022, SSLMultiviewTransform


def pretrain(args):
    utils.set_random_seeds(42)
    args.save_dir.mkdir(exist_ok=True, parents=True)
    if args.dino_base:
        backbone = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
    else:
        backbone = models.resnet50()
        backbone.fc = nn.Identity()
    student = utils.ProjectionNetwork(backbone, args.projection_dim)
    teacher = utils.ProjectionNetwork(
        backbone, args.projection_dim, head_norm_last_layer=True,
    )
    if torch.cuda.is_available():
        device = torch.device(torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    student = student.to(device=device)
    teacher = teacher.to(device=device)
    # teacher and student start with the same weights
    teacher.load_state_dict(student.state_dict())
    # there is no backpropagation through the teacher
    for p in teacher.parameters():
        p.requires_grad = False
    # Data
    train_ds, train_loader, valid_ds, valid_loader = _get_train_valid_data(args)
    # Prepare loss function
    loss_fn = utils.SSLLoss(
        projection_dim=args.projection_dim,
        num_local_views=args.num_local_views,
        teacher_temp=args.teacher_temp,
        teacher_temp_start=args.teacher_temp_start,
        teacher_temp_warmup_epochs=args.teacher_temp_warmup,
        num_epochs=args.epochs,
        student_temp=args.student_temp,
        center_momentum=args.center_momentum,
    )
    loss_fn.to(device=device)
    # Optim
    params_groups = utils.get_params_groups(student)
    if args.optim == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(params_groups, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {args.optim} not supported")
    # Schedulers
    base_lr = args.lr * args.batch_size / 256.0
    lr_schedule = utils.cosine_scheduler(
        base_value=base_lr,
        final_value=args.lr_end,
        epochs=args.epochs,
        niter_per_ep=len(train_loader),
        warmup_epochs=args.lr_warmup,
        start_warmup_value=min(1e-7, base_lr / 10),
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
    # Resume training
    to_restore = {
        "epoch": 0,
        "loss_history": np.zeros((args.epochs, 3), dtype=np.float32),
    }
    utils.restart_from_checkpoint(
        args.save_dir / "checkpoint.pth",
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        loss=loss_fn,
    )
    loss_history = to_restore["loss_history"]
    # Training
    start_time = time.monotonic()
    for epoch in range(to_restore["epoch"], args.epochs):
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
        loss_history[epoch, 0] = loss_log.min_value
        loss_history[epoch, 1] = loss_log.max_value
        loss_history[epoch, 2] = loss_log.global_avg
        # Save checkpoints
        states_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "loss_history": loss_history,
            "args": args,
            "loss": loss_fn.state_dict(),
        }
        utils.save_states(
            save_file=args.save_dir / "checkpoint.pth",
            states_dict=states_dict,
        )
        if args.checkpoint_freq and not (epoch % args.checkpoint_freq):
            utils.save_states(
                save_file=args.save_dir / f"checkpoint_{epoch:04d}.pth",
                states_dict=states_dict,
            )
        # Run eval
        pass
    # Save loss history and report duration
    np.savetxt(
        fname=(args.save_dir / "loss_history.csv"),
        X=loss_history,
        delimiter=",",
        fmt="%24.10e",
        header="min,max,avg",
    )
    train_duration = int(time.monotonic() - start_time)
    print(f"Training ended in {utils.get_time_string(train_duration)}!")
    return


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
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", utils.SmoothedValue(fmt="({value:10.6f} / {local_avg:10.6f} avg)"))
    metric_logger.add_meter("lr", utils.SmoothedValue(fmt="{value:.2e}"))
    metric_logger.add_meter("wd", utils.SmoothedValue(fmt="{value:.2e}"))
    header = f"Epoch: [{str(epoch).rjust(len(str(args.epochs)), '0')}/{args.epochs}]"
    for it, (global_views, local_views) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        it = len(data_loader) * epoch + it  # Global training iteration
        # Only the first group is regularized
        optimizer.param_groups[0]["weight_decay"] = wd_schedule[it]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[it]
        # Move images to device
        global_views = global_views.to(device=device, non_blocking=True)
        local_views = local_views.to(device=device, non_blocking=True)
        # Teacher and student forward passes && compute loss
        teacher_out = teacher(global_views)
        student_global_out = student(global_views)
        student_local_out = student(local_views)
        loss = loss_fn(teacher_out, student_global_out, student_local_out, epoch)
        # Loss explosion, unstable training
        if not math.isfinite(loss.item()):
            raise SystemExit(f"Loss exploded with value {loss.item()}")
        # Student update
        optimizer.zero_grad()
        loss.backward()
        if epoch < args.freeze_proj:
            # Freeze last layer of the projection head in this iteration
            student.head.last_layer.zero_grad(set_to_none=True)
        optimizer.step()
        # EMA teacher update
        with torch.no_grad():
            m = momentum_schedule[it]
            for param_stdt, param_tchr in zip(student.parameters(), teacher.parameters()):
                param_tchr.data.mul_(m).add_((1 - m) * param_stdt.detach().data)
        # Logging
        if device.type == "cuda":
            torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # print("Averaged stats:", metric_logger)
    return metric_logger.meters["loss"]


def get_args_parser():
    parser = argparse.ArgumentParser()
    # Architecture/weights
    # parser.add_argument("--archi", type=str, choices=["resnet50", "vits16"], default="resnet50")
    parser.add_argument("--dino-base", action="store_true")
    parser.add_argument("--projection-dim", type=int, default=16384)
    # Momentum coefficients
    parser.add_argument("--teacher-momentum", type=float, default=0.996)
    parser.add_argument("--center-momentum", type=float, default=0.9)
    # Sharpness temperature
    parser.add_argument("--student-temp", type=float, default=0.1)
    parser.add_argument("--teacher-temp", type=float, default=0.04)
    parser.add_argument("--teacher-temp-start", type=float, default=0.02)
    parser.add_argument("--teacher-temp-warmup", type=int, default=3)
    # Training/optim parameters
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-end", type=float, default=1e-6)
    parser.add_argument("--lr-warmup", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--weight-decay-end", type=float, default=0.4)
    parser.add_argument("--optim", type=str, choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--freeze-proj", type=int, default=1)
    # Multi-view
    parser.add_argument("--global-size", type=int, default=224)
    parser.add_argument("--global-scale", type=float, nargs=2, default=(0.75, 1.0))
    parser.add_argument("--local-size", type=int, default=96)
    parser.add_argument("--local-scale", type=float, nargs=2, default=(0.05, 0.4))
    parser.add_argument("--num-local-views", type=int, default=2)
    # Others
    parser.add_argument("--data", type=Path, default=Path("data/RDD2022_proc"))
    parser.add_argument("--save-dir", type=Path, default=Path("data/results/pretrain"))
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


def _get_train_valid_data(args):
    multiview_trans = SSLMultiviewTransform(
        global_size=args.global_size,
        global_scale=args.global_scale,
        local_size=args.local_size,
        local_scale=args.local_scale,
        num_local_views=args.num_local_views,
    )
    train_dataset = RDD2022(
        data_path=args.data,
        task="ssl",
        countries="all",
        split="train",
        transform=multiview_trans,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=SSLMultiviewTransform.multiview_collate_fn,
        pin_memory=False,
        drop_last=True,
    )
    valid_dataset = RDD2022(
        data_path=args.data,
        task="classification",
        countries="all",
        split="valid",
        transform=multiview_trans.teacher_global,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return train_dataset, train_loader, valid_dataset, valid_loader


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

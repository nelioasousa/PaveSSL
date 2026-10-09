from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import torch

from pavessl import models


def add_eval_args(parser: ArgumentParser, save_csv_name: str) -> ArgumentParser:
    # Data
    parser.add_argument("--data", type=Path, default=Path("data/RDD2022_proc"))
    parser.add_argument(
        "--target-source",
        choices=[
            "China_Drone",
            "China_MotorBike",
            "Czech",
            "India",
            "Japan",
            "Norway",
            "United_States",
        ],
        type=str,
        default="India",
    )
    # Target and save location
    parser.add_argument("-d", "--experiment-dir", type=Path, required=True)
    parser.add_argument("-k", "--checkpoint-key", type=str, required=True)
    parser.add_argument("--save-folder", type=str, default="evals")
    parser.add_argument("--save-file", type=str, default=save_csv_name)
    return parser


def load_backbone(state_file: Path, checkpoint_key: str):
    backbone = models.build_backbone(None)
    state_dict = torch.load(state_file, map_location="cpu", weights_only=False)
    state_dict = {
        key[9:]: state_tensor
        for key, state_tensor in state_dict[checkpoint_key].items()
        if key.startswith("backbone.")
    }
    backbone.load_state_dict(state_dict, strict=True)
    return backbone


def multi_value_classification_metrics(preds: np.ndarray, targets: np.ndarray):
    counts = targets.sum(axis=0)
    tps = (preds * targets).sum(axis=0)
    tns = ((1.0 - preds) * (1.0 - targets)).sum(axis=0)
    accuracy = (tps + tns) / len(targets)
    sensitivity = tps / counts
    specificity = tns / (len(targets) - counts)
    class_weights = counts / counts.sum()
    macro_accuracy = accuracy.mean().item()
    weighted_accuracy = (accuracy * class_weights).sum().item()
    balanced_accuracy = sensitivity.mean().item()
    macro_sensitivity = balanced_accuracy
    weighted_sensitivity = (sensitivity * class_weights).sum().item()
    macro_specificity = specificity.mean().item()
    weighted_specificity = (specificity * class_weights).sum().item()
    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "macro_accuracy": macro_accuracy,
        "weighted_accuracy": weighted_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_sensitivity": macro_sensitivity,
        "weighted_sensitivity": weighted_sensitivity,
        "macro_specificity": macro_specificity,
        "weighted_specificity": weighted_specificity,
    }

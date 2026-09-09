from pathlib import Path

import numpy as np
import torch


def set_random_seeds(seed: int = 42):
    """Set random seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_bboxes(annotation_path: Path):
    bboxes = {"boxes": [], "labels": []}
    for line in annotation_path.read_text().strip().splitlines():
        label, *box = line.split()
        bboxes["boxes"].append([int(coord) for coord in box])
        bboxes["labels"].append(int(label))
    return bboxes

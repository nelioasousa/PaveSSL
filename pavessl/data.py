from itertools import accumulate
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision.transforms.v2 as transforms_v2
import torchvision.transforms.v2.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import Transform, query_size

from pavessl.utils import load_bboxes


def get_rdd2022_sources(subset_level: Literal[1, 2, 3] = 3):
    subset_level = min(max(subset_level, 1), 3)
    sources = [
        "China_Drone",
        "China_MotorBike",
        "Japan",
        "Norway",
        "Czech",
        "United_States",
    ]
    return sources[:(2*subset_level)]


class RDD2022(Dataset):
    def __init__(
        self,
        data_path: Path,
        sources: str | list[str] | Literal["all"],
        split: Literal["train", "valid", "test"],
        task: Literal["ssl", "classification", "detection"],
        transform: Transform | None = None,
        independent_labels: bool = False,
    ):
        data_path = data_path.resolve()
        if not data_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {data_path}")
        all_sources = [s.name for s in data_path.iterdir() if s.is_dir()]
        if sources == "all":
            sources = all_sources
        else:
            if isinstance(sources, str):
                sources = [sources]
            unknown = frozenset(sources).difference(all_sources)
            if unknown:
                raise ValueError(f"Unknown source folders: {list(unknown)}")
        sources.sort()
        self.base = data_path
        self.split = split
        self.sources = sources
        self.task = task
        self.transform = transform
        self.classes = (data_path / "classes.txt").read_text().splitlines()
        self.independent_labels = independent_labels
        if independent_labels:
            self.local_index_map = {}
            self.local_classes = {}
            for source in sources:
                cls_file = data_path / source / "classes.txt"
                cls_idxs, cls_names = list(zip(*(
                    line.split(maxsplit=1)
                    for line in cls_file.read_text().splitlines()
                )))
                self.local_classes[source] = cls_names
                self.local_index_map[source] = {
                    int(idx): i for i, idx in enumerate(cls_idxs)
                }
        # Not all train images have annotations
        if split == "train" and task != "ssl":
            self.images = {
                source: [
                    f"{label.stem}.jpg"
                    for label in (
                        data_path / f"{source}/{split}/annotations"
                    ).iterdir()
                ]
                for source in sources
            }
        else:
            self.images = {
                source: [
                    image.name
                    for image in (
                        data_path / f"{source}/{split}/images"
                    ).iterdir()
                ]
                for source in sources
            }
        self.counts = [len(self.images[source]) for source in sources]
        self.acc_counts = list(accumulate(self.counts))
        self.acc_counts.insert(0, 0)

    def __len__(self):
        return self.acc_counts[-1]

    def _compute_from_global_index(self, index):
        for i, acc in enumerate(self.acc_counts[1:]):
            if index < acc:
                return self.sources[i], index - self.acc_counts[i]
        raise IndexError("Out of range")

    def _load_label_for_classification(self, label_path, source):
        labels = load_bboxes(label_path)["labels"]
        # Multi-hot encoding
        if self.independent_labels:
            labels = [self.local_index_map[source][idx] for idx in labels]
            multihot = torch.zeros(len(self.local_classes[source]))
        else:
            multihot = torch.zeros(len(self.classes))
        return multihot.index_fill_(
            dim=0, index=torch.tensor(labels, dtype=torch.long), value=1.0,
        )

    def _load_label_for_detection(self, label_path, dim, source):
        bboxes = load_bboxes(label_path)
        if self.independent_labels:
            bboxes["labels"] = [
                self.local_index_map[source][idx] for idx in bboxes["labels"]
            ]
        return {
            "boxes": tv_tensors.BoundingBoxes(
                bboxes["boxes"],
                format="XYXY",
                canvas_size=dim,
                clamping_mode="hard",
            ),
            "labels": torch.tensor(bboxes["labels"], dtype=torch.long),
        }

    def __getitem__(self, index):
        source, local_index = self._compute_from_global_index(index)
        image_name = self.images[source][local_index]
        split_dir = self.base / f"{source}/{self.split}"
        image = Image.open(split_dir / f"images/{image_name}").convert("RGB")
        image = F.to_dtype(F.to_image(image), dtype=torch.float32, scale=True)
        if self.task == "ssl":
            if self.transform is None:
                return image
            return self.transform(image)
        label_path = split_dir / f"annotations/{image_name[:-3]}txt"
        if self.task == "classification":
            label = self._load_label_for_classification(label_path, source)
        else:
            label = self._load_label_for_detection(
                label_path, image.shape[-2:], source,
            )
        if self.transform is None:
            return image, label
        return self.transform(image, label)


class BottomRandomResizedCrop(transforms_v2.RandomResizedCrop):
    _bottom_threshold = 0.5

    def make_params(self, flat_inputs):
        h, _ = query_size(flat_inputs)
        thres = round(h * self._bottom_threshold)
        params = super().make_params(flat_inputs)
        if params["top"] >= thres:
            return params
        limit = h - params["height"]
        params["top"] = (
            torch.randint(thres, limit, size=(1,)).item()
            if params["height"] < (h - thres)
            else limit
        )
        return params


class SSLMultiviewTransform(torch.nn.Module):
    def __init__(
        self,
        global_size: int,
        global_scale: tuple[float, float],
        local_size: int,
        local_scale: tuple[float, float],
        num_local_views: int = 2,
    ):
        super().__init__()
        self.global_size = global_size
        self.local_size = local_size
        self.num_local_views = num_local_views
        augs = self.get_multiview_augmentations(
            global_size,
            global_scale,
            local_size,
            local_scale,
        )
        self.teacher_global = augs[0]
        self.student_global = augs[1]
        self.student_local = augs[2]

    def forward(self, img, label=None):
        views = [self.teacher_global(img), self.student_global(img)]
        views.extend(
            [self.student_local(img) for _ in range(self.num_local_views)]
        )
        return views, label

    @staticmethod
    def get_multiview_augmentations(
        global_size: int = 224,
        global_scale: tuple[float, float] = (0.75, 1.0),
        local_size: int = 96,
        local_scale: tuple[float, float] = (0.05, 0.4),
    ):
        # Random cropping
        global_crop = BottomRandomResizedCrop(global_size, global_scale)
        local_crop = BottomRandomResizedCrop(local_size, local_scale)
        # Geometric transformations
        geometric_transfs = [
            transforms_v2.RandomHorizontalFlip(p=0.5),
            transforms_v2.RandomApply(
                [transforms_v2.RandomRotation(degrees=10, expand=False)],
                p=0.3,
            ),
        ]
        # Photometric transformations
        photometric_transfs = [
            transforms_v2.RandomApply(
                [
                    transforms_v2.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.2,
                        hue=0.1,
                    )
                ],
                p=0.8,
            ),
            transforms_v2.RandomPosterize(bits=6, p=0.3),
            transforms_v2.RandomSolarize(threshold=0.65, p=0.3),
            transforms_v2.RandomApply(
                [transforms_v2.GaussianBlur(kernel_size=5, sigma=(0.5, 1.2))],
                p=0.8,
            ),
            transforms_v2.RandomGrayscale(p=0.2),
        ]
        # Student
        ## Global
        student_global_aug = transforms_v2.Compose(
            [
                global_crop,
                *geometric_transfs,
                *photometric_transfs,
            ]
        )
        ## Local
        student_local_aug = transforms_v2.Compose(
            [
                local_crop,
                *geometric_transfs,
                *photometric_transfs,
            ]
        )
        # Teacher
        teacher_aug = transforms_v2.Compose(
            [
                global_crop,
                *geometric_transfs,
                photometric_transfs[0],
                *photometric_transfs[-2:],
            ]
        )
        return teacher_aug, student_global_aug, student_local_aug

    @staticmethod
    def multiview_collate_fn(batch):
        # `batch` = [([v1, ..., vn], None), ...]
        B = len(batch)  # batch size
        tensor_dtype = batch[0][0][0].dtype
        # Collating global views
        global_dim = batch[0][0][0].shape[-3:]
        global_views = torch.empty((2 * B, *global_dim), dtype=tensor_dtype)
        torch.stack(
            [sample[0][0] for sample in batch],
            dim=0,
            out=global_views[:B],
        )
        torch.stack(
            [sample[0][1] for sample in batch],
            dim=0,
            out=global_views[B:],
        )
        # Collating local views
        local_dim = batch[0][0][-1].shape[-3:]
        num_local_views = len(batch[0][0]) - 2
        local_views = torch.empty(
            (num_local_views * B, *local_dim),
            dtype=tensor_dtype,
        )
        for i in range(num_local_views):
            torch.stack(
                [sample[0][i + 2] for sample in batch],
                dim=0,
                out=local_views[i * B : (i + 1) * B],
            )
        return global_views, local_views


class MultiBottomSplit(transforms_v2.Transform):
    def __init__(self, size: int, num_crops: int = 3, overlap: float = 0.0):
        super().__init__()
        if num_crops < 2:
            raise ValueError("Only make sense for 2 or more crops.")
        if size < 1:
            raise ValueError("`size` must be positive.")
        if overlap >= 1.0:
            raise ValueError("`overlap` can not be >= 1.0")
        self._size = size
        self._num_crops = num_crops
        self._overlap = overlap
        self._local_cache = dict()

    def make_params(self, flat_inputs):
        h, w = query_size(flat_inputs)
        try:
            crop_size = self._local_cache[w]
        except KeyError:
            crop_size = min(h, self._compute_split_crop_size(w))
            self._local_cache[w] = crop_size
        top = h - crop_size
        crops = [
            dict(top=top, left=int(left), height=crop_size, width=crop_size)
            for left in np.linspace(0, w - crop_size, self._num_crops)
        ]
        return {"crops": crops}

    def transform(self, inpt, params):
        if not isinstance(inpt, torch.Tensor) or len(inpt.shape) == 1:
            return inpt
        crops = [
            transforms_v2.functional.resized_crop(
                inpt=inpt,
                size=self._size,
                **crop,
            )
            for crop in params["crops"]
        ]
        return crops

    def _compute_split_crop_size(self, width: int):
        n = self._num_crops
        overlap = self._overlap
        crop_size_no_overlap = int(width / n)
        if overlap == 0.0:
            return crop_size_no_overlap
        crop_size = crop_size_no_overlap
        increment = 1 if overlap > 0.0 else -1
        overlap = abs(overlap)
        while True:
            actual_overlap = (n * crop_size - width) / (crop_size * (n - 1))
            if actual_overlap >= overlap:
                break
            crop_size += increment
        return crop_size

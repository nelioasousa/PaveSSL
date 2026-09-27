from itertools import accumulate
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision.transforms.v2 as transforms_v2
import torchvision.transforms.v2.functional as F
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import tv_tensors
from torchvision.transforms.v2 import Transform, query_size


def get_rdd2022_sources(subset_level: Literal[1, 2, 3, 4]):
    subset_level = min(max(subset_level, 1), 4)
    sources = [
        "China_Drone",
        "China_MotorBike",
        "Japan",
        "Norway",
        "Czech",
        "United_States",
        "India",
    ]
    return sources[: (2 * subset_level)]


def load_bboxes(annotation_path: str | Path):
    bboxes = {"boxes": [], "labels": []}
    with open(annotation_path, mode="r") as ann_file:
        for line in ann_file.read().strip().splitlines():
            label, *box = line.split()
            bboxes["boxes"].append([int(coord) for coord in box])
            bboxes["labels"].append(int(label))
    return bboxes


class RDD2022(Dataset):
    def __init__(
        self,
        data_path: Path,
        sources: str | list[str] | Literal["all"],
        split: Literal["train", "valid", "test"],
        task: Literal["ssl", "classification", "detection"],
        transform: Transform | None = None,  # transforms v2 API
    ):
        data_path = data_path.resolve()
        if not data_path.is_dir():
            raise NotADirectoryError(f'Not a directory: "{data_path}"')
        all_sources = [s.name for s in data_path.iterdir() if s.is_dir()]
        if sources == "all":
            sources = all_sources
        else:
            sources = [sources] if isinstance(sources, str) else sources
            unknown = frozenset(sources).difference(all_sources)
            if unknown:
                raise ValueError(f"Unknown sources: {list(unknown)}")
        sources.sort()
        self.base_dir = data_path
        self.sources_base_dir = {
            source: str(data_path / source / split) for source in sources
        }
        self.split = split
        self.sources = sources
        self.task = task
        self.transform = transform
        # Handling sources with different classes
        self.sources_classes = {
            source: (data_path / source / "classes.txt")
            .read_text()
            .splitlines()
            for source in sources
        }
        self.classes = sorted(set().union(*self.sources_classes.values()))
        self.sources_classes = {
            source: [self.classes.index(cls_name) for cls_name in classes]
            for source, classes in self.sources_classes.items()
        }
        if task == "classification":
            self.sources_classification_mask = {
                source: torch.zeros(len(self.classes)).index_fill_(
                    dim=0,
                    index=torch.tensor(indexes, dtype=torch.long),
                    value=1.0,
                )
                for source, indexes in self.sources_classes.items()
            }
        # Not all train images have annotations
        if split == "train" and task != "ssl":
            self.images = {
                source: [
                    f"{label.stem}.jpg"
                    for label in (
                        (data_path / f"{source}/{split}/annotations").iterdir()
                    )
                ]
                for source in sources
            }
        else:
            self.images = {
                source: [
                    image.name
                    for image in (
                        (data_path / f"{source}/{split}/images").iterdir()
                    )
                ]
                for source in sources
            }
        self.counts = [len(self.images[source]) for source in sources]
        self.acc_counts = list(accumulate(self.counts))
        self.acc_counts.insert(0, 0)
        # Sources sample weights
        self.sources_weight = {
            source: self.acc_counts[-1] / (count * len(self.classes))
            for source, count in zip(sources, self.counts)
        }

    def __len__(self):
        return self.acc_counts[-1]

    def _compute_location_from_index(self, index):
        for i, acc in enumerate(self.acc_counts[1:]):
            if index < acc:
                return self.sources[i], index - self.acc_counts[i]
        raise IndexError("Out of range")

    def _load_classific_label(self, label_path, source):
        labels = [
            self.sources_classes[source][idx]
            for idx in load_bboxes(label_path)["labels"]
        ]
        return {
            "target": torch.zeros(len(self.classes)).index_fill_(
                dim=0,
                index=torch.tensor(labels, dtype=torch.long),
                value=1.0,
            ),
            "weight": self.sources_weight[source],
            "mask": self.sources_classification_mask[source].clone(),
        }

    def _load_detection_label(self, label_path, dim, source):
        bboxes = load_bboxes(label_path)
        bboxes["labels"] = [
            self.sources_classes[source][idx] for idx in bboxes["labels"]
        ]
        return {
            "boxes": tv_tensors.BoundingBoxes(
                data=bboxes["boxes"],
                format="XYXY",
                canvas_size=dim,
                clamping_mode="hard",
            ),
            "labels": torch.tensor(bboxes["labels"], dtype=torch.long),
            "weight": self.sources_weight[source],
        }

    def __getitem__(self, item_ref: int | tuple[str, int]):
        source, source_index = (
            self._compute_location_from_index(item_ref)
            if isinstance(item_ref, int)
            else item_ref
        )
        image_name = self.images[source][source_index]
        source_dir = self.sources_base_dir[source]
        image = Image.open(f"{source_dir}/images/{image_name}").convert("RGB")
        image = F.to_dtype(F.to_image(image), dtype=torch.float32, scale=True)
        if self.task == "ssl":
            if self.transform is None:
                return image, self.sources_weight[source]
            return self.transform(image, self.sources_weight[source])
        label_path = f"{source_dir}/annotations/{image_name[:-3]}txt"
        if self.task == "classification":
            label = self._load_classific_label(label_path, source)
        else:
            label = self._load_detection_label(
                label_path,
                image.shape[-2:],
                source,
            )
        if self.transform is None:
            return image, label
        return self.transform(image, label)

    def get_balanced_sampler(self, batch_size: int):
        return BalancedSampler(self, batch_size)


class BalancedSampler(Sampler[list[tuple[str, int]]]):
    def __init__(self, rdd_dataset: RDD2022, batch_size: int):
        self.batch_size = batch_size
        self._sources_count = {
            source: count
            for source, count in zip(rdd_dataset.sources, rdd_dataset.counts)
        }
        self._sources_participation = {
            source: max(1, round(batch_size * count / len(rdd_dataset)))
            for source, count in self._sources_count.items()
        }
        # Match participations to batch size exactly
        diff = self.batch_size - sum(self._sources_participation.values())
        inc = -1 if diff < 0 else 1
        inc_order = sorted(
            self._sources_participation,
            key=(lambda x: self._sources_participation[x]),
            reverse=(diff < 0),
        )
        for i in range(abs(diff)):
            self._sources_participation[inc_order[i]] += inc
        # For iteration
        self._total_batches = min(
            count // self._sources_participation[source]
            for source, count in self._sources_count.items()
        )
        self._batch = None
        self._indexes = {
            source: np.arange(count, dtype=np.int32)
            for source, count in self._sources_count.items()
        }

    def __len__(self):
        return self._total_batches

    def __iter__(self):
        self._batch = 0
        for indexes in self._indexes.values():
            np.random.shuffle(indexes)
        return self

    def __next__(self):
        if self._batch == self._total_batches:
            raise StopIteration("No more full batches")
        indices = []
        i = self._batch
        for source, size in self._sources_participation.items():
            indices.extend(
                (source, indx.item())
                for indx in self._indexes[source][i * size : (i + 1) * size]
            )
        self._batch += 1
        return indices


class BottomRandomResizedCrop(transforms_v2.RandomResizedCrop):
    _bottom_threshold = 0.6

    def make_params(self, flat_inputs):
        h, _ = query_size(flat_inputs)
        params = super().make_params(flat_inputs)
        thres = max(0, int(h * self._bottom_threshold - params["height"] / 2))
        if params["top"] > thres:
            return params
        limit = h - params["height"]
        params["top"] = (
            torch.randint(thres, limit, size=(1,)).item()
            if limit > thres
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
        augmentations = self.get_multiview_augmentations(
            global_size,
            global_scale,
            local_size,
            local_scale,
        )
        self.global1_aug = augmentations[0]
        self.global2_aug = augmentations[1]
        self.local_aug = augmentations[2]

    def forward(self, img, label=None):
        views = [self.global1_aug(img), self.global2_aug(img)]
        views.extend([self.local_aug(img) for _ in range(self.num_local_views)])
        return views, label

    @staticmethod
    def get_multiview_augmentations(
        global_size: int,
        global_scale: tuple[float, float],
        local_size: int,
        local_scale: tuple[float, float],
    ):
        # Geometric transformations
        local_crop = BottomRandomResizedCrop(local_size, local_scale)
        global_crop = BottomRandomResizedCrop(global_size, global_scale)
        hflip = transforms_v2.RandomHorizontalFlip(p=0.5)
        rotation = transforms_v2.RandomApply(
            [transforms_v2.RandomRotation(degrees=10, expand=False)],
            p=0.4,
        )
        # Photometric transformations
        ### CollorJitter is known to return NaN values
        ### Best to put it before any other photometric transformations
        ### Or apply it with the image in INT format/dtype
        collor_jitter = transforms_v2.RandomApply(
            [
                transforms_v2.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.2,
                    hue=0.1,
                )
            ],
            p=0.85,
        )
        solarize = transforms_v2.RandomSolarize(threshold=0.8, p=0.35)
        photometric_transforms = [
            transforms_v2.RandomPosterize(bits=6, p=0.35),
            transforms_v2.RandomApply(
                [transforms_v2.GaussianBlur(kernel_size=5, sigma=(0.2, 1.2))],
                p=0.75,
            ),
            transforms_v2.RandomGrayscale(p=0.25),
        ]
        # Global
        stronger_global_aug = transforms_v2.Compose(
            [
                rotation,
                global_crop,
                hflip,
                collor_jitter,
                solarize,
                *photometric_transforms,
            ]
        )
        weaker_global_aug = transforms_v2.Compose(
            [
                rotation,
                global_crop,
                hflip,
                collor_jitter,
                *photometric_transforms,
            ]
        )
        # Local
        local_aug = transforms_v2.Compose(
            [
                rotation,
                local_crop,
                hflip,
                collor_jitter,
                *photometric_transforms,
            ]
        )
        return weaker_global_aug, stronger_global_aug, local_aug

    @staticmethod
    def multiview_collate_fn(batch):
        # `batch` = [([v1, ..., vn], float), ...]
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
        # Collating sample weights
        weights = torch.tensor(
            [sample[1] for sample in batch],
            dtype=tensor_dtype,
        )
        return global_views, local_views, weights


class MultiCropBottomSplit(transforms_v2.Transform):
    _bottom_threshold = 0.6

    def __init__(
        self,
        crop_size: int,
        num_crops: int = 3,
        crops_overlap: float = 0.0,
    ):
        super().__init__()
        if num_crops < 2:
            raise ValueError("Only make sense for 2 or more crops.")
        if crop_size < 1:
            raise ValueError("`size` must be positive.")
        if crops_overlap >= 1.0:
            raise ValueError("`overlap` can not be >= 1.0")
        self.size = crop_size
        self.num_crops = num_crops
        self.overlap = crops_overlap
        self.local_cache = {}

    def make_params(self, flat_inputs):
        h, w = query_size(flat_inputs)
        try:
            crop_size = self.local_cache[w]
        except KeyError:
            crop_size = min(h, self._compute_split_crop_size(w))
            self.local_cache[w] = crop_size
        thres = max(0, int(h * self._bottom_threshold - crop_size / 2))
        limit = h - crop_size
        crops = [
            {
                "top": (
                    torch.randint(thres, limit, size=(1,)).item()
                    if limit > thres
                    else limit
                ),
                "left": int(left),
                "height": crop_size,
                "width": crop_size,
            }
            for left in np.linspace(0, w - crop_size, self.num_crops)
        ]
        return {"crops": crops}

    def transform(self, inpt, params):
        if not isinstance(inpt, torch.Tensor) or len(inpt.shape) == 1:
            return inpt
        crops = [
            transforms_v2.functional.resized_crop(
                inpt=inpt,
                size=self.size,
                **crop,
            )
            for crop in params["crops"]
        ]
        return crops

    def _compute_split_crop_size(self, width: int):
        n = self.num_crops
        crop_size = int(width / n)
        overlap = self.overlap
        increment = 1 if overlap > 0.0 else -1
        overlap = abs(overlap)
        while True:
            actual_overlap = (n * crop_size - width) / (crop_size * (n - 1))
            if actual_overlap >= overlap:
                break
            crop_size += increment
        return crop_size

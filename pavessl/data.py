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


def get_rdd2022_sources(subset_level: Literal[0, 1, 2, 3, 4]):
    subset_level = min(max(subset_level, 0), 4)
    if not subset_level:
        return ["India"]
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
        transform: Transform | torch.nn.Module | None = None,
    ):
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
        sources = sorted(sources)
        self.sources = sources
        self.split = split
        self.task = task
        self.transform = transform
        self.base_dir = data_path
        self.sources_base_dir = {
            source: (data_path / source / split).as_posix()
            for source in sources
        }
        # Handling sources with different classes
        self.sources_classes = {
            source: (
                (data_path / source / "classes.txt").read_text().splitlines()
            )
            for source in sources
        }
        self.classes = sorted(set().union(*self.sources_classes.values()))
        self.sources_classes = {
            source: [self.classes.index(cls_name) for cls_name in classes]
            for source, classes in self.sources_classes.items()
        }
        # Not all train images have annotations
        if split == "train" and task != "ssl":
            self.images = {
                source: [
                    f"{annotation.stem}.jpg"
                    for annotation in (
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
        self.weights = self._compute_weights()

    def _compute_weights(self):
        if self.split != "train" or self.task == "detection":
            weight = {"weight": 1.0}
            return {source: weight for source in self.sources}
        weights = {
            source: {
                "source_weight": self.acc_counts[-1]
                / (count * len(self.sources))
            }
            for source, count in zip(self.sources, self.counts)
        }
        if self.task == "ssl":
            return weights
        for source, count in zip(self.sources, self.counts):
            source_dir = self.sources_base_dir[source]
            classes_count = torch.zeros(len(self.classes))
            for image_name in self.images[source]:
                ann_path = f"{source_dir}/annotations/{image_name[:-3]}txt"
                try:
                    labels = [
                        self.sources_classes[source][idx]
                        for idx in load_bboxes(ann_path)["labels"]
                    ]
                except FileNotFoundError:
                    continue
                classes_count.add_(
                    torch.zeros(len(self.classes)).index_fill_(
                        dim=0,
                        index=torch.tensor(labels, dtype=torch.long),
                        value=1.0,
                    )
                )
            pos_weights = count / (2 * classes_count).clip(min=1)
            pos_weights = torch.where(classes_count > 0, pos_weights, 0)
            neg_weights = count / (2 * (count - classes_count)).clip(min=1)
            neg_weights = torch.where(classes_count > 0, neg_weights, 0)
            weights[source]["pos_weights"] = pos_weights
            weights[source]["neg_weights"] = neg_weights
        return weights

    def __len__(self):
        return self.acc_counts[-1]

    def _compute_location_from_index(self, index):
        for i, acc in enumerate(self.acc_counts[1:]):
            if index < acc:
                return self.sources[i], index - self.acc_counts[i]
        raise IndexError("Out of range")

    def _load_classification_label(self, annotation_path, source):
        labels = [
            self.sources_classes[source][idx]
            for idx in load_bboxes(annotation_path)["labels"]
        ]
        return {
            "target": torch.zeros(len(self.classes)).index_fill_(
                dim=0,
                index=torch.tensor(labels, dtype=torch.long),
                value=1.0,
            ),
            **self.weights[source],
        }

    def _load_detection_label(self, annotation_path, img_dim, source):
        bboxes = load_bboxes(annotation_path)
        bboxes["labels"] = [
            self.sources_classes[source][idx] for idx in bboxes["labels"]
        ]
        return {
            "boxes": tv_tensors.BoundingBoxes(
                data=bboxes["boxes"],
                format="XYXY",
                canvas_size=img_dim,
                clamping_mode="hard",
            ),
            "labels": torch.tensor(bboxes["labels"], dtype=torch.long),
            **self.weights[source],
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
            label = self.weights[source]
        else:
            label_path = f"{source_dir}/annotations/{image_name[:-3]}txt"
            label = (
                self._load_classification_label(label_path, source)
                if self.task == "classification"
                else self._load_detection_label(
                    label_path,
                    image.shape[-2:],
                    source,
                )
            )
        if self.transform is None:
            return image, label
        return self.transform(image, label)

    def get_balanced_sampler(self, batch_size: int):
        return SourceBalancedSampler(self, batch_size)


class SourceBalancedSampler(Sampler[list[tuple[str, int]]]):
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
        # `batch` = [([v1, ..., vn], dict[str, float]), ...]
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
        # Collating samples weight
        samples_weight = torch.tensor(
            [sample[1]["source_weight"] for sample in batch],
            dtype=tensor_dtype,
        )
        return global_views, local_views, samples_weight


class FourPointCrops(transforms_v2.Transform):
    def __init__(self, size: int, overlap: float):
        super().__init__()
        self.size = size
        self.overlap = overlap

    def make_params(self, flat_inputs):
        w, h = query_size(flat_inputs)
        crop_size = {
            "width": min(w, round(w * (1 + self.overlap) / 2)),
            "height": min(h, round(h * (1 + self.overlap) / 2)),
        }
        return [
            # Top-left
            {"top": 0, "left": 0, **crop_size},
            # Top-right
            {"top": 0, "left": w - crop_size["width"], **crop_size},
            # Bottom-left
            {"top": h - crop_size["height"], "left": 0, **crop_size},
            # Bottom-right
            {
                "top": h - crop_size["height"],
                "left": w - crop_size["width"],
                **crop_size,
            },
        ]

    def transform(self, inpt, params):
        if not isinstance(inpt, torch.Tensor) or len(inpt.shape) != 3:
            return inpt
        crops = [
            transforms_v2.functional.resized_crop(
                inpt=inpt,
                size=self.size,
                **crop,
            )
            for crop in params
        ]
        return crops


def get_augmentation(size: int, scale: tuple[float, float]):
    # Geometric transformations
    random_crop = BottomRandomResizedCrop(size, scale)
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
    return transforms_v2.Compose(
        [
            rotation,
            random_crop,
            hflip,
            collor_jitter,
            solarize,
            *photometric_transforms,
        ]
    )

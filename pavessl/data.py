from collections import defaultdict
from itertools import accumulate
from math import ceil
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
import torchvision.transforms.v2 as transforms_v2
import torchvision.transforms.v2.functional as F_v2
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import tv_tensors


def get_rdd2022_sources(subset_level: Literal[0, 1, 2, 3, 4]) -> list[str]:
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


class BBoxes(TypedDict):
    boxes: list[tuple[int, int, int, int]]
    labels: list[int]


def load_bboxes(annotation_path: str | Path) -> BBoxes:
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
        transform: transforms_v2.Transform | None = None,
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
                    for annotation in sorted(
                        (data_path / f"{source}/{split}/annotations").iterdir()
                    )
                ]
                for source in sources
            }
        else:
            self.images = {
                source: [
                    image.name
                    for image in sorted(
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
        if self.task != "classification" or self.split != "train":
            return None
        weights = defaultdict(dict)
        for source, count in zip(self.sources, self.counts):
            source_dir = self.sources_base_dir[source]
            classes_count = torch.zeros(len(self.classes))
            for image_name in self.images[source]:
                ann_path = f"{source_dir}/annotations/{image_name[:-3]}txt"
                labels = [
                    self.sources_classes[source][idx]
                    for idx in load_bboxes(ann_path)["labels"]
                ]
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
        classification_label = {
            "target": torch.zeros(len(self.classes)).index_fill_(
                dim=0,
                index=torch.tensor(labels, dtype=torch.long),
                value=1.0,
            ),
        }
        if self.weights is not None:
            classification_label.update(self.weights[source])
        return classification_label

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
        image = F_v2.to_dtype(
            F_v2.to_image(image),
            dtype=torch.float32,
            scale=True,
        )
        if self.task == "ssl":
            label = None
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

    def get_sources_sampler(self, batch_size: int):
        return SourcesSampler(self, batch_size // len(self.sources))


class SourcesSampler(Sampler[list[tuple[str, int]]]):
    def __init__(self, rdd_dataset: RDD2022, images_per_source: int):
        self.images_per_source = images_per_source
        self.sources = rdd_dataset.sources
        self.counts = rdd_dataset.counts
        # For iteration
        self._total_batches = min(self.counts) // self.images_per_source
        self._current_batch = self._total_batches
        self._indexes = tuple(
            np.arange(count, dtype=np.int32) for count in self.counts
        )

    def __len__(self):
        return self._total_batches

    def __iter__(self):
        self._current_batch = 0
        for indexes in self._indexes:
            np.random.shuffle(indexes)
        return self

    def __next__(self):
        if self._current_batch == self._total_batches:
            raise StopIteration("No more full batches")
        batch = []
        start = self._current_batch * self.images_per_source
        end = start + self.images_per_source
        for i, source in enumerate(self.sources):
            batch.extend(
                (source, idx.item())
                for idx in self._indexes[i][start:end]
            )
        self._current_batch += 1
        return batch


class BottomRandomResizedCrop(transforms_v2.RandomResizedCrop):
    _bottom_threshold = 0.6

    def make_params(self, flat_inputs):
        h, _ = transforms_v2.query_size(flat_inputs)
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


def get_augmentation(
    size: int,
    scale: tuple[float, float],
    stronger: bool = True,
) -> transforms_v2.Transform:
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
    photometric_transforms = [
        transforms_v2.RandomApply(
            [
                transforms_v2.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.2,
                    hue=0.1,
                )
            ],
            p=0.85,
        ),
    ]
    if stronger:
        photometric_transforms.append(
            transforms_v2.RandomSolarize(threshold=0.85, p=0.25),
        )
    photometric_transforms.extend(
        [
            transforms_v2.RandomApply(
                [transforms_v2.GaussianBlur(kernel_size=5, sigma=(0.2, 1.0))],
                p=0.7,
            ),
            transforms_v2.RandomGrayscale(p=0.25),
        ],
    )
    # Augmentations composition
    augmentations = transforms_v2.Compose(
        [
            rotation,
            random_crop,
            hflip,
            *photometric_transforms,
        ]
    )
    return augmentations


class SSLMultiviewTransform(transforms_v2.Transform):
    def __init__(
        self,
        global_size: int,
        global_scale: tuple[float, float],
        local_size: int,
        local_scale: tuple[float, float],
        num_local_views: int,
    ):
        super().__init__()
        self.num_local_views = num_local_views
        self.global_aug_1 = get_augmentation(
            global_size,
            global_scale,
            stronger=True,
        )
        self.global_aug_2 = get_augmentation(
            global_size,
            global_scale,
            stronger=False,
        )
        self.local_aug = get_augmentation(
            local_size,
            local_scale,
            stronger=False,
        )

    def transform(self, inpt, params):
        if not isinstance(inpt, torch.Tensor) or len(inpt.shape) != 3:
            return inpt
        views = [self.global_aug_1(inpt), self.global_aug_2(inpt)]
        views.extend(
            [self.local_aug(inpt) for _ in range(self.num_local_views)]
        )
        return views

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


class FourCornersCrop(transforms_v2.Transform):
    def __init__(self, size: int, overlap: float = 0.1):
        super().__init__()
        self.size = size
        self.overlap = overlap

    def make_params(self, flat_inputs):
        w, h = transforms_v2.query_size(flat_inputs)
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

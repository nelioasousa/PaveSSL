from pathlib import Path
from typing import Literal
from itertools import accumulate

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import Transform, query_size
import torchvision.transforms.v2 as transforms_v2
import torchvision.transforms.v2.functional as F

from pavessl.utils import load_bboxes


class RDD2022(Dataset):
    def __init__(
        self,
        data_path: Path,
        countries: str | list[str] | Literal["all"],
        split: Literal["train", "valid", "test"],
        task: Literal["ssl", "classification", "detection"],
        transform: Transform | None = None,
    ):
        data_path = data_path.resolve()
        if not data_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {data_path}")
        all_countries = [c.name for c in data_path.iterdir() if c.is_dir()]
        if countries == "all":
            countries = all_countries
        else:
            if isinstance(countries, str):
                countries = [countries]
            unknown = frozenset(countries).difference(all_countries)
            if unknown:
                raise ValueError(f"Unknown country folders: {list(unknown)}")
        self.base = data_path
        self.split = split
        self.countries = countries
        self.task = task
        self.transform = transform
        self.classes = (data_path / "classes.txt").read_text().splitlines()
        # Not all train images have annotations
        if split == "train" and task != "ssl":
            self.images = {
                country: [
                    f"{label.stem}.jpg"
                    for label
                    in (data_path / f"{country}/{split}/annotations").iterdir()
                ]
                for country in countries
            }
        else:
            self.images = {
                country: [
                    image.name
                    for image
                    in (data_path / f"{country}/{split}/images").iterdir()
                ]
                for country in countries
            }
        self.acc_counts = list(accumulate(
            len(self.images[country]) for country in countries
        ))
        self.acc_counts.insert(0, 0)
        return

    def __len__(self):
        return self.acc_counts[-1]

    def _compute_from_global_index(self, index):
        for i, acc in enumerate(self.acc_counts[1:]):
            if index < acc:
                return self.countries[i], index - self.acc_counts[i]
        raise IndexError("Out of range")

    def _load_label_for_classification(self, label_path):
        labels = load_bboxes(label_path)["labels"]
        # Multi-hot encoding
        onehot = [0] * len(self.classes)
        onehot[labels] = 1
        return torch.tensor(onehot)

    def _load_label_for_detection(self, label_path, dim):
        bboxes = load_bboxes(label_path)
        return {
            "boxes": tv_tensors.BoundingBoxes(
                bboxes["boxes"], format="XYXY",
                canvas_size=dim, clamping_mode="hard",
            ),
            "labels": torch.tensor(bboxes["labels"]),
        }

    def __getitem__(self, index):
        country, local_index = self._compute_from_global_index(index)
        image_name = self.images[country][local_index]
        split_dir = self.base / f"{country}/{self.split}"
        image = Image.open(split_dir / f"images/{image_name}").convert("RGB")
        image = F.to_dtype(F.to_image(image), dtype=torch.float32, scale=True)
        if self.task == "ssl":
            if self.transform is None:
                return image
            return self.transform(image)
        label_path = split_dir / f"annotations/{image_name[:-3]}txt"
        if self.task == "classification":
            label = self._load_label_for_classification(label_path)
        else:
            label = self._load_label_for_detection(label_path, image.shape[-2:])
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
            global_size, global_scale, local_size, local_scale,
        )
        self.teacher_global = augs[0]
        self.student_global = augs[1]
        self.student_local = augs[2]

    def forward(self, img, label=None):
        views = [self.teacher_global(img), self.student_global(img)]
        views.extend([
            self.student_local(img) for _ in range(self.num_local_views)
        ])
        return views, label

    @staticmethod
    def get_multiview_augmentations(
        global_size: int = 224,
        global_scale: tuple[float, float] = (0.75, 1.0),
        local_size: int = 96,
        local_scale: tuple[float, float] = (0.05, 0.4),
    ):
        # Random cropping
        global_crop = transforms_v2.RandomResizedCrop(
            size=global_size,
            scale=global_scale,
        )
        local_crop = BottomRandomResizedCrop(size=local_size, scale=local_scale)
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
        student_global_aug = transforms_v2.Compose([
            global_crop,
            *geometric_transfs,
            *photometric_transfs,
        ])
        ## Local
        student_local_aug = transforms_v2.Compose([
            local_crop,
            *geometric_transfs,
            *photometric_transfs,
        ])
        # Teacher
        teacher_aug = transforms_v2.Compose([
            global_crop,
            *geometric_transfs,
            photometric_transfs[0],
            *photometric_transfs[-2:],
        ])
        return teacher_aug, student_global_aug, student_local_aug

    @staticmethod
    def multiview_collate_fn(batch):
        B = len(batch)  # batch size
        tensor_dtype = batch[0][0].dtype
        # Collating global views
        global_dim = batch[0][0].shape[-3:]
        global_views = torch.empty((2 * B, *global_dim), dtype=tensor_dtype)
        torch.stack([views[0] for views in batch], dim=0, out=global_views[:B])
        torch.stack([views[1] for views in batch], dim=0, out=global_views[B:])
        # Collating local views
        local_dim = batch[0][-1].shape[-3:]
        num_local_views = len(batch[0]) - 2
        local_views = torch.empty((num_local_views * B, *local_dim), dtype=tensor_dtype)
        for i in range(num_local_views):
            torch.stack([views[i+2] for views in batch], dim=0, out=local_views[i*B:(i+1)*B])
        return global_views, local_views

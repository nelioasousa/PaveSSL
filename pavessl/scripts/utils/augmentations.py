import argparse
from pathlib import Path

import torch
import torchvision.transforms.v2.functional as F_v2
from PIL import Image

from pavessl import GLOBAL_SCALE, GLOBAL_SIZE, LOCAL_SCALE, data


def _main(args):
    args.save_dir.mkdir(exist_ok=True, parents=True)
    half_size = GLOBAL_SIZE // 2
    g_size = int(1.01 * GLOBAL_SIZE)
    global_transform = data.get_augmentation(g_size, GLOBAL_SCALE)
    local_transform = data.get_augmentation(half_size, LOCAL_SCALE)
    for processed, img_path in enumerate(args.images.glob(args.glob)):
        if processed >= args.max_images:
            break
        save_dir = args.save_dir / img_path.stem
        save_dir.mkdir(exist_ok=True)
        img = Image.open(img_path).convert("RGB")
        img.save(save_dir / img_path.name)
        img = F_v2.to_dtype(F_v2.to_image(img), dtype=torch.float32, scale=True)
        # Multi-crop strategy
        orig_resized = F_v2.resize(img, int(1.52 * GLOBAL_SIZE))
        c, h, w = orig_resized.shape
        grid_img = orig_resized.new_zeros((c, h, int(w + 1.02 * GLOBAL_SIZE)))
        grid_img[:, :, :w].copy_(orig_resized)
        grid_img[:, :g_size, -g_size:].copy_(global_transform(img))
        for i in range(2):
            start = grid_img.shape[-1] - g_size + i * int(1.02 * half_size)
            grid_img[:, -half_size:, start : start + half_size].copy_(
                local_transform(img)
            )
        grid_img = F_v2.to_pil_image(grid_img)
        grid_img.save(save_dir / "multi_views.jpg")


def _get_args_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Apply augmentations to images for visualization purposes."
        ),
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing images to be augmentated.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        required=True,
        help="Directory path in which to save the augmentations.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.jpg",
        help="Glob pattern for image search in the target directory.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Limit the number of images to be augmented.",
    )
    return parser


def _run():
    try:
        parser = _get_args_parser()
        args = parser.parse_args()
        _main(args)
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())

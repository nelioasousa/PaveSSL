import argparse
from pathlib import Path

import torch
import torchvision.transforms.v2.functional as F_v2
from PIL import Image

from pavessl.data import MultiCropBottomSplit, SSLMultiviewTransform
from pavessl.scripts.pretrain.ssl_pretrain import add_multiview_args


def _main(args):
    args.save_dir.mkdir(exist_ok=True, parents=True)
    multiview = SSLMultiviewTransform(
        global_size=args.global_size,
        global_scale=args.global_scale,
        local_size=args.local_size,
        local_scale=args.local_scale,
        num_local_views=args.num_local_views,
    )
    bottomsplit = MultiCropBottomSplit(
        crop_size=args.crop_size,
        num_crops=args.num_crops,
        crops_overlap=args.crops_overlap,
    )
    for processed, img_path in enumerate(args.images.glob("*.jpg")):
        if processed >= args.max_images:
            break
        save_dir = args.save_dir / img_path.stem
        save_dir.mkdir(exist_ok=True)
        img = Image.open(img_path).convert("RGB")
        img.save(save_dir / img_path.name)
        img = F_v2.to_dtype(F_v2.to_image(img), dtype=torch.float32, scale=True)
        multiview_augs = multiview(img)[0]
        for i, global_view in enumerate(multiview_augs[:2], start=1):
            global_view = F_v2.to_pil_image(global_view)
            save_path = save_dir / f"global_{i}.jpg"
            global_view.save(save_path)
        for i, local_view in enumerate(multiview_augs[2:], start=3):
            local_view = F_v2.to_pil_image(local_view)
            save_path = save_dir / f"local_{i}.jpg"
            local_view.save(save_path)
        img_splits = bottomsplit(img)
        for i, split in enumerate(img_splits, start=1):
            split = F_v2.to_pil_image(split)
            save_path = save_dir / f"split_{i}.jpg"
            split.save(save_path)


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Apply augmentations to images for visualization purposes."
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Dictory containing images to be augmentated.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        required=True,
        help="Directory path in which to save the augmentations.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Limit the number of images to be augmented.",
    )
    # Multi-view
    parser = add_multiview_args(parser)
    # k-NN eval
    parser.add_argument(
        "--num-crops",
        type=int,
        default=3,
        help="Split images in N >= 2 crops for kNN evaluation.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=448,
        help="Resize crops to this dimension.",
    )
    parser.add_argument(
        "--crops-overlap",
        type=float,
        default=0.3,
        help="Percentual overlap between split crops.",
    )
    return parser


def run():
    try:
        parser = get_args_parser()
        args = parser.parse_args()
        _main(args)
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(run())

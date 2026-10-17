import argparse
from pathlib import Path

import torch
import torchvision.transforms.v2.functional as F_v2
from PIL import Image

from pavessl.data import FourPointCrops, SSLMultiviewTransform, get_augmentation
from pavessl.scripts.pretrain.ssl_pretrain import add_multiview_transform_args
from pavessl.scripts.pretrain.sup_pretrain import add_supervised_transform_args
from pavessl.scripts.eval.knn_eval import add_knn_transform_args


def _main(args):
    args.save_dir.mkdir(exist_ok=True, parents=True)
    multiview_transform = SSLMultiviewTransform(
        global_size=args.mv_global_size,
        global_scale=args.mv_global_scale,
        local_size=(args.mv_global_size // 2),
        local_scale=args.mv_local_scale,
        num_local_views=4,
    )
    knn_transform = FourPointCrops(
        size=args.knn_crops_size,
        overlap=args.knn_crops_overlap,
    )
    sup_transform = get_augmentation(args.sup_image_size, args.sup_crop_scale)
    for processed, img_path in enumerate(args.images.glob(args.glob)):
        if processed >= args.max_images:
            break
        save_dir = args.save_dir / img_path.stem
        save_dir.mkdir(exist_ok=True)
        img = Image.open(img_path).convert("RGB")
        img.save(save_dir / img_path.name)
        img = F_v2.to_dtype(F_v2.to_image(img), dtype=torch.float32, scale=True)
        # Multiview SSL pre-train
        multiview_aug = multiview_transform(img)[0]
        multiview_img = torch.cat(multiview_aug[:2], dim=-1)
        multiview_img = torch.cat(
            (multiview_img, torch.cat(multiview_aug[2:], dim=-1)),
            dim=-2,
        )
        multiview_img = F_v2.to_pil_image(multiview_img)
        multiview_img.save(save_dir / "ssl_multiview.jpg")
        # kNN
        img_corners = knn_transform(img)
        c, h, w = img_corners[0].shape
        knn_img = img_corners[0].new_zeros((c, 2 * h, 2 * w))
        knn_img[:, :h, :w].copy_(img_corners[0])
        knn_img[:, :h, -w:].copy_(img_corners[1])
        knn_img[:, -h:, :w].copy_(img_corners[2])
        knn_img[:, -h:, -w:].copy_(img_corners[3])
        knn_img = F_v2.to_pil_image(knn_img)
        knn_img.save(save_dir / "knn_eval.jpg")
        # Supervised pre-train
        sup_img = sup_transform(img)
        sup_img = F_v2.to_pil_image(sup_img)
        sup_img.save(save_dir / "sup_pretrain.jpg")


def get_args_parser():
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
    parser = add_multiview_transform_args(parser)
    parser = add_supervised_transform_args(parser)
    parser = add_knn_transform_args(parser)
    return parser


def _run():
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
    raise SystemExit(_run())

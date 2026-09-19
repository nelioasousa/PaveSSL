import argparse
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from math import ceil, floor
from pathlib import Path

from PIL import Image
from tqdm import tqdm

random.seed(42)


CLASS_NAMES_MAP = {
    "D00": "crack_longitudinal",
    "D0w0": "crack_longitudinal",
    "D10": "crack_transverse",
    "D20": "crack_interconnected",
    "D40": "pothole",
    "Block crack": "crack_interconnected",
    "Repair": "repair",
    "D01": "construction_joint",  # longitudinal
    "D11": "construction_joint",  # transverse
    "D50": "manhole",
    "D43": "blur_crosswalk",
    "D44": "blur_white_line",
}

CLASSES = sorted(set(CLASS_NAMES_MAP.values()))

CLASS_INDEXES = {class_name: i for i, class_name in enumerate(CLASSES)}

NORWAY_TARGET_SIZE = 2000


# To cut Norway images
def _cut_image_and_fix_bboxes(img, bboxes):
    img_height = img.size[1]
    diff = img_height - NORWAY_TARGET_SIZE
    cut_bboxes = []
    for class_name, xmin, ymin, xmax, ymax in bboxes:
        box_width = xmax - xmin
        box_remain = img_height - xmin
        # Discard bbox if remaining width is small
        if box_remain < (box_width / 16):
            continue
        cut_bboxes.append(
            (
                class_name,
                xmin,
                ymin - diff,
                min(img_height, xmax),
                ymax - diff,
            )
        )
    return img.crop((0, diff, NORWAY_TARGET_SIZE, img_height)), cut_bboxes


def _get_bboxes_from_xml_label(label_path: Path):
    label = ET.parse(label_path)
    bboxes = []
    for object in label.iter("object"):
        class_name = object.findtext("name", None)
        bbox = object.find("bndbox")
        if class_name is None or bbox is None:
            continue
        try:
            xmin = round(float(bbox.findtext("xmin")))
            ymin = round(float(bbox.findtext("ymin")))
            xmax = round(float(bbox.findtext("xmax")))
            ymax = round(float(bbox.findtext("ymax")))
        except TypeError:
            continue
        bboxes.append((CLASS_NAMES_MAP[class_name], xmin, ymin, xmax, ymax))
    return bboxes


def _save_bboxes_to_annotation(bboxes, ann_save_path):
    label_lines = [
        "{:02d} {:04d} {:04d} {:04d} {:04d}\n".format(
            CLASS_INDEXES[cls_name],
            *coords,
        )
        for cls_name, *coords in bboxes
    ]
    with open(ann_save_path, mode="w") as label_file:
        label_file.writelines(label_lines)


def _get_train_test_samples(source_folder: Path):
    train_samples = [
        (
            img_path,
            img_path.parents[1] / f"annotations/xmls/{img_path.stem}.xml",
        )
        for img_path in (source_folder / "train/images").iterdir()
    ]
    if not (source_folder / "test").is_dir():
        return train_samples, []
    test_samples = list((source_folder / "test/images").iterdir())
    return train_samples, test_samples


def _get_stratified_splits(
    image_label_pairs: list[tuple[Path, Path]],
    train_size: int,
    valid_size: int,
    test_size: int,
):
    pairs_by_class_group = defaultdict(list)
    for pair in image_label_pairs:
        bboxes = _get_bboxes_from_xml_label(pair[1])
        class_group = tuple(
            sorted(frozenset(CLASS_INDEXES[name] for name, *_ in bboxes))
        )
        pairs_by_class_group[class_group].append(pair)
    train_pairs, valid_pairs, test_pairs = [], [], []
    total_size = train_size + valid_size + test_size
    test_proportion = test_size / total_size
    valid_proportion = valid_size / total_size
    for pairs in pairs_by_class_group.values():
        test_num_samples = ceil(len(pairs) * test_proportion)
        valid_num_samples = floor(len(pairs) * valid_proportion)
        test_valid_size = test_num_samples + valid_num_samples
        random.shuffle(pairs)
        test_pairs.extend(pairs[:test_num_samples])
        valid_pairs.extend(pairs[test_num_samples:test_valid_size])
        train_pairs.extend(pairs[test_valid_size:])
    return train_pairs, valid_pairs, test_pairs


def _get_available_classes(img_ann_pairs: tuple[Path, Path]):
    classes = set()
    for _, ann in img_ann_pairs:
        bboxes = _get_bboxes_from_xml_label(ann)
        classes.update(cls_name for cls_name, *_ in bboxes)
    return sorted(classes)


def _train_valid_test_split(
    source_folder: Path,
    train_proportion: float,
    valid_proportion: float,
):
    if (
        train_proportion < 0.0
        or valid_proportion < 0.0
        or (train_proportion + valid_proportion > 1.0)
    ):
        raise ValueError("Invalid proportions")
    train_samples, test_samples = _get_train_test_samples(source_folder)
    classes = _get_available_classes(train_samples)
    total_samples = len(train_samples) + len(test_samples)
    test_size = ceil(
        total_samples * (1.0 - train_proportion - valid_proportion)
    )
    valid_size = floor(total_samples * valid_proportion)
    train_size = total_samples - test_size - valid_size
    train_split = [(img_path, None) for img_path in test_samples]
    train_size -= len(train_split)
    if train_size < 0:
        raise ValueError(
            "Train size must be at least {} (proportion: {:.2f})".format(
                len(test_samples),
                len(test_samples) / total_samples,
            )
        )
    inc_train, valid_split, test_split = _get_stratified_splits(
        train_samples,
        train_size,
        valid_size,
        test_size,
    )
    train_split.extend(inc_train)
    return classes, train_split, valid_split, test_split


def _prepare_dst_folder(dest_folder: Path):
    dest_folder.mkdir(exist_ok=True)
    for split in ("train", "valid", "test"):
        split_folder = dest_folder / split
        split_folder.mkdir(exist_ok=True)
        (split_folder / "images").mkdir(exist_ok=True)
        (split_folder / "annotations").mkdir(exist_ok=True)


def main(data_path: Path, train_prop: float, valid_prop: float, save_dir: Path):
    save_dir.mkdir(exist_ok=True, parents=True)
    for source_folder in tqdm(list(data_path.iterdir())):
        dst_folder = save_dir / source_folder.name
        _prepare_dst_folder(dst_folder)
        classes, train_split, valid_split, test_split = _train_valid_test_split(
            source_folder,
            train_prop,
            valid_prop,
        )
        (dst_folder / "classes.txt").write_text(
            "\n".join(f"{CLASS_INDEXES[cls]} {cls}" for cls in classes)
        )
        for img_path, label_path in train_split:
            img = Image.open(img_path)
            bboxes = (
                []
                if label_path is None
                else _get_bboxes_from_xml_label(label_path)
            )
            if source_folder.name == "Norway":
                img, bboxes = _cut_image_and_fix_bboxes(img, bboxes)
            img.save(dst_folder / f"train/images/{img_path.name}")
            if label_path is None:
                continue
            ann_dst = dst_folder / f"train/annotations/{label_path.stem}.txt"
            _save_bboxes_to_annotation(bboxes, ann_dst)
        for split, pairs in (("valid", valid_split), ("test", test_split)):
            for img_path, label_path in pairs:
                img = Image.open(img_path)
                bboxes = _get_bboxes_from_xml_label(label_path)
                if source_folder.name == "Norway":
                    img, bboxes = _cut_image_and_fix_bboxes(img, bboxes)
                img.save(dst_folder / f"{split}/images/{img_path.name}")
                ann_dst = (
                    dst_folder / f"{split}/annotations/{label_path.stem}.txt"
                )
                _save_bboxes_to_annotation(bboxes, ann_dst)
    (save_dir / "classes.txt").write_text("\n".join(CLASSES))


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the RDD2022 dataset for "
            "pretraining, fine-tuning and evaluation."
        ),
    )
    parser.add_argument(
        "-tp",
        "--train_proportion",
        type=float,
        default=0.75,
        help="Proportion of the train split.",
    )
    parser.add_argument(
        "-vp",
        "--valid_proportion",
        type=float,
        default=0.1,
        help="Proportion of the validation split.",
    )
    parser.add_argument(
        "data",
        type=Path,
        default="data/RDD2022",
        nargs="?",
        help="Path to the RDD2022 dataset.",
    )
    parser.add_argument(
        "save_dir",
        type=Path,
        default="data/RDD2022_proc",
        nargs="?",
        help="Where to save the prepared dataset.",
    )
    return parser


def run():
    try:
        parser = get_arg_parser()
        args = parser.parse_args()
        main(
            args.data.resolve(),
            args.train_proportion,
            args.valid_proportion,
            args.save_dir.resolve(),
        )
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(run())

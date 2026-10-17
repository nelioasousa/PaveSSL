import argparse
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from json import dump as json_dump
from pathlib import Path
from shutil import copy

from tqdm import tqdm


def _main(args):
    args.save_dir.mkdir(exist_ok=True, parents=True)
    sizes_count = defaultdict(lambda: defaultdict(int))
    objects_count = defaultdict(lambda: defaultdict(int))
    samples_count = defaultdict(lambda: defaultdict(int))
    for source_dir in tqdm(list(args.data.iterdir())):
        source_name = source_dir.name
        samples_dir = args.save_dir / source_name
        samples_dir.mkdir(exist_ok=True)
        annotations_folder = source_dir / "train/annotations/xmls"
        images_folder = source_dir / "train/images"
        for label in annotations_folder.iterdir():
            root = ET.parse(label)
            size_tag = root.find("size")
            size = (
                "?x?"
                if size_tag is None
                else "{}x{}".format(
                    size_tag.findtext("width", "?"),
                    size_tag.findtext("height", "?"),
                )
            )
            sizes_count[source_name][size] += 1
            label_obj_count = Counter(
                obj.findtext("name", "?") for obj in root.iter("object")
            )
            for obj, count in label_obj_count.items():
                objects_count[source_name][obj] += count
                if samples_count[source_name][obj] < args.num_samples:
                    sample_dir = samples_dir / "_".join(obj.split())
                    sample_dir.mkdir(exist_ok=True)
                    image_name = f"{label.stem}.jpg"
                    copy(images_folder / image_name, sample_dir / image_name)
                    samples_count[source_name][obj] += 1
    with (args.save_dir / "overview.json").open("w") as f:
        json_dump([sizes_count, objects_count], f, indent=2, sort_keys=True)


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description="A simple overview of the RDD2022 dataset training sets.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/RDD2022"),
        help="Path to the RDD2022 dataset.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("data/RDD2022_overview"),
        help="Where to save the overview results.",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples selected per object class.",
    )
    return parser


def _run():
    try:
        parser = get_arg_parser()
        args = parser.parse_args()
        _main(args)
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())

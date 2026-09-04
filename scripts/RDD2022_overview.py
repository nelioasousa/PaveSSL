import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict, Counter
from json import dump
from shutil import copy
from tqdm import tqdm


def main(data_path: Path, save_dir: Path, num_samples: int):
    save_dir.mkdir(exist_ok=True, parents=True)
    sizes_count: dict[str, dict[str, int]] = defaultdict((lambda : defaultdict(int)))
    objects_count: dict[str, dict[str, int]] = defaultdict((lambda : defaultdict(int)))
    samples_count: dict[str, dict[str, int]] = defaultdict((lambda : defaultdict(int)))
    for country_dir in tqdm(list(data_path.iterdir())):
        country_name = country_dir.name
        samples_dir = save_dir / country_name
        samples_dir.mkdir(exist_ok=True)
        annotations_folder = country_dir / "train/annotations/xmls"
        images_folder = country_dir / "train/images"
        for label in annotations_folder.iterdir():
            root = ET.parse(label)
            size_tag = root.find("size")
            size = "?x?" if size_tag is None else "%sx%s" %(
                size_tag.findtext("width", "?"),
                size_tag.findtext("height", "?"),
            )
            sizes_count[country_name][size] += 1
            label_obj_count = Counter(
                obj.findtext("name", "?")
                for obj in root.iter("object")
            )
            for obj, count in label_obj_count.items():
                objects_count[country_name][obj] += count
                if samples_count[country_name][obj] < num_samples:
                    sample_dir = samples_dir / "_".join(obj.split())
                    sample_dir.mkdir(exist_ok=True)
                    image_name = f"{label.stem}.jpg"
                    copy(images_folder / image_name, sample_dir / image_name)
                    samples_count[country_name][obj] += 1
    with (save_dir / "overview.json").open("w") as f:
        dump([sizes_count, objects_count], f, indent=2, sort_keys=True)
    return


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data",
        type=str, default="data/RDD2022/", nargs='?',
        help="Path to the RDD2022 dataset relative to the script parent folder.",
    )
    parser.add_argument(
        "save_dir",
        type=str, default="data/RDD2022_overview/", nargs='?',
        help="Where to save the overview results relative to the script parent folder.",
    )
    parser.add_argument(
        "-n", "--num_samples",
        type=int, default=50,
        help="Number of samples selected per object class.",
    )

    args = parser.parse_args()

    data: str = args.data
    save_dir: str = args.save_dir
    num_samples: int = args.num_samples
    base_dir = Path(__file__).absolute().resolve().parent
    data_path = base_dir / data.removeprefix("/") # Input
    save_path = base_dir / save_dir.removeprefix("/") # Output

    try:
        main(data_path, save_path, num_samples)
        sys.exit(0)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)

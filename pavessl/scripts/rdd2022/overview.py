import argparse
from collections import Counter, defaultdict
from json import dump as json_dump
from pathlib import Path

from PIL import Image, ImageDraw

from pavessl.scripts.rdd2022.prepare import get_bboxes_from_xml_label


def _main(data_path: Path, save_dir: Path, num_samples: int):
    save_dir.mkdir(exist_ok=True, parents=True)
    for source_dir in sorted(data_path.iterdir()):
        sizes_count = defaultdict(int)
        objects_count = defaultdict(int)
        samples_count = defaultdict(int)
        source_save_dir = save_dir / source_dir.name
        source_save_dir.mkdir(exist_ok=True)
        source_samples_dir = source_save_dir / "samples"
        source_samples_dir.mkdir(exist_ok=True)
        images_dir = source_dir / "train/images"
        annots_dir = source_dir / "train/annotations/xmls"
        classes = set()
        for label in annots_dir.iterdir():
            img = Image.open(images_dir / f"{label.stem}.jpg")
            sizes_count["{}x{}".format(*img.size)] += 1
            bboxes = get_bboxes_from_xml_label(label)
            objs_count = Counter(bbox[0] for bbox in bboxes)
            for obj_name, count in objs_count.items():
                classes.add(obj_name)
                objects_count[obj_name] += count
                if samples_count[obj_name] < num_samples:
                    sample_img = img.copy()
                    draw = ImageDraw.Draw(sample_img)
                    for name, *coords in bboxes:
                        draw.rectangle(
                            coords,
                            outline=("red" if name == obj_name else "cyan"),
                            width=2,
                        )
                    obj_folder = "__".join(obj_name.strip().split())
                    obj_samples_dir = source_samples_dir / obj_folder
                    try:
                        sample_img.save(obj_samples_dir / f"{label.stem}.jpg")
                    except FileNotFoundError:
                        obj_samples_dir.mkdir()
                        sample_img.save(obj_samples_dir / f"{label.stem}.jpg")
                    samples_count[obj_name] += 1
        with (source_save_dir / "classes.txt").open("w") as classes_file:
            classes_file.write("\n".join(sorted(classes)))
        with (source_save_dir / "summary.json").open("w") as summary_file:
            json_dump(
                obj={"sizes": sizes_count, "objects": objects_count},
                fp=summary_file,
                indent=2,
            )


def _get_args_parser():
    parser = argparse.ArgumentParser(
        description="A simple overview of the RDD2022 dataset training sets.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/RDD2022"),
        help="Path to the original RDD2022 dataset.",
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
        parser = _get_args_parser()
        args = parser.parse_args()
        _main(
            data_path=args.data,
            save_dir=args.save_dir,
            num_samples=args.num_samples,
        )
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())

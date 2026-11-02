"""Split the raw ASL dataset into train / val / test folders.

The dataset has 10 people (P1-P10). For every sign, each person was recorded
as 100 consecutive video frames, so neighbouring images are nearly identical.
Splitting images at random puts almost the same picture in train and test,
which makes test accuracy look much better than it really is.

By default this script splits by PERSON instead: the val and test people are
never seen during training, so the test score shows how well the model works
on someone new (which is what the webcam demo needs).

Examples:
    python data_splitting.py --source path/to/ASL_Raw_Images/asl_dataset
    python data_splitting.py --source path/to/asl_dataset --dry-run
    python data_splitting.py --source path/to/asl_dataset --mode random --dest dataset_random

Default person split: train P1-P7, val P8, test P9-P10 (about 70/10/20).
Filenames look like P10_1_940.jpg -> person P10, sign 1, frame 940.
"""

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Split the ASL dataset")
    p.add_argument("--source", type=Path, default=Path("asl_dataset"),
                   help="Folder with one subfolder per class (0-9, A-Z)")
    p.add_argument("--dest", type=Path, default=Path("dataset"),
                   help="Output folder (gets train/, val/, test/)")
    p.add_argument("--mode", choices=["person", "random"], default="person",
                   help="person = hold out whole people (default); random = old per-image split")
    p.add_argument("--val", nargs="+", default=["P8"], help="People used for validation")
    p.add_argument("--test", nargs="+", default=["P9", "P10"], help="People used for testing")
    p.add_argument("--train-ratio", type=float, default=0.70, help="random mode only")
    p.add_argument("--val-ratio", type=float, default=0.15, help="random mode only")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="Only print counts, copy nothing")
    return p.parse_args()


def person_of(path):
    return path.name.split("_", 1)[0].upper()


def split_by_person(images, val_people, test_people):
    splits = {"train": [], "val": [], "test": []}
    for img in images:
        person = person_of(img)
        if person in test_people:
            splits["test"].append(img)
        elif person in val_people:
            splits["val"].append(img)
        else:
            splits["train"].append(img)
    return splits


def split_randomly(images, train_ratio, val_ratio, rng):
    images = images[:]
    rng.shuffle(images)
    n_train = int(len(images) * train_ratio)
    n_val = int(len(images) * val_ratio)
    return {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }


def main():
    args = parse_args()
    val_people = {p.upper() for p in args.val}
    test_people = {p.upper() for p in args.test}
    if val_people & test_people:
        raise SystemExit(f"These people are in both val and test: {val_people & test_people}")
    if not args.source.is_dir():
        raise SystemExit(f"Source folder not found: {args.source}")
    if not args.dry_run and args.dest.exists() and any(args.dest.iterdir()):
        raise SystemExit(f"{args.dest} already has files in it. Delete it first or use --dest "
                         "with a new folder, so old and new splits don't get mixed.")

    rng = random.Random(args.seed)
    totals = Counter()
    people_per_split = {"train": set(), "val": set(), "test": set()}

    for class_folder in sorted(args.source.iterdir()):
        if not class_folder.is_dir():
            continue
        images = sorted(p for p in class_folder.iterdir() if p.suffix.lower() in EXTENSIONS)

        if args.mode == "person":
            splits = split_by_person(images, val_people, test_people)
        else:
            splits = split_randomly(images, args.train_ratio, args.val_ratio, rng)

        for split_name, files in splits.items():
            totals[split_name] += len(files)
            people_per_split[split_name].update(person_of(f) for f in files)
            if args.dry_run:
                continue
            out = args.dest / split_name / class_folder.name
            out.mkdir(parents=True, exist_ok=True)
            for file in files:
                shutil.copy2(file, out / file.name)

        counts = ", ".join(f"{k} {len(v)}" for k, v in splits.items())
        print(f"{class_folder.name}: {len(images)} images -> {counts}")

    print(f"\nMode: {args.mode}")
    for split_name in ("train", "val", "test"):
        people = sorted(people_per_split[split_name], key=lambda s: (len(s), s))
        print(f"{split_name:5s}: {totals[split_name]:6d} images | people: {', '.join(people)}")
    if not totals["val"] or not totals["test"]:
        print("Warning: val or test is empty. Check the --val/--test names match the filenames.")
    print("Dry run, nothing copied." if args.dry_run else f"Dataset split completed -> {args.dest}")


if __name__ == "__main__":
    main()

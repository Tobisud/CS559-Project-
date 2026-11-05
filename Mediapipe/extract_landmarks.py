"""Turn folders of hand-sign images into a CSV of MediaPipe hand landmarks.

Expected layout (label = name of the folder that directly holds the image):
    asl_dataset/A/P1_A_1.jpg
    webcam_dataset/train/A/xxx.jpg      (train/val/test is recorded in the "split" column)

Output CSV columns:
    path, label, person, split, handedness, score,
    x0, y0, z0, ..., x20, y20, z20        (raw MediaPipe coordinates)
    nx0, ny0, nz0, ..., nx20, ny20, nz20  (normalized: wrist at origin, scaled by hand size,
                                           left hands mirrored to look like right hands)

Requires: mediapipe (Tasks API), opencv-python, numpy
Examples:
    python extract_landmarks.py asl_dataset
    python extract_landmarks.py asl_dataset webcam_dataset --output landmarks_all.csv
"""

import argparse
import csv
import re
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = {"train", "val", "valid", "validation", "test"}
NUM_LANDMARKS = 21


def normalize_landmarks(points, handedness):
    """points: (21, 3) array. Returns a (21, 3) array that ignores position and scale.

    Use this same function in the webcam script so training and inference match.
    """
    pts = np.asarray(points, dtype=np.float32).copy()
    pts -= pts[0]                                   # wrist -> origin
    if handedness == "Left":                        # mirror left hands to look like right hands
        pts[:, 0] *= -1
    scale = np.linalg.norm(pts[9, :2])              # wrist -> middle-finger knuckle distance
    if scale < 1e-6:
        scale = np.max(np.linalg.norm(pts[:, :2], axis=1)) or 1.0
    return pts / scale


def person_from_name(name):
    """'P1_A_10.jpg' -> 'P1'. Returns '' if the filename has no P<number> prefix."""
    m = re.match(r"(P\d+)_", name)
    return m.group(1) if m else ""


def split_from_path(path, root):
    for part in path.relative_to(root).parts[:-1]:
        if part.lower() in SPLIT_NAMES:
            return part.lower()
    return ""


def detect(landmarker, bgr, pad_px):
    """Try the image as-is; if no hand is found, retry with a border added around it."""
    attempts = [0] + ([pad_px] if pad_px > 0 else [])
    for pad in attempts:
        img = bgr
        if pad:
            img = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        if result.hand_landmarks:
            h, w = img.shape[:2]
            pts = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
            if pad:  # map coordinates back to the original (unpadded) image
                pts[:, 0] = (pts[:, 0] * w - pad) / bgr.shape[1]
                pts[:, 1] = (pts[:, 1] * h - pad) / bgr.shape[0]
            cat = result.handedness[0][0]
            return pts, cat.category_name, cat.score
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe hand landmarks from labeled image folders")
    parser.add_argument("folders", nargs="+", type=Path, help="Dataset root folder(s)")
    parser.add_argument("--output", type=Path, default=Path("landmarks.csv"))
    parser.add_argument("--failed", type=Path, default=Path("landmarks_failed.txt"),
                        help="List of images where no hand was detected")
    parser.add_argument("--hand-model", type=Path, default=Path(__file__).with_name("hand_landmarker.task"))
    parser.add_argument("--min-confidence", type=float, default=0.3,
                        help="Detection confidence (lower than webcam since these are single still images)")
    parser.add_argument("--pad", type=int, default=80, help="Border (px) to add on retry; 0 disables retry")
    parser.add_argument("--flip", action="store_true",
                        help="Mirror images horizontally first (match a webcam script that uses cv2.flip)")
    args = parser.parse_args()

    if not args.hand_model.is_file():
        parser.error(f"Missing hand landmark model: {args.hand_model}")

    images = []
    for root in args.folders:
        if not root.is_dir():
            parser.error(f"Not a folder: {root}")
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in IMAGE_EXTS and p.is_file():
                images.append((root, p))
    if not images:
        parser.error("No images found")
    print(f"Found {len(images)} images in {len(args.folders)} folder(s)")

    header = ["path", "label", "person", "split", "handedness", "score"]
    header += [f"{a}{i}" for i in range(NUM_LANDMARKS) for a in ("x", "y", "z")]
    header += [f"n{a}{i}" for i in range(NUM_LANDMARKS) for a in ("x", "y", "z")]

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.hand_model.resolve())),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=args.min_confidence,
    )

    ok = 0
    failed = []
    per_label = {}
    start = time.monotonic()
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker, \
            open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, (root, path) in enumerate(images, 1):
            label = path.parent.name.upper()
            stats = per_label.setdefault(label, [0, 0])
            stats[1] += 1

            bgr = cv2.imread(str(path))
            pts = None
            if bgr is not None:
                if args.flip:
                    bgr = cv2.flip(bgr, 1)
                pts, hand, score = detect(landmarker, bgr, args.pad)

            if pts is None:
                failed.append(str(path))
            else:
                norm = normalize_landmarks(pts, hand)
                writer.writerow(
                    [path.as_posix(), label, person_from_name(path.name), split_from_path(path, root), hand,
                     f"{score:.4f}"]
                    + [f"{v:.6f}" for v in pts.flatten()]
                    + [f"{v:.6f}" for v in norm.flatten()]
                )
                ok += 1
                stats[0] += 1

            if i % 200 == 0 or i == len(images):
                rate = i / max(time.monotonic() - start, 1e-6)
                print(f"  {i}/{len(images)}  detected={ok}  failed={len(failed)}  ({rate:.0f} img/s)")

    args.failed.write_text("\n".join(failed) + ("\n" if failed else ""))

    print(f"\nSaved {ok} rows to {args.output.resolve()}")
    print(f"No hand detected in {len(failed)} images (listed in {args.failed.resolve()})")
    print("\nDetection rate per label:")
    for label in sorted(per_label):
        got, total = per_label[label]
        flag = "  <-- low" if total and got / total < 0.8 else ""
        print(f"  {label}: {got}/{total} ({got / total:.0%}){flag}")


if __name__ == "__main__":
    main()

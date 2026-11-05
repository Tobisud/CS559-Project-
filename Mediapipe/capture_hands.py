"""Capture cropped hand images once per second for 150 seconds.

Requires: opencv-python, mediapipe (Tasks API), numpy
Place hand_landmarker.task alongside this script, or pass --hand-model.
Example: python capture_hands.py --label 1
"""

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Capture labeled webcam hand crops")
    parser.add_argument("--label", required=True, help="Gesture label, e.g. 1, A, W")
    parser.add_argument("--duration", type=int, default=100, help="Recording duration in seconds")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between captures")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--padding", type=float, default=0.20, help="Padding fraction of bounding-box maximum dimension")
    parser.add_argument("--hand-model", type=Path, default=Path(__file__).with_name("hand_landmarker.task"))
    parser.add_argument("--output", type=Path, default=Path("webcam_captures"))
    parser.add_argument("--no-flip", action="store_true", help="Do not mirror webcam frame")
    args = parser.parse_args()

    if not args.label.isalnum() or len(args.label) != 1 or args.label.upper() not in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        parser.error("--label must be one digit (0-9) or letter (A-Z)")
    if args.duration <= 0 or args.interval <= 0 or args.padding < 0:
        parser.error("duration and interval must be positive; padding cannot be negative")
    if not args.hand_model.is_file():
        parser.error(f"Missing hand landmark model: {args.hand_model}")

    label = args.label.upper()
    save_dir = args.output / label
    save_dir.mkdir(parents=True, exist_ok=True)

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.hand_model.resolve())),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    start = time.monotonic()
    next_capture = start + args.interval
    last_timestamp_ms = -1
    saved = 0
    print(f"Capturing gesture {label!r} for {args.duration}s, every {args.interval}s")
    print(f"Saving to {save_dir.resolve()}. Press Q to stop early.")

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                now = time.monotonic()
                elapsed = now - start
                if elapsed >= args.duration:
                    break

                ok, frame = cap.read()
                if not ok:
                    print("Camera frame unavailable; stopping.")
                    break
                if not args.no_flip:
                    frame = cv2.flip(frame, 1)

                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                timestamp_ms = max(int(elapsed * 1000), last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                crop = None
                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    xs = [p.x for p in landmarks]
                    ys = [p.y for p in landmarks]
                    left = int(min(xs) * width)
                    top = int(min(ys) * height)
                    right = int(max(xs) * width)
                    bottom = int(max(ys) * height)
                    pad = int(max(right - left, bottom - top) * args.padding)
                    x1, y1 = max(0, left - pad), max(0, top - pad)
                    x2, y2 = min(width, right + pad), min(height, bottom + pad)
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2].copy()
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                now = time.monotonic()
                if now >= next_capture:
                    if crop is not None:
                        filename = save_dir / f"{label}_{time.time_ns()}.jpg"
                        if cv2.imwrite(str(filename), crop):
                            saved += 1
                            print(f"Saved {saved}: {filename.name}")
                        else:
                            print("Warning: failed to save image")
                    else:
                        print(f"At {now-start:.1f}s: no hand detected; skipped")
                    # Do not attempt to catch up by saving duplicate frames.
                    next_capture += (int((now - next_capture) // args.interval) + 1) * args.interval

                remaining = max(0, args.duration - (now - start))
                cv2.putText(frame, f"Class: {label} | Saved: {saved} | Left: {remaining:.0f}s", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Hand Dataset Capture (Q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user")
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Finished. Saved {saved} images to {save_dir.resolve()}")


if __name__ == "__main__":
    main()

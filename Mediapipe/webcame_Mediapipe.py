import cv2
import time
import torch
import mediapipe as mp
import numpy as np

from model_mlp import create_model
from extract_landmarks import normalize_landmarks

# ---------- CONFIG ----------
MODEL_PATH = "gesture_mlp.pth"
HAND_MODEL_PATH = "hand_landmarker.task"
CONFIDENCE_THRESHOLD = 0.70
PADDING = 30

# ---------- LOAD CNN ----------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

###using CNN
# checkpoint = torch.load(
#     MODEL_PATH,
#     map_location=device,
#     weights_only=True
# )

# classes = checkpoint["classes"]

# model = create_model(len(classes))
# model.load_state_dict(checkpoint["model_state_dict"])
# model = model.to(device)
# model.eval()
###END

###Using MLP
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
classes = checkpoint["classes"]
use_z = checkpoint["use_z"]

model = create_model(
    checkpoint["num_inputs"],        # 42 for x,y only (63 if trained with --use-z)
    len(classes),
    tuple(checkpoint["hidden"]),
    checkpoint["dropout"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()
#END

print("Model classes:", classes)

for i, name in enumerate(classes):
    print(i, "->", name)

# ---------- MEDIAPIPE TASKS ----------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.3
)

# Connections between the 21 hand landmarks
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

from collections import deque
import numpy as np

SMOOTHING_WINDOW = 30
STABLE_FRAMES = 10
CONFIDENCE_THRESHOLD = 0.70

prediction_history = deque(maxlen=SMOOTHING_WINDOW)

displayed_gesture = "Unknown"
candidate_gesture = None
candidate_count = 0

# ---------- WEBCAM ----------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Webcam started. Press Q to quit.")

start_time = time.monotonic()
last_timestamp_ms = -1

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            # Create MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb)
            )

            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )
            timestamp_ms = max(
                timestamp_ms, last_timestamp_ms + 1
            )
            last_timestamp_ms = timestamp_ms

            # Detect hand landmarks
            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            for hand_idx, landmarks in enumerate(result.hand_landmarks):

                # ---------- BOUNDING BOX ----------
                xs = [lm.x for lm in landmarks]
                ys = [lm.y for lm in landmarks]

                x1 = max(0, int(min(xs) * width) - PADDING)
                y1 = max(0, int(min(ys) * height) - PADDING)
                x2 = min(width, int(max(xs) * width) + PADDING)
                y2 = min(height, int(max(ys) * height) + PADDING)

                if x2 <= x1 or y2 <= y1:
                    continue

                # ---------- MLP CLASSIFICATION ----------
                hand_label = result.handedness[hand_idx][0].category_name   # "Left" / "Right"
                pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
                feats = normalize_landmarks(pts, hand_label)
                if not use_z:
                    feats = feats[:, :2]
                input_tensor = torch.from_numpy(feats.reshape(1, -1)).to(device)

                with torch.inference_mode():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)[0]

                    # Get top 5 predictions
                    top_probs, top_indices = torch.topk(
                        probabilities,
                        k=5
                    )

                # Print top 5 predictions
                print(f"\nTop 5 Predictions ({hand_label} hand):")

                for probability, index in zip(top_probs, top_indices):
                    print(
                        f"{classes[index.item()]}: "
                        f"{probability.item():.2%}"
                    )

                # Keep the highest prediction for webcam display
                confidence = top_probs[0].item()
                gesture = classes[top_indices[0].item()]

                if confidence < CONFIDENCE_THRESHOLD:
                    gesture = "Unknown"

                # ---------- STABILITY CHECK ----------
                if gesture == candidate_gesture:
                    candidate_count += 1
                else:
                    candidate_gesture = gesture
                    candidate_count = 1

                # Only update after consecutive stable predictions
                if candidate_count >= STABLE_FRAMES:
                    displayed_gesture = candidate_gesture

                # Use the stabilized prediction for display
                gesture = displayed_gesture


                # ---------- DRAW BOUNDING BOX ----------
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{gesture} ({confidence:.1%})",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                # ---------- DRAW LANDMARKS ----------
                points = []

                for lm in landmarks:
                    px = int(lm.x * width)
                    py = int(lm.y * height)
                    points.append((px, py))

                    cv2.circle(
                        frame,
                        (px, py),
                        4,
                        (0, 0, 255),
                        -1
                    )

                for start, end in HAND_CONNECTIONS:
                    cv2.line(
                        frame,
                        points[start],
                        points[end],
                        (255, 255, 0),
                        2
                    )

            cv2.imshow("Hand Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()

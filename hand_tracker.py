"""
hand_tracker.py
---------------
Robust MediaPipe HandLandmarker wrapper.
- Uses RunningMode.IMAGE for zero timestamp-desync issues.
- Tolerant detection threshold (0.32) for all lighting conditions.
- Tracks any visible hand (left or right).
- Provides skeleton drawing for the PiP overlay so the user can verify detection.
"""

import math
import urllib.request
from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Hand landmark indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Standard hand skeleton connections
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Palm base
    (0, 17)
]

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"


def _ensure_model() -> None:
    """Download the hand-landmark model on first run if absent."""
    if not _MODEL_PATH.exists():
        print(f"[HandTracker] Downloading model -> {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
        print("[HandTracker] Download complete.")


class HandTracker:
    def __init__(self, max_hands: int = 1,
                 detection_confidence: float = 0.32,
                 tracking_confidence: float = 0.32):
        _ensure_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(_MODEL_PATH)
            ),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=tracking_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def process(self, frame):
        """
        Run hand detection on `frame` (BGR).
        Returns a dict:
            landmarks: list of 21 (x, y) pixel coordinates, or None
            norm_landmarks: list of 21 (x, y, z) normalized coordinates, or None
            wrist, thumb_tip, index_tip, middle_tip: convenience (x, y) tuples
            handedness: 'Right' or 'Left' or 'Hand' or None
            hand_size: scalar approximate pixel size
        """
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_contiguous = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_contiguous)
        results = self._landmarker.detect(mp_image)

        output = {
            "landmarks": None,
            "norm_landmarks": None,
            "wrist": None,
            "thumb_tip": None,
            "index_tip": None,
            "middle_tip": None,
            "handedness": None,
            "hand_size": 0.0,
        }

        if not results.hand_landmarks:
            return output

        # Prefer right hand if available, otherwise take first detected hand
        chosen_idx = 0
        handedness_label = "Hand"
        if results.handedness:
            for i, hd_list in enumerate(results.handedness):
                if hd_list and len(hd_list) > 0:
                    name = hd_list[0].category_name
                    handedness_label = name
                    if name.lower() == "right":
                        chosen_idx = i
                        break

        hand_lm = results.hand_landmarks[chosen_idx]
        pts = [(lm.x * w, lm.y * h) for lm in hand_lm]
        norm_pts = [(lm.x, lm.y, lm.z) for lm in hand_lm]

        # Calculate hand pixel size (wrist to middle MCP)
        w_pt = pts[WRIST]
        m_pt = pts[MIDDLE_MCP]
        hand_size = math.hypot(w_pt[0] - m_pt[0], w_pt[1] - m_pt[1])

        output["landmarks"] = pts
        output["norm_landmarks"] = norm_pts
        output["wrist"] = pts[WRIST]
        output["thumb_tip"] = pts[THUMB_TIP]
        output["index_tip"] = pts[INDEX_TIP]
        output["middle_tip"] = pts[MIDDLE_TIP]
        output["handedness"] = handedness_label
        output["hand_size"] = max(hand_size, 10.0)

        return output

    @staticmethod
    def draw_skeleton(image, landmarks, color=(0, 255, 160), point_color=(255, 255, 255)):
        """Draws hand skeleton connections and joints on an image (e.g. PiP feed)."""
        if landmarks is None:
            return image

        overlay = image.copy()
        # Draw connection lines
        for p1_idx, p2_idx in HAND_CONNECTIONS:
            if p1_idx < len(landmarks) and p2_idx < len(landmarks):
                p1 = (int(landmarks[p1_idx][0]), int(landmarks[p1_idx][1]))
                p2 = (int(landmarks[p2_idx][0]), int(landmarks[p2_idx][1]))
                cv2.line(overlay, p1, p2, color, 2, cv2.LINE_AA)

        # Draw joints
        for i, pt in enumerate(landmarks):
            center = (int(pt[0]), int(pt[1]))
            radius = 4 if i in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, WRIST) else 2
            cv2.circle(overlay, center, radius, point_color, -1, cv2.LINE_AA)

        return overlay

    def close(self):
        self._landmarker.close()



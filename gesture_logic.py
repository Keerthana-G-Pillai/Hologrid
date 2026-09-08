"""
gesture_logic.py
-----------------
Translates 21-point hand landmarks into:
1. 3D Grid Cursor Position (gx, gy, gz) in whichever direction the hand moves.
2. Full 3D View Rotation (rot_x, rot_y, rot_z) when user makes a fist and moves hand.
3. Block Placement Triggers (Draw/Paint mode, Fist grab, Pinch).
"""

import math
import numpy as np


class EMA:
    def __init__(self, alpha=0.35, initial=None):
        self.alpha = alpha
        self.value = initial

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = (1.0 - self.alpha) * self.value + self.alpha * new_value
        return self.value


class GestureProcessor:
    # MediaPipe landmark indices
    WRIST = 0
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_TIP = 20

    def __init__(self, frame_w=640, frame_h=480, grid_range=(7, 6, 5)):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.max_gx, self.max_gy, self.max_gz = grid_range

        # Cursor smoothing
        self.ema_cx = EMA(alpha=0.45, initial=0.0)
        self.ema_cy = EMA(alpha=0.45, initial=0.0)
        self.ema_cz = EMA(alpha=0.35, initial=0.0)

        # 3D View rotation angles (accumulated)
        self.rot_x = 18.0
        self.rot_y = -25.0
        self.rot_z = 0.0

        # Scale
        self.scale = 1.0
        self.ema_scale = EMA(alpha=0.25, initial=1.0)

        # Grab / Fist detection
        self.ema_grab = EMA(alpha=0.5, initial=0.0)
        self.prev_hand_pos = None

        # Base reference hand size for depth
        self.ref_hand_size = 110.0

        self.last_state = {
            "grid_pos": (0, 0, 0),
            "rot_x": self.rot_x,
            "rot_y": self.rot_y,
            "rot_z": self.rot_z,
            "scale": 1.0,
            "grab": False,
            "is_pinch": False,
            "hand_visible": False,
        }

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _hand_size(self, lm):
        return max(self._dist(lm[self.WRIST], lm[self.MIDDLE_MCP]), 10.0)

    def _is_fist(self, lm):
        """Fist detection: fingers curled close to wrist relative to knuckles."""
        wrist = lm[self.WRIST]
        pairs = [
            (self.INDEX_TIP, self.INDEX_MCP),
            (self.MIDDLE_TIP, self.MIDDLE_MCP),
            (self.RING_TIP, self.RING_MCP),
            (self.PINKY_TIP, self.PINKY_MCP),
        ]
        curled = 0
        for tip_idx, mcp_idx in pairs:
            tip_d = self._dist(wrist, lm[tip_idx])
            mcp_d = self._dist(wrist, lm[mcp_idx])
            if tip_d < mcp_d * 1.18:
                curled += 1
        return curled >= 3

    def update(self, landmarks):
        if landmarks is None or len(landmarks) < 21:
            return self.hold()

        lm = landmarks
        wrist = lm[self.WRIST]
        index_tip = lm[self.INDEX_TIP]
        thumb_tip = lm[self.THUMB_TIP]
        middle_mcp = lm[self.MIDDLE_MCP]

        size = self._hand_size(lm)

        # 1. Grab (Fist) detection
        raw_fist = 1.0 if self._is_fist(lm) else 0.0
        grab = self.ema_grab.update(raw_fist) >= 0.5

        # 2. Pinch detection (thumb tip close to index tip)
        pinch_dist = self._dist(thumb_tip, index_tip) / max(size, 1.0)
        is_pinch = pinch_dist < 0.30

        # Hand center of interaction: index tip if open hand, palm center if fist
        center_x = (wrist[0] + middle_mcp[0]) * 0.5 if grab else index_tip[0]
        center_y = (wrist[1] + middle_mcp[1]) * 0.5 if grab else index_tip[1]

        # Normalized coordinates [-1.0, 1.0] from screen center
        norm_x = (center_x - self.frame_w * 0.5) / (self.frame_w * 0.45)
        norm_y = -(center_y - self.frame_h * 0.5) / (self.frame_h * 0.45)
        norm_x = max(-1.2, min(1.2, norm_x))
        norm_y = max(-1.2, min(1.2, norm_y))

        # Depth estimation from hand size:
        # Closer hand = bigger size -> positive Z (towards user)
        # Farther hand = smaller size -> negative Z (into the scene)
        norm_z = (size - self.ref_hand_size) / 45.0
        norm_z = max(-1.2, min(1.2, norm_z))

        # Smooth cursor coordinates
        smooth_x = self.ema_cx.update(norm_x)
        smooth_y = self.ema_cy.update(norm_y)
        smooth_z = self.ema_cz.update(norm_z)

        # Discrete 3D Grid coordinates for block placement
        gx = int(round(smooth_x * self.max_gx))
        gy = int(round(smooth_y * self.max_gy))
        gz = int(round(smooth_z * self.max_gz))
        gx = max(-self.max_gx, min(self.max_gx, gx))
        gy = max(-self.max_gy, min(self.max_gy, gy))
        gz = max(-self.max_gz, min(self.max_gz, gz))

        # 3. View Rotation Handling:
        # If user forms a fist ("grab"), moving hand turns the entire 3D view!
        curr_hand_pos = (center_x, center_y)
        if grab and self.prev_hand_pos is not None:
            dx = curr_hand_pos[0] - self.prev_hand_pos[0]
            dy = curr_hand_pos[1] - self.prev_hand_pos[1]

            # Sensitivity for rotating the whole 3D world
            rot_speed = 0.55
            self.rot_y += dx * rot_speed
            self.rot_x -= dy * rot_speed

            # Wrist tilt angle for roll
            angle = math.degrees(math.atan2(middle_mcp[0] - wrist[0], -(middle_mcp[1] - wrist[1])))
            self.rot_z = max(-45.0, min(45.0, angle * 0.6))
        elif not grab:
            # Subtle hand tilt adds slight parallax when open palm
            target_roll = (middle_mcp[0] - wrist[0]) * 0.2
            self.rot_z = 0.9 * self.rot_z + 0.1 * max(-30.0, min(30.0, target_roll))

        self.prev_hand_pos = curr_hand_pos

        self.last_state = {
            "grid_pos": (gx, gy, gz),
            "rot_x": self.rot_x,
            "rot_y": self.rot_y,
            "rot_z": self.rot_z,
            "scale": self.scale,
            "grab": grab,
            "is_pinch": is_pinch,
            "hand_visible": True,
        }
        return self.last_state

    def hold(self):
        """When no hand is visible, freeze pose gracefully."""
        self.prev_hand_pos = None
        self.last_state["grab"] = False
        self.last_state["hand_visible"] = False
        return dict(self.last_state)

    def reset_view(self):
        self.rot_x = 20.0
        self.rot_y = -35.0
        self.rot_z = 0.0
        self.scale = 0.80

    def set_rotation(self, rot_x, rot_y, rot_z=0.0):
        self.rot_x = rot_x
        self.rot_y = rot_y
        self.rot_z = rot_z



"""
gesture_logic.py
-----------------
3D Painting & Drawing Engine for Hologrid Builder:

Interaction Model:
1. START PAINTING:
   - User pinches INDEX + THUMB.
   - Detected and confirmed for 2 consecutive frames -> Enters PAINTING mode.
   - Index finger becomes the continuous 3D brush.
   - One pinch = Start of ONE complete drawing stroke / figure.

2. CONTINUOUS 3D PATH GENERATION (ZERO GAPS):
   - As the index finger moves in 3D space, intermediate grid cells between frames
     are interpolated using 3D Digital Differential Analyzer (DDA).
   - Generates unbroken, contiguous straight lines, curves, rings, and shapes.
   - Duplicates are prevented; interior is NOT automatically filled.

3. REAL-TIME PAINT PREVIEW:
   - The in-progress stroke is displayed as a glowing 3D ghost/wireframe path
     trailing behind the index brush tip in real time.

4. RELEASE = FINISH & COMMIT STROKE:
   - When the user releases the pinch, the complete stroke is committed atomically.
   - Moving the index finger after release does NOT modify previous strokes.
   - Pinching again begins a fresh new stroke.

5. GESTURE HIERARCHY:
   - 🖐 OPEN PALM: Rotate entire 3D hologrid (tumble)
   - 👐 TWO HANDS: Zoom in / out
   - ✊ FIST: Cancel current stroke / Standby
   - ☝ INDEX + PINCH: 3D Paint
"""

import math
import numpy as np


def interpolate_3d_line(p0, p1):
    """
    3D Digital Differential Analyzer (DDA).
    Generates an unbroken sequence of grid cells between p0 and p1 with ZERO gaps.
    Guarantees every adjacent step differs by at most 1 on any axis.
    """
    x0, y0, z0 = p0
    x1, y1, z1 = p1

    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0

    steps = max(abs(dx), abs(dy), abs(dz))
    if steps == 0:
        return [p0]

    cells = []
    for i in range(steps + 1):
        t = i / float(steps)
        gx = int(round(x0 + t * dx))
        gy = int(round(y0 + t * dy))
        gz = int(round(z0 + t * dz))
        cells.append((gx, gy, gz))

    return cells


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

    # Gesture states & labels
    STATE_READY       = "READY • PINCH TO PAINT"
    STATE_PAINTING    = "PAINTING • STROKE ACTIVE"
    STATE_COMMITTED   = "STROKE COMMITTED"
    STATE_CANCELLED   = "CANCELLED • FIST"
    STATE_ROTATE      = "ROTATING • ROTATE ↻"
    STATE_ZOOM_IN     = "ZOOMING IN • ZOOM IN ↑"
    STATE_ZOOM_OUT    = "ZOOMING OUT • ZOOM OUT ↓"
    STATE_ZOOM_HOLD   = "ZOOM MODE"
    STATE_FIST        = "STOPPED • FIST"
    STATE_NOHAND      = "IDLE • NO HAND"

    MODE_ROTATE = "MODE_ROTATE"
    MODE_INDEX  = "MODE_INDEX"
    MODE_FIST   = "MODE_FIST"

    def __init__(self, frame_w=640, frame_h=480, grid_range=(7, 5, 4)):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.max_gx, self.max_gy, self.max_gz = grid_range

        # Smoothing for world grid target position
        self.ema_cur_x = EMA(alpha=0.40, initial=0.0)
        self.ema_cur_y = EMA(alpha=0.40, initial=0.0)
        self.ema_cur_z = EMA(alpha=0.30, initial=0.0)

        # Smoothing for palm rotation tracking (slow and smooth)
        self.ema_palm_x = EMA(alpha=0.35, initial=None)
        self.ema_palm_y = EMA(alpha=0.35, initial=None)

        # 3D View rotation angles (accumulated)
        self.rot_x = 18.0
        self.rot_y = -30.0
        self.rot_z = 0.0

        # Zoom limits and sensitivity
        self.scale = 0.85
        self.MIN_SCALE = 0.35
        self.MAX_SCALE = 2.40
        self.ZOOM_SENSITIVITY = 0.0035
        self.ref_hand_size = 115.0

        # State debounce / confirmation
        self.current_mode = self.MODE_FIST
        self.candidate_mode = self.MODE_FIST
        self.candidate_frames = 0
        self.CONFIRM_FRAMES = 2

        # 3D PAINTING STROKE SYSTEM
        self.is_painting = False
        self.stroke_count = 0
        self.current_stroke = []        # list of (gx, gy, gz) in order
        self.current_stroke_set = set() # fast O(1) duplicate check
        self.last_stroke_cell = None

        self.pinch_start_frames = 0
        self.pinch_release_frames = 0
        self.committed_timer = 0

        # Two-hand tracking for Zoom
        self.prev_hands_dist = None
        self.zoom_action_label = "ZOOM MODE"

        # Palm position tracking for rotation
        self.prev_palm_center = None

        # Stable quantized grid cell with hysteresis
        self.current_gx = 0
        self.current_gy = 0
        self.current_gz = 0

        self.last_state = {
            "state_label": self.STATE_NOHAND,
            "mode_badge": "MODE: READY",
            "action_badge": "",
            "gesture": "STOP",
            "grid_pos": (0, 0, 0),
            "rot_x": self.rot_x,
            "rot_y": self.rot_y,
            "rot_z": self.rot_z,
            "scale": self.scale,
            "is_painting": False,
            "active_stroke": [],
            "commit_stroke": [],
            "preview_active": False,
            "is_rotating": False,
            "is_zooming": False,
            "num_hands": 0,
            "hand_visible": False,
            "index_pos": None,
            "palm_pos": None,
            "confidence": 0,
            "pinch_ratio": 1.0,
            "stroke_count": 0,
            "stroke_len": 0,
        }

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _hand_size(self, lm):
        return max(self._dist(lm[self.WRIST], lm[self.MIDDLE_MCP]), 10.0)

    @staticmethod
    def _rot_matrix_x(deg):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

    @staticmethod
    def _rot_matrix_y(deg):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

    @staticmethod
    def _rot_matrix_z(deg):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    def get_rotation_matrix(self):
        return (self._rot_matrix_z(self.rot_z) @
                self._rot_matrix_y(self.rot_y) @
                self._rot_matrix_x(self.rot_x))

    def _classify_fingers(self, lm, norm_lm=None):
        wrist = lm[self.WRIST]

        def dist2(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        fingers = {
            "index": (self.INDEX_MCP, self.INDEX_PIP, self.INDEX_DIP, self.INDEX_TIP),
            "middle": (self.MIDDLE_MCP, self.MIDDLE_PIP, self.MIDDLE_DIP, self.MIDDLE_TIP),
            "ring": (self.RING_MCP, self.RING_PIP, self.RING_DIP, self.RING_TIP),
            "pinky": (self.PINKY_MCP, self.PINKY_PIP, self.PINKY_DIP, self.PINKY_TIP),
        }

        extended = {}
        for name, (mcp_i, pip_i, dip_i, tip_i) in fingers.items():
            mcp = lm[mcp_i]
            pip = lm[pip_i]
            tip = lm[tip_i]

            d_wrist_tip = dist2(wrist, tip)
            d_wrist_pip = dist2(wrist, pip)
            d_mcp_tip = dist2(mcp, tip)
            d_mcp_pip = dist2(mcp, pip)

            is_ext_3d = None
            if norm_lm is not None and len(norm_lm) > tip_i:
                nmcp = norm_lm[mcp_i]
                npip = norm_lm[pip_i]
                ntip = norm_lm[tip_i]
                d3_mcp_tip = math.sqrt((nmcp[0]-ntip[0])**2 + (nmcp[1]-ntip[1])**2 + (nmcp[2]-ntip[2])**2)
                d3_mcp_pip = math.sqrt((nmcp[0]-npip[0])**2 + (nmcp[1]-npip[1])**2 + (nmcp[2]-npip[2])**2)
                if d3_mcp_pip > 1e-4:
                    is_ext_3d = (d3_mcp_tip / d3_mcp_pip) > 1.35

            is_ext_2d = (d_wrist_tip > d_wrist_pip * 1.08) and (d_mcp_tip > d_mcp_pip * 1.12)

            if is_ext_3d is not None:
                extended[name] = is_ext_3d and is_ext_2d
            else:
                extended[name] = is_ext_2d

        # Thumb extension
        t_tip = lm[self.THUMB_TIP]
        t_ip = lm[self.THUMB_IP]
        d_wrist_ttip = dist2(wrist, t_tip)
        d_wrist_tip_ip = dist2(wrist, t_ip)
        d_imcp_ttip = dist2(lm[self.INDEX_MCP], t_tip)
        d_imcp_tip_ip = dist2(lm[self.INDEX_MCP], t_ip)

        thumb_ext = (d_wrist_ttip > d_wrist_tip_ip * 1.10) and (d_imcp_ttip > d_imcp_tip_ip * 1.15)
        extended["thumb"] = thumb_ext

        extended["extended_count"] = sum([
            extended["index"], extended["middle"], extended["ring"], extended["pinky"]
        ])
        extended["total_extended"] = extended["extended_count"] + (1 if thumb_ext else 0)

        return extended

    def _determine_raw_mode(self, lm, norm_lm=None):
        f = self._classify_fingers(lm, norm_lm)
        ext_count = f["extended_count"]
        tot_count = f["total_extended"]

        # PRIORITY 1: WHOLE OPEN PALM (3 or 4 fingers extended) -> ROTATE
        if ext_count >= 3 or tot_count >= 4:
            return self.MODE_ROTATE

        # PRIORITY 2: INDEX FINGER (index extended, others folded) -> INDEX
        if f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]:
            return self.MODE_INDEX

        # PRIORITY 3: FIST (no main fingers extended) -> FIST
        if ext_count == 0:
            return self.MODE_FIST

        return self.MODE_FIST

    def update(self, landmarks, norm_landmarks=None, all_hands=None):
        if landmarks is None or len(landmarks) < 21:
            return self.hold()

        lm = landmarks
        wrist = lm[self.WRIST]
        index_tip = lm[self.INDEX_TIP]
        middle_mcp = lm[self.MIDDLE_MCP]
        size = self._hand_size(lm)
        palm_x = (wrist[0] + middle_mcp[0]) * 0.5
        palm_y = (wrist[1] + middle_mcp[1]) * 0.5
        d_pinch = self._dist(lm[self.THUMB_TIP], lm[self.INDEX_TIP])
        pinch_ratio = d_pinch / size

        num_hands = len(all_hands) if all_hands is not None else 1

        # -------------------------------------------------------------
        # 4. INTERACTION STATE MACHINE (ZOOM vs ROTATE vs PAINT vs FIST)
        # -------------------------------------------------------------
        # RULE 1: When TWO hands are detected:
        # -> ZOOM MODE (rotation disabled, painting disabled)
        #
        # RULE 2: When ONE open palm is detected:
        # -> ROTATE MODE (zoom disabled, painting disabled)
        #
        # RULE 3: When painting:
        # -> PAINT MODE (rotation disabled, zoom disabled)
        #
        # RULE 4: Fist cancels active action / Stop.
        # -------------------------------------------------------------

        is_zooming = False
        zoom_badge = ""
        is_rotating = False
        commit_stroke = []
        mode_badge = "MODE: READY"
        action_badge = ""
        state_label = self.STATE_READY

        # CHECK TWO HANDS FOR ZOOM MODE
        if num_hands >= 2 and all_hands is not None and len(all_hands) >= 2:
            # Cancel or commit any painting stroke if user suddenly presents two hands
            if self.is_painting and self.current_stroke:
                commit_stroke = list(self.current_stroke)
                self.is_painting = False
                self.current_stroke = []
                self.current_stroke_set = set()
                self.last_stroke_cell = None

            w0 = all_hands[0][0]
            w1 = all_hands[1][0]
            current_dist = math.hypot(w0[0] - w1[0], w0[1] - w1[1])

            if self.prev_hands_dist is not None:
                delta_d = current_dist - self.prev_hands_dist
                # Deadzone: ignore jitter < 2.5px
                if abs(delta_d) > 2.5:
                    # Low sensitivity zoom step
                    zoom_change = delta_d * self.ZOOM_SENSITIVITY
                    # Max zoom change per frame clamp (0.045) to prevent jumping
                    zoom_change = max(-0.045, min(0.045, zoom_change))
                    new_scale = self.scale * (1.0 + zoom_change)
                    # Min and Max limits: prevent entering hologrid or zooming infinitely far
                    self.scale = max(self.MIN_SCALE, min(self.MAX_SCALE, new_scale))
                    is_zooming = True

                    if delta_d > 0:
                        mode_badge = "MODE: ZOOMING IN"
                        action_badge = "ZOOM IN ↑"
                        state_label = self.STATE_ZOOM_IN
                    else:
                        mode_badge = "MODE: ZOOMING OUT"
                        action_badge = "ZOOM OUT ↓"
                        state_label = self.STATE_ZOOM_OUT
                else:
                    is_zooming = True
                    mode_badge = "MODE: ZOOM"
                    action_badge = "ZOOM MODE"
                    state_label = self.STATE_ZOOM_HOLD

            self.prev_hands_dist = current_dist
            self.prev_palm_center = None
            self.pinch_start_frames = 0
            self.pinch_release_frames = 0

        else:
            self.prev_hands_dist = None

            # ONE HAND DETECTED: Determine gesture mode
            raw_mode = self._determine_raw_mode(lm, norm_landmarks)

            if raw_mode == self.candidate_mode:
                self.candidate_frames += 1
                if self.candidate_frames >= self.CONFIRM_FRAMES:
                    self.current_mode = raw_mode
            else:
                self.candidate_mode = raw_mode
                self.candidate_frames = 1

            palm_x = (wrist[0] + middle_mcp[0]) * 0.5
            palm_y = (wrist[1] + middle_mcp[1]) * 0.5

            d_pinch = self._dist(lm[self.THUMB_TIP], lm[self.INDEX_TIP])
            pinch_ratio = d_pinch / size

            # ---------------------------------------------------------
            # MODE: ROTATE — ONE OPEN PALM
            # ---------------------------------------------------------
            if self.current_mode == self.MODE_ROTATE and not self.is_painting:
                is_rotating = True
                mode_badge = "MODE: ROTATING"
                action_badge = "ROTATE ↻"
                state_label = self.STATE_ROTATE

                self.pinch_start_frames = 0
                self.pinch_release_frames = 0

                smooth_px = self.ema_palm_x.update(palm_x)
                smooth_py = self.ema_palm_y.update(palm_y)

                if self.prev_palm_center is not None:
                    dx = smooth_px - self.prev_palm_center[0]
                    dy = smooth_py - self.prev_palm_center[1]

                    # Deadzone (2.0px) to eliminate MediaPipe jitter
                    # Clamp max rotation per frame (2.5 deg) for slow, natural rotation
                    ROT_SENSITIVITY = 0.28
                    MAX_ROT_STEP = 2.5

                    if abs(dx) > 2.0:
                        rot_step_y = max(-MAX_ROT_STEP, min(MAX_ROT_STEP, dx * ROT_SENSITIVITY))
                        self.rot_y += rot_step_y

                    if abs(dy) > 2.0:
                        rot_step_x = max(-MAX_ROT_STEP, min(MAX_ROT_STEP, -dy * ROT_SENSITIVITY))
                        self.rot_x += rot_step_x

                self.prev_palm_center = (smooth_px, smooth_py)

            else:
                self.prev_palm_center = None
                self.ema_palm_x.value = None
                self.ema_palm_y.value = None

            # ---------------------------------------------------------
            # MODE: FIST — STOP / CANCEL
            # ---------------------------------------------------------
            if self.current_mode == self.MODE_FIST and not is_rotating:
                if self.is_painting:
                    # Cancel in-progress stroke on fist!
                    self.is_painting = False
                    self.current_stroke = []
                    self.current_stroke_set = set()
                    self.last_stroke_cell = None
                    mode_badge = "MODE: STOPPED"
                    action_badge = "FIST CANCEL"
                    state_label = self.STATE_CANCELLED
                else:
                    mode_badge = "MODE: STOPPED"
                    action_badge = "FIST"
                    state_label = self.STATE_FIST

                self.pinch_start_frames = 0
                self.pinch_release_frames = 0

            # ---------------------------------------------------------
            # MODE: INDEX FINGER — 3D CONTINUOUS PAINTING
            # ---------------------------------------------------------
            elif self.current_mode == self.MODE_INDEX and not is_rotating:
                # 1. Screen / camera coordinates -> 3D Hologrid space
                view_x = (index_tip[0] - self.frame_w * 0.5) / (self.frame_w * 0.38)
                view_y = -(index_tip[1] - self.frame_h * 0.5) / (self.frame_h * 0.38)
                view_x = max(-1.25, min(1.25, view_x))
                view_y = max(-1.25, min(1.25, view_y))

                view_z = (size - self.ref_hand_size) / 45.0
                view_z = max(-1.25, min(1.25, view_z))

                target_cam = np.array([view_x * self.max_gx,
                                       view_y * self.max_gy,
                                       view_z * self.max_gz], dtype=np.float64)

                rot_mat = self.get_rotation_matrix()
                target_world = rot_mat.T @ target_cam

                smooth_wx = self.ema_cur_x.update(target_world[0])
                smooth_wy = self.ema_cur_y.update(target_world[1])
                smooth_wz = self.ema_cur_z.update(target_world[2])

                raw_gx = int(round(smooth_wx))
                raw_gy = int(round(smooth_wy))
                raw_gz = int(round(smooth_wz))

                raw_gx = max(-self.max_gx, min(self.max_gx, raw_gx))
                raw_gy = max(-self.max_gy, min(self.max_gy, raw_gy))
                raw_gz = max(-self.max_gz, min(self.max_gz, raw_gz))

                if abs(smooth_wx - self.current_gx) > 0.52:
                    self.current_gx = raw_gx
                if abs(smooth_wy - self.current_gy) > 0.52:
                    self.current_gy = raw_gy
                if abs(smooth_wz - self.current_gz) > 0.52:
                    self.current_gz = raw_gz

                current_pos = (self.current_gx, self.current_gy, self.current_gz)

                # PINCH DETECTION & STROKE STATE MACHINE
                PINCH_START_THRESHOLD   = 0.30
                PINCH_RELEASE_THRESHOLD = 0.48

                if not self.is_painting:
                    # Idle / Ready: check if user starts a new paint stroke
                    if pinch_ratio < PINCH_START_THRESHOLD:
                        self.pinch_start_frames += 1
                        if self.pinch_start_frames >= 2:
                            # START PAINTING SESSION!
                            self.is_painting = True
                            self.stroke_count += 1
                            self.current_stroke = [current_pos]
                            self.current_stroke_set = {current_pos}
                            self.last_stroke_cell = current_pos
                            self.pinch_start_frames = 0
                            self.pinch_release_frames = 0
                            mode_badge = "MODE: PAINTING"
                            action_badge = "STROKE ACTIVE"
                            state_label = f"PAINTING • STROKE {self.stroke_count:02d}"
                    else:
                        self.pinch_start_frames = 0
                        mode_badge = "MODE: READY"
                        action_badge = "PINCH TO DRAW"
                        if self.committed_timer > 0:
                            self.committed_timer -= 1
                            state_label = "STROKE COMPLETE • PINCH TO DRAW AGAIN"
                        else:
                            state_label = self.STATE_READY

                else:
                    # Currently PAINTING: user moves index finger to draw continuous path!
                    if pinch_ratio > PINCH_RELEASE_THRESHOLD:
                        self.pinch_release_frames += 1
                        if self.pinch_release_frames >= 2:
                            # FINISH & COMMIT STROKE!
                            self.is_painting = False
                            commit_stroke = list(self.current_stroke)
                            self.current_stroke = []
                            self.current_stroke_set = set()
                            self.last_stroke_cell = None
                            self.pinch_release_frames = 0
                            self.committed_timer = 30
                            mode_badge = "MODE: READY"
                            action_badge = "COMMITTED"
                            state_label = f"STROKE {self.stroke_count:02d} COMMITTED ({len(commit_stroke)} BLOCKS)"
                    else:
                        self.pinch_release_frames = 0
                        # Continuously connect every movement with ZERO GAPS (3D DDA)
                        if current_pos != self.last_stroke_cell and self.last_stroke_cell is not None:
                            segment = interpolate_3d_line(self.last_stroke_cell, current_pos)
                            for cell in segment:
                                if cell not in self.current_stroke_set:
                                    self.current_stroke.append(cell)
                                    self.current_stroke_set.add(cell)
                            self.last_stroke_cell = current_pos

                        mode_badge = "MODE: PAINTING"
                        action_badge = "STROKE ACTIVE"
                        state_label = f"PAINTING • STROKE: {self.stroke_count:02d} ({len(self.current_stroke)} BLOCKS)"

        confidence_pct = min(100, int((self.candidate_frames / self.CONFIRM_FRAMES) * 100)) if self.candidate_frames > 0 else 100

        self.last_state = {
            "state_label": state_label,
            "mode_badge": mode_badge,
            "action_badge": action_badge,
            "gesture": "ROTATE" if is_rotating else ("ZOOM" if is_zooming else ("PAINT" if self.is_painting else "IDLE")),
            "grid_pos": (self.current_gx, self.current_gy, self.current_gz),
            "rot_x": self.rot_x,
            "rot_y": self.rot_y,
            "rot_z": self.rot_z,
            "scale": self.scale,
            "is_painting": self.is_painting,
            "active_stroke": list(self.current_stroke),
            "commit_stroke": commit_stroke,
            "preview_active": True if (self.current_mode == self.MODE_INDEX) else False,
            "is_rotating": is_rotating,
            "is_zooming": is_zooming,
            "num_hands": num_hands,
            "hand_visible": True,
            "index_pos": (int(index_tip[0]), int(index_tip[1])),
            "palm_pos": (int(palm_x), int(palm_y)),
            "confidence": confidence_pct,
            "pinch_ratio": round(pinch_ratio, 2),
            "stroke_count": self.stroke_count,
            "stroke_len": len(self.current_stroke),
        }
        return self.last_state

    def hold(self):
        """When hand leaves camera view: stop everything immediately."""
        commit_stroke = []
        if self.is_painting and self.current_stroke:
            commit_stroke = list(self.current_stroke)

        self.current_mode = self.MODE_FIST
        self.candidate_mode = self.MODE_FIST
        self.candidate_frames = 0
        self.is_painting = False
        self.current_stroke = []
        self.current_stroke_set = set()
        self.last_stroke_cell = None
        self.pinch_start_frames = 0
        self.pinch_release_frames = 0
        self.prev_palm_center = None
        self.prev_hands_dist = None

        self.last_state["state_label"] = self.STATE_NOHAND
        self.last_state["mode_badge"] = "MODE: STOPPED"
        self.last_state["action_badge"] = "NO HAND"
        self.last_state["gesture"] = "STOP"
        self.last_state["is_painting"] = False
        self.last_state["active_stroke"] = []
        self.last_state["commit_stroke"] = commit_stroke
        self.last_state["preview_active"] = False
        self.last_state["is_rotating"] = False
        self.last_state["is_zooming"] = False
        self.last_state["num_hands"] = 0
        self.last_state["hand_visible"] = False
        self.last_state["confidence"] = 0
        self.last_state["pinch_ratio"] = 1.0
        return dict(self.last_state)

    def reset_view(self):
        self.rot_x = 18.0
        self.rot_y = -30.0
        self.rot_z = 0.0

    def set_rotation(self, rot_x, rot_y, rot_z=0.0):
        self.rot_x = rot_x
        self.rot_y = rot_y
        self.rot_z = rot_z

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


class GeometricStrokeBuilder:
    """
    Segment-Based Sticky-Axis 3D Geometric Drawing Engine.
    
    Principles:
    1. INTENT WINDOW:
       - Collects initial fingertip samples at start of stroke/segment.
       - Locks dominant direction: 'H' (Horizontal, lock Y), 'V' (Vertical, lock X), or 'CURVE'.
    2. STICKY AXIS LOCK:
       - In 'H': Y is STRICTLY locked to the initial row. MediaPipe vertical jitter is 100% ignored.
       - In 'V': X is STRICTLY locked to the initial column. MediaPipe horizontal jitter is 100% ignored.
       - Depth Z is locked to the stroke plane.
    3. SEGMENT-BASED CORNER DETECTION:
       - Sustained perpendicular motion (> DIRECTION_CHANGE_THRESHOLD for CORNER_SUSTAIN_FRAMES)
         ends the previous segment cleanly at the corner block and starts the next perpendicular segment.
       - Creates razor-sharp 90° right angles (no rounded corners, no diagonal transitions).
    4. PERFECT VOXEL RECTANGLES / SQUARES:
       - A 4-sided loop snaps closed when returning within 1 cell of the start point.
       - Exactly one block thick with zero gaps, zero clustering, and zero zigzags.
    """
    def __init__(self):
        self.DIRECTION_CHANGE_THRESHOLD = 0.85  # grid units required to turn a corner
        self.CORNER_SUSTAIN_FRAMES = 2         # consecutive frames of perpendicular movement required
        self.INTENT_MIN_DIST = 0.35            # movement required to decide initial segment direction
        self.reset()

    def reset(self):
        self.all_completed_segments = []       # list of (seg_start_cell, seg_end_cell)
        self.start_cell = None
        self.anchor_cell = None
        self.anchor_float = None
        self.locked_axis = None                # 'H', 'V', 'CURVE', or None
        self.locked_coord = None               # locked Y for 'H', locked X for 'V'
        self.depth_z = 0
        self.max_reach = 0
        self.corner_sustain_count = 0
        self.active_cell = None
        self.stroke_cells = []
        self.stroke_set = set()
        self.last_cell = None

    def start(self, float_pos):
        fx, fy, fz = float(float_pos[0]), float(float_pos[1]), float(float_pos[2])
        gx, gy, gz = int(round(fx)), int(round(fy)), int(round(fz))
        start_cell = (gx, gy, gz)
        self.all_completed_segments = []
        self.start_cell = start_cell
        self.anchor_cell = start_cell
        self.anchor_float = (fx, fy, fz)
        self.locked_axis = None
        self.locked_coord = None
        self.depth_z = gz
        self.max_reach = gx
        self.corner_sustain_count = 0
        self.active_cell = start_cell
        self.last_cell = start_cell
        self.stroke_cells = [start_cell]
        self.stroke_set = {start_cell}

    def add_point(self, float_pos):
        if self.anchor_cell is None:
            self.start(float_pos)
            return self.active_cell

        fx, fy, fz = float(float_pos[0]), float(float_pos[1]), float(float_pos[2])
        ax, ay, az = self.anchor_float

        dx = fx - ax
        dy = fy - ay
        dist = math.hypot(dx, dy)

        # 1. Intent Window: determine initial axis if not yet set
        if self.locked_axis is None:
            if dist < self.INTENT_MIN_DIST:
                return self.anchor_cell
            if abs(dx) >= 1.30 * abs(dy):
                self.locked_axis = 'H'
                self.locked_coord = self.anchor_cell[1] # lock Y row
                self.max_reach = self.anchor_cell[0]
            elif abs(dy) >= 1.30 * abs(dx):
                self.locked_axis = 'V'
                self.locked_coord = self.anchor_cell[0] # lock X col
                self.max_reach = self.anchor_cell[1]
            else:
                self.locked_axis = 'CURVE'

        # 2. Process according to active locked axis
        if self.locked_axis == 'H':
            locked_y = self.locked_coord
            cur_gx = int(round(fx))
            delta_y = fy - locked_y

            # Check for deliberate corner turn (sustained vertical intent)
            if abs(delta_y) >= self.DIRECTION_CHANGE_THRESHOLD:
                self.corner_sustain_count += 1
                if self.corner_sustain_count >= self.CORNER_SUSTAIN_FRAMES:
                    # Finalize current horizontal segment at corner!
                    corner_x = self.max_reach
                    corner_cell = (corner_x, locked_y, self.depth_z)
                    self.all_completed_segments.append((self.anchor_cell, corner_cell))

                    # Start new vertical segment at corner_cell
                    self.anchor_cell = corner_cell
                    self.anchor_float = (float(corner_x), float(locked_y), float(self.depth_z))
                    self.locked_axis = 'V'
                    self.locked_coord = corner_x # lock X to corner_x
                    self.max_reach = corner_cell[1]
                    self.corner_sustain_count = 0
                    return self.add_point(float_pos)
            else:
                self.corner_sustain_count = 0
                reach_dir = 1 if (cur_gx >= self.anchor_cell[0]) else -1
                if reach_dir == 1 and cur_gx > self.max_reach:
                    self.max_reach = cur_gx
                elif reach_dir == -1 and cur_gx < self.max_reach:
                    self.max_reach = cur_gx

            active_cell = (cur_gx, locked_y, self.depth_z)

        elif self.locked_axis == 'V':
            locked_x = self.locked_coord
            cur_gy = int(round(fy))
            delta_x = fx - locked_x

            # Check for deliberate corner turn (sustained horizontal intent)
            if abs(delta_x) >= self.DIRECTION_CHANGE_THRESHOLD:
                self.corner_sustain_count += 1
                if self.corner_sustain_count >= self.CORNER_SUSTAIN_FRAMES:
                    # Finalize current vertical segment at corner!
                    corner_y = self.max_reach
                    corner_cell = (locked_x, corner_y, self.depth_z)
                    self.all_completed_segments.append((self.anchor_cell, corner_cell))

                    # Start new horizontal segment at corner_cell
                    self.anchor_cell = corner_cell
                    self.anchor_float = (float(locked_x), float(corner_y), float(self.depth_z))
                    self.locked_axis = 'H'
                    self.locked_coord = corner_y # lock Y to corner_y
                    self.max_reach = corner_cell[0]
                    self.corner_sustain_count = 0
                    return self.add_point(float_pos)
            else:
                self.corner_sustain_count = 0
                reach_dir = 1 if (cur_gy >= self.anchor_cell[1]) else -1
                if reach_dir == 1 and cur_gy > self.max_reach:
                    self.max_reach = cur_gy
                elif reach_dir == -1 and cur_gy < self.max_reach:
                    self.max_reach = cur_gy

            active_cell = (locked_x, cur_gy, self.depth_z)

        else:
            # CURVE mode: follows smoothed continuous path
            cur_gx = int(round(fx))
            cur_gy = int(round(fy))
            active_cell = (cur_gx, cur_gy, self.depth_z)

        # Snap closed loop if near start point after drawing at least 3 completed segments
        if len(self.all_completed_segments) >= 3 and self.start_cell is not None:
            d_start = max(abs(active_cell[0] - self.start_cell[0]),
                          abs(active_cell[1] - self.start_cell[1]))
            if d_start <= 1:
                active_cell = self.start_cell

        self.active_cell = active_cell
        self.last_cell = active_cell

        # Build full deterministic stroke path from all completed segments + active segment
        cells = []
        seen = set()
        for (seg_p0, seg_p1) in self.all_completed_segments:
            for c in interpolate_3d_line(seg_p0, seg_p1):
                if c not in seen:
                    seen.add(c)
                    cells.append(c)

        for c in interpolate_3d_line(self.anchor_cell, self.active_cell):
            if c not in seen:
                seen.add(c)
                cells.append(c)

        self.stroke_cells = cells
        self.stroke_set = seen
        return active_cell


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

    # Strict Action States
    STATE_IDLE     = "IDLE"
    STATE_PAINT    = "PAINT"
    STATE_ROTATE   = "ROTATE"
    STATE_ZOOM_IN  = "ZOOM_IN"
    STATE_ZOOM_OUT = "ZOOM_OUT"
    STATE_ERASE    = "ERASE"

    def __init__(self, frame_w=640, frame_h=480, grid_range=(7, 5, 4)):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.max_gx, self.max_gy, self.max_gz = grid_range

        # Active State Machine
        self.state = self.STATE_IDLE

        # Gesture Confirmation Debounce
        self.candidate_raw = self.STATE_IDLE
        self.candidate_frames = 0
        self.CONFIRM_FRAMES_DEFAULT = 3
        self.CONFIRM_FRAMES_ERASE = 6

        # Smoothing for world grid target position
        self.ema_cur_x = EMA(alpha=0.40, initial=0.0)
        self.ema_cur_y = EMA(alpha=0.40, initial=0.0)
        self.ema_cur_z = EMA(alpha=0.30, initial=0.0)

        # Palm center smoothing for rotation
        self.ema_palm_x = EMA(alpha=0.30, initial=None)
        self.ema_palm_y = EMA(alpha=0.30, initial=None)
        self.prev_palm_pos = None

        # 3D View rotation angles (degrees)
        self.rot_x = 22.0
        self.rot_y = -30.0
        self.rot_z = 0.0

        # Rotation smoothness parameters
        self.ROT_SENSITIVITY = 0.40
        self.ROT_SMOOTHING = 0.30
        self.MAX_ROT_PER_FRAME = 4.5
        self.ROT_DEADZONE = 2.0

        # Zoom limits and sensitivity
        self.scale = 0.85
        self.MIN_SCALE = 0.35
        self.MAX_SCALE = 2.40
        self.ZOOM_STEP_RATE = 0.012  # Controlled zoom step per frame

        self.ref_hand_size = 115.0

        # Geometric Stroke Engine (straight line lock, sharp corners, curve preservation, zero gaps)
        self.stroke_builder = GeometricStrokeBuilder()
        self.is_painting = False
        self.stroke_count = 0
        self.current_stroke = []        # list of (gx, gy, gz)
        self.current_stroke_set = set() # O(1) duplicate check
        self.last_stroke_cell = None

        # Erase system
        self.erase_target = None
        self.erase_hold_frames = 0
        self.ERASE_HOLD_REQUIRED = 8
        self.erase_cooldown = 0
        self.last_erased_cell = None

        # Stable quantized grid cell
        self.current_gx = 0
        self.current_gy = 0
        self.current_gz = 0

        # Directional feedback strings
        self.rotation_dir_str = ""
        self.zoom_dir_str = ""

        # Last output state cache
        self.last_state = {
            "state": self.STATE_IDLE,
            "mode_badge": "READY",
            "action_badge": "PINCH TO PAINT",
            "sub_badge": "STOPPED",
            "direction_badge": "",
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
            "is_erasing": False,
            "erase_cell": None,
            "num_hands": 0,
            "hand_visible": False,
            "index_pos": None,
            "palm_pos": None,
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
                    is_ext_3d = (d3_mcp_tip / d3_mcp_pip) > 1.32

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

    def _detect_hand_fist(self, lm):
        """Returns True if a hand is tightly clenched into a fist."""
        f = self._classify_fingers(lm)
        return f["extended_count"] == 0

    def _detect_hand_open_palm(self, lm):
        """Returns True if a hand is an open palm (4 fingers extended)."""
        f = self._classify_fingers(lm)
        return f["extended_count"] >= 3

    def _detect_raw_candidate(self, lm, norm_lm, all_hands):
        """
        Evaluates raw hand geometry and returns a raw candidate state:
        STATE_ZOOM_IN, STATE_ZOOM_OUT, STATE_ROTATE, STATE_ERASE, STATE_PAINT, or STATE_IDLE.
        """
        num_hands = len(all_hands) if all_hands is not None else 1

        # ---------------------------------------------------------
        # 1. TWO-HAND GESTURES (ZOOM IN vs ZOOM OUT)
        # ---------------------------------------------------------
        if num_hands >= 2 and all_hands is not None and len(all_hands) >= 2:
            h0 = all_hands[0]
            h1 = all_hands[1]
            if len(h0) >= 21 and len(h1) >= 21:
                h0_palm = self._detect_hand_open_palm(h0)
                h1_palm = self._detect_hand_open_palm(h1)
                h0_fist = self._detect_hand_fist(h0)
                h1_fist = self._detect_hand_fist(h1)

                # GESTURE: TWO OPEN PALMS -> ZOOM IN
                if h0_palm and h1_palm:
                    return self.STATE_ZOOM_IN

                # GESTURE: TWO CLOSED FISTS -> ZOOM OUT
                if h0_fist and h1_fist:
                    return self.STATE_ZOOM_OUT

            return self.STATE_IDLE

        # ---------------------------------------------------------
        # 2. SINGLE-HAND GESTURES
        # ---------------------------------------------------------
        f = self._classify_fingers(lm, norm_lm)
        ext_count = f["extended_count"]
        size = self._hand_size(lm)

        d_index_thumb = self._dist(lm[self.THUMB_TIP], lm[self.INDEX_TIP])
        ratio_index_thumb = d_index_thumb / size

        d_index_middle = self._dist(lm[self.INDEX_TIP], lm[self.MIDDLE_TIP])
        ratio_index_middle = d_index_middle / size

        # GESTURE: ONE OPEN PALM -> ROTATE
        # 3 or 4 fingers extended, thumb free, no pinch
        if ext_count >= 3 and ratio_index_thumb > 0.40 and ratio_index_middle > 0.28:
            return self.STATE_ROTATE

        # GESTURE: INDEX + MIDDLE TOGETHER (PINCH) -> ERASE
        # Index & middle are extended, ring and pinky are folded,
        # index and middle tips are pinched together (< 0.22 hand size),
        # thumb is separated from index (> 0.35 hand size) so not a thumb pinch!
        if (f["index"] and f["middle"] and not f["ring"] and not f["pinky"]
                and ratio_index_middle < 0.24 and ratio_index_thumb > 0.35):
            return self.STATE_ERASE

        # GESTURE: INDEX + THUMB PINCH -> PAINT
        # Index extended, middle/ring/pinky folded, thumb pinched to index tip
        if f["index"] and not f["ring"] and not f["pinky"] and ratio_index_thumb < 0.32:
            return self.STATE_PAINT

        # If finger is pointing (Index extended only) without pinch:
        # If we are already painting, we remain in PAINT until release hysteresis!
        if self.state == self.STATE_PAINT:
            # Check release threshold (0.48):
            if ratio_index_thumb > 0.48:
                return self.STATE_IDLE
            else:
                return self.STATE_PAINT

        # Default single hand state
        return self.STATE_IDLE

    def update(self, landmarks, norm_landmarks=None, all_hands=None, existing_blocks=None, camera_r=None):
        if landmarks is None or len(landmarks) < 21:
            return self.hold()

        lm = landmarks
        wrist = lm[self.WRIST]
        index_tip = lm[self.INDEX_TIP]
        middle_mcp = lm[self.MIDDLE_MCP]
        size = self._hand_size(lm)

        palm_x = (wrist[0] + middle_mcp[0]) * 0.5
        palm_y = (wrist[1] + middle_mcp[1]) * 0.5
        num_hands = len(all_hands) if all_hands is not None else 1

        # Decrement erase cooldown if active
        if self.erase_cooldown > 0:
            self.erase_cooldown -= 1

        # -------------------------------------------------------------
        # STEP 1: DETECT RAW GESTURE CANDIDATE
        # -------------------------------------------------------------
        raw_candidate = self._detect_raw_candidate(lm, norm_landmarks, all_hands)

        # -------------------------------------------------------------
        # STEP 2: CONFIRMATION DEBOUNCE (NO RANDOM TRIGGERING)
        # -------------------------------------------------------------
        # Require consecutive frames before committing a state change
        required_frames = self.CONFIRM_FRAMES_ERASE if raw_candidate == self.STATE_ERASE else self.CONFIRM_FRAMES_DEFAULT

        if raw_candidate == self.candidate_raw:
            self.candidate_frames += 1
        else:
            self.candidate_raw = raw_candidate
            self.candidate_frames = 1

        # Determine if candidate is confirmed
        target_state = self.state
        if self.candidate_frames >= required_frames:
            target_state = self.candidate_raw

        # -------------------------------------------------------------
        # STEP 3: STRICT STATE MACHINE TRANSITIONS (MUTUALLY EXCLUSIVE)
        # -------------------------------------------------------------
        commit_stroke = []
        erase_cell = None

        # Exit handler for PAINT
        if self.state == self.STATE_PAINT and target_state != self.STATE_PAINT:
            if self.current_stroke:
                commit_stroke = list(self.current_stroke)
            self.is_painting = False
            self.current_stroke = []
            self.current_stroke_set = set()
            self.last_stroke_cell = None
            self.stroke_builder.reset()

        # Exit handler for ROTATE
        if self.state == self.STATE_ROTATE and target_state != self.STATE_ROTATE:
            self.prev_palm_pos = None
            self.ema_palm_x.value = None
            self.ema_palm_y.value = None
            self.rotation_dir_str = ""

        # Exit handler for ERASE
        if self.state == self.STATE_ERASE and target_state != self.STATE_ERASE:
            self.erase_target = None
            self.erase_hold_frames = 0

        # Exit handler for ZOOM
        if self.state in (self.STATE_ZOOM_IN, self.STATE_ZOOM_OUT) and target_state not in (self.STATE_ZOOM_IN, self.STATE_ZOOM_OUT):
            self.zoom_dir_str = ""

        # Commit confirmed state
        self.state = target_state

        # -------------------------------------------------------------
        # STEP 4: 3D GRID COORDINATE MAPPING (CURSOR)
        # -------------------------------------------------------------
        view_x = (index_tip[0] - self.frame_w * 0.5) / (self.frame_w * 0.38)
        view_y = -(index_tip[1] - self.frame_h * 0.5) / (self.frame_h * 0.38)
        view_x = max(-1.25, min(1.25, view_x))
        view_y = max(-1.25, min(1.25, view_y))

        view_z = (size - self.ref_hand_size) / 45.0
        view_z = max(-1.25, min(1.25, view_z))

        target_cam = np.array([view_x * self.max_gx,
                               view_y * self.max_gy,
                               view_z * self.max_gz], dtype=np.float64)

        if camera_r is not None:
            rot_mat = camera_r
        else:
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

        current_grid_pos = (self.current_gx, self.current_gy, self.current_gz)

        # -------------------------------------------------------------
        # STEP 5: EXECUTE ACTIVE MUTUALLY EXCLUSIVE ACTION
        # -------------------------------------------------------------
        mode_badge = "READY"
        action_badge = "PINCH TO PAINT"
        sub_badge = "STOPPED"
        direction_badge = ""

        # === STATE: PAINT ===
        if self.state == self.STATE_PAINT:
            float_pos = (smooth_wx, smooth_wy, smooth_wz)
            if not self.is_painting:
                # Enter paint stroke
                self.is_painting = True
                self.stroke_count += 1
                self.stroke_builder.start(float_pos)
                active_cursor_cell = self.stroke_builder.active_cell
                self.current_stroke = list(self.stroke_builder.stroke_cells)
                self.current_stroke_set = set(self.stroke_builder.stroke_set)
                self.last_stroke_cell = self.stroke_builder.last_cell
            else:
                # Intelligent Segment-Based Sticky-Axis Geometric Path Generation
                active_cursor_cell = self.stroke_builder.add_point(float_pos)
                self.current_stroke = list(self.stroke_builder.stroke_cells)
                self.current_stroke_set = set(self.stroke_builder.stroke_set)
                self.last_stroke_cell = self.stroke_builder.last_cell

            # Magnetically lock the cursor position on the grid
            current_grid_pos = active_cursor_cell
            self.current_gx, self.current_gy, self.current_gz = active_cursor_cell

            # Axis lock status indicator
            if self.stroke_builder.locked_axis == 'H':
                axis_str = f"LOCKED ROW Y={self.stroke_builder.locked_coord}"
            elif self.stroke_builder.locked_axis == 'V':
                axis_str = f"LOCKED COL X={self.stroke_builder.locked_coord}"
            elif self.stroke_builder.locked_axis == 'CURVE':
                axis_str = "FREEHAND CURVE"
            else:
                axis_str = "ALIGNING INTENT..."

            mode_badge = "PAINTING"
            action_badge = "DRAWING STROKE"
            sub_badge = f"STROKE {self.stroke_count:02d} ({len(self.current_stroke)} BLOCKS)"
            direction_badge = axis_str

        # === STATE: ROTATE ===
        elif self.state == self.STATE_ROTATE:
            smooth_px = self.ema_palm_x.update(palm_x)
            smooth_py = self.ema_palm_y.update(palm_y)

            rot_pitch_delta = 0.0
            rot_yaw_delta = 0.0

            # Rotation lock: first frame captures reference
            if self.prev_palm_pos is not None:
                dx = smooth_px - self.prev_palm_pos[0]
                dy = smooth_py - self.prev_palm_pos[1]

                h_dir = ""
                v_dir = ""

                # Move Palm Right -> Rotate Horizontally Right (+yaw)
                # Move Palm Left  -> Rotate Horizontally Left (-yaw)
                if abs(dx) > self.ROT_DEADZONE:
                    rot_yaw_delta = max(-self.MAX_ROT_PER_FRAME, min(self.MAX_ROT_PER_FRAME, dx * self.ROT_SENSITIVITY))
                    self.rot_y = (self.rot_y + rot_yaw_delta) % 360.0
                    h_dir = "ROTATE RIGHT →" if dx > 0 else "← ROTATE LEFT"

                # Move Palm Up    -> Rotate Vertically Up (+pitch)
                # Move Palm Down  -> Rotate Vertically Down (-pitch)
                if abs(dy) > self.ROT_DEADZONE:
                    rot_pitch_delta = max(-self.MAX_ROT_PER_FRAME, min(self.MAX_ROT_PER_FRAME, -dy * self.ROT_SENSITIVITY))
                    self.rot_x = max(-75.0, min(85.0, self.rot_x + rot_pitch_delta))
                    v_dir = "↑ ROTATE UP" if dy < 0 else "↓ ROTATE DOWN"

                if h_dir and v_dir:
                    self.rotation_dir_str = f"{h_dir}  |  {v_dir}"
                elif h_dir:
                    self.rotation_dir_str = h_dir
                elif v_dir:
                    self.rotation_dir_str = v_dir
                else:
                    self.rotation_dir_str = "PALM STEADY"

            self.prev_palm_pos = (smooth_px, smooth_py)

            mode_badge = "ROTATING"
            action_badge = "← MOVE PALM →"
            sub_badge = "ONE OPEN PALM"
            direction_badge = self.rotation_dir_str

        # === STATE: ZOOM IN ===
        elif self.state == self.STATE_ZOOM_IN:
            new_scale = self.scale * (1.0 + self.ZOOM_STEP_RATE)
            self.scale = max(self.MIN_SCALE, min(self.MAX_SCALE, new_scale))

            mode_badge = "ZOOM IN"
            action_badge = "↑ CLOSER"
            sub_badge = "TWO OPEN PALMS"
            direction_badge = f"ZOOM IN: {self.scale:.2f}x"

        # === STATE: ZOOM OUT ===
        elif self.state == self.STATE_ZOOM_OUT:
            new_scale = self.scale * (1.0 - self.ZOOM_STEP_RATE)
            self.scale = max(self.MIN_SCALE, min(self.MAX_SCALE, new_scale))

            mode_badge = "ZOOM OUT"
            action_badge = "↓ FARTHER"
            sub_badge = "TWO CLOSED FISTS"
            direction_badge = f"ZOOM OUT: {self.scale:.2f}x"

        # === STATE: ERASE ===
        elif self.state == self.STATE_ERASE:
            mode_badge = "ERASING"
            action_badge = "SELECT BLOCK"
            sub_badge = "INDEX + MIDDLE"

            # Check if cursor is on an existing block
            target_pos = current_grid_pos
            if existing_blocks is not None and target_pos in existing_blocks:
                if target_pos == self.erase_target:
                    self.erase_hold_frames += 1
                else:
                    self.erase_target = target_pos
                    self.erase_hold_frames = 1

                direction_badge = f"HOLD OVER ({target_pos[0]}, {target_pos[1]}, {target_pos[2]}) [{self.erase_hold_frames}/{self.ERASE_HOLD_REQUIRED}]"

                # One confirmed erase action -> One block removed
                if self.erase_hold_frames >= self.ERASE_HOLD_REQUIRED and self.erase_cooldown == 0:
                    erase_cell = target_pos
                    self.last_erased_cell = target_pos
                    self.erase_cooldown = 20  # Cooldown before another erase
                    self.erase_hold_frames = 0
                    self.erase_target = None
                    direction_badge = f"BLOCK ({target_pos[0]}, {target_pos[1]}, {target_pos[2]}) ERASED"
            else:
                self.erase_target = None
                self.erase_hold_frames = 0
                direction_badge = "HOVER OVER TARGET BLOCK"

        # === STATE: IDLE ===
        else:
            mode_badge = "READY"
            action_badge = "PINCH TO PAINT"
            sub_badge = "STOPPED"
            direction_badge = "AWAITING GESTURE"

        self.last_state = {
            "state": self.state,
            "mode_badge": mode_badge,
            "action_badge": action_badge,
            "sub_badge": sub_badge,
            "direction_badge": direction_badge,
            "grid_pos": current_grid_pos,
            "rot_x": self.rot_x,
            "rot_y": self.rot_y,
            "rot_z": self.rot_z,
            "pitch": self.rot_x,
            "yaw": self.rot_y,
            "scale": self.scale,
            "zoom_direction": "IN" if self.state == self.STATE_ZOOM_IN else ("OUT" if self.state == self.STATE_ZOOM_OUT else None),
            "is_painting": (self.state == self.STATE_PAINT),
            "active_stroke": list(self.current_stroke),
            "commit_stroke": commit_stroke,
            "preview_active": (self.state in (self.STATE_PAINT, self.STATE_ERASE, self.STATE_IDLE)),
            "is_rotating": (self.state == self.STATE_ROTATE),
            "is_zooming": (self.state in (self.STATE_ZOOM_IN, self.STATE_ZOOM_OUT)),
            "is_erasing": (self.state == self.STATE_ERASE),
            "erase_cell": erase_cell,
            "num_hands": num_hands,
            "hand_visible": True,
            "index_pos": (int(index_tip[0]), int(index_tip[1])),
            "palm_pos": (int(palm_x), int(palm_y)),
            "stroke_count": self.stroke_count,
            "stroke_len": len(self.current_stroke),
        }
        return self.last_state

    def hold(self):
        """When hand leaves camera view: stop everything safely."""
        commit_stroke = []
        if self.is_painting and self.current_stroke:
            commit_stroke = list(self.current_stroke)

        self.state = self.STATE_IDLE
        self.candidate_raw = self.STATE_IDLE
        self.candidate_frames = 0
        self.is_painting = False
        self.current_stroke = []
        self.current_stroke_set = set()
        self.last_stroke_cell = None
        self.prev_palm_pos = None
        self.erase_target = None
        self.erase_hold_frames = 0
        self.rotation_dir_str = ""
        self.zoom_dir_str = ""

        self.last_state = {
            "state": self.STATE_IDLE,
            "mode_badge": "STOPPED",
            "action_badge": "NO HAND",
            "sub_badge": "IDLE",
            "direction_badge": "HAND NOT DETECTED",
            "grid_pos": (0, 0, 0),
            "rot_x": self.rot_x,
            "rot_y": self.rot_y,
            "rot_z": self.rot_z,
            "scale": self.scale,
            "is_painting": False,
            "active_stroke": [],
            "commit_stroke": commit_stroke,
            "preview_active": False,
            "is_rotating": False,
            "is_zooming": False,
            "is_erasing": False,
            "erase_cell": None,
            "num_hands": 0,
            "hand_visible": False,
            "index_pos": None,
            "palm_pos": None,
            "stroke_count": self.stroke_count,
            "stroke_len": 0,
        }
        return dict(self.last_state)

    def reset_view(self):
        self.rot_x = 22.0
        self.rot_y = -30.0
        self.rot_z = 0.0
        self.stroke_builder.reset()

    def set_rotation(self, rot_x, rot_y, rot_z=0.0):
        self.rot_x = rot_x
        self.rot_y = rot_y
        self.rot_z = rot_z

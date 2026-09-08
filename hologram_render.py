"""
hologram_render.py
-------------------
Clean, minimal holographic wireframe renderer matching the Tony Stark project:
- Deep dark space with rotating 3D perspective ground grid plane.
- Crisp cyber-cyan wireframe cubes with double-pass bloom.
- Continuous 3D Painting preview: draws live translucent preview ribbon/path along the user's stroke.
- Holographic Gesture & Direction HUD on the left side.
- Minimalist uncluttered interface: clean [CLEAR] button and bottom-right camera PiP.
"""

import time
import cv2
import numpy as np


class HologramRenderer:
    def __init__(self, width=1024, height=720):
        self.width = width
        self.height = height

        self.scanline_y = 0.0
        self.scanline_speed = 100.0
        self._last_time = time.time()

        # Futuristic Electric Blue Holographic Palette (BGR)
        self.bg_color = (14, 10, 6)               # deep dark futuristic space
        self.grid_color = (70, 48, 16)            # subtle dark cyan/blue grid lines
        self.grid_highlight = (140, 95, 30)       # glowing blue grid accents

        # OPAQUE SHINING ELECTRIC BLUE BLOCKS (SOLID ENERGY UNITS)
        # Deep opaque electric blue core faces with high contrast directional shading
        self.face_base_blue = np.array([235, 115, 15], dtype=np.float64)  # Solid electric blue
        self.light_dir = np.array([0.35, 0.75, 0.55], dtype=np.float64)   # directional light vector
        self.light_dir /= np.linalg.norm(self.light_dir)

        # Luminous cyan-blue edges and outer border
        self.edge_cyan_core = (255, 245, 45)      # bright electric cyan edges
        self.edge_cyan_glow = (220, 130, 0)       # outer cyan bloom
        
        # In-progress stroke preview colors (distinct bright cyan-preview trail)
        self.preview_face_blue = np.array([190, 85, 10], dtype=np.float64)
        self.preview_edge_core = (255, 250, 100)
        self.preview_edge_glow = (200, 140, 0)

        # Erase target highlight color (intense red/orange contrasting wireframe & fill)
        self.erase_face_col = (30, 30, 180)
        self.erase_edge_col = (40, 60, 255)

        # Cursor
        self.cursor_edge = (240, 240, 240)

    def _draw_ground_grid(self, canvas, pts2d, edges):
        """Draws the rotating 3D horizontal CAD ground grid with subtle blue perspective lines."""
        for (i, j) in edges:
            if i >= len(pts2d) or j >= len(pts2d):
                continue
            p1, p2 = pts2d[i], pts2d[j]
            if not (np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))):
                continue
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            if (pt1[0] < -100 or pt1[0] > self.width + 100 or
                pt1[1] < -100 or pt1[1] > self.height + 100):
                continue
            cv2.line(canvas, pt1, pt2, self.grid_color, 1, cv2.LINE_AA)

    def _draw_opaque_blocks(self, canvas, block_items, is_preview=False, erase_target_cell=None):
        """
        Renders OPAQUE, SHINING ELECTRIC BLUE 3D BLOCKS with bright luminous cyan edges.
        100% OPAQUE - NO ALPHA BLENDING OR TRANSPARENT GLASS LOOK.
        Uses Painter's algorithm (back-to-front depth sorting) and directional lighting.
        Bevel separation (92% block size) ensures distinct physical units.
        """
        if not block_items:
            return

        # Sort blocks by average camera depth (avg_z) ascending (furthest first, closest last)
        sorted_blocks = sorted(block_items, key=lambda b: b.get("avg_z", 0.0))

        for block in sorted_blocks:
            pts2d = block["pts2d"]
            faces = block.get("faces", [])
            edges = block.get("edges", [])
            pos = block.get("grid_pos")

            is_erase_target = (erase_target_cell is not None and pos == erase_target_cell)

            # 1. Render 100% OPAQUE shaded faces sorted by face depth
            sorted_faces = sorted(faces, key=lambda f: f.get("avg_z", 0.0))
            for f in sorted_faces:
                f_pts = f["pts"]
                f_norm = f["normal"]

                # Back-face culling check
                if f_norm[2] < -0.20:
                    continue

                poly = np.array([[int(p[0]), int(p[1])] for p in f_pts], dtype=np.int32)
                if len(poly) < 3:
                    continue

                if is_erase_target:
                    fill_color = self.erase_face_col
                else:
                    # Directional lighting calculation: diffuse + specular highlight
                    diffuse = max(0.25, float(np.dot(f_norm, self.light_dir)))
                    specular = 0.0
                    half_vec = self.light_dir + np.array([0, 0, 1], dtype=np.float64)
                    half_vec /= np.linalg.norm(half_vec)
                    n_dot_h = max(0.0, float(np.dot(f_norm, half_vec)))
                    if n_dot_h > 0.65:
                        specular = (n_dot_h ** 10) * 110.0

                    base = self.preview_face_blue if is_preview else self.face_base_blue
                    b = min(255, int(base[0] * diffuse + specular + 40))
                    g = min(255, int(base[1] * diffuse + specular * 0.6 + 20))
                    r = min(255, int(base[2] * diffuse + specular * 0.3 + 10))
                    fill_color = (b, g, r)

                # DIRECT SOLID FILL (100% Opaque, alpha = 1.0)
                cv2.fillConvexPoly(canvas, poly, fill_color, lineType=cv2.LINE_AA)

            # 2. Render luminous cyan outer edge borders with subtle glow
            if is_erase_target:
                core_col = self.erase_edge_col
                border_thick = 3
            else:
                core_col = self.preview_edge_core if is_preview else self.edge_cyan_core
                border_thick = 1 if is_preview else 2

                # Subtle emissive outer glow pass around solid block
                if not is_preview:
                    for (i, j) in edges:
                        p1, p2 = pts2d[i], pts2d[j]
                        if not (np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))):
                            continue
                        pt1 = (int(p1[0]), int(p1[1]))
                        pt2 = (int(p2[0]), int(p2[1]))
                        cv2.line(canvas, pt1, pt2, (180, 110, 15), 4, cv2.LINE_AA)

            # Sharp physical solid core edge
            for (i, j) in edges:
                p1, p2 = pts2d[i], pts2d[j]
                if not (np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))):
                    continue
                pt1 = (int(p1[0]), int(p1[1]))
                pt2 = (int(p2[0]), int(p2[1]))
                cv2.line(canvas, pt1, pt2, core_col, border_thick, cv2.LINE_AA)

    def _draw_cursor(self, canvas, pts2d, edges, state="IDLE"):
        """Renders 3D cursor block at index brush position."""
        if state == "PAINT":
            edge_col = (0, 255, 180)
            thickness = 2
        elif state == "ERASE":
            edge_col = (50, 80, 255)
            thickness = 2
        else:
            edge_col = self.cursor_edge
            thickness = 1

        for (i, j) in edges:
            p1, p2 = pts2d[i], pts2d[j]
            if not (np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))):
                continue
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            cv2.line(canvas, pt1, pt2, edge_col, thickness, cv2.LINE_AA)

    def _draw_clear_button(self, canvas, btn_rect, is_hovered=False):
        """Minimal clean [CLEAR] button in top-left."""
        bx, by, bw, bh = btn_rect
        btn_overlay = canvas.copy()
        fill_col = (55, 45, 20) if is_hovered else (25, 20, 12)
        cv2.rectangle(btn_overlay, (bx, by), (bx + bw, by + bh), fill_col, -1)
        cv2.addWeighted(btn_overlay, 0.75, canvas, 0.25, 0, dst=canvas)

        border_col = (255, 235, 40) if is_hovered else (140, 100, 30)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), border_col, 1, cv2.LINE_AA)

        label = "CLEAR"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.46, 1)
        tx = bx + (bw - tw) // 2
        ty = by + (bh + th) // 2 - 1
        txt_col = (255, 255, 255) if is_hovered else (200, 220, 220)
        cv2.putText(canvas, label, (tx, ty), font, 0.46, txt_col, 1, cv2.LINE_AA)

    def _draw_gesture_control_panel(self, canvas, gesture_info):
        """
        Permanent on-screen HELP / CONTROL PANEL in bottom-left.
        Clearly displays all 5 precise mutually exclusive actions:
        - 🤏 INDEX + THUMB  -> PAINT / DRAW
        - 🖐 ONE OPEN PALM  -> ROTATE HORIZONTALLY / VERTICALLY
        - 👐 TWO OPEN PALMS -> ZOOM IN
        - ✊ TWO FISTS      -> ZOOM OUT
        - ☝🖕 INDEX + MIDDLE -> ERASE
        """
        pw, ph = 246, 395
        px = 20
        py = max(70, self.height - ph - 44)

        # Glassmorphic dark blue-tint backdrop
        hud_bg = canvas.copy()
        cv2.rectangle(hud_bg, (px, py), (px + pw, py + ph), (18, 14, 8), -1)
        cv2.addWeighted(hud_bg, 0.80, canvas, 0.20, 0, dst=canvas)

        # Futuristic glowing cyan border
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (90, 70, 24), 1, cv2.LINE_AA)

        # Tech corner brackets
        tlen = 8
        cyan_bracket = (255, 220, 30)
        cv2.line(canvas, (px, py), (px + tlen, py), cyan_bracket, 2)
        cv2.line(canvas, (px, py), (px, py + tlen), cyan_bracket, 2)
        cv2.line(canvas, (px + pw, py), (px + pw - tlen, py), cyan_bracket, 2)
        cv2.line(canvas, (px + pw, py), (px + pw, py + tlen), cyan_bracket, 2)
        cv2.line(canvas, (px, py + ph), (px + tlen, py + ph), cyan_bracket, 2)
        cv2.line(canvas, (px, py + ph), (px, py + ph - tlen), cyan_bracket, 2)
        cv2.line(canvas, (px + pw, py + ph), (px + pw - tlen, py + ph), cyan_bracket, 2)
        cv2.line(canvas, (px + pw, py + ph), (px + pw, py + ph - tlen), cyan_bracket, 2)

        # Panel Header
        cv2.putText(canvas, "HOLOGRID CONTROLS", (px + 14, py + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 230, 40), 1, cv2.LINE_AA)
        cv2.line(canvas, (px + 10, py + 28), (px + pw - 10, py + 28), (70, 52, 20), 1)

        cur_state = gesture_info.get("state", "IDLE") if gesture_info else "IDLE"

        cards = [
            {
                "icon": "🤏 INDEX + THUMB",
                "act": "PAINT / DRAW",
                "sub": "PINCH -> DRAW -> RELEASE",
                "active": (cur_state == "PAINT"),
                "color": (0, 255, 180),
            },
            {
                "icon": "🖐 ONE OPEN PALM",
                "act": "ROTATE 360 ORBIT",
                "sub": "<- LEFT / RIGHT -> | ^ UP / DOWN v",
                "active": (cur_state == "ROTATE"),
                "color": (255, 220, 30),
            },
            {
                "icon": "👐 TWO OPEN PALMS",
                "act": "ZOOM IN (CLOSER)",
                "sub": "TOWARD CAMERA -> ZOOM IN",
                "active": (cur_state == "ZOOM_IN"),
                "color": (255, 140, 240),
            },
            {
                "icon": "✊ TWO CLOSED FISTS",
                "act": "ZOOM OUT (FARTHER)",
                "sub": "TWO FISTS -> ZOOM OUT",
                "active": (cur_state == "ZOOM_OUT"),
                "color": (240, 110, 180),
            },
            {
                "icon": "☝🖕 INDEX + MIDDLE",
                "act": "ERASE BLOCK",
                "sub": "INDEX + MIDDLE -> ERASE",
                "active": (cur_state == "ERASE"),
                "color": (50, 90, 255),
            },
        ]

        item_y = py + 34
        item_h = 49
        for card in cards:
            box_x = px + 10
            box_w = pw - 20

            if card["active"]:
                cbg = canvas.copy()
                cv2.rectangle(cbg, (box_x, item_y), (box_x + box_w, item_y + item_h), (40, 32, 16), -1)
                cv2.addWeighted(cbg, 0.70, canvas, 0.30, 0, dst=canvas)
                cv2.rectangle(canvas, (box_x, item_y), (box_x + box_w, item_y + item_h), card["color"], 1, cv2.LINE_AA)
                cv2.circle(canvas, (box_x + 8, item_y + 13), 4, card["color"], -1, cv2.LINE_AA)
                title_col = (255, 255, 255)
            else:
                cv2.rectangle(canvas, (box_x, item_y), (box_x + box_w, item_y + item_h), (45, 36, 18), 1)
                cv2.circle(canvas, (box_x + 8, item_y + 13), 3, (80, 70, 45), -1)
                title_col = (190, 200, 200)

            cv2.putText(canvas, card["icon"], (box_x + 18, item_y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, title_col, 1, cv2.LINE_AA)
            cv2.putText(canvas, card["act"], (box_x + 18, item_y + 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, card["color"] if card["active"] else (220, 180, 40), 1, cv2.LINE_AA)
            cv2.putText(canvas, card["sub"], (box_x + 18, item_y + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (140, 160, 160), 1, cv2.LINE_AA)

            item_y += item_h + 5

        # Footer notes
        item_y += 2
        cv2.line(canvas, (px + 10, item_y), (px + pw - 10, item_y), (70, 52, 20), 1)
        item_y += 14
        cv2.putText(canvas, "RELEASE -> STOP CURRENT ACTION", (px + 14, item_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 220, 220), 1, cv2.LINE_AA)
        item_y += 13
        cv2.putText(canvas, "MUTUALLY EXCLUSIVE ACTIONS", (px + 14, item_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.29, (140, 170, 170), 1, cv2.LINE_AA)

    def _draw_top_banners(self, canvas, gesture_info, screen_data):
        """
        Draws TOP-LEFT App Title and TOP-RIGHT Prominent Active Mode & Gesture Indicators.
        Includes live directional cues: ROTATE RIGHT →, ← ROTATE LEFT, ↑ ROTATE UP, ZOOM IN ↑ CLOSER, etc.
        """
        # TOP LEFT: App Title & Subtitle
        cv2.putText(canvas, "HOLOGRID BUILDER", (135, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 235, 30), 2, cv2.LINE_AA)
        cv2.putText(canvas, "3D HOLOGRAPHIC SCULPTOR • PRECISION FSM", (135, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 150, 60), 1, cv2.LINE_AA)

        # TOP RIGHT: Active Mode & Action Badge
        mode_badge = gesture_info.get("mode_badge", "READY") if gesture_info else "READY"
        action_badge = gesture_info.get("action_badge", "") if gesture_info else ""
        direction_badge = gesture_info.get("direction_badge", "") if gesture_info else ""
        cur_state = gesture_info.get("state", "IDLE") if gesture_info else "IDLE"

        # Color coding for mode
        if cur_state == "PAINT":
            badge_border = (0, 255, 180)     # electric cyan/green
            badge_fill = (35, 50, 25)
        elif cur_state == "ROTATE":
            badge_border = (255, 220, 30)    # electric cyan-blue
            badge_fill = (45, 38, 18)
        elif cur_state == "ZOOM_IN":
            badge_border = (255, 140, 240)   # magenta/violet
            badge_fill = (45, 22, 42)
        elif cur_state == "ZOOM_OUT":
            badge_border = (240, 110, 180)   # deep rose
            badge_fill = (45, 18, 32)
        elif cur_state == "ERASE":
            badge_border = (50, 90, 255)     # electric red/amber
            badge_fill = (45, 20, 20)
        else:
            badge_border = (160, 140, 50)    # calm blue-grey
            badge_fill = (22, 18, 12)

        # Render prominent badge in top right (expanded for direction feedback)
        bx, by, bw, bh = self.width - 320, 14, 300, 56
        bg_card = canvas.copy()
        cv2.rectangle(bg_card, (bx, by), (bx + bw, by + bh), badge_fill, -1)
        cv2.addWeighted(bg_card, 0.72, canvas, 0.28, 0, dst=canvas)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), badge_border, 1, cv2.LINE_AA)

        # Text inside badge
        header_text = f"STATE: {mode_badge}"
        cv2.putText(canvas, header_text, (bx + 14, by + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        
        detail_text = f"{action_badge}"
        if direction_badge:
            detail_text += f" | {direction_badge}"
        if len(detail_text) > 34:
            detail_text = detail_text[:34]

        cv2.putText(canvas, detail_text, (bx + 14, by + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, badge_border, 1, cv2.LINE_AA)

    def render(self, screen_data, gesture_info=None, clear_btn_rect=(20, 20, 95, 34),
               is_btn_hovered=False, canvas_w=None, canvas_h=None):
        if canvas_w is not None and canvas_h is not None:
            self.width = canvas_w
            self.height = canvas_h

        canvas = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)

        # 1. 3D Perspective Ground Grid (centered in viewport)
        if "ground_grid" in screen_data:
            g_pts2d, g_edges = screen_data["ground_grid"]
            self._draw_ground_grid(canvas, g_pts2d, g_edges)

        # 2. In-Progress Stroke Preview (distinct glowing blue preview path)
        stroke_preview = screen_data.get("stroke_preview", [])
        if stroke_preview:
            self._draw_opaque_blocks(canvas, stroke_preview, is_preview=True)

        # 3. Committed Blocks (100% Opaque, shining electric blue blocks with cyan luminous edges)
        placed_blocks = screen_data.get("blocks", [])
        cur_state = gesture_info.get("state", "IDLE") if gesture_info else "IDLE"
        erase_target_cell = gesture_info.get("grid_pos") if cur_state == "ERASE" else None
        
        if placed_blocks:
            self._draw_opaque_blocks(canvas, placed_blocks, is_preview=False, erase_target_cell=erase_target_cell)

        # 4. Active Brush Cursor at index fingertip
        preview_active = gesture_info.get("preview_active", False) if gesture_info else False

        if "cursor" in screen_data and preview_active:
            cur_pts2d, cur_edges = screen_data["cursor"]
            self._draw_cursor(canvas, cur_pts2d, cur_edges, state=cur_state)

        # 5. Clean minimal [CLEAR] button in top-left
        self._draw_clear_button(canvas, clear_btn_rect, is_btn_hovered)

        # 6. Top Banners: Title (Top Left) & Prominent Active Mode Indicator (Top Right)
        self._draw_top_banners(canvas, gesture_info, screen_data)

        # 7. Permanent Futuristic Gesture Help / Control Panel (Corner HUD)
        self._draw_gesture_control_panel(canvas, gesture_info)

        # 8. Clean bottom status strip (blocks, strokes, in-progress count, coordinates, camera orbit)
        count = screen_data.get("count", 0)
        strokes = screen_data.get("stroke_count", 0)
        drawing_len = screen_data.get("active_stroke_len", 0)
        cpos = screen_data.get("cursor_pos", (0, 0, 0))
        cam_dist = screen_data.get("camera_dist", 9.8)
        yaw = screen_data.get("yaw", 0.0)
        pitch = screen_data.get("pitch", 0.0)

        bottom_bar = canvas.copy()
        cv2.rectangle(bottom_bar, (0, self.height - 34), (self.width, self.height), (8, 6, 4), -1)
        cv2.addWeighted(bottom_bar, 0.85, canvas, 0.15, 0, dst=canvas)

        status_left = f"BLOCKS: {count}  |  STROKES: {strokes}  |  TRAIL: {drawing_len}  |  CELL: ({cpos[0]:+d}, {cpos[1]:+d}, {cpos[2]:+d})"
        cv2.putText(canvas, status_left, (20, self.height - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 220, 220), 1, cv2.LINE_AA)

        # Orbit & Camera info on right
        orbit_info = f"ORBIT YAW: {yaw:.0f}°  PITCH: {pitch:.0f}°  ZOOM DIST: {cam_dist:.1f}"
        cv2.putText(canvas, orbit_info, (max(20, self.width - 440), self.height - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 200, 220), 1, cv2.LINE_AA)

        return canvas

    def add_pip(self, canvas, raw_frame, hand_landmarks=None, gesture_info=None,
                pip_w=200, pip_h=150, margin=14):
        """Composites clean PIP camera in bottom-right corner matching video reference."""
        h, w = canvas.shape[:2]
        pip = cv2.resize(raw_frame, (pip_w, pip_h))

        state_label = gesture_info.get("state_label", "STOP — NO HAND") if gesture_info else "STOP — NO HAND"
        is_painting = gesture_info.get("is_painting", False) if gesture_info else False

        if is_painting:
            skel_col = (0, 255, 160)
        elif "COMMITTED" in state_label:
            skel_col = (255, 230, 40)
        elif "ROTATE" in state_label:
            skel_col = (0, 220, 255)
        elif "ZOOM" in state_label:
            skel_col = (255, 120, 230)
        elif "FIST" in state_label:
            skel_col = (60, 120, 255)
        else:
            skel_col = (100, 120, 100)

        if hand_landmarks is not None and len(hand_landmarks) > 0:
            orig_h, orig_w = raw_frame.shape[:2]
            scale_x = pip_w / orig_w
            scale_y = pip_h / orig_h
            pip_lm = [(pt[0] * scale_x, pt[1] * scale_y) for pt in hand_landmarks]
            from hand_tracker import HandTracker
            pip = HandTracker.draw_skeleton(pip, pip_lm, color=skel_col, point_color=(255, 255, 255))

        x0 = w - pip_w - margin
        y0 = h - pip_h - 40 - margin

        cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x0 + pip_w + 1, y0 + pip_h + 1), skel_col, 1, cv2.LINE_AA)
        canvas[y0:y0 + pip_h, x0:x0 + pip_w] = pip

        disp_label = state_label.replace("—", "-").replace("☝ ", "").replace("✓ ", "").replace("•", "|")
        if len(disp_label) > 22:
            disp_label = disp_label[:22]
        cv2.putText(canvas, disp_label, (x0 + 6, y0 + pip_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, skel_col, 1, cv2.LINE_AA)

        return canvas

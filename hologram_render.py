"""
hologram_render.py
-------------------
Renders multi-block 3D voxels as glowing neon cyber holograms:
- Glowing double-pass bloom for placed blocks (cyan/teal).
- Distinct pulsating cursor wireframe (bright emerald/amber).
- Receding sci-fi grid floor and drifting scanlines.
- Enhanced PiP webcam inset showing real-time hand skeleton overlay.
- Sci-fi HUD overlay with active mode, grid coordinates, and controls.
"""

import time
import cv2
import numpy as np


class HologramRenderer:
    def __init__(self, width=1024, height=720):
        self.width = width
        self.height = height

        self.scanline_y = 0.0
        self.scanline_speed = 130.0  # pixels / sec
        self._last_time = time.time()

        # Neon Cyber Palette (BGR)
        self.block_core = (255, 250, 100)      # bright cyan-white core
        self.block_glow = (210, 150, 0)        # deep cyan bloom
        self.cursor_core = (100, 255, 180)     # neon emerald green core
        self.cursor_glow = (40, 180, 80)       # green glow
        self.grid_color = (80, 60, 10)         # deep space grid
        self.bg_color = (14, 10, 6)            # dark void background

    def _draw_grid_floor(self, canvas, horizon_y, vanish_x):
        h, w = canvas.shape[:2]
        n_lines = 9
        for i in range(1, n_lines + 1):
            t = i / n_lines
            y = int(horizon_y + (t ** 1.6) * (h - horizon_y))
            alpha = max(0.12, 1.0 - t * 0.75)
            color = tuple(int(c * alpha) for c in self.grid_color)
            cv2.line(canvas, (0, y), (w, y), color, 1, cv2.LINE_AA)

        n_verts = 14
        for i in range(n_verts + 1):
            x_top = int(vanish_x + (i - n_verts / 2) * (w / n_verts) * 0.22)
            x_bottom = int(vanish_x + (i - n_verts / 2) * (w / n_verts) * 2.2)
            cv2.line(canvas, (x_top, horizon_y), (x_bottom, h), self.grid_color, 1, cv2.LINE_AA)

    def _draw_scanline(self, canvas, dt):
        h, w = canvas.shape[:2]
        self.scanline_y += self.scanline_speed * dt
        if self.scanline_y > h:
            self.scanline_y = 0.0
        y = int(self.scanline_y)
        overlay = canvas.copy()
        cv2.line(overlay, (0, y), (w, y), (180, 220, 255), 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, dst=canvas)

    def _draw_wireframe(self, canvas, pts2d, edges, core_col, glow_col, glow_w=7, core_w=2):
        overlay = canvas.copy()
        # Glow pass
        for (i, j) in edges:
            p1, p2 = pts2d[i], pts2d[j]
            if not (np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))):
                continue
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            cv2.line(overlay, pt1, pt2, glow_col, glow_w, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.40, canvas, 0.60, 0, dst=canvas)

        # Core crisp line pass
        for (i, j) in edges:
            p1, p2 = pts2d[i], pts2d[j]
            if not (np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))):
                continue
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            cv2.line(canvas, pt1, pt2, core_col, core_w, cv2.LINE_AA)

        # Corner joints
        for p in pts2d:
            if np.all(np.isfinite(p)):
                cv2.circle(canvas, (int(p[0]), int(p[1])), 3, core_col, -1, cv2.LINE_AA)

    def _draw_control_panel(self, canvas, buttons, hovered_btn=None):
        """Renders on-screen cyber control panel and clickable buttons."""
        # Panel backdrop on left side
        panel_x, panel_y, panel_w, panel_h = 14, 68, 178, 595
        panel_overlay = canvas.copy()
        cv2.rectangle(panel_overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                      (18, 14, 8), -1)
        cv2.addWeighted(panel_overlay, 0.75, canvas, 0.25, 0, dst=canvas)

        # Panel cyber border and corner accents
        cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                      (80, 70, 20), 1, cv2.LINE_AA)
        cv2.line(canvas, (panel_x, panel_y), (panel_x + 20, panel_y), (0, 255, 160), 2, cv2.LINE_AA)
        cv2.line(canvas, (panel_x, panel_y), (panel_x, panel_y + 20), (0, 255, 160), 2, cv2.LINE_AA)
        cv2.line(canvas, (panel_x + panel_w - 20, panel_y + panel_h), (panel_x + panel_w, panel_y + panel_h), (0, 255, 160), 2, cv2.LINE_AA)
        cv2.line(canvas, (panel_x + panel_w, panel_y + panel_h - 20), (panel_x + panel_w, panel_y + panel_h), (0, 255, 160), 2, cv2.LINE_AA)

        # Draw each button
        for btn in buttons:
            bx, by, bw, bh = btn["rect"]
            bid = btn["id"]
            label = btn["label"]
            is_active = btn.get("active", False)
            is_hovered = (bid == hovered_btn)

            # Button background fill
            btn_overlay = canvas.copy()
            if is_hovered:
                fill_col = (50, 90, 40) if "paint" in bid else (40, 70, 90)
                alpha = 0.65
            elif is_active:
                fill_col = (30, 80, 40)
                alpha = 0.50
            else:
                fill_col = (25, 20, 14)
                alpha = 0.40

            cv2.rectangle(btn_overlay, (bx, by), (bx + bw, by + bh), fill_col, -1)
            cv2.addWeighted(btn_overlay, alpha, canvas, 1.0 - alpha, 0, dst=canvas)

            # Button border
            if is_hovered:
                border_col = (0, 255, 220)
                b_thick = 2
            elif is_active:
                border_col = (80, 255, 120)
                b_thick = 2
            else:
                border_col = (110, 100, 50)
                b_thick = 1

            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), border_col, b_thick, cv2.LINE_AA)

            # Text centering
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.40 if len(label) > 8 else 0.44
            (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
            tx = bx + max(2, (bw - tw) // 2)
            ty = by + (bh + th) // 2 - 1
            txt_col = (255, 255, 255) if (is_hovered or is_active) else (200, 215, 215)
            cv2.putText(canvas, label, (tx, ty), font, scale, txt_col, 1, cv2.LINE_AA)

    def render(self, screen_data, status_text="", mode_text="PAINTING", hand_detected=True,
               buttons=None, hovered_btn=None):
        now = time.time()
        dt = now - self._last_time
        self._last_time = now

        canvas = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)

        # 1. Perspective grid floor
        horizon_y = int(self.height * 0.72)
        self._draw_grid_floor(canvas, horizon_y, self.width // 2)

        # 2. Render all placed blocks
        for pts2d, edges in screen_data.get("blocks", []):
            self._draw_wireframe(canvas, pts2d, edges, self.block_core, self.block_glow, glow_w=6, core_w=2)

        # 3. Render active 3D hand cursor
        if "cursor" in screen_data and hand_detected:
            cur_pts2d, cur_edges = screen_data["cursor"]
            pulse = 0.5 + 0.5 * np.sin(now * 8.0)
            cursor_c = (int(self.cursor_core[0] * 0.7 + self.cursor_core[0] * 0.3 * pulse),
                        int(self.cursor_core[1]),
                        int(self.cursor_core[2]))
            self._draw_wireframe(canvas, cur_pts2d, cur_edges, cursor_c, self.cursor_glow, glow_w=9, core_w=2)

        # 4. Drifting scanline
        self._draw_scanline(canvas, dt)

        # 5. On-screen Control Panel
        if buttons:
            self._draw_control_panel(canvas, buttons, hovered_btn)

        # 6. HUD Status Bar at bottom
        count = screen_data.get("count", 0)
        cpos = screen_data.get("cursor_pos", (0, 0, 0))
        coord_str = f"GRID: (X:{cpos[0]:+d}, Y:{cpos[1]:+d}, Z:{cpos[2]:+d}) | BLOCKS: {count}"

        # Semi-transparent bottom HUD bar
        bar_overlay = canvas.copy()
        cv2.rectangle(bar_overlay, (0, self.height - 42), (self.width, self.height), (10, 8, 4), -1)
        cv2.addWeighted(bar_overlay, 0.7, canvas, 0.3, 0, dst=canvas)

        cv2.putText(canvas, coord_str, (200, self.height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)

        # Mode indicator badge
        mode_col = (0, 220, 255) if "ROTAT" in mode_text else (80, 255, 120)
        cv2.putText(canvas, f"MODE: {mode_text}", (self.width - 340, self.height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_col, 2, cv2.LINE_AA)

        if status_text:
            cv2.putText(canvas, status_text, (200, self.height - 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 200, 255), 1, cv2.LINE_AA)

        return canvas

    def add_pip(self, canvas, raw_frame, hand_landmarks=None, hand_detected=False,
                pip_w=210, pip_h=150, margin=14):
        """Composites PIP camera with real-time hand skeleton overlay."""
        h, w = canvas.shape[:2]
        pip = cv2.resize(raw_frame, (pip_w, pip_h))

        # Scale landmarks to PIP dimensions
        if hand_landmarks is not None and len(hand_landmarks) > 0:
            orig_h, orig_w = raw_frame.shape[:2]
            scale_x = pip_w / orig_w
            scale_y = pip_h / orig_h
            pip_lm = [(pt[0] * scale_x, pt[1] * scale_y) for pt in hand_landmarks]
            from hand_tracker import HandTracker
            pip = HandTracker.draw_skeleton(pip, pip_lm, color=(0, 255, 120), point_color=(255, 255, 255))

        x0 = w - pip_w - margin
        y0 = margin

        # Border
        border_col = (0, 255, 160) if hand_detected else (80, 80, 160)
        cv2.rectangle(canvas, (x0 - 2, y0 - 2), (x0 + pip_w + 2, y0 + pip_h + 2),
                      border_col, 2, cv2.LINE_AA)
        canvas[y0:y0 + pip_h, x0:x0 + pip_w] = pip

        # Badge under PiP
        badge_text = "HAND DETECTED" if hand_detected else "LOOKING FOR HAND"
        badge_col = (0, 255, 120) if hand_detected else (100, 100, 220)
        cv2.putText(canvas, badge_text, (x0 + 10, y0 + pip_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, badge_col, 1, cv2.LINE_AA)

        return canvas



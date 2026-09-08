"""
main.py
-------
HoloGrid 3D Studio — 3D Painting & Sculpting Edition.

Features:
- Continuous 3D Stroke Painting: Pinch index + thumb to start drawing; move index finger
  freely to sculpt unbroken lines, curves, rings, and geometric figures.
- Real-Time 3D Paint Preview: Shows an active glowing ghost trail along the user's trajectory.
- Release to Commit: Opening the pinch commits the complete figure atomically to the hologrid.
- Zero Gaps: 3D Bresenham / DDA interpolation connects every movement with no disconnected voxels.
- Whole Palm Rotation: Tumbles the complete 3D structure around its central pivot.
- Two-Hand Zoom: Spreading or bringing hands together zooms in and out.
- Minimalist Iron Man CAD UI: Side HUD, clean [CLEAR] button, and camera PiP.
"""

import time
import cv2
import ctypes
import numpy as np

from hand_tracker import HandTracker
from gesture_logic import GestureProcessor
from cube_engine import CubeEngine
from hologram_render import HologramRenderer

try:
    _user32 = ctypes.windll.user32
    SCREEN_MONITOR_W = _user32.GetSystemMetrics(0)
    SCREEN_MONITOR_H = _user32.GetSystemMetrics(1)
except Exception:
    SCREEN_MONITOR_W = 1280
    SCREEN_MONITOR_H = 720

WIN_W = min(1280, SCREEN_MONITOR_W - 60)
WIN_H = min(720, SCREEN_MONITOR_H - 100)

CLEAR_BTN_RECT = (20, 20, 95, 34)

# Mouse interaction state
mouse_state = {
    "x": 0,
    "y": 0,
    "is_down": False,
    "drag_start": None,
    "clicked_clear": False,
    "is_hovered": False,
}


def _point_in_rect(px, py, rect):
    rx, ry, rw, rh = rect
    return rx <= px <= rx + rw and ry <= py <= ry + rh


def _on_mouse(event, x, y, flags, param):
    global mouse_state
    mouse_state["x"] = x
    mouse_state["y"] = y
    mouse_state["is_hovered"] = _point_in_rect(x, y, CLEAR_BTN_RECT)

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_state["is_down"] = True
        mouse_state["drag_start"] = (x, y)
        if mouse_state["is_hovered"]:
            mouse_state["clicked_clear"] = True

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_state["is_down"] = False
        mouse_state["drag_start"] = None

    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            param["cube"].zoom_in()
        else:
            param["cube"].zoom_out()


def _open_camera():
    """Try camera indices 0 and 1 with a short warmup loop."""
    for idx in (0, 1):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(25):
            ret, frame = cap.read()
            if ret and frame is not None and frame.mean() > 2:
                print(f"[Camera] Opened index {idx} successfully ({frame.shape[1]}x{frame.shape[0]}).")
                return cap, frame
            time.sleep(0.06)
        cap.release()
    return None, None


def main():
    cap, first_frame = _open_camera()
    if cap is None:
        print("Could not open or read from any webcam (tried indices 0 and 1).")
        return

    cam_h, cam_w = first_frame.shape[:2]

    # Initialize tracking supporting up to 2 hands (for drawing + two-hand zoom)
    tracker = HandTracker(max_hands=2, detection_confidence=0.32, tracking_confidence=0.32)
    gestures = GestureProcessor(frame_w=cam_w, frame_h=cam_h, grid_range=(7, 5, 4))
    cube = CubeEngine(block_size=0.35, grid_step=0.35)
    renderer = HologramRenderer(width=WIN_W, height=WIN_H)

    win_name = "HoloGrid 3D Studio"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, WIN_W, WIN_H)

    mouse_params = {"cube": cube}
    cv2.setMouseCallback(win_name, _on_mouse, mouse_params)

    prev_drag_pos = None

    print("\n=======================================================")
    print("   HOLOGRID BUILDER — PROFESSIONAL FSM 3D STUDIO")
    print("=======================================================")
    print(" 🤏 INDEX + THUMB PINCH  : START PAINTING")
    print(" ☝ MOVE INDEX FINGER     : Draw continuous 3D block path")
    print(" 🤏 RELEASE PINCH        : Commit stroke & return to IDLE")
    print(" 🖐 ONE OPEN PALM        : 360° Smooth Hologrid Orbit")
    print(" 👐 TWO OPEN PALMS       : ZOOM IN (Toward Camera)")
    print(" ✊ TWO CLOSED FISTS     : ZOOM OUT (Farther / Overview)")
    print(" ☝🖕 INDEX + MIDDLE PINCH : ERASE TARGET BLOCK (Hold to remove)")
    print(" 'f'                     : Toggle Fullscreen Display")
    print(" 'c' / [CLEAR]           : Clear structure | 'u': Undo stroke")
    print(" 'r'                     : Reset view | Mouse Drag: Rotate")
    print("=======================================================\n")

    is_fullscreen = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame = cv2.flip(frame, 1)

            # Query dynamic window dimensions (handles window maximizing, resizing, fullscreen)
            fs_prop = cv2.getWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN)
            if fs_prop == cv2.WINDOW_FULLSCREEN or is_fullscreen:
                cur_win_w = SCREEN_MONITOR_W
                cur_win_h = SCREEN_MONITOR_H
            else:
                rect = cv2.getWindowImageRect(win_name)
                if rect is not None and rect[2] > 100 and rect[3] > 100:
                    cur_win_w = rect[2]
                    cur_win_h = rect[3]
                else:
                    cur_win_w = WIN_W
                    cur_win_h = WIN_H

            # Handle clear button click
            if mouse_state["clicked_clear"]:
                mouse_state["clicked_clear"] = False
                cube.clear()
                print("[Action] Cleared all blocks.")

            # Handle mouse drag rotation (secondary control)
            if mouse_state["is_down"] and mouse_state["drag_start"]:
                cur_x, cur_y = mouse_state["x"], mouse_state["y"]
                if not mouse_state["is_hovered"]:
                    if prev_drag_pos is not None:
                        pdx = cur_x - prev_drag_pos[0]
                        pdy = cur_y - prev_drag_pos[1]
                        cube.rotate_orbit(pitch_delta=-pdy * 0.40, yaw_delta=pdx * 0.40)
                        gestures.set_rotation(cube.pitch, cube.yaw)
                    prev_drag_pos = (cur_x, cur_y)
            else:
                prev_drag_pos = None

            # Process hand tracking with synchronized camera view matrix
            result = tracker.process(frame)
            landmarks = result["landmarks"]
            norm_landmarks = result.get("norm_landmarks")
            all_hands = result.get("all_hands")

            cam_r = cube.get_camera_r()
            if landmarks is not None:
                state = gestures.update(landmarks, norm_landmarks, all_hands, existing_blocks=cube.blocks, camera_r=cam_r)
            else:
                state = gestures.hold()

            # Update 3D cursor position
            gx, gy, gz = state["grid_pos"]
            cube.set_cursor(gx, gy, gz)

            # Sync in-progress preview stroke for live rendering
            cube.set_active_stroke(state.get("active_stroke", []))

            # Commit stroke if completed
            commit_cells = state.get("commit_stroke", [])
            if commit_cells:
                new_blocks = cube.commit_stroke(commit_cells)
                print(f"[Stroke Committed] Added {new_blocks} blocks. Total: {len(cube.blocks)} across {len(cube.stroke_history)} strokes.")

            # Dedicated erase action execution (one confirmed block removed)
            erase_cell = state.get("erase_cell")
            if erase_cell is not None:
                cube.remove_block(erase_cell[0], erase_cell[1], erase_cell[2])
                print(f"[Erase Committed] Removed block at {erase_cell}. Remaining: {len(cube.blocks)}.")

            # Apply 3D scene orbit rotation when One Open Palm is active (full 360 wrap)
            if state["is_rotating"]:
                cube.set_orbit(state["pitch"], state["yaw"])

            # Handle dedicated Zoom In / Zoom Out gestures
            zoom_dir = state.get("zoom_direction")
            if zoom_dir == "IN":
                cube.zoom_in(factor=0.985)
            elif zoom_dir == "OUT":
                cube.zoom_out(factor=1.015)

            # Project 3D geometry with dynamic viewport resolution and correct aspect ratio
            screen_data = cube.get_screen_data(cur_win_w, cur_win_h)

            # Render clean Tony Stark holographic scene with dynamic canvas dimensions
            canvas = renderer.render(
                screen_data,
                gesture_info=state,
                clear_btn_rect=CLEAR_BTN_RECT,
                is_btn_hovered=mouse_state["is_hovered"],
                canvas_w=cur_win_w,
                canvas_h=cur_win_h,
            )

            # Composite clean PiP camera in bottom-right corner with skeleton
            canvas = renderer.add_pip(
                canvas,
                frame,
                hand_landmarks=landmarks,
                gesture_info=state,
            )

            cv2.imshow(win_name, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('f'):
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(win_name, WIN_W, WIN_H)
            elif key == ord('c'):
                cube.clear()
            elif key == ord('r'):
                cube.set_preset_view("iso")
                gestures.reset_view()
            elif key == ord('u'):
                cube.undo_stroke()
            elif key in (ord('+'), ord('=')):
                cube.zoom_in()
            elif key in (ord('-'), ord('_')):
                cube.zoom_out()

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
main.py
-------
Project HoloGrid 3D Builder
Interactive 3D Voxel Hologram Studio with On-Screen UI & Hand Gesture Controls.

Features:
- On-Screen Clickable Buttons: Clear, Undo, Paint Toggle, Zoom -/+, View Presets (ISO, Front, Top, Side), Rotation controls, and Auto-Spin.
- Zoomed-Out Default View: Clean perspective fitting all 3D blocks comfortably.
- Air Hand Drawing: Move hand in 3D air to paint blocks in whichever direction you move.
- Physical Fist-Grab: Turn the whole 3D world by making a fist and moving hand.
- Mouse Support: Click buttons, drag in 3D viewport to tumble/rotate, scroll to zoom.
"""

import time
import cv2

from hand_tracker import HandTracker
from gesture_logic import GestureProcessor
from cube_engine import CubeEngine
from hologram_render import HologramRenderer

WIN_W = 1024
WIN_H = 720

# Global mouse state
mouse_state = {
    "x": 0,
    "y": 0,
    "is_down": False,
    "drag_start": None,
    "clicked_btn": None,
    "hovered_btn": None,
}


def _get_buttons(paint_enabled, auto_spin):
    bx = 20
    bw = 166
    bw_half = 80
    bw_gap = 6

    return [
        # Actions
        {"id": "clear",        "label": "CLEAR ALL",       "rect": (bx, 82,  bw, 32), "active": False},
        {"id": "undo",         "label": "UNDO BLOCK",      "rect": (bx, 120, bw, 30), "active": False},
        {"id": "paint_toggle", "label": f"PAINT: {'ON' if paint_enabled else 'OFF'}",
                               "rect": (bx, 156, bw, 32), "active": paint_enabled},

        # Zoom
        {"id": "zoom_out",     "label": "ZOOM -",          "rect": (bx, 206, bw_half, 30), "active": False},
        {"id": "zoom_in",      "label": "ZOOM +",          "rect": (bx + bw_half + bw_gap, 206, bw_half, 30), "active": False},

        # 3D Presets
        {"id": "view_iso",     "label": "3D ISO VIEW",     "rect": (bx, 256, bw, 28), "active": False},
        {"id": "view_front",   "label": "FRONT",           "rect": (bx, 290, bw_half, 28), "active": False},
        {"id": "view_top",     "label": "TOP",             "rect": (bx + bw_half + bw_gap, 290, bw_half, 28), "active": False},
        {"id": "view_side",    "label": "SIDE VIEW",       "rect": (bx, 324, bw, 28), "active": False},

        # Nudge Rotation
        {"id": "rot_left",     "label": "< ROT-L",         "rect": (bx, 372, bw_half, 28), "active": False},
        {"id": "rot_right",    "label": "ROT-R >",         "rect": (bx + bw_half + bw_gap, 372, bw_half, 28), "active": False},
        {"id": "tilt_up",      "label": "^ TILT-U",        "rect": (bx, 406, bw_half, 28), "active": False},
        {"id": "tilt_down",    "label": "v TILT-D",        "rect": (bx + bw_half + bw_gap, 406, bw_half, 28), "active": False},

        # Auto-spin & Reset
        {"id": "auto_spin",    "label": f"SPIN: {'ON' if auto_spin else 'OFF'}",
                               "rect": (bx, 454, bw, 30), "active": auto_spin},
        {"id": "reset_view",   "label": "RESET VIEW",      "rect": (bx, 490, bw, 30), "active": False},
    ]


def _point_in_rect(px, py, rect):
    rx, ry, rw, rh = rect
    return rx <= px <= rx + rw and ry <= py <= ry + rh


def _on_mouse(event, x, y, flags, param):
    global mouse_state
    mouse_state["x"] = x
    mouse_state["y"] = y

    buttons = param.get("buttons", [])

    # Check hover
    hovered = None
    for btn in buttons:
        if _point_in_rect(x, y, btn["rect"]):
            hovered = btn["id"]
            break
    mouse_state["hovered_btn"] = hovered

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_state["is_down"] = True
        mouse_state["drag_start"] = (x, y)
        if hovered:
            mouse_state["clicked_btn"] = hovered

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_state["is_down"] = False
        mouse_state["drag_start"] = None

    elif event == cv2.EVENT_MOUSEWHEEL:
        # High-order word indicates scroll direction
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

    tracker = HandTracker(max_hands=1, detection_confidence=0.32, tracking_confidence=0.32)
    gestures = GestureProcessor(frame_w=cam_w, frame_h=cam_h, grid_range=(8, 6, 5))
    cube = CubeEngine(block_size=0.35, grid_step=0.35)
    renderer = HologramRenderer(width=WIN_W, height=WIN_H)

    paint_enabled = True
    auto_spin = False
    fps = 0.0
    prev_time = time.time()

    win_name = "Project HoloGrid 3D Builder"
    cv2.namedWindow(win_name)

    mouse_params = {"buttons": [], "cube": cube}
    cv2.setMouseCallback(win_name, _on_mouse, mouse_params)

    prev_drag_pos = None

    print("\n=======================================================")
    print("Project HoloGrid 3D Builder running with On-Screen UI!")
    print(" - On-screen buttons: Clear, Undo, Zoom -/+, Rotate, Spin")
    print(" - Hand in air: Paint blocks in whichever direction you move")
    print(" - Closed fist: Grab and turn the whole 3D world")
    print(" - Mouse: Click buttons, drag 3D view, or scroll to zoom")
    print("=======================================================\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame = cv2.flip(frame, 1)

            # 1. Update on-screen buttons list
            buttons = _get_buttons(paint_enabled, auto_spin)
            mouse_params["buttons"] = buttons

            # 2. Handle button clicks
            if mouse_state["clicked_btn"]:
                cid = mouse_state["clicked_btn"]
                mouse_state["clicked_btn"] = None

                if cid == "clear":
                    cube.clear()
                    print("[UI] Cleared all blocks.")
                elif cid == "undo":
                    cube.undo()
                    print("[UI] Undid last block.")
                elif cid == "paint_toggle":
                    paint_enabled = not paint_enabled
                    print(f"[UI] Paint toggled: {'ON' if paint_enabled else 'OFF'}")
                elif cid == "zoom_in":
                    cube.zoom_in()
                elif cid == "zoom_out":
                    cube.zoom_out()
                elif cid == "view_iso":
                    cube.set_preset_view("iso")
                    gestures.set_rotation(cube.rot_x, cube.rot_y, cube.rot_z)
                elif cid == "view_front":
                    cube.set_preset_view("front")
                    gestures.set_rotation(cube.rot_x, cube.rot_y, cube.rot_z)
                elif cid == "view_top":
                    cube.set_preset_view("top")
                    gestures.set_rotation(cube.rot_x, cube.rot_y, cube.rot_z)
                elif cid == "view_side":
                    cube.set_preset_view("side")
                    gestures.set_rotation(cube.rot_x, cube.rot_y, cube.rot_z)
                elif cid == "rot_left":
                    cube.rotate_nudge(yaw_delta=-18.0)
                    gestures.set_rotation(cube.rot_x, cube.rot_y, cube.rot_z)
                elif cid == "rot_right":
                    cube.rotate_nudge(yaw_delta=18.0)
                    gestures.set_rotation(cube.rot_x, cube.rot_y, cube.rot_z)
                elif cid == "tilt_up":
                    cube.rotate_nudge(pitch_delta=-18.0)
                    gestures.set_rotation(cube.rot_x, cube.rot_y, cube.rot_z)
                elif cid == "tilt_down":
                    cube.rotate_nudge(pitch_delta=18.0)
                    gestures.set_rotation(cube.rot_x, cube.rot_y, cube.rot_z)
                elif cid == "auto_spin":
                    auto_spin = not auto_spin
                    print(f"[UI] Auto-spin: {'ON' if auto_spin else 'OFF'}")
                elif cid == "reset_view":
                    cube.set_preset_view("iso")
                    cube.scale = 0.80
                    gestures.reset_view()
                    auto_spin = False
                    print("[UI] Reset view.")

            # 3. Handle mouse drag rotation (when dragging outside the button panel)
            if mouse_state["is_down"] and mouse_state["drag_start"]:
                cur_x, cur_y = mouse_state["x"], mouse_state["y"]
                if cur_x > 200:  # outside the left control panel
                    if prev_drag_pos is not None:
                        pdx = cur_x - prev_drag_pos[0]
                        pdy = cur_y - prev_drag_pos[1]
                        cube.rotate_nudge(pitch_delta=-pdy * 0.45, yaw_delta=pdx * 0.45)
                        gestures.set_rotation(cube.rot_x, cube.rot_y, cube.rot_z)
                    prev_drag_pos = (cur_x, cur_y)
            else:
                prev_drag_pos = None

            # 4. Auto-spin rotation
            if auto_spin:
                cube.rotate_nudge(yaw_delta=0.85)
                gestures.set_rotation(cube.rot_x, cube.rot_y, cube.rot_z)

            # 5. Hand tracking
            result = tracker.process(frame)
            hand_present = result["landmarks"] is not None

            if hand_present:
                state = gestures.update(result["landmarks"])
                gx, gy, gz = state["grid_pos"]
                cube.set_cursor(gx, gy, gz)

                if state["grab"]:
                    mode_str = "ROTATING 3D VIEW (Fist Grab)"
                    status_str = f"Rotating scene | Hand at ({gx:+d}, {gy:+d}, {gz:+d})"
                    cube.set_transform(state["rot_x"], state["rot_y"], state["rot_z"])
                else:
                    if paint_enabled:
                        cube.add_block(gx, gy, gz)
                        mode_str = "3D AIR PAINTING"
                        status_str = f"Painting voxel at ({gx:+d}, {gy:+d}, {gz:+d}) | Fist to rotate"
                    else:
                        mode_str = "CURSOR NAVIGATION (Paused)"
                        status_str = f"Cursor at ({gx:+d}, {gy:+d}, {gz:+d}) | Click [PAINT] to draw"
            else:
                state = gestures.hold()
                mode_str = "NO HAND DETECTED"
                status_str = "Show hand or use on-screen buttons to build & rotate"

            # 6. Project 3D blocks & cursor to 2D screen coordinates
            screen_data = cube.get_screen_data(WIN_W, WIN_H)

            # 7. Render hologram scene with on-screen buttons
            canvas = renderer.render(
                screen_data,
                status_text=status_str,
                mode_text=mode_str,
                hand_detected=hand_present,
                buttons=buttons,
                hovered_btn=mouse_state["hovered_btn"],
            )

            # 8. Composite PiP camera with skeleton tracking overlay
            canvas = renderer.add_pip(
                canvas,
                frame,
                hand_landmarks=result["landmarks"],
                hand_detected=hand_present,
            )

            # 9. FPS and Header HUD
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                inst = 1.0 / dt
                fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst

            cv2.putText(canvas, "HOLOGRID 3D BUILDER", (210, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 120), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"FPS: {fps:.1f} | Zoom: {cube.scale:.2f}x | Click on-screen buttons or drag mouse", (210, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (120, 240, 255), 1, cv2.LINE_AA)

            cv2.imshow(win_name, canvas)

            # 10. Key shortcuts (backup alongside on-screen buttons)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('c'):
                cube.clear()
            elif key == ord('r'):
                cube.set_preset_view("iso")
                cube.scale = 0.80
                gestures.reset_view()
                auto_spin = False
            elif key == ord('u'):
                cube.undo()
            elif key == ord(' '):
                paint_enabled = not paint_enabled
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



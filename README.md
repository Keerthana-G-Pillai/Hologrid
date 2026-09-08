# HoloGrid 3D Builder — Hand-Tracked Holographic Voxel Studio

An interactive webcam-powered 3D holographic block builder rendered entirely with OpenCV and NumPy (no heavy OpenGL/3D game engines). Track your hand movements in real-time with MediaPipe to paint 3D voxel structures in mid-air and rotate the entire 3D world with physical hand gestures!

---

## What's New & Features

1. **Air Drawing in Any Direction (X, Y, Z)**:
   - Move your hand naturally in the air to paint glowing sci-fi blocks in whichever direction your hand travels.
   - Depth (Z-axis) dynamically responds to moving your hand closer or further from the camera.

2. **360° 3D View Rotation**:
   - Close your hand into a **Fist** to grab and turn the entire 3D scene in real-time.
   - Release your fist to lock the angle and continue building from that new perspective.

3. **Compact Modular Block Size**:
   - Blocks have been tuned to a sleek, modular size (`0.35` units) so you can sculpt rich 3D structures without giant blocks cluttering the screen.

4. **Enhanced Hand Detection & Live Skeleton PiP**:
   - Uses MediaPipe Tasks `RunningMode.IMAGE` with sensitive detection thresholds (`0.32`) to ensure your hand is picked up immediately regardless of lighting or handedness.
   - The top-right Picture-in-Picture webcam feed displays your real-time 21-point hand skeleton in neon green so you can immediately see tracking status.

## On-Screen Cyber Control Panel

All features can now be controlled directly on the screen by clicking with your mouse (or using hand gestures / hotkeys):

### 1. Actions
- **`[CLEAR ALL]`**: Clears all blocks back to the center anchor.
- **`[UNDO BLOCK]`**: Undoes the last placed voxel.
- **`[PAINT: ON/OFF]`**: Toggles active air painting on or off (pause painting to navigate the cursor freely).

### 2. View Zoom
- **`[ZOOM -]`** / **`[ZOOM +]`**: Smoothly zooms the 3D scene in or out. (You can also scroll the mouse wheel or pinch fingers).
- **Zoomed-Out Default View**: The camera distance and initial scale have been zoomed out so your 3D voxel creations fit comfortably in the frame.

### 3. 3D View Angles & Presets
- **`[3D ISO VIEW]`**: Isometric 3D angle.
- **`[FRONT]`**: Direct front elevation view.
- **`[TOP]`**: Direct bird's-eye top-down plan view.
- **`[SIDE VIEW]`**: Side profile view.

### 4. Rotation Controls
- **`[< ROT-L]`** / **`[ROT-R >]`**: Step rotate view left/right.
- **`[^ TILT-U]`** / **`[v TILT-D]`**: Step tilt view up/down.
- **`[SPIN: ON/OFF]`**: Automatic continuous rotation so you can view your creation hands-free.
- **`[RESET VIEW]`**: Snaps orientation and zoom back to default.
- **Mouse Drag in 3D Viewport**: Click and drag anywhere in the 3D area with the mouse to freely tumble and inspect your blocks from any angle!



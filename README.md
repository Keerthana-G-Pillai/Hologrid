# HoloGrid 3D Studio — Tony Stark / Iron Man Holographic Builder

An uncluttered, interactive 3D holographic voxel workshop inspired by Tony Stark's CAD interface, rendered entirely in Python using OpenCV and NumPy with MediaPipe hand tracking.

---

## 3D Continuous Painting Controls

### 1. 🤏 Pinch (Index + Thumb) to Start Painting
- Bring your **Index Fingertip and Thumb Tip together**.
- The system enters **`PAINTING`** mode and locks the stroke.
- The index finger becomes your continuous 3D brush.

### 2. ☝ Move Index Finger: Sculpt Unbroken 3D Figures
- Simply move your index finger in 3D space.
- The system connects every movement using **3D DDA interpolation with ZERO GAPS**.
- Creates straight beams, smooth curves, circles, rings, and complex 3D geometry.
- A real-time **ghost preview trail** renders along the entire path as you draw.

### 3. 💨 Release Pinch: Commit Complete Stroke
- Open your fingers to release the pinch.
- The entire path is **atomically committed** as glowing cyber-cyan blocks.
- Moving the index finger after release does **NOT** modify previous strokes.
- Pinch again whenever you want to begin a fresh new stroke.

### 4. 🖐 Whole Open Palm: Rotate 3D Scene
- Show an **Open Palm** (3+ fingers extended) to tumble the entire 3D structure around its center pivot.

### 5. 👐 Two Hands: Zoom In / Out
- Present **both hands** and separate them to zoom in, or bring them together to zoom out.

### 6. ✊ Closed Fist: Cancel / Standby
- Making a fist during drawing immediately **cancels** the current in-progress stroke.

### 7. Keyboard Shortcuts
- **`u`**: Undo last complete stroke.
- **`c`**: Clear all blocks.
- **`r`**: Reset view to isometric preset.
- **Mouse Drag**: Secondary rotation | **Scroll**: Zoom.

---

## Clean & Minimal Interface

- **Pure 3D Viewport**: All clumsy sidebar buttons have been removed for an authentic, clean holographic space.
- **3D Perspective Ground Grid**: Horizontal CAD ground plane that rotates with the scene for spatial orientation.
- **Top-Left `[CLEAR]` Button**: Single clean button to reset blocks (or press `c` on the keyboard).
- **Bottom-Right Camera Inset**: Minimal PiP webcam feed with live hand skeleton overlay.
- **Mouse Controls**: Click & drag in the 3D space to rotate; scroll to zoom in/out.





"""
cube_engine.py
--------------
Holographic 3D Voxel Engine matching the Tony Stark / Iron Man reference:
- Modular cyan wireframe cubes (size 0.35, grid step 0.35).
- Rotating horizontal 3D ground grid plane (XZ plane) for CAD depth cues.
- Linear structural placement: ensures straight lines, frames, and geometric shapes.
- Clean white/grey active cursor box.
"""

import numpy as np


class CubeEngine:
    def __init__(self, block_size=0.35, grid_step=0.35):
        self.block_size = block_size
        self.grid_step = grid_step

        # Template unit cube centered at origin
        # Slightly scaled for bevel / gap separation (92% of grid step) so adjacent blocks don't fuse
        s = (block_size * 0.92) / 2.0
        self.unit_vertices = np.array([
            [-s, -s, -s],  # 0
            [ s, -s, -s],  # 1
            [ s,  s, -s],  # 2
            [-s,  s, -s],  # 3
            [-s, -s,  s],  # 4
            [ s, -s,  s],  # 5
            [ s,  s,  s],  # 6
            [-s,  s,  s],  # 7
        ], dtype=np.float64)

        # 6 Quad faces (indices into unit_vertices) with outward normals for directional shading
        self.cube_faces = [
            ([0, 1, 2, 3], np.array([ 0,  0, -1], dtype=np.float64)),  # Back
            ([4, 5, 6, 7], np.array([ 0,  0,  1], dtype=np.float64)),  # Front
            ([0, 3, 7, 4], np.array([-1,  0,  0], dtype=np.float64)),  # Left
            ([1, 2, 6, 5], np.array([ 1,  0,  0], dtype=np.float64)),  # Right
            ([2, 3, 7, 6], np.array([ 0,  1,  0], dtype=np.float64)),  # Top
            ([0, 1, 5, 4], np.array([ 0, -1,  0], dtype=np.float64)),  # Bottom
        ]

        self.cube_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # back face
            (4, 5), (5, 6), (6, 7), (7, 4),  # front face
            (0, 4), (1, 5), (2, 6), (3, 7),  # connecting edges
        ]

        # Collection of blocks: set of (gx, gy, gz)
        self.blocks = set()
        self.history = []
        self.stroke_history = []
        self.active_stroke = []
        self.last_placed_pos = None

        # Active hand cursor position in grid coordinates
        self.cursor = (0, 0, 0)

        # -------------------------------------------------------------
        # PROPER 3D ORBIT CAMERA SYSTEM
        # -------------------------------------------------------------
        # Camera orbits around the stable central pivot of the hologrid
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Yaw (horizontal orbit 0..360°) and Pitch (vertical orbit)
        self.yaw = -30.0    # degrees, full 360 continuous wrap
        self.pitch = 22.0   # degrees, safe vertical orbit without camera inversion
        self.roll = 0.0

        # Camera distance: controlled by zoom
        # Default distance chosen so the entire hologrid plane and blocks comfortably fit viewport on startup
        self.DEFAULT_DIST = 9.5
        self.MIN_DIST = 4.0   # Zoomed in close inspection
        self.MAX_DIST = 22.0  # Zoomed out wide overview
        self.distance = self.DEFAULT_DIST

        # Vertical Field of View in degrees
        self.fov_deg = 46.0

        # Generate 3D centered ground grid lines at Y = 0.0 plane (dead center)
        self.ground_grid_edges = []
        self.ground_grid_vertices = []
        self._build_ground_grid(span=8, step=0.35, y_level=0.0)

    def _build_ground_grid(self, span=8, step=0.35, y_level=0.0):
        """
        Generates 3D CAD ground grid centered exactly around (0, y_level, 0).
        Comfortable span ensures the full working plane fits on screen at default distance.
        """
        verts = []
        edges = []
        idx = 0
        extent = span * step
        # Grid lines along X
        for i in range(-span, span + 1):
            z = i * step
            verts.append([-extent, y_level, z])
            verts.append([ extent, y_level, z])
            edges.append((idx, idx + 1))
            idx += 2
        # Grid lines along Z
        for i in range(-span, span + 1):
            x = i * step
            verts.append([x, y_level, -extent])
            verts.append([x, y_level,  extent])
            edges.append((idx, idx + 1))
            idx += 2
        self.ground_grid_vertices = np.array(verts, dtype=np.float64)
        self.ground_grid_edges = edges

    def add_block(self, gx: int, gy: int, gz: int):
        pos = (int(gx), int(gy), int(gz))
        if pos not in self.blocks:
            self.blocks.add(pos)
            self.history.append(pos)
            self.last_placed_pos = pos
            return True
        return False

    def remove_block(self, gx: int, gy: int, gz: int):
        pos = (int(gx), int(gy), int(gz))
        if pos in self.blocks:
            self.blocks.discard(pos)
            if pos in self.history:
                self.history.remove(pos)
            return True
        return False

    def set_active_stroke(self, cells):
        """Sets the in-progress preview stroke cells."""
        self.active_stroke = list(cells)

    def clear_active_stroke(self):
        """Discards in-progress preview stroke without committing."""
        self.active_stroke = []

    def commit_stroke(self, cells):
        """
        Commits an entire continuous painted stroke into the hologrid at once.
        Eliminates duplicates, appends to stroke history, and returns count of new blocks.
        """
        new_cells = []
        for cell in cells:
            pos = (int(cell[0]), int(cell[1]), int(cell[2]))
            if pos not in self.blocks:
                self.blocks.add(pos)
                self.history.append(pos)
                new_cells.append(pos)
        if new_cells:
            self.stroke_history.append(new_cells)
            self.last_placed_pos = new_cells[-1]
        self.active_stroke = []
        return len(new_cells)

    def undo_stroke(self):
        """Undoes the most recent entire painting stroke."""
        if self.stroke_history:
            last_stroke = self.stroke_history.pop()
            for pos in last_stroke:
                self.blocks.discard(pos)
            self.last_placed_pos = self.stroke_history[-1][-1] if self.stroke_history else None
            return True
        elif self.history:
            self.undo()
            return True
        return False

    def undo(self):
        if self.history:
            last = self.history.pop()
            self.blocks.discard(last)
            self.last_placed_pos = self.history[-1] if self.history else None

    def clear(self):
        self.blocks.clear()
        self.history.clear()
        self.stroke_history.clear()
        self.active_stroke.clear()
        self.last_placed_pos = None

    def zoom_in(self, factor=0.88):
        """Moves camera closer along orbit line (reveals more detail)."""
        self.distance = max(self.MIN_DIST, self.distance * factor)

    def zoom_out(self, factor=1.14):
        """Moves camera farther along orbit line (reveals more of the plane)."""
        self.distance = min(self.MAX_DIST, self.distance * factor)

    def set_zoom_distance(self, dist):
        """Directly set camera orbit distance clamped to safe limits."""
        self.distance = max(self.MIN_DIST, min(self.MAX_DIST, float(dist)))

    def set_preset_view(self, preset="iso"):
        if preset == "iso":
            self.pitch, self.yaw = 24.0, -30.0
        elif preset == "front":
            self.pitch, self.yaw = 0.0, 0.0
        elif preset == "top":
            self.pitch, self.yaw = 85.0, 0.0
        self.distance = self.DEFAULT_DIST

    def rotate_orbit(self, pitch_delta=0.0, yaw_delta=0.0):
        """
        Orbits the camera smoothly around the hologrid center.
        Yaw rotates full 360° continuously.
        Pitch is bounded between -75° and +85° to prevent camera flipping.
        """
        self.yaw = (self.yaw + yaw_delta) % 360.0
        self.pitch = max(-75.0, min(85.0, self.pitch + pitch_delta))

    def set_orbit(self, pitch_deg, yaw_deg):
        self.yaw = yaw_deg % 360.0
        self.pitch = max(-75.0, min(85.0, pitch_deg))

    def set_cursor(self, gx: int, gy: int, gz: int):
        self.cursor = (int(gx), int(gy), int(gz))

    def get_view_matrix(self):
        """
        Computes standard 4x4 LookAt View Matrix for camera orbiting target:
        eye = target + distance * [cos(pitch)*sin(yaw), sin(pitch), cos(pitch)*cos(yaw)]
        """
        p_rad = np.radians(self.pitch)
        y_rad = np.radians(self.yaw)

        # Eye position in 3D world space
        eye_x = self.target[0] + self.distance * np.cos(p_rad) * np.sin(y_rad)
        eye_y = self.target[1] + self.distance * np.sin(p_rad)
        eye_z = self.target[2] + self.distance * np.cos(p_rad) * np.cos(y_rad)
        eye = np.array([eye_x, eye_y, eye_z], dtype=np.float64)

        # Forward vector (from eye to target)
        forward = self.target - eye
        norm_f = np.linalg.norm(forward)
        forward = forward / norm_f if norm_f > 1e-6 else np.array([0, 0, -1], dtype=np.float64)

        # Up vector (world +Y)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        # Right vector
        right = np.cross(forward, world_up)
        norm_r = np.linalg.norm(right)
        if norm_r < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            right = right / norm_r

        # Recompute orthonormal true up
        true_up = np.cross(right, forward)

        # Camera rotation matrix (World -> Camera space)
        # Camera convention: X=right, Y=up, Z=-forward (looking down -Z)
        R = np.vstack([right, true_up, -forward])
        t = -R @ eye

        return R, t, eye

    def get_camera_r(self):
        """Returns the 3x3 Camera Rotation Matrix (World -> Camera space)."""
        R, _, _ = self.get_view_matrix()
        return R

    def project_points(self, pts3d, screen_w, screen_h):
        """
        Perspective project 3D world points -> 2D screen coordinates.
        Maintains true aspect ratio and vertical FOV across any window resolution.
        """
        R, t, _ = self.get_view_matrix()
        # Transform points to Camera Space: P_cam = pts3d @ R.T + t
        cam_pts = pts3d @ R.T + t

        # Focal length derived from vertical FOV
        fov_rad = np.radians(self.fov_deg)
        focal = (screen_h * 0.5) / np.tan(fov_rad * 0.5)

        pts2d = np.zeros((len(pts3d), 2), dtype=np.float64)
        cam_z = cam_pts[:, 2]

        for i in range(len(pts3d)):
            z = -cam_z[i] # distance in front of camera
            z = z if z > 0.1 else 0.1
            factor = focal / z
            pts2d[i, 0] = cam_pts[i, 0] * factor + screen_w * 0.5
            pts2d[i, 1] = -cam_pts[i, 1] * factor + screen_h * 0.5

        return pts2d, cam_pts

    def get_screen_data(self, screen_w, screen_h):
        """Calculates projected coordinates for ground grid, placed blocks, active preview, and cursor."""
        R, t, eye = self.get_view_matrix()

        # 1. 3D Ground Grid (horizontal CAD floor centered around target)
        grid_pts2d, _ = self.project_points(self.ground_grid_vertices, screen_w, screen_h)

        # 2. Placed blocks (sorted by camera depth for Painter's algorithm)
        block_items = []
        for (gx, gy, gz) in self.blocks:
            offset = np.array([gx * self.grid_step,
                               gy * self.grid_step,
                               gz * self.grid_step], dtype=np.float64)
            local_v = self.unit_vertices + offset
            pts2d, cam_v = self.project_points(local_v, screen_w, screen_h)

            # Distance from camera eye to block center for back-to-front sorting
            center_world = offset
            dist_to_eye = float(np.linalg.norm(center_world - eye))

            # Faces in camera space
            faces = []
            for face_idx, normal in self.cube_faces:
                cam_n = normal @ R.T
                face_pts = [pts2d[idx] for idx in face_idx]
                face_avg_dist = float(np.mean([np.linalg.norm(local_v[idx] - eye) for idx in face_idx]))
                faces.append({
                    "pts": face_pts,
                    "normal": cam_n,
                    "avg_z": -face_avg_dist, # for sorted_faces ascending (furthest first)
                })

            block_items.append({
                "pts2d": pts2d,
                "edges": self.cube_edges,
                "faces": faces,
                "avg_z": -dist_to_eye, # furthest first
                "grid_pos": (gx, gy, gz),
            })

        # 3. Active cursor block
        cgx, cgy, cgz = self.cursor
        cursor_offset = np.array([cgx * self.grid_step,
                                  cgy * self.grid_step,
                                  cgz * self.grid_step], dtype=np.float64)
        cursor_v = (self.unit_vertices * 1.05) + cursor_offset
        cursor_pts2d, _ = self.project_points(cursor_v, screen_w, screen_h)

        # 4. In-progress stroke preview
        stroke_items = []
        for (sgx, sgy, sgz) in self.active_stroke:
            offset = np.array([sgx * self.grid_step,
                               sgy * self.grid_step,
                               sgz * self.grid_step], dtype=np.float64)
            local_v = self.unit_vertices + offset
            pts2d, cam_v = self.project_points(local_v, screen_w, screen_h)
            dist_to_eye = float(np.linalg.norm(offset - eye))

            faces = []
            for face_idx, normal in self.cube_faces:
                cam_n = normal @ R.T
                face_pts = [pts2d[idx] for idx in face_idx]
                face_avg_dist = float(np.mean([np.linalg.norm(local_v[idx] - eye) for idx in face_idx]))
                faces.append({
                    "pts": face_pts,
                    "normal": cam_n,
                    "avg_z": -face_avg_dist,
                })

            stroke_items.append({
                "pts2d": pts2d,
                "edges": self.cube_edges,
                "faces": faces,
                "avg_z": -dist_to_eye,
                "grid_pos": (sgx, sgy, sgz),
            })

        return {
            "ground_grid": (grid_pts2d, self.ground_grid_edges),
            "blocks": block_items,
            "cursor": (cursor_pts2d, self.cube_edges),
            "stroke_preview": stroke_items,
            "count": len(self.blocks),
            "stroke_count": len(self.stroke_history),
            "active_stroke_len": len(self.active_stroke),
            "cursor_pos": (cgx, cgy, cgz),
            "camera_dist": self.distance,
            "yaw": self.yaw,
            "pitch": self.pitch,
        }




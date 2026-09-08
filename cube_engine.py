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

        # 3D View orientation: slight pitch and yaw for 3D isometric look
        self.rot_x = 18.0
        self.rot_y = -30.0
        self.rot_z = 0.0
        self.scale = 0.85
        self.camera_dist = 5.8

        # Generate 3D horizontal ground grid lines (Y = -2.0 plane)
        self.ground_grid_edges = []
        self.ground_grid_vertices = []
        self._build_ground_grid()

    def _build_ground_grid(self, span=12, step=0.35, y_level=-1.8):
        """Generates static 3D line segments for the ground grid plane."""
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

    def add_linear_step(self, target_gx: int, target_gy: int, target_gz: int):
        """
        Places the next block along the dominant straight-line axis from the
        last placed position, producing clean linear beams and rectangular frames.
        """
        if not self.blocks or self.last_placed_pos is None:
            return self.add_block(target_gx, target_gy, target_gz)

        lx, ly, lz = self.last_placed_pos
        dx = target_gx - lx
        dy = target_gy - ly
        dz = target_gz - lz

        # If already at the last placed position, nothing to add
        if dx == 0 and dy == 0 and dz == 0:
            return False

        # Find dominant axis to enforce straight line movement
        abs_dx, abs_dy, abs_dz = abs(dx), abs(dy), abs(dz)
        if abs_dx >= abs_dy and abs_dx >= abs_dz:
            step_pos = (lx + (1 if dx > 0 else -1), ly, lz)
        elif abs_dy >= abs_dx and abs_dy >= abs_dz:
            step_pos = (lx, ly + (1 if dy > 0 else -1), lz)
        else:
            step_pos = (lx, ly, lz + (1 if dz > 0 else -1))

        return self.add_block(step_pos[0], step_pos[1], step_pos[2])

    def remove_block(self, gx: int, gy: int, gz: int):
        pos = (int(gx), int(gy), int(gz))
        self.blocks.discard(pos)

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

    def zoom_in(self):
        self.scale = min(3.0, self.scale * 1.15)

    def zoom_out(self):
        self.scale = max(0.25, self.scale * 0.85)

    def set_preset_view(self, preset="iso"):
        if preset == "iso":
            self.rot_x, self.rot_y, self.rot_z = 18.0, -30.0, 0.0
        elif preset == "front":
            self.rot_x, self.rot_y, self.rot_z = 0.0, 0.0, 0.0
        elif preset == "top":
            self.rot_x, self.rot_y, self.rot_z = 88.0, 0.0, 0.0

    def rotate_nudge(self, pitch_delta=0.0, yaw_delta=0.0):
        self.rot_x += pitch_delta
        self.rot_y += yaw_delta

    def set_cursor(self, gx: int, gy: int, gz: int):
        self.cursor = (int(gx), int(gy), int(gz))

    def set_transform(self, rot_x_deg, rot_y_deg, rot_z_deg):
        self.rot_x = rot_x_deg
        self.rot_y = rot_y_deg
        self.rot_z = rot_z_deg

    @staticmethod
    def _rot_matrix_x(deg):
        a = np.radians(deg)
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

    @staticmethod
    def _rot_matrix_y(deg):
        a = np.radians(deg)
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

    @staticmethod
    def _rot_matrix_z(deg):
        a = np.radians(deg)
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    def get_rotation_matrix(self):
        return (self._rot_matrix_z(self.rot_z) @
                self._rot_matrix_y(self.rot_y) @
                self._rot_matrix_x(self.rot_x))

    def project_points(self, pts3d, screen_w, screen_h):
        """Perspective project 3D points -> 2D screen coordinates."""
        focal_length = screen_w * 0.95
        pts2d = np.zeros((len(pts3d), 2), dtype=np.float64)
        for i, (x, y, z) in enumerate(pts3d):
            denom = self.camera_dist - z
            denom = denom if abs(denom) > 1e-4 else 1e-4
            factor = focal_length / denom
            pts2d[i, 0] = x * factor + screen_w / 2.0
            pts2d[i, 1] = -y * factor + screen_h / 2.0
        return pts2d

    def get_screen_data(self, screen_w, screen_h):
        """Calculates projected coordinates for ground grid, placed blocks, and cursor."""
        rot_mat = self.get_rotation_matrix()

        # 1. 3D Ground Grid (horizontal CAD floor)
        world_grid = (self.ground_grid_vertices @ rot_mat.T) * self.scale
        grid_pts2d = self.project_points(world_grid, screen_w, screen_h)

        # 2. Placed blocks (sorted by camera depth for Painter's algorithm)
        block_items = []
        for (gx, gy, gz) in self.blocks:
            offset = np.array([gx * self.grid_step,
                               gy * self.grid_step,
                               gz * self.grid_step], dtype=np.float64)
            local_v = self.unit_vertices + offset
            world_v = (local_v @ rot_mat.T) * self.scale
            pts2d = self.project_points(world_v, screen_w, screen_h)
            
            # Average depth in camera space (Z component of world_v)
            # Since camera looks from +Z, objects with smaller camera_dist - Z are closer
            avg_z = float(np.mean(world_v[:, 2]))
            
            # Extract projected faces with outward world normals
            faces = []
            for face_idx, normal in self.cube_faces:
                world_n = (normal @ rot_mat.T)
                face_pts = [pts2d[idx] for idx in face_idx]
                face_avg_z = float(np.mean([world_v[idx, 2] for idx in face_idx]))
                faces.append({
                    "pts": face_pts,
                    "normal": world_n,
                    "avg_z": face_avg_z,
                })
            
            block_items.append({
                "pts2d": pts2d,
                "edges": self.cube_edges,
                "faces": faces,
                "avg_z": avg_z,
                "grid_pos": (gx, gy, gz),
            })

        # 3. Active cursor block (white/grey wireframe)
        cgx, cgy, cgz = self.cursor
        cursor_offset = np.array([cgx * self.grid_step,
                                  cgy * self.grid_step,
                                  cgz * self.grid_step], dtype=np.float64)
        cursor_v = (self.unit_vertices * 1.05) + cursor_offset
        cursor_world = (cursor_v @ rot_mat.T) * self.scale
        cursor_pts2d = self.project_points(cursor_world, screen_w, screen_h)

        # 4. In-progress stroke preview (glowing blue preview blocks)
        stroke_items = []
        for (sgx, sgy, sgz) in self.active_stroke:
            offset = np.array([sgx * self.grid_step,
                               sgy * self.grid_step,
                               sgz * self.grid_step], dtype=np.float64)
            local_v = self.unit_vertices + offset
            world_v = (local_v @ rot_mat.T) * self.scale
            pts2d = self.project_points(world_v, screen_w, screen_h)
            avg_z = float(np.mean(world_v[:, 2]))
            
            faces = []
            for face_idx, normal in self.cube_faces:
                world_n = (normal @ rot_mat.T)
                face_pts = [pts2d[idx] for idx in face_idx]
                face_avg_z = float(np.mean([world_v[idx, 2] for idx in face_idx]))
                faces.append({
                    "pts": face_pts,
                    "normal": world_n,
                    "avg_z": face_avg_z,
                })
            
            stroke_items.append({
                "pts2d": pts2d,
                "edges": self.cube_edges,
                "faces": faces,
                "avg_z": avg_z,
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
        }




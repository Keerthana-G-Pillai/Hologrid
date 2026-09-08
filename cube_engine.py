"""
cube_engine.py
--------------
Multi-block 3D Voxel / Hologram Engine.
- Supports placing multiple blocks in 3D grid space (X, Y, Z).
- Crisp, smaller modular block size so creations don't overwhelm the viewport.
- Real-time 360-degree rotation of the entire 3D scene (Pitch, Yaw, Roll).
- Visual cursor block indicating where the next block will be placed.
"""

import numpy as np


class CubeEngine:
    def __init__(self, block_size=0.35, grid_step=0.35):
        self.block_size = block_size
        self.grid_step = grid_step

        # Template unit cube centered at origin
        s = block_size / 2.0
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

        self.cube_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # back face
            (4, 5), (5, 6), (6, 7), (7, 4),  # front face
            (0, 4), (1, 5), (2, 6), (3, 7),  # connecting edges
        ]

        # Collection of blocks: set of (gx, gy, gz) grid coordinate tuples
        self.blocks = set()
        self.history = []  # for undo

        # Add initial starter block at center origin
        self.add_block(0, 0, 0)

        # Active hand cursor position in grid coordinates
        self.cursor = (0, 0, 0)

        # Scene camera / orientation transform (zoomed out for wider field of view)
        self.rot_x = 20.0
        self.rot_y = -35.0
        self.rot_z = 0.0
        self.scale = 0.80
        self.camera_dist = 6.0

    def add_block(self, gx: int, gy: int, gz: int):
        pos = (int(gx), int(gy), int(gz))
        if pos not in self.blocks:
            self.blocks.add(pos)
            self.history.append(pos)

    def remove_block(self, gx: int, gy: int, gz: int):
        pos = (int(gx), int(gy), int(gz))
        self.blocks.discard(pos)

    def undo(self):
        if self.history:
            last = self.history.pop()
            self.blocks.discard(last)

    def clear(self):
        self.blocks.clear()
        self.history.clear()
        # Keep center anchor block
        self.add_block(0, 0, 0)

    def zoom_in(self):
        self.scale = min(3.0, self.scale * 1.2)

    def zoom_out(self):
        self.scale = max(0.25, self.scale * 0.82)

    def set_preset_view(self, preset):
        preset = preset.lower()
        if preset == "iso":
            self.rot_x, self.rot_y, self.rot_z = 20.0, -35.0, 0.0
        elif preset == "front":
            self.rot_x, self.rot_y, self.rot_z = 0.0, 0.0, 0.0
        elif preset == "top":
            self.rot_x, self.rot_y, self.rot_z = 85.0, 0.0, 0.0
        elif preset == "side":
            self.rot_x, self.rot_y, self.rot_z = 0.0, 90.0, 0.0

    def rotate_nudge(self, pitch_delta=0.0, yaw_delta=0.0):
        self.rot_x += pitch_delta
        self.rot_y += yaw_delta

    def set_cursor(self, gx: int, gy: int, gz: int):
        self.cursor = (int(gx), int(gy), int(gz))

    def set_transform(self, rot_x_deg, rot_y_deg, rot_z_deg, scale=None):
        self.rot_x = rot_x_deg
        self.rot_y = rot_y_deg
        self.rot_z = rot_z_deg
        if scale is not None:
            self.scale = max(0.2, min(3.0, scale))

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
        """
        Calculates projected edges and points for:
        1. All placed blocks
        2. The cursor block
        Returns:
            dict containing:
              - 'blocks': list of (pts2d, edges)
              - 'cursor': (pts2d, edges)
              - 'count': total number of blocks
        """
        rot_mat = self.get_rotation_matrix()

        # Render placed blocks
        block_items = []
        for (gx, gy, gz) in self.blocks:
            offset = np.array([gx * self.grid_step,
                               gy * self.grid_step,
                               gz * self.grid_step], dtype=np.float64)
            # Transform vertices: rotate relative to origin, then scale
            local_v = self.unit_vertices + offset
            world_v = (local_v @ rot_mat.T) * self.scale
            pts2d = self.project_points(world_v, screen_w, screen_h)
            block_items.append((pts2d, self.cube_edges))

        # Render cursor block (slightly larger pulsating wireframe)
        cgx, cgy, cgz = self.cursor
        cursor_offset = np.array([cgx * self.grid_step,
                                  cgy * self.grid_step,
                                  cgz * self.grid_step], dtype=np.float64)
        cursor_v = (self.unit_vertices * 1.08) + cursor_offset
        cursor_world = (cursor_v @ rot_mat.T) * self.scale
        cursor_pts2d = self.project_points(cursor_world, screen_w, screen_h)

        return {
            "blocks": block_items,
            "cursor": (cursor_pts2d, self.cube_edges),
            "count": len(self.blocks),
            "cursor_pos": (cgx, cgy, cgz),
        }



"""
3D tetrahedron mask renderer that tracks jaw and head movement.
Renders a single tetrahedron projecting from the face like a crocodile's jaw.
Uses accumulated rotation tracking like the JavaScript prism implementation.
"""

import time

import cv2
import numpy as np


class TetrahedronRenderer:
    """Render 3D tetrahedron mask that projects from face and follows head movement."""

    def __init__(self, color=(0, 255, 136), opacity=0.6, smoothing=0.7, y_offset=-0.4):
        """
        Initialize tetrahedron renderer.

        Args:
            color: BGR color tuple (default: cyan-green)
            opacity: Transparency level 0-1 (default: 0.6)
            smoothing: Smoothing factor 0-1, higher = smoother but laggier (default: 0.7)
            y_offset: Vertical offset as fraction of jaw_width, negative moves up (default: -0.4)
        """
        self.color = color
        self.opacity = opacity
        self.smoothing = smoothing
        self.y_offset = y_offset

        # Current rotation angles (set from pose estimation)
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.rot_z = 0.0

        # Neutral pose offsets (calibrated to user's natural resting position)
        self.neutral_pitch = None
        self.neutral_yaw = None
        self.neutral_roll = None
        self.calibration_frames = 0
        self.calibration_samples = []

        # Smoothed rotation matrix for temporal filtering
        self.smoothed_rotation = None

        # Frame timing for smooth transitions
        self.last_time = time.time()

    def render(self, frame, jaw_points, jaw_width, pose_data):
        """
        Render tetrahedron mask on frame.

        Args:
            frame: Input BGR image
            jaw_points: Dictionary with 'chin', 'left_jaw', 'right_jaw' positions
            jaw_width: Width of jaw in pixels
            pose_data: Tuple of (rotation_matrix, (yaw, pitch, roll)) from pose estimation

        Returns:
            numpy.ndarray: Frame with tetrahedron mask rendered
        """
        if jaw_points is None or pose_data is None:
            return frame

        rotation_matrix, (yaw, pitch, roll) = pose_data

        # Calibrate neutral rotation matrix during first 30 frames
        if self.neutral_pitch is None:
            self.calibration_samples.append(rotation_matrix.copy())
            self.calibration_frames += 1

            if self.calibration_frames >= 30:
                # Calculate average neutral rotation matrix
                neutral_rmat = np.mean(self.calibration_samples, axis=0)
                # Store it for relative rotation
                self.neutral_rotation_matrix = neutral_rmat
                # Extract Euler angles for display
                sy = np.sqrt(neutral_rmat[0, 0] ** 2 + neutral_rmat[1, 0] ** 2)
                self.neutral_pitch = np.degrees(np.arctan2(-neutral_rmat[2, 0], sy))
                self.neutral_yaw = np.degrees(
                    np.arctan2(neutral_rmat[1, 0], neutral_rmat[0, 0])
                )
                self.neutral_roll = np.degrees(
                    np.arctan2(neutral_rmat[2, 1], neutral_rmat[2, 2])
                )
                print(
                    f"Calibrated neutral pose: pitch={self.neutral_pitch:.1f}°, yaw={self.neutral_yaw:.1f}°, roll={self.neutral_roll:.1f}°"
                )
            else:
                # Still calibrating - return frame unchanged
                return frame

        # Calculate relative rotation: R_relative = R_neutral^T @ R_current
        # This gives rotation from neutral pose to current pose
        relative_rotation = self.neutral_rotation_matrix.T @ rotation_matrix

        # Invert the rotation (transpose) to fix left/right inversion
        relative_rotation = relative_rotation.T

        # Apply temporal smoothing to reduce jitter
        if self.smoothed_rotation is None:
            # First frame - initialize
            self.smoothed_rotation = relative_rotation.copy()
        else:
            # Exponential moving average (EMA) on rotation matrix
            # Higher smoothing = more lag but smoother motion
            alpha = (
                1.0 - self.smoothing
            )  # Convert smoothing to alpha (0=smooth, 1=responsive)

            # Interpolate rotation matrix
            interpolated = (
                self.smoothing * self.smoothed_rotation + alpha * relative_rotation
            )

            # Re-orthogonalize to ensure it's a valid rotation matrix
            # Using SVD decomposition: A = U * S * V^T, then orthogonal = U * V^T
            U, _, Vt = np.linalg.svd(interpolated)
            self.smoothed_rotation = U @ Vt

            # Ensure determinant is +1 (proper rotation, not reflection)
            if np.linalg.det(self.smoothed_rotation) < 0:
                U[:, -1] *= -1
                self.smoothed_rotation = U @ Vt

        # Use smoothed rotation for rendering
        relative_rotation = self.smoothed_rotation

        # Extract chin for anchor point
        chin = np.array(jaw_points["chin"][:2], dtype=np.float64)

        # Apply vertical offset to shift prism up/down (negative = up, positive = down)
        vertical_shift = jaw_width * self.y_offset
        chin[1] += vertical_shift  # Adjust Y coordinate

        # Calculate tetrahedron size based on jaw width
        base_size = jaw_width * 2.5  # Large base
        apex_distance = jaw_width * 1.5  # Forward projection distance

        # Define tetrahedron vertices in local coordinates
        # Default orientation: apex points forward (positive Z), inverted triangle (point down)
        vertices_local = np.array(
            [
                # Apex (forward-pointing vertex, like a bill)
                [0, 0, apex_distance],
                # Base triangle (large, wrapping around head/jaw/neck) - INVERTED
                [-base_size * 0.6, base_size * 0.3, -base_size * 0.8],  # Left-top-back
                [base_size * 0.6, base_size * 0.3, -base_size * 0.8],  # Right-top-back
                [0, -base_size * 0.7, -base_size * 0.8],  # Bottom-back (below chin)
            ],
            dtype=np.float64,
        )

        # Apply rotation matrix directly to vertices
        # This is mathematically clean - no Euler angle ambiguity
        vertices_rotated = vertices_local @ relative_rotation.T

        # Use chin as anchor point
        anchor_x, anchor_y = int(chin[0]), int(chin[1])

        # Project to 2D screen coordinates
        vertices_2d = self._project_to_2d(
            vertices_rotated, anchor_x, anchor_y, frame.shape
        )

        # Draw solid black fill directly on frame (fully opaque, no transparency)
        self._draw_filled_faces(frame, vertices_2d, vertices_rotated)

        # Create overlay for transparent wireframe only
        overlay = frame.copy()

        # Draw tetrahedron wireframe on overlay
        self._draw_wireframe(overlay, vertices_2d)

        # Blend only the wireframe with transparency
        frame = cv2.addWeighted(overlay, self.opacity, frame, 1 - self.opacity, 0)

        return frame

    def _draw_filled_faces(self, img, vertices_2d, vertices_3d):
        """
        Draw filled black faces of the tetrahedron.

        Args:
            img: Image to draw on
            vertices_2d: (4, 2) array of 2D vertex positions
            vertices_3d: (4, 3) array of 3D vertex positions for depth sorting
        """
        # Define the 4 triangular faces of the tetrahedron
        # vertices: [0]=apex, [1]=left-top-back, [2]=right-top-back, [3]=bottom-back
        faces = [
            (0, 1, 2),  # Front-left face
            (0, 2, 3),  # Front-right face
            (0, 3, 1),  # Front-bottom face
            (1, 2, 3),  # Back base triangle
        ]

        # Sort faces by average Z depth (back to front for proper occlusion)
        face_depths = []
        for face in faces:
            avg_z = np.mean([vertices_3d[i][2] for i in face])
            face_depths.append((avg_z, face))

        # Sort by depth (furthest first)
        face_depths.sort(key=lambda x: x[0])

        # Draw each face
        for _, face in face_depths:
            # Get 2D points for this face
            points = np.array([vertices_2d[i] for i in face], dtype=np.int32)

            # Back-face culling: only draw if face is visible (counter-clockwise in screen space)
            # Calculate cross product of two edges
            edge1 = points[1] - points[0]
            edge2 = points[2] - points[0]
            cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]

            # If counter-clockwise (cross > 0), face is visible
            if cross > 0:
                # Draw filled triangle with solid black
                cv2.fillPoly(img, [points], (0, 0, 0), lineType=cv2.LINE_AA)

    def _project_to_2d(self, vertices_3d, center_x, center_y, frame_shape):
        """
        Project 3D vertices to 2D screen coordinates with perspective.

        Args:
            vertices_3d: (N, 3) array of 3D coordinates
            center_x: X coordinate of anchor point (chin)
            center_y: Y coordinate of anchor point (chin)
            frame_shape: (height, width, channels) of frame

        Returns:
            numpy.ndarray: (N, 2) array of 2D pixel coordinates
        """
        h, w = frame_shape[:2]
        vertices_2d = np.zeros((len(vertices_3d), 2), dtype=np.int32)

        for i, (x, y, z) in enumerate(vertices_3d):
            # Perspective projection (like JavaScript)
            # focal_length controls perspective strength
            focal_length = w * 0.8
            perspective = 2.0 / (2.0 - z / focal_length * 0.3)

            # Project to 2D with perspective
            px = int(center_x + x * perspective)
            py = int(center_y - y * perspective * 0.8)  # Y inverted, slight compression

            # Clamp to frame bounds
            px = np.clip(px, 0, w - 1)
            py = np.clip(py, 0, h - 1)

            vertices_2d[i] = [px, py]

        return vertices_2d

    def _draw_wireframe(self, img, vertices_2d):
        """
        Draw tetrahedron wireframe with glow effect.

        Args:
            img: Image to draw on
            vertices_2d: (4, 2) array of 2D vertex positions
                         [0] = apex, [1] = left-bottom, [2] = right-bottom, [3] = top
        """
        # Define edges connecting vertices (6 edges for tetrahedron)
        edges = [
            (0, 1),  # Apex to left-bottom
            (0, 2),  # Apex to right-bottom
            (0, 3),  # Apex to top
            (1, 2),  # Left-bottom to right-bottom (base edge)
            (2, 3),  # Right-bottom to top (base edge)
            (3, 1),  # Top to left-bottom (base edge)
        ]

        # Draw edges with glow effect
        for edge in edges:
            pt1 = tuple(vertices_2d[edge[0]])
            pt2 = tuple(vertices_2d[edge[1]])

            # Multi-pass glow effect
            for thickness in [10, 7, 5, 3]:
                alpha = 0.3 * (11 - thickness) / 10
                color_adjusted = tuple([int(c * alpha * 1.8) for c in self.color])
                cv2.line(img, pt1, pt2, color_adjusted, thickness, cv2.LINE_AA)

            # Final bright line
            cv2.line(img, pt1, pt2, self.color, 2, cv2.LINE_AA)

        # Draw vertices as glowing points
        for i, vertex in enumerate(vertices_2d):
            vertex_tuple = tuple(vertex)

            # Apex is brighter and larger
            if i == 0:
                # Glow layers for apex
                cv2.circle(
                    img,
                    vertex_tuple,
                    12,
                    tuple([int(c * 0.3) for c in self.color]),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    img,
                    vertex_tuple,
                    8,
                    tuple([int(c * 0.6) for c in self.color]),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.circle(img, vertex_tuple, 5, self.color, -1, cv2.LINE_AA)
                cv2.circle(img, vertex_tuple, 3, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                # Base vertices
                cv2.circle(
                    img,
                    vertex_tuple,
                    6,
                    tuple([int(c * 0.5) for c in self.color]),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.circle(img, vertex_tuple, 3, self.color, -1, cv2.LINE_AA)

    def draw_debug_info(self, frame, jaw_points, jaw_width, pose_data, fps):
        """
        Draw debug information on frame.

        Args:
            frame: Input frame
            jaw_points: Jaw landmark points
            jaw_width: Jaw width in pixels
            pose_data: Tuple of (rotation_matrix, (yaw, pitch, roll))
            fps: Current FPS

        Returns:
            numpy.ndarray: Frame with debug info
        """
        if jaw_points is None or pose_data is None:
            cv2.putText(
                frame,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            return frame

        _, (yaw, pitch, roll) = pose_data

        # Calculate relative angles for display
        if self.neutral_pitch is not None:
            rel_pitch = pitch - self.neutral_pitch
            rel_yaw = yaw - self.neutral_yaw
            rel_roll = roll - self.neutral_roll
        else:
            rel_pitch = pitch
            rel_yaw = yaw
            rel_roll = roll

        # Draw text info including pose angles
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Jaw Width: {jaw_width:.0f}px",
            f"Pitch: {rel_pitch:.1f}° (raw: {pitch:.1f}°)",
            f"Yaw: {rel_yaw:.1f}° (raw: {yaw:.1f}°)",
            f"Roll: {rel_roll:.1f}° (raw: {roll:.1f}°)",
        ]

        y_offset = 30
        for line in info_lines:
            cv2.putText(
                frame,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            y_offset += 25

        # Draw jaw landmarks
        chin_x, chin_y, _ = jaw_points["chin"]
        left_x, left_y, _ = jaw_points["left_jaw"]
        right_x, right_y, _ = jaw_points["right_jaw"]

        cv2.circle(frame, (chin_x, chin_y), 5, (0, 255, 255), -1)
        cv2.circle(frame, (left_x, left_y), 5, (255, 0, 255), -1)
        cv2.circle(frame, (right_x, right_y), 5, (255, 0, 255), -1)

        return frame

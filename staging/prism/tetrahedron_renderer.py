"""
3D tetrahedron mask renderer that tracks jaw and head movement.
Renders a single tetrahedron projecting from the face like a crocodile's jaw.
Uses accumulated rotation tracking like the JavaScript prism implementation.
"""

import cv2
import numpy as np
import time


class TetrahedronRenderer:
    """Render 3D tetrahedron mask that projects from face and follows head movement."""

    def __init__(self, color=(0, 255, 136), opacity=0.6, smoothing=0.7):
        """
        Initialize tetrahedron renderer.

        Args:
            color: BGR color tuple (default: cyan-green)
            opacity: Transparency level 0-1 (default: 0.6)
            smoothing: Smoothing factor 0-1, higher = smoother but laggier (default: 0.7)
        """
        self.color = color
        self.opacity = opacity
        self.smoothing = smoothing

        # Accumulated rotation angles (like globalRotX, globalRotY, globalRotZ in JS)
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.rot_z = 0.0

        # Rotation velocities
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0

        # Previous jaw orientation for tracking changes
        self.prev_jaw_center = None
        self.prev_jaw_right_vec = None
        self.prev_jaw_up_vec = None

        # Frame timing for velocity calculations
        self.last_time = time.time()

    def render(self, frame, jaw_points, jaw_width, pose_angles):
        """
        Render tetrahedron mask on frame.

        Args:
            frame: Input BGR image
            jaw_points: Dictionary with 'chin', 'left_jaw', 'right_jaw' positions
            jaw_width: Width of jaw in pixels
            pose_angles: Tuple of (yaw, pitch, roll) in degrees (used for reference)

        Returns:
            numpy.ndarray: Frame with tetrahedron mask rendered
        """
        if jaw_points is None:
            return frame

        # Extract jaw landmarks
        chin = np.array(jaw_points['chin'][:2], dtype=np.float64)
        left_jaw = np.array(jaw_points['left_jaw'][:2], dtype=np.float64)
        right_jaw = np.array(jaw_points['right_jaw'][:2], dtype=np.float64)

        # Calculate current jaw orientation
        jaw_center = (left_jaw + right_jaw) / 2.0
        jaw_right_vec = right_jaw - left_jaw  # Left to right vector
        jaw_up_vec = jaw_center - chin  # Chin to jaw center vector

        # Normalize
        jaw_right_vec = jaw_right_vec / (np.linalg.norm(jaw_right_vec) + 1e-6)
        jaw_up_vec = jaw_up_vec / (np.linalg.norm(jaw_up_vec) + 1e-6)

        # Calculate time delta for velocity
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        # Detect rotation from jaw orientation changes
        if self.prev_jaw_center is not None and dt > 0:
            # Calculate angular velocities from frame-to-frame changes
            self._update_rotation_from_jaw_changes(
                jaw_center, jaw_right_vec, jaw_up_vec, dt
            )

        # Store current orientation for next frame
        self.prev_jaw_center = jaw_center.copy()
        self.prev_jaw_right_vec = jaw_right_vec.copy()
        self.prev_jaw_up_vec = jaw_up_vec.copy()

        # Apply rotation velocities to accumulated angles
        self.rot_x += self.vel_x
        self.rot_y += self.vel_y
        self.rot_z += self.vel_z

        # Calculate tetrahedron size based on jaw width
        base_size = jaw_width * 2.5  # Large base
        apex_distance = jaw_width * 1.5  # Forward projection distance

        # Define tetrahedron vertices in local coordinates
        vertices_local = np.array([
            # Apex (forward-pointing vertex, like a bill)
            [0, 0, apex_distance],

            # Base triangle (large, wrapping around head/jaw/neck)
            [-base_size * 0.6, -base_size * 0.3, -base_size * 0.8],  # Left-bottom-back
            [base_size * 0.6, -base_size * 0.3, -base_size * 0.8],   # Right-bottom-back
            [0, base_size * 0.7, -base_size * 0.8]                    # Top-back (above head)
        ], dtype=np.float64)

        # Apply 3D rotations (like JavaScript rotate3D)
        vertices_rotated = self._rotate_vertices_3d(vertices_local)

        # Project to 2D screen coordinates
        chin_x, chin_y = int(chin[0]), int(chin[1])
        vertices_2d = self._project_to_2d(vertices_rotated, chin_x, chin_y, frame.shape)

        # Create overlay for transparency
        overlay = frame.copy()

        # Draw tetrahedron wireframe
        self._draw_wireframe(overlay, vertices_2d)

        # Blend overlay with original frame
        frame = cv2.addWeighted(overlay, self.opacity, frame, 1 - self.opacity, 0)

        return frame

    def _update_rotation_from_jaw_changes(self, jaw_center, jaw_right_vec, jaw_up_vec, dt):
        """
        Calculate rotation velocities from jaw orientation changes.

        Args:
            jaw_center: Current jaw center position
            jaw_right_vec: Current right vector (normalized)
            jaw_up_vec: Current up vector (normalized)
            dt: Time delta in seconds
        """
        # Position change (translational movement)
        center_delta = jaw_center - self.prev_jaw_center

        # Detect YAW (left-right turn) from right vector rotation in XY plane
        # When face turns left, right vector rotates counterclockwise
        prev_angle_right = np.arctan2(self.prev_jaw_right_vec[1], self.prev_jaw_right_vec[0])
        curr_angle_right = np.arctan2(jaw_right_vec[1], jaw_right_vec[0])
        yaw_change = curr_angle_right - prev_angle_right

        # Normalize angle to [-pi, pi]
        yaw_change = np.arctan2(np.sin(yaw_change), np.cos(yaw_change))

        # Detect PITCH (up-down tilt) from up vector length and direction changes
        # When face tilts up, jaw center moves up relative to chin
        prev_angle_up = np.arctan2(self.prev_jaw_up_vec[1], self.prev_jaw_up_vec[0])
        curr_angle_up = np.arctan2(jaw_up_vec[1], jaw_up_vec[0])
        pitch_change = curr_angle_up - prev_angle_up

        # Normalize angle
        pitch_change = np.arctan2(np.sin(pitch_change), np.cos(pitch_change))

        # Detect ROLL (head tilt) from right vector tilt
        # Cross product magnitude indicates roll
        cross = np.cross(self.prev_jaw_right_vec, jaw_right_vec)
        roll_change = cross * 0.5  # Scale down

        # Convert changes to velocities
        # Scale factors tuned for responsive but smooth tracking
        yaw_velocity = -yaw_change * 0.5  # Inverted and scaled
        pitch_velocity = pitch_change * 0.5
        roll_velocity = roll_change * 0.3

        # Apply smoothing to velocities
        smooth = self.smoothing
        self.vel_x = smooth * self.vel_x + (1 - smooth) * pitch_velocity
        self.vel_y = smooth * self.vel_y + (1 - smooth) * yaw_velocity
        self.vel_z = smooth * self.vel_z + (1 - smooth) * roll_velocity

        # Damping to prevent runaway rotation
        self.vel_x *= 0.95
        self.vel_y *= 0.95
        self.vel_z *= 0.95

    def _rotate_vertices_3d(self, vertices):
        """
        Apply 3D rotations to vertices (like JavaScript rotate3D function).

        Args:
            vertices: (N, 3) array of vertex coordinates

        Returns:
            numpy.ndarray: Rotated vertices
        """
        rotated = vertices.copy()

        for i in range(len(vertices)):
            x, y, z = vertices[i]

            # Rotate around X axis (pitch)
            cos_x = np.cos(self.rot_x)
            sin_x = np.sin(self.rot_x)
            y1 = y * cos_x - z * sin_x
            z1 = y * sin_x + z * cos_x
            y = y1
            z = z1

            # Rotate around Y axis (yaw)
            cos_y = np.cos(self.rot_y)
            sin_y = np.sin(self.rot_y)
            x1 = x * cos_y + z * sin_y
            z2 = -x * sin_y + z * cos_y
            x = x1
            z = z2

            # Rotate around Z axis (roll)
            cos_z = np.cos(self.rot_z)
            sin_z = np.sin(self.rot_z)
            x2 = x * cos_z - y * sin_z
            y2 = x * sin_z + y * cos_z

            rotated[i] = [x2, y2, z]

        return rotated

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
                cv2.circle(img, vertex_tuple, 12,
                          tuple([int(c * 0.3) for c in self.color]), -1, cv2.LINE_AA)
                cv2.circle(img, vertex_tuple, 8,
                          tuple([int(c * 0.6) for c in self.color]), -1, cv2.LINE_AA)
                cv2.circle(img, vertex_tuple, 5, self.color, -1, cv2.LINE_AA)
                cv2.circle(img, vertex_tuple, 3, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                # Base vertices
                cv2.circle(img, vertex_tuple, 6,
                          tuple([int(c * 0.5) for c in self.color]), -1, cv2.LINE_AA)
                cv2.circle(img, vertex_tuple, 3, self.color, -1, cv2.LINE_AA)

    def draw_debug_info(self, frame, jaw_points, jaw_width, pose_angles, fps):
        """
        Draw debug information on frame.

        Args:
            frame: Input frame
            jaw_points: Jaw landmark points
            jaw_width: Jaw width in pixels
            pose_angles: (yaw, pitch, roll) tuple
            fps: Current FPS

        Returns:
            numpy.ndarray: Frame with debug info
        """
        if jaw_points is None:
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame

        yaw, pitch, roll = pose_angles if pose_angles else (0, 0, 0)

        # Draw text info including rotation angles
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Jaw Width: {jaw_width:.0f}px",
            f"RotX: {np.degrees(self.rot_x):.1f}° VelX: {self.vel_x:.3f}",
            f"RotY: {np.degrees(self.rot_y):.1f}° VelY: {self.vel_y:.3f}",
            f"RotZ: {np.degrees(self.rot_z):.1f}° VelZ: {self.vel_z:.3f}"
        ]

        y_offset = 30
        for line in info_lines:
            cv2.putText(frame, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 25

        # Draw jaw landmarks
        chin_x, chin_y, _ = jaw_points['chin']
        left_x, left_y, _ = jaw_points['left_jaw']
        right_x, right_y, _ = jaw_points['right_jaw']

        cv2.circle(frame, (chin_x, chin_y), 5, (0, 255, 255), -1)
        cv2.circle(frame, (left_x, left_y), 5, (255, 0, 255), -1)
        cv2.circle(frame, (right_x, right_y), 5, (255, 0, 255), -1)

        return frame

"""
3D sphere mask renderer that tracks jaw movement.
Renders semi-transparent sphere with glow effect and rotation.
"""

import cv2
import numpy as np
import time


class SphereRenderer:
    """Render 3D sphere mask that follows jaw movement."""

    def __init__(self, color=(0, 255, 136), opacity=0.6):
        """
        Initialize sphere renderer.

        Args:
            color: BGR color tuple (default: cyan-green)
            opacity: Transparency level 0-1 (default: 0.6)
        """
        self.color = color
        self.opacity = opacity
        self.start_time = time.time()

    def render(self, frame, jaw_points, jaw_width, pose_angles):
        """
        Render sphere mask on frame.

        Args:
            frame: Input BGR image
            jaw_points: Dictionary with 'chin', 'left_jaw', 'right_jaw' positions
            jaw_width: Width of jaw in pixels
            pose_angles: Tuple of (yaw, pitch, roll) in degrees

        Returns:
            numpy.ndarray: Frame with sphere mask rendered
        """
        if jaw_points is None or pose_angles is None:
            return frame

        # Extract chin position
        chin_x, chin_y, chin_z = jaw_points['chin']
        yaw, pitch, roll = pose_angles

        # Calculate sphere size based on jaw width
        base_radius = int(jaw_width * 0.4)

        # Add pulsing effect
        pulse = self._get_pulse_scale()
        radius = int(base_radius * pulse)

        # Create overlay for transparency
        overlay = frame.copy()

        # Draw main sphere with gradient effect
        self._draw_gradient_sphere(overlay, chin_x, chin_y, radius, yaw)

        # Draw wireframe for 3D effect
        self._draw_wireframe_sphere(overlay, chin_x, chin_y, radius, yaw, pitch, roll)

        # Draw glowing tendrils
        self._draw_tendrils(overlay, chin_x, chin_y, radius, yaw)

        # Blend overlay with original frame
        frame = cv2.addWeighted(overlay, self.opacity, frame, 1 - self.opacity, 0)

        return frame

    def _draw_gradient_sphere(self, img, cx, cy, radius, yaw):
        """Draw sphere with radial gradient."""
        # Create gradient effect with multiple circles
        for i in range(5, 0, -1):
            r = int(radius * i / 5)
            alpha = 0.3 * (6 - i) / 5
            color_adjusted = tuple([int(c * alpha) for c in self.color])

            cv2.circle(img, (cx, cy), r, color_adjusted, -1, cv2.LINE_AA)

    def _draw_wireframe_sphere(self, img, cx, cy, radius, yaw, pitch, roll):
        """Draw 3D wireframe sphere with rotation."""
        # Number of latitude and longitude lines
        n_lat = 6
        n_lon = 8

        # Convert angles to radians
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        roll_rad = np.radians(roll)

        # Draw latitude circles
        for i in range(1, n_lat):
            lat_angle = np.pi * i / n_lat - np.pi / 2
            circle_radius = radius * np.cos(lat_angle)
            circle_y_offset = radius * np.sin(lat_angle)

            # Apply pitch rotation to y offset
            rotated_y = circle_y_offset * np.cos(pitch_rad)

            if circle_radius > 0:
                # Draw ellipse for perspective
                axes = (int(circle_radius), int(circle_radius * 0.7))
                cv2.ellipse(img, (cx, int(cy + rotated_y)), axes,
                           angle=np.degrees(yaw_rad), startAngle=0, endAngle=360,
                           color=self.color, thickness=2, lineType=cv2.LINE_AA)

        # Draw longitude lines (meridians)
        for i in range(n_lon):
            lon_angle = 2 * np.pi * i / n_lon
            points = []

            for j in range(20):
                lat = np.pi * j / 19 - np.pi / 2

                # Sphere coordinates
                x = radius * np.cos(lat) * np.cos(lon_angle)
                y = radius * np.sin(lat)
                z = radius * np.cos(lat) * np.sin(lon_angle)

                # Apply rotations
                # Yaw (Y-axis rotation)
                x_rot = x * np.cos(yaw_rad) - z * np.sin(yaw_rad)
                z_rot = x * np.sin(yaw_rad) + z * np.cos(yaw_rad)

                # Pitch (X-axis rotation)
                y_rot = y * np.cos(pitch_rad) - z_rot * np.sin(pitch_rad)

                # Project to 2D
                px = int(cx + x_rot)
                py = int(cy + y_rot)

                points.append((px, py))

            # Draw polyline for this meridian
            points = np.array(points, dtype=np.int32)
            cv2.polylines(img, [points], False, self.color, 2, cv2.LINE_AA)

    def _draw_tendrils(self, img, cx, cy, radius, yaw):
        """Draw animated glowing tendrils emanating from sphere."""
        n_tendrils = 6
        t = time.time() - self.start_time

        for i in range(n_tendrils):
            # Calculate tendril angle with rotation based on yaw
            angle = (2 * np.pi * i / n_tendrils) + np.radians(yaw * 0.5)

            # Pulsing length
            pulse = 0.8 + 0.2 * np.sin(t * 2 + i * 0.5)
            length = radius * 1.5 * pulse

            # End point
            end_x = int(cx + length * np.cos(angle))
            end_y = int(cy + length * np.sin(angle))

            # Draw with glow effect (multiple lines with decreasing thickness)
            for thickness in [8, 6, 4, 2]:
                alpha = 0.5 * (9 - thickness) / 8
                color_adjusted = tuple([int(c * alpha) for c in self.color])
                cv2.line(img, (cx, cy), (end_x, end_y),
                        color_adjusted, thickness, cv2.LINE_AA)

    def _get_pulse_scale(self):
        """Get pulsing scale factor based on time."""
        t = time.time() - self.start_time
        return 1.0 + 0.1 * np.sin(t * 3)

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
        if jaw_points is None or pose_angles is None:
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame

        yaw, pitch, roll = pose_angles

        # Draw text info
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Jaw Width: {jaw_width:.0f}px",
            f"Yaw: {yaw:.1f}°",
            f"Pitch: {pitch:.1f}°",
            f"Roll: {roll:.1f}°"
        ]

        y_offset = 30
        for line in info_lines:
            cv2.putText(frame, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30

        # Draw jaw landmarks
        chin_x, chin_y, _ = jaw_points['chin']
        left_x, left_y, _ = jaw_points['left_jaw']
        right_x, right_y, _ = jaw_points['right_jaw']

        cv2.circle(frame, (chin_x, chin_y), 5, (0, 255, 255), -1)
        cv2.circle(frame, (left_x, left_y), 5, (255, 0, 255), -1)
        cv2.circle(frame, (right_x, right_y), 5, (255, 0, 255), -1)

        return frame

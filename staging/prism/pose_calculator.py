"""
Head pose estimation using Perspective-n-Point (PnP) algorithm.
Calculates yaw, pitch, and roll angles from facial landmarks.
"""

import cv2
import numpy as np


class PoseCalculator:
    """Calculate head pose (yaw, pitch, roll) from facial landmarks."""

    def __init__(self, image_width=1280, image_height=720):
        """
        Initialize pose calculator with camera calibration.

        Args:
            image_width: Frame width in pixels
            image_height: Frame height in pixels
        """
        self.image_width = image_width
        self.image_height = image_height

        # Camera matrix (simplified calibration)
        focal_length = image_width
        self.camera_matrix = np.array([
            [focal_length, 0, image_width / 2],
            [0, focal_length, image_height / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        # Distortion coefficients (assume no distortion)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    def calculate_pose(self, points_2d, points_3d):
        """
        Calculate head pose from 2D and 3D landmark points.

        Args:
            points_2d: (N, 2) array of 2D pixel coordinates
            points_3d: (N, 3) array of 3D coordinates

        Returns:
            tuple: (rotation_matrix, euler_angles_for_debug)
                   rotation_matrix: 3x3 numpy array
                   euler_angles_for_debug: (yaw, pitch, roll) in degrees for display
        """
        if points_2d is None or points_3d is None:
            return None, (None, None, None)

        # Solve PnP to get rotation and translation vectors
        success, rot_vec, trans_vec = cv2.solvePnP(
            points_3d,
            points_2d,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None, (None, None, None)

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rot_vec)

        # Extract Euler angles for debug display only
        sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(-rmat[2, 0], sy)
            yaw = np.arctan2(rmat[1, 0], rmat[0, 0])
            roll = np.arctan2(rmat[2, 1], rmat[2, 2])
        else:
            pitch = np.arctan2(-rmat[2, 0], sy)
            yaw = 0
            roll = np.arctan2(-rmat[1, 2], rmat[1, 1])

        # Convert to degrees for debug
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)
        roll_deg = np.degrees(roll)

        return rmat, (yaw_deg, pitch_deg, roll_deg)

    def get_rotation_matrix(self, yaw, pitch, roll):
        """
        Get rotation matrix from euler angles.

        Args:
            yaw: Rotation around Y axis (degrees)
            pitch: Rotation around X axis (degrees)
            roll: Rotation around Z axis (degrees)

        Returns:
            numpy.ndarray: 3x3 rotation matrix
        """
        # Convert to radians
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        roll_rad = np.radians(roll)

        # Rotation matrix around Y (yaw)
        R_y = np.array([
            [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
            [0, 1, 0],
            [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
        ])

        # Rotation matrix around X (pitch)
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ])

        # Rotation matrix around Z (roll)
        R_z = np.array([
            [np.cos(roll_rad), -np.sin(roll_rad), 0],
            [np.sin(roll_rad), np.cos(roll_rad), 0],
            [0, 0, 1]
        ])

        # Combined rotation matrix
        R = R_z @ R_y @ R_x
        return R

    def normalize_angle(self, angle):
        """
        Normalize angle to [-180, 180] range.

        Args:
            angle: Angle in degrees

        Returns:
            float: Normalized angle
        """
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

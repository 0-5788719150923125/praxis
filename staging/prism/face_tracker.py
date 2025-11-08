"""
OpenCV YuNet face detector for detecting faces and 5 key facial landmarks.
Fast and efficient, works with Python 3.13.
"""

import os
import urllib.request

import cv2
import numpy as np


class FaceTracker:
    """OpenCV YuNet-based face tracker with 5 facial landmarks."""

    # YuNet model URL (from OpenCV model zoo)
    MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    MODEL_NAME = "face_detection_yunet_2023mar.onnx"

    # Landmark indices from YuNet output
    # YuNet returns: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm]
    # Where: re=right eye, le=left eye, nt=nose tip, rcm=right corner mouth, lcm=left corner mouth
    RIGHT_EYE_IDX = (4, 5)
    LEFT_EYE_IDX = (6, 7)
    NOSE_TIP_IDX = (8, 9)
    RIGHT_MOUTH_IDX = (10, 11)
    LEFT_MOUTH_IDX = (12, 13)

    def __init__(self, model_path=None, score_threshold=0.6, nms_threshold=0.3):
        """
        Initialize YuNet face detector.

        Args:
            model_path: Path to YuNet model file (downloads if not exists)
            score_threshold: Detection confidence threshold (0-1)
            nms_threshold: Non-maximum suppression threshold
        """
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        # Download model if needed
        if model_path is None:
            model_path = self._get_model_path()

        if not os.path.exists(model_path):
            print(f"Downloading YuNet model to {model_path}...")
            self._download_model(model_path)
            print("Download complete.")

        # Initialize detector
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),  # Will be updated with actual frame size
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
        )

    def _get_model_path(self):
        """Get path for model file (in models subdirectory)."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        return os.path.join(models_dir, self.MODEL_NAME)

    def _download_model(self, model_path):
        """Download YuNet model from OpenCV zoo."""
        try:
            urllib.request.urlretrieve(self.MODEL_URL, model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download YuNet model: {e}")

    def detect(self, frame):
        """
        Detect face landmarks in frame.

        Args:
            frame: BGR image from OpenCV

        Returns:
            tuple: (face_data, image_shape) or (None, None) if no face detected
                   face_data: dict with 'bbox', 'landmarks', 'confidence'
                   image_shape: (height, width) of input frame
        """
        h, w = frame.shape[:2]

        # Update input size if needed
        self.detector.setInputSize((w, h))

        # Detect faces
        _, faces = self.detector.detect(frame)

        if faces is None or len(faces) == 0:
            return None, None

        # Get first (most confident) face
        face = faces[0]

        # Parse detection results
        # face format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, confidence]
        bbox = face[0:4].astype(int)  # [x, y, w, h]

        landmarks = {
            "right_eye": (int(face[4]), int(face[5])),
            "left_eye": (int(face[6]), int(face[7])),
            "nose_tip": (int(face[8]), int(face[9])),
            "right_mouth": (int(face[10]), int(face[11])),
            "left_mouth": (int(face[12]), int(face[13])),
        }

        confidence = face[14] if len(face) > 14 else 1.0

        face_data = {"bbox": bbox, "landmarks": landmarks, "confidence": confidence}

        return face_data, (h, w)

    def get_jaw_points(self, face_data, image_shape):
        """
        Estimate jaw landmark positions from face data.

        Args:
            face_data: Face detection data from detect()
            image_shape: (height, width) tuple

        Returns:
            dict: Dictionary with 'chin', 'left_jaw', 'right_jaw' as (x, y, z) tuples
                  z is estimated depth (0 for this detector)
        """
        if face_data is None:
            return None

        landmarks = face_data["landmarks"]
        bbox = face_data["bbox"]

        # Estimate chin position (below mouth, centered)
        mouth_center_x = (landmarks["left_mouth"][0] + landmarks["right_mouth"][0]) // 2
        mouth_center_y = (landmarks["left_mouth"][1] + landmarks["right_mouth"][1]) // 2

        # Chin is approximately 0.6-0.7 of face height below nose
        face_height = bbox[3]
        chin_y = mouth_center_y + int(face_height * 0.25)
        chin_x = mouth_center_x

        # Estimate jaw corners (sides of lower face)
        jaw_width_factor = 0.7  # Jaw is typically 70% of face width
        jaw_width = int(bbox[2] * jaw_width_factor)
        jaw_y = mouth_center_y + int(face_height * 0.15)

        left_jaw_x = mouth_center_x - jaw_width // 2
        right_jaw_x = mouth_center_x + jaw_width // 2

        jaw_points = {
            "chin": (chin_x, chin_y, 0),
            "left_jaw": (left_jaw_x, jaw_y, 0),
            "right_jaw": (right_jaw_x, jaw_y, 0),
        }

        return jaw_points

    def get_pose_points(self, face_data, image_shape):
        """
        Extract key landmarks for pose estimation.

        Args:
            face_data: Face detection data from detect()
            image_shape: (height, width) tuple

        Returns:
            tuple: (points_2d, points_3d) as numpy arrays for cv2.solvePnP
                   points_2d: (N, 2) pixel coordinates (needs at least 6 points)
                   points_3d: (N, 3) 3D model coordinates
        """
        if face_data is None:
            return None, None

        landmarks = face_data["landmarks"]
        bbox = face_data["bbox"]

        # Estimate chin position (6th point needed for solvePnP)
        mouth_center_x = (landmarks["left_mouth"][0] + landmarks["right_mouth"][0]) / 2
        mouth_center_y = (landmarks["left_mouth"][1] + landmarks["right_mouth"][1]) / 2
        face_height = bbox[3]
        chin_x = mouth_center_x
        chin_y = mouth_center_y + face_height * 0.25

        # 2D points from detected landmarks (6 points for solvePnP DLT algorithm)
        points_2d = np.array(
            [
                landmarks["nose_tip"],
                landmarks["left_eye"],
                landmarks["right_eye"],
                landmarks["left_mouth"],
                landmarks["right_mouth"],
                [chin_x, chin_y],  # Estimated chin point
            ],
            dtype=np.float64,
        )

        # 3D model points (generic face model)
        # Coordinates in mm, approximate human face proportions
        points_3d = np.array(
            [
                [0.0, 0.0, 0.0],  # Nose tip
                [-30.0, -30.0, -20.0],  # Left eye
                [30.0, -30.0, -20.0],  # Right eye
                [-20.0, 30.0, -10.0],  # Left mouth corner
                [20.0, 30.0, -10.0],  # Right mouth corner
                [0.0, 50.0, -15.0],  # Chin
            ],
            dtype=np.float64,
        )

        return points_2d, points_3d

    def calculate_jaw_width(self, jaw_points):
        """
        Calculate jaw width in pixels.

        Args:
            jaw_points: Dictionary from get_jaw_points()

        Returns:
            float: Distance between left and right jaw points
        """
        if jaw_points is None:
            return 0

        left = jaw_points["left_jaw"]
        right = jaw_points["right_jaw"]

        # Euclidean distance in 2D
        width = np.sqrt((right[0] - left[0]) ** 2 + (right[1] - left[1]) ** 2)
        return width

    def close(self):
        """Release resources (YuNet doesn't need explicit cleanup)."""
        pass
        pass

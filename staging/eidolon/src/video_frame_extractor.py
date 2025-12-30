"""
Video frame extractor for on-demand frame extraction.

Provides utilities for extracting frames from video files without bulk extraction.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Dict, Any

from utils import get_video_info


class VideoFrameExtractor:
    """Extract frames on-demand from video files."""

    def __init__(self, video_path: str, target_fps: int = 5):
        """
        Initialize video frame extractor.

        Args:
            video_path: Path to video file
            target_fps: Target frame rate for extraction (default: 5 fps)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(
                f"Video file not found: {video_path}\n"
                "Please ensure the video file exists and the path is correct."
            )

        self.video_path = video_path
        self.target_fps = target_fps

        # Get video metadata
        self.video_info = get_video_info(video_path)
        self.source_fps = self.video_info['fps']
        self.frame_interval = int(self.source_fps / target_fps)

    def get_frame_at_timestamp(self, timestamp: float) -> np.ndarray:
        """
        Extract frame at specific timestamp.

        Args:
            timestamp: Timestamp in seconds

        Returns:
            Frame as numpy array (BGR format)

        Raises:
            ValueError: If frame extraction fails
        """
        # Calculate source frame number
        source_frame = int(timestamp * self.source_fps)

        # Open video and seek
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        try:
            # Try frame-based seeking first
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ret, frame = cap.read()

            if not ret:
                # Try alternate seek method (timestamp-based)
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ret, frame = cap.read()

            if not ret:
                raise ValueError(f"Failed to extract frame at {timestamp}s")

            return frame

        except Exception as e:
            raise RuntimeError(f"Video seek error: {e}")

        finally:
            cap.release()

    def get_frame_at_index(self, frame_idx: int) -> np.ndarray:
        """
        Extract frame by logical index (0-based at target_fps).

        Args:
            frame_idx: Frame index at target FPS (0-based)

        Returns:
            Frame as numpy array (BGR format)

        Raises:
            ValueError: If frame extraction fails
        """
        # Calculate source frame number
        source_frame = frame_idx * self.frame_interval
        timestamp = source_frame / self.source_fps

        return self.get_frame_at_timestamp(timestamp)

    def get_total_frames(self) -> int:
        """
        Calculate total frames at target FPS.

        Returns:
            Total number of frames at target FPS
        """
        return int(self.video_info['duration'] * self.target_fps)

    def iter_frames(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        """
        Iterate through all frames at target FPS (sequential reading).

        This method is optimized for sequential access (e.g., labeling tool)
        and is much faster than repeated seeking.

        Yields:
            Tuple of (frame_index, timestamp, frame)
            - frame_index: 0-based index at target FPS
            - timestamp: Timestamp in seconds
            - frame: Frame as numpy array (BGR format)

        Raises:
            RuntimeError: If video cannot be opened
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        frame_count = 0
        extracted_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Only yield frames at our target intervals
                if frame_count % self.frame_interval == 0:
                    timestamp = frame_count / self.source_fps
                    yield extracted_count, timestamp, frame
                    extracted_count += 1

                frame_count += 1

        finally:
            cap.release()

    def extract_batch_at_timestamps(
        self,
        timestamps: list[float]
    ) -> list[Tuple[float, np.ndarray]]:
        """
        Extract multiple frames at specific timestamps (optimized for batch extraction).

        This method opens the video once and extracts all requested frames.
        Timestamps should be sorted for best performance.

        Args:
            timestamps: List of timestamps in seconds (should be sorted)

        Returns:
            List of (timestamp, frame) tuples

        Raises:
            RuntimeError: If video cannot be opened
        """
        if not timestamps:
            return []

        # Sort timestamps to enable sequential reading optimization
        sorted_timestamps = sorted(timestamps)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        results = []

        try:
            for timestamp in sorted_timestamps:
                source_frame = int(timestamp * self.source_fps)

                # Seek and read
                cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
                ret, frame = cap.read()

                if not ret:
                    # Try alternate method
                    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                    ret, frame = cap.read()

                if ret:
                    results.append((timestamp, frame))
                else:
                    print(f"WARNING: Failed to extract frame at {timestamp}s")

        finally:
            cap.release()

        return results

    def get_video_metadata(self) -> Dict[str, Any]:
        """
        Get video metadata.

        Returns:
            Dictionary with video info (width, height, fps, duration, total_frames)
        """
        return self.video_info.copy()

"""
Utility functions shared across the pipeline.
"""

import os
import json
import yaml
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_json(data: Any, filepath: str) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Extract video metadata using ffprobe.

    Returns dict with: width, height, fps, duration, total_frames
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,duration,nb_frames',
        '-of', 'json',
        video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    stream = data['streams'][0]

    # Parse frame rate (format: "60000/1001" or "30/1")
    fps_parts = stream['r_frame_rate'].split('/')
    fps = float(fps_parts[0]) / float(fps_parts[1])

    # Calculate total frames if not provided
    duration = float(stream.get('duration', 0))
    total_frames = int(stream.get('nb_frames', 0))
    if total_frames == 0 and duration > 0:
        total_frames = int(duration * fps)

    return {
        'width': int(stream['width']),
        'height': int(stream['height']),
        'fps': fps,
        'duration': duration,
        'total_frames': total_frames
    }


def seconds_to_frames(seconds: float, fps: float) -> int:
    """Convert timestamp in seconds to frame number."""
    return int(seconds * fps)


def frames_to_seconds(frame: int, fps: float) -> float:
    """Convert frame number to timestamp in seconds."""
    return frame / fps


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def get_absolute_path(path: str) -> str:
    """Convert to absolute path."""
    return str(Path(path).resolve())

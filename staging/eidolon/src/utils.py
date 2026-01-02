"""
Utility functions shared across the pipeline.
"""

import json
import os
import shutil
import subprocess
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_json(data: Any, filepath: str) -> None:
    """Save data to JSON file."""
    dirname = os.path.dirname(filepath)
    if dirname:  # Only create directory if there is a parent directory
        os.makedirs(dirname, exist_ok=True)
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


# === Task Configuration Functions ===

DEFAULT_TASK_CONFIG = {
    "version": "1.0",
    "task_name": "binary_classification",
    "task_description": "Generic binary classification task",
    "labels": {
        "positive": {
            "display_name": "true",
            "description": "Positive class"
        },
        "negative": {
            "display_name": "false",
            "description": "Negative class"
        }
    }
}


def load_task_config(config_path: str = 'task_config.json') -> Dict[str, Any]:
    """
    Load task configuration from JSON file.

    If file doesn't exist, returns default config.

    Args:
        config_path: Path to task_config.json

    Returns:
        Task configuration dictionary
    """
    if os.path.exists(config_path):
        try:
            return load_json(config_path)
        except Exception as e:
            print(f"Warning: Could not load {config_path}: {e}")
            print("Using default task configuration")
            return DEFAULT_TASK_CONFIG.copy()
    return DEFAULT_TASK_CONFIG.copy()


def internal_to_display(label: str, task_config: Dict[str, Any]) -> str:
    """
    Convert internal label ('true'/'false') to display label.

    Args:
        label: Internal label ('true' or 'false')
        task_config: Task configuration dictionary

    Returns:
        Display label (e.g., 'touching', 'not_touching')
    """
    if label == 'true':
        return task_config['labels']['positive']['display_name']
    elif label == 'false':
        return task_config['labels']['negative']['display_name']
    return label


def display_to_internal(label: str, task_config: Dict[str, Any]) -> str:
    """
    Convert display label to internal ('true'/'false').

    Supports legacy labels ('touching'/'not_touching') for backward compatibility.

    Args:
        label: Display label or legacy label
        task_config: Task configuration dictionary

    Returns:
        Internal label ('true' or 'false')
    """
    pos_name = task_config['labels']['positive']['display_name']
    neg_name = task_config['labels']['negative']['display_name']

    # Check current task labels
    if label == pos_name:
        return 'true'
    elif label == neg_name:
        return 'false'

    # Legacy support for old hardcoded labels (backward compatibility)
    if label == 'touching':
        return 'true'
    elif label == 'not_touching':
        return 'false'

    return label


def migrate_labels_to_internal(df: pd.DataFrame, backup: bool = True, backup_path: str = None) -> pd.DataFrame:
    """
    Migrate labels from old formats to internal format ('true'/'false').

    Handles:
    - Legacy hardcoded labels: 'touching'/'not_touching'
    - Python boolean strings: 'True'/'False'
    - Already correct: 'true'/'false'

    Args:
        df: DataFrame with 'label' column
        backup: Whether to create backup (only if file path is known)
        backup_path: Path to backup file (optional)

    Returns:
        DataFrame with migrated labels
    """
    if 'label' not in df.columns:
        return df

    # Convert label column to string type to handle any boolean values
    df['label'] = df['label'].astype(str)

    unique_labels = df['label'].unique()
    needs_migration = any(label in ['touching', 'not_touching', 'True', 'False'] for label in unique_labels)

    if not needs_migration:
        return df

    print("  Migrating labels to task-agnostic format ('true'/'false')...")

    # Create backup if path provided
    if backup and backup_path:
        try:
            shutil.copy(backup_path, f"{backup_path}.pre_migration_backup")
            print(f"    ✓ Created backup: {backup_path}.pre_migration_backup")
        except Exception as e:
            print(f"    Warning: Could not create backup: {e}")

    # Migrate labels
    label_map = {
        'touching': 'true',
        'not_touching': 'false',
        'True': 'true',     # Python boolean string
        'False': 'false',   # Python boolean string
    }

    df['label'] = df['label'].map(lambda x: label_map.get(x, x))

    migrated_count = (df['label'].isin(['true', 'false'])).sum()
    print(f"    ✓ Migrated {migrated_count}/{len(df)} labels")

    return df


def create_task_config_from_labels(labels_csv: str, config_path: str = 'task_config.json') -> Dict[str, Any]:
    """
    Create task_config.json from detected labels in CSV (if missing).

    This supports migration from legacy installations with hardcoded labels.

    Args:
        labels_csv: Path to labels CSV file
        config_path: Path where task_config.json should be created

    Returns:
        Created task configuration dictionary
    """
    if os.path.exists(config_path):
        print(f"  task_config.json already exists")
        return load_task_config(config_path)

    print("  Creating task_config.json from detected labels...")

    # Read labels to detect old format
    if not os.path.exists(labels_csv):
        print(f"    Labels file not found: {labels_csv}")
        print("    Using default task configuration")
        task_config = DEFAULT_TASK_CONFIG.copy()
    else:
        try:
            df = pd.read_csv(labels_csv)
            unique_labels = df['label'].unique() if 'label' in df.columns else []

            # Detect if this uses legacy hardcoded labels
            if 'touching' in unique_labels or 'not_touching' in unique_labels:
                task_config = {
                    "version": "1.0",
                    "task_name": "nose_touch_detection",
                    "task_description": "Binary classification for detecting when subject touches their nose",
                    "labels": {
                        "positive": {
                            "display_name": "touching",
                            "description": "Subject is touching their nose"
                        },
                        "negative": {
                            "display_name": "not_touching",
                            "description": "Subject is not touching their nose"
                        }
                    },
                    "created_at": datetime.now().isoformat(),
                    "last_modified": datetime.now().isoformat()
                }
            else:
                # Unknown labels - use generic config
                task_config = DEFAULT_TASK_CONFIG.copy()
                task_config["created_at"] = datetime.now().isoformat()
                task_config["last_modified"] = datetime.now().isoformat()
        except Exception as e:
            print(f"    Warning: Could not read labels file: {e}")
            task_config = DEFAULT_TASK_CONFIG.copy()
            task_config["created_at"] = datetime.now().isoformat()
            task_config["last_modified"] = datetime.now().isoformat()

    save_json(task_config, config_path)
    print(f"    ✓ Created: {config_path}")

    return task_config

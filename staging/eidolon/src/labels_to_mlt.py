#!/usr/bin/env python3
"""
Generate Shotcut MLT project from labeled frames.

Usage:
    python src/labels_to_mlt.py --video videos/my_video.mp4
    python src/labels_to_mlt.py --video videos/my_video.mp4 --output project.mlt
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from utils import load_config, get_video_info, ensure_dir
from generate_mlt import create_mlt_project


def labels_to_events(labels_df: pd.DataFrame, video_path: str, config: dict, video_fps: float) -> list:
    """
    Convert labeled frames to event format (similar to detect_events in infer_video.py).

    Args:
        labels_df: DataFrame with columns: frame_path, label, timestamp
        video_path: Path to source video
        config: Configuration dict
        video_fps: Video frame rate

    Returns:
        List of event dicts with start_time, end_time, etc.
    """
    # Get video name to filter labels
    video_name = Path(video_path).stem
    frames_path = config['paths']['frames']
    frames_path_pattern = os.path.join(frames_path, video_name)

    # Filter labels for this video
    video_labels = labels_df[labels_df['frame_path'].str.contains(frames_path_pattern, na=False, regex=False)]

    if len(video_labels) == 0:
        print(f"No labels found for video: {video_name}")
        return []

    # Sort by timestamp
    video_labels = video_labels.sort_values('timestamp')

    # Convert to list of predictions format
    predictions = []
    for _, row in video_labels.iterrows():
        # Calculate frame index from timestamp
        frame_idx = int(row['timestamp'] * video_fps)

        predictions.append({
            'frame_idx': frame_idx,
            'timestamp': row['timestamp'],
            'predicted_class': row['label'],  # Already 'touching' or 'not_touching' string
            'probability': 1.0  # Labels are 100% confident
        })

    # Apply event detection logic (same as infer_video.py)
    min_duration = config['inference'].get('min_event_duration', 0.0)

    events = []
    current_event = None

    print("\nDetecting events from labels...")

    for pred in predictions:
        if pred['predicted_class'] == 'touching':
            if current_event is None:
                # Start new event
                current_event = {
                    'start_time': pred['timestamp'],
                    'end_time': pred['timestamp'],
                    'start_frame': pred['frame_idx'],
                    'end_frame': pred['frame_idx'],
                    'frames': [pred]
                }
            else:
                # Extend current event
                current_event['end_time'] = pred['timestamp']
                current_event['end_frame'] = pred['frame_idx']
                current_event['frames'].append(pred)
        else:
            if current_event is not None:
                # End current event
                duration = current_event['end_time'] - current_event['start_time']

                if duration >= min_duration:
                    # Calculate average confidence
                    confidences = [f['probability'] for f in current_event['frames']]
                    current_event['avg_confidence'] = np.mean(confidences)
                    current_event['max_confidence'] = np.max(confidences)
                    current_event['num_frames'] = len(current_event['frames'])

                    # Remove detailed frame list to save space
                    del current_event['frames']

                    events.append(current_event)

                current_event = None

    # Handle event at end of video
    if current_event is not None:
        duration = current_event['end_time'] - current_event['start_time']
        if duration >= min_duration:
            confidences = [f['probability'] for f in current_event['frames']]
            current_event['avg_confidence'] = np.mean(confidences)
            current_event['max_confidence'] = np.max(confidences)
            current_event['num_frames'] = len(current_event['frames'])

            del current_event['frames']
            events.append(current_event)

    # Add event IDs
    for i, event in enumerate(events):
        event['event_id'] = i + 1

    return events


def generate_mlt_from_labels(video_path: str, output_path: str = None, config_path: str = 'config.yaml'):
    """
    Generate MLT project from labeled frames.

    Args:
        video_path: Path to source video file
        output_path: Output MLT path (optional)
        config_path: Path to config file (default: config.yaml)
    """
    # Load config
    config = load_config(config_path)
    marker_buffer = config.get('mlt', {}).get('marker_buffer', 2.0)

    # Load labels
    labels_path = config['paths']['labels']
    if not os.path.exists(labels_path):
        print(f"Error: No labels found at {labels_path}")
        print("Please label some frames first.")
        return

    print(f"Loading labels from: {labels_path}")
    labels_df = pd.read_csv(labels_path)

    # Get video info
    video_info = get_video_info(video_path)
    video_fps = video_info['fps']

    # Convert labels to events
    events = labels_to_events(labels_df, video_path, config, video_fps)

    if not events:
        print("No touching events found in labels!")
        return

    print(f"\nFound {len(events)} labeled events:")
    for event in events:
        from utils import format_timestamp
        print(f"  Event {event['event_id']}: "
              f"{format_timestamp(event['start_time'])} - {format_timestamp(event['end_time'])} "
              f"({event['num_frames']} frames)")

    # Determine output path
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = f"outputs/projects/{video_name}_from_labels.mlt"

    # Generate MLT
    print(f"\nGenerating MLT project...")
    create_mlt_project(video_path, events, video_fps, output_path, marker_buffer)


def main():
    parser = argparse.ArgumentParser(description='Generate Shotcut MLT from labels')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--output', help='Output MLT path (default: outputs/projects/<video>_from_labels.mlt)')
    parser.add_argument('--config', default='config.yaml', help='Config file path')

    args = parser.parse_args()

    generate_mlt_from_labels(args.video, args.output, args.config)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Process raw predictions to detect events with configurable threshold and duration filters.

This script loads raw predictions (probabilities) from inference and applies threshold
and min_event_duration filters to generate events. This allows quick re-processing
with different parameters without re-running slow inference.

Usage:
    python src/process_events.py \\
        --predictions outputs/predictions/video_predictions.json \\
        --threshold 0.5 \\
        --min-duration 0.4
"""

import os
import argparse
import numpy as np
from pathlib import Path
from utils import load_config, load_json, save_json, ensure_dir, format_timestamp, get_video_info


def apply_threshold_to_predictions(predictions: list, threshold: float) -> list:
    """
    Apply classification threshold to raw predictions.

    Args:
        predictions: List of prediction dicts with 'probability' field
        threshold: Classification threshold (0.0-1.0)

    Returns:
        Same list with 'predicted_class' field added based on threshold
    """
    processed = []
    for pred in predictions:
        prob = pred['probability']
        predicted_class = 'touching' if prob >= threshold else 'not_touching'

        processed.append({
            'frame_idx': pred['frame_idx'],
            'timestamp': pred['timestamp'],
            'probability': prob,
            'predicted_class': predicted_class
        })

    return processed


def detect_events(predictions: list, min_duration: float) -> list:
    """
    Convert frame-level predictions to discrete events.

    Args:
        predictions: List of prediction dicts with 'predicted_class' field
        min_duration: Minimum event duration in seconds

    Returns:
        List of event dicts with start/end times
    """
    events = []
    current_event = None

    print("\nDetecting events...")

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


def process_events_from_predictions(
    predictions_file: str,
    threshold: float,
    min_duration: float,
    output_file: str = None,
    config: dict = None
) -> dict:
    """
    Process raw predictions to generate events with specified parameters.

    Args:
        predictions_file: Path to predictions JSON file
        threshold: Classification threshold (0.0-1.0)
        min_duration: Minimum event duration in seconds
        output_file: Output path for events JSON (optional)
        config: Configuration dict (optional)

    Returns:
        Events data dict
    """
    print("=" * 80)
    print("EVENT DETECTION FROM PREDICTIONS")
    print("=" * 80)
    print()

    # Load predictions
    if not os.path.exists(predictions_file):
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

    print(f"Loading predictions from: {predictions_file}")
    raw_predictions = load_json(predictions_file)

    print(f"Loaded {len(raw_predictions)} frame predictions")
    print()

    # Show probability statistics
    probs = [p['probability'] for p in raw_predictions]
    print(f"Probability statistics:")
    print(f"  Min: {min(probs):.4f}")
    print(f"  Max: {max(probs):.4f}")
    print(f"  Mean: {np.mean(probs):.4f}")
    print(f"  Std: {np.std(probs):.4f}")
    print()

    # Apply threshold
    print(f"Parameters:")
    print(f"  Classification threshold: {threshold}")
    print(f"  Min event duration: {min_duration}s")
    print()

    print("Applying threshold to predictions...")
    predictions = apply_threshold_to_predictions(raw_predictions, threshold)

    touching_frames = sum(1 for p in predictions if p['predicted_class'] == 'touching')
    print(f"Touching frames: {touching_frames}/{len(predictions)} ({touching_frames/len(predictions)*100:.1f}%)")

    # Detect events
    events = detect_events(predictions, min_duration)

    print(f"\nDetected {len(events)} events")

    if events:
        print("\nEvents:")
        for event in events[:10]:  # Show first 10
            print(f"  Event {event['event_id']}: "
                  f"{format_timestamp(event['start_time'])} - {format_timestamp(event['end_time'])} "
                  f"({event['num_frames']} frames, "
                  f"avg conf: {event['avg_confidence']:.2f})")
        if len(events) > 10:
            print(f"  ... and {len(events) - 10} more events")

    # Get video info from predictions filename
    video_name = Path(predictions_file).stem.replace('_predictions', '')

    # Try to get video path - check common locations
    video_path = None
    for ext in ['.mp4', '.mkv', '.avi', '.mov']:
        candidate = f"videos/{video_name}{ext}"
        if os.path.exists(candidate):
            video_path = os.path.abspath(candidate)
            break

    # Get video info if we found the video
    video_info = {}
    if video_path and os.path.exists(video_path):
        video_info = get_video_info(video_path)
    else:
        # Extract from predictions data
        if raw_predictions:
            video_info = {
                'fps': raw_predictions[-1]['frame_idx'] / raw_predictions[-1]['timestamp'] if raw_predictions[-1]['timestamp'] > 0 else 30.0,
                'duration': raw_predictions[-1]['timestamp']
            }

    # Prepare events data
    events_data = {
        'video_path': video_path or f"videos/{video_name}.mp4",
        'video_info': video_info,
        'predictions_file': os.path.abspath(predictions_file),
        'config': {
            'threshold': threshold,
            'min_event_duration': min_duration,
            'marker_buffer': config.get('mlt', {}).get('marker_buffer', 2.0) if config else 2.0
        },
        'num_events': len(events),
        'events': events
    }

    # Determine output path
    if output_file is None:
        output_file = predictions_file.replace('/predictions/', '/events/').replace('_predictions.json', '_events.json')

    # Save events
    ensure_dir(os.path.dirname(output_file))
    save_json(events_data, output_file)
    print(f"\nEvents saved to: {output_file}")

    print()
    print("=" * 80)
    print("EVENT DETECTION COMPLETE")
    print("=" * 80)
    print()

    return events_data


def main():
    parser = argparse.ArgumentParser(
        description='Process predictions to detect events with configurable parameters'
    )
    parser.add_argument('--predictions', required=True, help='Path to predictions JSON file')
    parser.add_argument('--threshold', type=float, help='Classification threshold (default: from config)')
    parser.add_argument('--min-duration', type=float, help='Minimum event duration in seconds (default: from config)')
    parser.add_argument('--output', help='Output events JSON path (default: auto-generate from predictions path)')
    parser.add_argument('--config', default='config.yaml', help='Config file path')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Get parameters from args or config
    threshold = args.threshold if args.threshold is not None else config['inference']['threshold']
    min_duration = args.min_duration if args.min_duration is not None else config['inference'].get('min_event_duration', 0.0)

    # Process events
    process_events_from_predictions(
        predictions_file=args.predictions,
        threshold=threshold,
        min_duration=min_duration,
        output_file=args.output,
        config=config
    )


if __name__ == '__main__':
    main()

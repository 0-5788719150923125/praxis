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
from scipy.ndimage import median_filter
from utils import (
    load_config,
    load_json,
    save_json,
    ensure_dir,
    format_timestamp,
    get_video_info,
    load_task_config
)


def apply_temporal_smoothing(predictions: list, method: str, window_size: int, consensus_threshold: float = 0.6) -> list:
    """
    Apply temporal smoothing to reduce false positives from single-frame misclassifications.

    Args:
        predictions: List of prediction dicts with 'probability' field
        method: Smoothing method - "median", "moving_average", or "consensus"
        window_size: Number of frames to consider in the window
        consensus_threshold: For consensus mode - fraction of frames that must agree (0.0-1.0)

    Returns:
        Smoothed predictions with updated 'probability' field
    """
    if len(predictions) == 0:
        return predictions

    # Extract probabilities as numpy array
    probs = np.array([p['probability'] for p in predictions])
    original_probs = probs.copy()

    if method == "median":
        # Median filter: Replace each value with median of surrounding window
        # This is very effective at removing isolated spikes/false positives
        smoothed_probs = median_filter(probs, size=window_size, mode='nearest')

    elif method == "moving_average":
        # Moving average: Smooth by averaging neighboring frames
        # Creates smoother transitions but can blur true boundaries
        smoothed_probs = np.convolve(probs, np.ones(window_size)/window_size, mode='same')

    elif method == "consensus":
        # Consensus voting: Require multiple frames in window to agree
        # Most aggressive - good for very noisy predictions
        smoothed_probs = np.zeros_like(probs)
        half_window = window_size // 2

        for i in range(len(probs)):
            start = max(0, i - half_window)
            end = min(len(probs), i + half_window + 1)
            window = probs[start:end]

            # Count how many frames in window are above 0.5
            positive_fraction = np.mean(window > 0.5)

            # If consensus threshold met, use original probability, else suppress
            if positive_fraction >= consensus_threshold:
                smoothed_probs[i] = probs[i]
            else:
                smoothed_probs[i] = 0.0

    else:
        raise ValueError(f"Unknown smoothing method: {method}")

    # Update predictions with smoothed probabilities
    smoothed_predictions = []
    for i, pred in enumerate(predictions):
        smoothed_predictions.append({
            'frame_idx': pred['frame_idx'],
            'timestamp': pred['timestamp'],
            'probability': float(smoothed_probs[i]),
            'original_probability': float(original_probs[i])  # Keep original for debugging
        })

    # Print smoothing statistics
    diff = np.abs(smoothed_probs - original_probs)
    print(f"\nTemporal smoothing applied ({method}, window={window_size}):")
    print(f"  Mean absolute change: {np.mean(diff):.4f}")
    print(f"  Max change: {np.max(diff):.4f}")
    print(f"  Frames significantly changed (>0.2): {np.sum(diff > 0.2)}/{len(probs)}")

    return smoothed_predictions


def apply_threshold_to_predictions(predictions: list, threshold: float, task_config: dict = None) -> list:
    """
    Apply classification threshold to raw predictions.

    Args:
        predictions: List of prediction dicts with 'probability' field
        threshold: Classification threshold (0.0-1.0)
        task_config: Task configuration dictionary

    Returns:
        Same list with 'predicted_class' field added based on threshold
    """
    task_config = task_config or load_task_config()
    pos_label = task_config['labels']['positive']['display_name']
    neg_label = task_config['labels']['negative']['display_name']

    processed = []
    for pred in predictions:
        prob = pred['probability']
        # Use display labels for predictions
        predicted_class = pos_label if prob >= threshold else neg_label

        processed.append({
            'frame_idx': pred['frame_idx'],
            'timestamp': pred['timestamp'],
            'probability': prob,
            'predicted_class': predicted_class
        })

    return processed


def detect_events(predictions: list, min_duration: float, task_config: dict = None) -> list:
    """
    Convert frame-level predictions to discrete events.

    Args:
        predictions: List of prediction dicts with 'predicted_class' field
        min_duration: Minimum event duration in seconds
        task_config: Task configuration dictionary

    Returns:
        List of event dicts with start/end times
    """
    task_config = task_config or load_task_config()
    pos_label = task_config['labels']['positive']['display_name']

    events = []
    current_event = None

    print("\nDetecting events...")

    for pred in predictions:
        if pred['predicted_class'] == pos_label:
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
    # Load task config
    task_config = load_task_config()
    pos_label = task_config['labels']['positive']['display_name']

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
    print(f"Probability statistics (raw):")
    print(f"  Min: {min(probs):.4f}")
    print(f"  Max: {max(probs):.4f}")
    print(f"  Mean: {np.mean(probs):.4f}")
    print(f"  Std: {np.std(probs):.4f}")
    print()

    # Apply temporal smoothing if enabled
    predictions_to_threshold = raw_predictions
    if config and config.get('inference', {}).get('temporal_smoothing', {}).get('enabled', False):
        smoothing_config = config['inference']['temporal_smoothing']
        method = smoothing_config.get('method', 'median')
        window_size = smoothing_config.get('window_size', 5)
        consensus_threshold_val = smoothing_config.get('consensus_threshold', 0.6)

        print(f"Temporal smoothing enabled:")
        print(f"  Method: {method}")
        print(f"  Window size: {window_size} frames")
        if method == 'consensus':
            print(f"  Consensus threshold: {consensus_threshold_val}")

        predictions_to_threshold = apply_temporal_smoothing(
            raw_predictions,
            method=method,
            window_size=window_size,
            consensus_threshold=consensus_threshold_val
        )
    else:
        print("Temporal smoothing: disabled")

    # Apply threshold
    print(f"\nClassification parameters:")
    print(f"  Classification threshold: {threshold}")
    print(f"  Min event duration: {min_duration}s")
    print()

    print("Applying threshold to predictions...")
    predictions = apply_threshold_to_predictions(predictions_to_threshold, threshold, task_config)

    positive_frames = sum(1 for p in predictions if p['predicted_class'] == pos_label)
    print(f"{pos_label} frames: {positive_frames}/{len(predictions)} ({positive_frames/len(predictions)*100:.1f}%)")

    # Detect events
    events = detect_events(predictions, min_duration, task_config)

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

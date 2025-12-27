#!/usr/bin/env python3
"""
End-to-end pipeline: inference + event detection + MLT generation in one command.

Usage:
    python src/pipeline.py --video videos/my_video.mp4 --model models/deit-small-nose-touch/final
    python src/pipeline.py --video videos/my_video.mp4 --model models/model/final --threshold 0.3 --min-duration 0.5
"""

import os
import argparse
from pathlib import Path
from infer_video import infer_video
from process_events import process_events_from_predictions
from generate_mlt import create_mlt_project
from utils import load_config, load_json


def run_pipeline(video_path: str, model_path: str, config: dict, output_mlt: str = None):
    """
    Run complete pipeline: inference + event detection + MLT generation.

    Args:
        video_path: Path to video file
        model_path: Path to trained model
        config: Configuration dict
        output_mlt: Output MLT path (optional)
    """
    print("=" * 80)
    print("EIDOLON PIPELINE - Nose-Touch Detection")
    print("=" * 80)
    print()

    video_name = Path(video_path).stem

    # Step 1: Run inference
    print("STEP 1: Running raw inference on video...")
    print("-" * 80)
    predictions_data = infer_video(video_path, model_path, config, 'outputs')

    print()
    print("=" * 80)

    # Step 2: Process events with threshold and min_duration
    print("STEP 2: Processing predictions to detect events...")
    print("-" * 80)

    predictions_file = predictions_data['predictions_file']
    threshold = config['inference']['threshold']
    min_duration = config['inference'].get('min_event_duration', 0.0)

    events_data = process_events_from_predictions(
        predictions_file=predictions_file,
        threshold=threshold,
        min_duration=min_duration,
        config=config
    )

    if events_data['num_events'] == 0:
        print("\n" + "!" * 80)
        print("WARNING: No events detected!")
        print("!" * 80)
        print("\nPossible solutions:")
        print("  - Lower the classification threshold (try --threshold 0.3)")
        print("  - Reduce minimum event duration (try --min-duration 0.2)")
        print("  - Check that the video contains nose-touching moments")
        print("  - Verify model was trained correctly")
        return

    print()
    print("=" * 80)

    # Step 3: Generate MLT
    print("STEP 3: Generating Shotcut project...")
    print("-" * 80)

    events_file = f"outputs/events/{video_name}_events.json"

    if output_mlt is None:
        output_mlt = f"outputs/projects/{video_name}_project.mlt"

    # Load events data
    data = load_json(events_file)
    marker_buffer = config.get('mlt', {}).get('marker_buffer', 2.0)
    create_mlt_project(
        data['video_path'],
        data['events'],
        data['video_info']['fps'],
        output_mlt,
        marker_buffer
    )

    print()
    print("=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print()
    print(f"Summary:")
    print(f"  Video: {video_path}")
    print(f"  Predictions: {predictions_data['num_predictions']} frames")
    print(f"  Events detected: {events_data['num_events']}")
    print(f"  MLT project: {output_mlt}")
    print()
    print("Next steps:")
    print(f"  1. Open Shotcut: shotcut {output_mlt}")
    print(f"  2. Timeline shows full video with cuts at predicted moments")
    print(f"  3. Export: File → Export")
    print()
    print("To re-generate with different parameters:")
    print(f"  python src/process_events.py --predictions {predictions_file} --threshold 0.3")
    print(f"  python src/generate_mlt.py --events {events_file} --marker-buffer 3.0")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Run complete pipeline: inference + event detection + MLT generation'
    )
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--output', help='Output MLT path (default: outputs/projects/<video>.mlt)')
    parser.add_argument('--threshold', type=float, help='Classification threshold (default: from config)')
    parser.add_argument('--min-duration', type=float, help='Minimum event duration in seconds (default: from config)')
    parser.add_argument('--marker-buffer', type=float, help='Marker buffer in seconds (default: from config)')
    parser.add_argument('--config', default='config.yaml', help='Config file path')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override parameters if specified
    if args.threshold is not None:
        config['inference']['threshold'] = args.threshold
    if args.min_duration is not None:
        config['inference']['min_event_duration'] = args.min_duration
    if args.marker_buffer is not None:
        config['mlt']['marker_buffer'] = args.marker_buffer

    # Run pipeline
    run_pipeline(args.video, args.model, config, args.output)


if __name__ == '__main__':
    main()

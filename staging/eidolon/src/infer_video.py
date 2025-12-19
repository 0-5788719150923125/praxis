#!/usr/bin/env python3
"""
Run inference on video to detect nose-touch events.

Usage:
    python src/infer_video.py --video videos/my_video.mp4 --model models/deit-small-nose-touch/final
"""

import os
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from utils import load_config, save_json, get_video_info, ensure_dir, format_timestamp


def detect_events(predictions: list, config: dict, video_fps: float) -> list:
    """
    Convert frame-level predictions to discrete events.

    Args:
        predictions: List of prediction dicts
        config: Configuration dict
        video_fps: Video frame rate

    Returns:
        List of event dicts with start/end times
    """
    buffer_before = config['mlt']['buffer_before']
    buffer_after = config['mlt']['buffer_after']
    min_duration = config['inference'].get('min_event_duration', 0.0)

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
                    # Apply buffers
                    current_event['start_time'] = max(0, current_event['start_time'] - buffer_before)
                    current_event['end_time'] = current_event['end_time'] + buffer_after

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
            current_event['start_time'] = max(0, current_event['start_time'] - buffer_before)
            current_event['end_time'] = current_event['end_time'] + buffer_after

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


def infer_video(video_path: str, model_path: str, config: dict, output_dir: str):
    """
    Run inference on video to detect nose-touch events.

    Args:
        video_path: Path to video file
        model_path: Path to trained model
        config: Configuration dict
        output_dir: Output directory for results
    """
    # Get video info
    video_info = get_video_info(video_path)
    video_fps = video_info['fps']
    duration = video_info['duration']

    print(f"Video: {video_path}")
    print(f"Resolution: {video_info['width']}x{video_info['height']}")
    print(f"FPS: {video_fps:.2f}")
    print(f"Duration: {format_timestamp(duration)}")
    print()

    # Load model and processor
    print(f"Loading model from: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() and config['inference']['use_gpu'] else "cpu")
    print(f"Using device: {device}")

    image_processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Get label mapping
    id2label = model.config.id2label
    print(f"Labels: {id2label}")
    print()

    # Extract and classify frames
    target_fps = config['extraction']['fps']
    frame_interval = int(video_fps / target_fps)
    batch_size = config['inference']['batch_size']
    threshold = config['inference']['threshold']

    print(f"Extracting frames at {target_fps} fps...")
    print(f"Batch size: {batch_size}")
    print(f"Classification threshold: {threshold}")
    print()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    predictions = []
    batch_frames = []
    batch_metadata = []
    frame_count = 0

    total_frames = video_info['total_frames']
    pbar = tqdm(total=total_frames, desc="Processing")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process at intervals
        if frame_count % frame_interval == 0:
            timestamp = frame_count / video_fps

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            batch_frames.append(pil_image)
            batch_metadata.append({
                'frame_idx': frame_count,
                'timestamp': timestamp
            })

            # Process batch
            if len(batch_frames) >= batch_size:
                # Preprocess images
                inputs = image_processor(batch_frames, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Run inference
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.nn.functional.softmax(logits, dim=-1)

                # Process predictions
                for i, (prob, meta) in enumerate(zip(probs, batch_metadata)):
                    touching_prob = prob[model.config.label2id['touching']].item()
                    predicted_label = 'touching' if touching_prob >= threshold else 'not_touching'

                    predictions.append({
                        'frame_idx': meta['frame_idx'],
                        'timestamp': meta['timestamp'],
                        'predicted_class': predicted_label,
                        'probability': touching_prob
                    })

                # Clear batch
                batch_frames = []
                batch_metadata = []

        frame_count += 1
        pbar.update(1)

    # Process remaining frames
    if batch_frames:
        inputs = image_processor(batch_frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

        for i, (prob, meta) in enumerate(zip(probs, batch_metadata)):
            touching_prob = prob[model.config.label2id['touching']].item()
            predicted_label = 'touching' if touching_prob >= threshold else 'not_touching'

            predictions.append({
                'frame_idx': meta['frame_idx'],
                'timestamp': meta['timestamp'],
                'predicted_class': predicted_label,
                'probability': touching_prob
            })

    pbar.close()
    cap.release()

    print(f"\nProcessed {len(predictions)} frames")

    # Calculate statistics
    touching_frames = sum(1 for p in predictions if p['predicted_class'] == 'touching')
    print(f"Touching frames: {touching_frames} ({touching_frames/len(predictions)*100:.1f}%)")

    # Detect events
    events = detect_events(predictions, config, video_fps)

    print(f"\nDetected {len(events)} events")

    if events:
        print("\nEvents:")
        for event in events:
            print(f"  Event {event['event_id']}: "
                  f"{format_timestamp(event['start_time'])} - {format_timestamp(event['end_time'])} "
                  f"({event['num_frames']} frames, "
                  f"avg conf: {event['avg_confidence']:.2f})")

    # Save results
    ensure_dir(output_dir)

    video_name = Path(video_path).stem

    # Save predictions
    predictions_path = os.path.join(output_dir, 'predictions', f'{video_name}_predictions.json')
    save_json(predictions, predictions_path)
    print(f"\nPredictions saved to: {predictions_path}")

    # Save events
    events_path = os.path.join(output_dir, 'events', f'{video_name}_events.json')
    events_data = {
        'video_path': os.path.abspath(video_path),
        'video_info': video_info,
        'model_path': model_path,
        'config': {
            'threshold': threshold,
            'buffer_before': config['mlt']['buffer_before'],
            'buffer_after': config['mlt']['buffer_after'],
            'min_event_duration': config['inference'].get('min_event_duration', 0.0)
        },
        'num_events': len(events),
        'events': events
    }
    save_json(events_data, events_path)
    print(f"Events saved to: {events_path}")

    return events_data


def main():
    parser = argparse.ArgumentParser(description='Run inference on video')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--output', help='Output directory (default: outputs/)')
    parser.add_argument('--threshold', type=float, help='Classification threshold (default: from config)')
    parser.add_argument('--config', default='config.yaml', help='Config file path')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override threshold if specified
    if args.threshold is not None:
        config['inference']['threshold'] = args.threshold

    # Determine output directory
    output_dir = args.output if args.output else 'outputs'

    # Run inference
    results = infer_video(args.video, args.model, config, output_dir)

    print("\nInference complete!")


if __name__ == '__main__':
    main()

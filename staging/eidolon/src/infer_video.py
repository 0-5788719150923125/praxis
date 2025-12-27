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
from peft import PeftModel
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
    # No buffers needed - we just detect the events themselves
    # Marker buffer is applied later in MLT generation
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


def infer_video(video_path: str, model_path: str, config: dict, output_dir: str):
    """
    Run inference on video to detect nose-touch events.

    Args:
        video_path: Path to video file
        model_path: Path to trained model
        config: Configuration dict
        output_dir: Output directory for results
    """
    # VALIDATE CONFIG FIRST - fail fast before processing
    print("Validating configuration...")
    try:
        _ = config['inference']['threshold']
        _ = config['inference']['batch_size']
        _ = config['inference']['use_gpu']
        _ = config['extraction']['fps']
        _ = config['mlt']['marker_buffer']
        print("✓ Configuration valid\n")
    except KeyError as e:
        raise ValueError(
            f"Missing required config key: {e}\n"
            "Please check your config.yaml file has all required fields."
        )

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

    # Check model directory contents
    print(f"\nModel directory contents:")
    if os.path.exists(model_path):
        for item in os.listdir(model_path):
            item_path = os.path.join(model_path, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                print(f"  - {item} ({size:,} bytes)")
            else:
                print(f"  - {item}/ (directory)")
    else:
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    # Check if this is a PEFT model
    is_peft = os.path.exists(os.path.join(model_path, 'is_peft_model'))
    has_adapter_config = os.path.exists(os.path.join(model_path, 'adapter_config.json'))

    print(f"\nModel type detection:")
    print(f"  is_peft_model marker: {is_peft}")
    print(f"  adapter_config.json: {has_adapter_config}")

    if is_peft or has_adapter_config:
        print("\n>>> Loading PEFT model (LoRA adapter)...")

        # Read the base model ID
        base_model_id_path = os.path.join(model_path, 'base_model_id.txt')
        if os.path.exists(base_model_id_path):
            with open(base_model_id_path, 'r') as f:
                base_model_id = f.read().strip()
            print(f"  Base model ID: {base_model_id}")
        else:
            # Fallback for models trained before this fix
            print("  ⚠ WARNING: base_model_id.txt not found, using default")
            base_model_id = "facebook/deit-small-patch16-224"

        # Load the original pretrained model from HuggingFace
        print(f"  Loading original pretrained model: {base_model_id}")
        model = AutoModelForImageClassification.from_pretrained(
            base_model_id,
            num_labels=2,  # Binary classification
            id2label={0: 'not_touching', 1: 'touching'},
            label2id={'not_touching': 0, 'touching': 1},
            ignore_mismatched_sizes=True
        )

        # Apply the LoRA adapter
        print(f"  Loading LoRA adapter from: {model_path}")
        try:
            model = PeftModel.from_pretrained(model, model_path)
            print("  ✓ LoRA adapter loaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to load LoRA adapter: {e}")
            raise

        # Verify adapter is active
        print(f"\n  Verifying adapter parameters:")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"    Total parameters: {total_params:,}")

        # List adapter modules
        adapter_modules = [name for name, _ in model.named_modules() if 'lora' in name.lower()]
        if adapter_modules:
            print(f"    LoRA modules found: {len(adapter_modules)}")
            print(f"    Sample: {adapter_modules[0]}")
        else:
            print("    ⚠ WARNING: No LoRA modules found in loaded model!")
    else:
        print("\n>>> Loading standard fine-tuned model...")
        model = AutoModelForImageClassification.from_pretrained(model_path)
        print("  ✓ Model loaded")

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

    # Calculate which frames we actually need to process
    total_frames = video_info['total_frames']
    frames_to_process = list(range(0, total_frames, frame_interval))
    total_to_process = len(frames_to_process)

    print(f"Will process {total_to_process} frames (every {frame_interval}th frame)")
    pbar = tqdm(total=total_to_process, desc="Inference", unit="frames")

    frame_count = 0
    last_frame_idx = -1

    # Read frames sequentially, skip unwanted frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process frames at our target intervals
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

            # Process batch when full
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
                batch_start_idx = len(predictions)
                for i, (prob, meta) in enumerate(zip(probs, batch_metadata)):
                    touching_prob = prob[model.config.label2id['touching']].item()
                    predicted_label = 'touching' if touching_prob >= threshold else 'not_touching'

                    predictions.append({
                        'frame_idx': meta['frame_idx'],
                        'timestamp': meta['timestamp'],
                        'predicted_class': predicted_label,
                        'probability': touching_prob
                    })

                # Early diagnostics on first batch
                if batch_start_idx == 0:
                    first_probs = [p['probability'] for p in predictions]
                    print(f"\n>>> First batch diagnostics (SANITY CHECK):")
                    print(f"    Processed: {len(predictions)} frames")
                    print(f"    Probability range: {min(first_probs):.4f} - {max(first_probs):.4f}")
                    print(f"    Mean probability: {sum(first_probs)/len(first_probs):.4f}")
                    print(f"    Threshold: {threshold}")

                    prob_range = max(first_probs) - min(first_probs)
                    if prob_range < 0.2:
                        print(f"\n    ⚠⚠⚠ CRITICAL WARNING ⚠⚠⚠")
                        print(f"    Probability range is very narrow: {prob_range:.4f}")
                        print(f"    Model is outputting near-random predictions!")
                        print(f"    Expected: Wide range (0.0-1.0) with confident predictions")
                        print(f"    Got: Narrow range around 0.5 (model unsure about everything)")
                        print(f"\n    Likely causes:")
                        print(f"      1. LoRA adapter not loaded (using untrained base model)")
                        print(f"      2. Wrong model checkpoint loaded")
                        print(f"      3. Preprocessing mismatch")
                        print(f"\n    Aborting inference to save time...")
                        cap.release()
                        pbar.close()
                        raise RuntimeError("Model confidence check failed - see warnings above")
                    else:
                        print(f"    ✓ Probability distribution looks reasonable")
                    print()

                # Clear batch and update progress
                batch_frames = []
                batch_metadata = []
                pbar.update(batch_size)

        frame_count += 1

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

        pbar.update(len(batch_frames))

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
            'marker_buffer': config['mlt']['marker_buffer'],
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

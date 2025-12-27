#!/usr/bin/env python3
"""
Extract frames from video for training or inference.

Usage:
    python src/extract_frames.py --video videos/my_video.mp4
    python src/extract_frames.py --video videos/my_video.mp4 --fps 10
"""

import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
from utils import load_config, save_json, get_video_info, ensure_dir, format_timestamp


def extract_frames(video_path: str, output_dir: str, target_fps: int, quality: int = 95) -> dict:
    """
    Extract frames from video at specified FPS.

    Args:
        video_path: Path to source video
        output_dir: Directory to save extracted frames
        target_fps: Target frames per second to extract
        quality: JPEG quality (1-100)

    Returns:
        Dictionary with metadata about extracted frames
    """
    # Create output directory
    ensure_dir(output_dir)

    # Get video information
    video_info = get_video_info(video_path)
    source_fps = video_info['fps']
    duration = video_info['duration']
    total_frames = video_info['total_frames']

    print(f"Video info:")
    print(f"  Resolution: {video_info['width']}x{video_info['height']}")
    print(f"  FPS: {source_fps:.2f}")
    print(f"  Duration: {format_timestamp(duration)}")
    print(f"  Total frames: {total_frames}")
    print(f"\nExtracting at {target_fps} fps...")

    # Calculate frame interval
    frame_interval = int(source_fps / target_fps)
    if frame_interval < 1:
        frame_interval = 1
        print(f"Warning: Target FPS ({target_fps}) is higher than source FPS ({source_fps:.2f})")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Extract frames
    frame_count = 0
    extracted_count = 0
    metadata = []

    pbar = tqdm(total=total_frames, desc="Extracting frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frame at interval
        if frame_count % frame_interval == 0:
            # Calculate timestamp
            timestamp = frame_count / source_fps

            # Save frame
            frame_filename = f"frame_{extracted_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)

            # Save with specified quality
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

            # Store metadata
            metadata.append({
                'frame_id': extracted_count,
                'source_frame': frame_count,
                'timestamp': timestamp,
                'filename': frame_filename
            })

            extracted_count += 1

        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    print(f"\nExtracted {extracted_count} frames from {frame_count} total frames")

    # Save metadata
    metadata_dict = {
        'video_path': os.path.abspath(video_path),
        'video_info': video_info,
        'extraction_fps': target_fps,
        'frame_interval': frame_interval,
        'total_extracted': extracted_count,
        'frames': metadata
    }

    metadata_path = os.path.join(output_dir, 'metadata.json')
    save_json(metadata_dict, metadata_path)
    print(f"Metadata saved to: {metadata_path}")

    return metadata_dict


def main():
    parser = argparse.ArgumentParser(description='Extract frames from video')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--output', help='Output directory (default: data/frames/<video_name>)')
    parser.add_argument('--fps', type=int, help='Target FPS (default: from config.yaml)')
    parser.add_argument('--quality', type=int, default=95, help='JPEG quality 1-100 (default: 95)')
    parser.add_argument('--config', default='config.yaml', help='Config file path')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        video_name = Path(args.video).stem
        frames_dir = config['paths']['frames']
        output_dir = os.path.join(frames_dir, video_name)

    # Determine FPS
    fps = args.fps if args.fps else config['extraction']['fps']

    # Extract frames
    print(f"Input video: {args.video}")
    print(f"Output directory: {output_dir}")
    print()

    metadata = extract_frames(args.video, output_dir, fps, args.quality)

    print(f"\nDone! Frames saved to: {output_dir}")


if __name__ == '__main__':
    main()

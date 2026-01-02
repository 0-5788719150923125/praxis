#!/usr/bin/env python3
"""
Prepare labeled frames for training by splitting into train/val/test sets.

Extracts labeled frames on-demand from video files (no bulk extraction needed).

Usage:
    python src/prepare_dataset.py --labels data/labels.csv
"""

import os
import cv2
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Optional
from utils import (
    load_config,
    ensure_dir,
    load_task_config,
    internal_to_display,
    display_to_internal,
    migrate_labels_to_internal,
    create_task_config_from_labels
)
from video_frame_extractor import VideoFrameExtractor


def find_video_file(video_name: str) -> Optional[str]:
    """
    Find video file matching video name.

    Args:
        video_name: Video filename stem (without extension)

    Returns:
        Absolute path to video file, or None if not found
    """
    videos_dir = "videos"
    for ext in ['.mp4', '.mkv', '.avi', '.mov', '.webm']:
        video_path = os.path.join(videos_dir, video_name + ext)
        if os.path.exists(video_path):
            return os.path.abspath(video_path)
    return None


def migrate_old_labels_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert old frame_path schema to new video_path schema.

    Args:
        df: DataFrame with old schema (frame_path, filename, label, timestamp)

    Returns:
        DataFrame with new schema (video_path, frame_index, label, timestamp)
    """
    print("  Migrating old labels schema to new format...")

    new_rows = []

    for _, row in df.iterrows():
        frame_path = row['frame_path']

        # Extract video name from path: "data/frames/@EphemeralRift.../frame_000099.jpg"
        parts = Path(frame_path).parts
        if 'frames' in parts:
            idx = parts.index('frames')
            if idx + 1 < len(parts):
                video_name = parts[idx + 1]

                # Find corresponding video file
                video_path = find_video_file(video_name)

                if video_path:
                    # Extract frame index from filename: frame_000099.jpg
                    filename = row['filename']
                    frame_idx = int(filename.split('_')[1].split('.')[0])

                    new_rows.append({
                        'video_path': video_path,
                        'frame_index': frame_idx,
                        'label': row['label'],
                        'timestamp': row['timestamp']
                    })
                else:
                    print(f"  WARNING: Video not found for: {video_name}")
        else:
            print(f"  WARNING: Could not parse frame path: {frame_path}")

    migrated_df = pd.DataFrame(new_rows)
    print(f"  Migrated {len(migrated_df)}/{len(df)} labels successfully")

    return migrated_df


def prepare_dataset(labels_csv: str, output_dir: str, config: dict):
    """
    Split labeled data into train/val/test sets.

    Args:
        labels_csv: Path to labels CSV
        output_dir: Output directory for dataset
        config: Configuration dictionary
    """
    # Load task config
    task_config = load_task_config()
    pos_label = task_config['labels']['positive']['display_name']
    neg_label = task_config['labels']['negative']['display_name']

    # Load labels
    print(f"Loading labels from: {labels_csv}")
    df = pd.read_csv(labels_csv)

    # Migrate labels to internal format ('true'/'false')
    df = migrate_labels_to_internal(df, backup=True, backup_path=labels_csv)

    # Save migrated labels back to CSV
    df.to_csv(labels_csv, index=False)

    # Create task_config.json if missing (infer from detected labels)
    if not os.path.exists('task_config.json'):
        create_task_config_from_labels(labels_csv)

    # Detect and handle old schema if needed
    if 'frame_path' in df.columns and 'video_path' not in df.columns:
        print("\nDetected old labels.csv schema")
        df = migrate_old_labels_schema(df)
        if df.empty:
            raise ValueError("Migration failed - no labels could be migrated")

    # Verify required columns
    required_cols = ['video_path', 'timestamp', 'label']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"labels.csv missing required columns: {required_cols}")

    # Remove skipped frames (only keep binary labeled samples)
    df = df[df['label'].isin(['true', 'false'])]  # Use internal labels

    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"\nClass distribution:")

    # Convert to display labels for user output
    display_labels = df['label'].map(lambda x: internal_to_display(x, task_config))
    print(display_labels.value_counts())
    print(display_labels.value_counts(normalize=True))

    # Check class balance
    class_counts = df['label'].value_counts()
    if len(class_counts) < 2:
        raise ValueError(f"Need at least 2 classes ({pos_label} and {neg_label})")

    imbalance_ratio = class_counts.max() / class_counts.min()
    if imbalance_ratio > 10:
        print(f"\nWarning: High class imbalance (ratio: {imbalance_ratio:.1f}:1)")
        print("Consider using class weights during training")

    # Get split ratios from config
    train_ratio = config['training']['train_split']
    val_ratio = config['training']['val_split']
    test_ratio = config['training']['test_split']

    # Verify ratios sum to 1.0
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Split ratios must sum to 1.0 (got {total})")

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df['label'],
        random_state=42
    )

    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        stratify=temp_df['label'],
        random_state=42
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    # Create directory structure (use display labels for directories)
    for split in ['train', 'val', 'test']:
        for label in [pos_label, neg_label]:
            split_dir = os.path.join(output_dir, split, label)
            ensure_dir(split_dir)

    # Extract labeled frames from video and save to dataset
    def extract_and_save_split(split_df, split_name):
        """Extract labeled frames from video and save to dataset."""
        print(f"\nExtracting {split_name} set...")

        # Group by video to minimize video reopening
        for video_path, group in split_df.groupby('video_path'):
            if not os.path.exists(video_path):
                print(f"  WARNING: Video not found: {video_path}")
                print(f"    Skipping {len(group)} labels from this video")
                continue

            print(f"  Processing {Path(video_path).name} ({len(group)} frames)...")

            # Create extractor for this video
            try:
                extractor = VideoFrameExtractor(video_path, target_fps=5)
            except Exception as e:
                print(f"    ERROR: Failed to open video: {e}")
                continue

            # Extract each labeled frame
            for _, row in group.iterrows():
                try:
                    # Extract frame at timestamp
                    timestamp = row['timestamp']
                    frame = extractor.get_frame_at_timestamp(timestamp)

                    # Convert internal label to display label for directory name
                    internal_label = row['label']  # 'true' or 'false'
                    display_label = internal_to_display(internal_label, task_config)

                    # Use timestamp-based filename for uniqueness
                    filename = f"{Path(video_path).stem}_frame_{int(timestamp*1000):08d}.jpg"
                    dst = os.path.join(output_dir, split_name, display_label, filename)

                    # Write frame
                    cv2.imwrite(dst, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                except Exception as e:
                    print(f"    WARNING: Failed to extract frame at {timestamp}s: {e}")
                    continue

    extract_and_save_split(train_df, 'train')
    extract_and_save_split(val_df, 'val')
    extract_and_save_split(test_df, 'test')

    # Save split metadata
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

    for split_name, split_df in splits.items():
        csv_path = os.path.join(output_dir, f'{split_name}.csv')
        split_df.to_csv(csv_path, index=False)
        print(f"Saved {split_name} metadata to: {csv_path}")

    print(f"\nDataset prepared successfully!")
    print(f"Output directory: {output_dir}")
    print(f"\nDirectory structure:")
    print(f"  {output_dir}/")
    print(f"    train/")
    print(f"      {pos_label}/")
    print(f"      {neg_label}/")
    print(f"    val/")
    print(f"      {pos_label}/")
    print(f"      {neg_label}/")
    print(f"    test/")
    print(f"      {pos_label}/")
    print(f"      {neg_label}/")


def main():
    parser = argparse.ArgumentParser(description='Prepare labeled dataset')
    parser.add_argument('--labels', required=True, help='Path to labels CSV')
    parser.add_argument('--output', help='Output directory (default: data/dataset)')
    parser.add_argument('--config', default='config.yaml', help='Config file path')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Determine output directory
    output_dir = args.output if args.output else config['paths']['dataset']

    # Prepare dataset
    prepare_dataset(args.labels, output_dir, config)


if __name__ == '__main__':
    main()

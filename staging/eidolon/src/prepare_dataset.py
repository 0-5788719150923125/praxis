#!/usr/bin/env python3
"""
Prepare labeled frames for training by splitting into train/val/test sets.

Usage:
    python src/prepare_dataset.py --labels data/labels.csv
"""

import os
import shutil
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from utils import load_config, ensure_dir


def prepare_dataset(labels_csv: str, output_dir: str, config: dict):
    """
    Split labeled data into train/val/test sets.

    Args:
        labels_csv: Path to labels CSV
        output_dir: Output directory for dataset
        config: Configuration dictionary
    """
    # Load labels
    print(f"Loading labels from: {labels_csv}")
    df = pd.read_csv(labels_csv)

    # Remove skipped frames (only keep labeled ones)
    df = df[df['label'].isin(['touching', 'not_touching'])]

    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"\nClass distribution:")
    print(df['label'].value_counts())
    print(df['label'].value_counts(normalize=True))

    # Check class balance
    class_counts = df['label'].value_counts()
    if len(class_counts) < 2:
        raise ValueError("Need at least 2 classes (touching and not_touching)")

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

    # Create directory structure
    for split in ['train', 'val', 'test']:
        for label in ['touching', 'not_touching']:
            split_dir = os.path.join(output_dir, split, label)
            ensure_dir(split_dir)

    # Copy files to appropriate directories
    def copy_split(split_df, split_name):
        print(f"\nCopying {split_name} set...")
        for _, row in split_df.iterrows():
            src = row['frame_path']
            label = row['label']
            filename = os.path.basename(src)
            dst = os.path.join(output_dir, split_name, label, filename)

            if not os.path.exists(src):
                print(f"Warning: Source file not found: {src}")
                continue

            shutil.copy2(src, dst)

    copy_split(train_df, 'train')
    copy_split(val_df, 'val')
    copy_split(test_df, 'test')

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
    print(f"      touching/")
    print(f"      not_touching/")
    print(f"    val/")
    print(f"      touching/")
    print(f"      not_touching/")
    print(f"    test/")
    print(f"      touching/")
    print(f"      not_touching/")


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

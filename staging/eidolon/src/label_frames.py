#!/usr/bin/env python3
"""
Simple GUI tool for manually labeling extracted frames.

Controls:
    T - Mark as True (positive class - touching nose)
    F - Mark as False (negative class - not touching)
    S - Skip (unsure/ambiguous)
    Q - Quit and save
    Left Arrow - Go back to previous frame
    Right Arrow - Skip to next frame

Usage:
    python src/label_frames.py --frames data/frames/my_video
"""

import os
import cv2
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from utils import load_json, ensure_dir


class FrameLabeler:
    """Interactive frame labeling tool."""

    def __init__(self, frames_dir: str, labels_csv: str):
        self.frames_dir = frames_dir
        self.labels_csv = labels_csv

        # Load frame metadata
        metadata_path = os.path.join(frames_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        self.metadata = load_json(metadata_path)

        # Get all frame files
        self.frames = sorted(glob(os.path.join(frames_dir, 'frame_*.jpg')))
        if not self.frames:
            raise FileNotFoundError(f"No frames found in: {frames_dir}")

        print(f"Found {len(self.frames)} frames")

        # Load existing labels if present
        self.labels = {}
        if os.path.exists(labels_csv):
            df = pd.read_csv(labels_csv)
            for _, row in df.iterrows():
                self.labels[row['frame_path']] = row['label']
            print(f"Loaded {len(self.labels)} existing labels")

        self.current_idx = 0
        self.window_name = "Frame Labeler"

        # Timeline scrubber state
        self.timeline_height = 30
        self.timeline_margin = 10
        self.dragging = False

        # Find first unlabeled frame
        for i, frame_path in enumerate(self.frames):
            if frame_path not in self.labels:
                self.current_idx = i
                break

    def get_label_stats(self):
        """Calculate current label distribution."""
        touching = sum(1 for label in self.labels.values() if label == 'touching')
        not_touching = sum(1 for label in self.labels.values() if label == 'not_touching')
        total_labeled = touching + not_touching

        if total_labeled == 0:
            return touching, not_touching, 0, 0, 0.0

        touching_pct = (touching / total_labeled) * 100
        not_touching_pct = (not_touching / total_labeled) * 100

        return touching, not_touching, total_labeled, touching_pct, not_touching_pct

    def get_current_frame_info(self):
        """Get information about current frame."""
        frame_path = self.frames[self.current_idx]
        frame_filename = os.path.basename(frame_path)

        # Find metadata for this frame
        frame_meta = None
        for f in self.metadata['frames']:
            if f['filename'] == frame_filename:
                frame_meta = f
                break

        return frame_path, frame_meta

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for timeline scrubbing."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is in timeline area
            if hasattr(self, 'timeline_y_start') and hasattr(self, 'timeline_y_end'):
                if self.timeline_y_start <= y <= self.timeline_y_end:
                    self.dragging = True
                    self.seek_to_position(x)
                    self.display_frame()  # Immediately show the new frame
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.seek_to_position(x)
            self.display_frame()  # Update display while dragging

    def seek_to_position(self, x):
        """Seek to a frame based on x position in timeline."""
        if not hasattr(self, 'timeline_x_start') or not hasattr(self, 'timeline_width'):
            return

        # Calculate relative position (0.0 to 1.0)
        relative_x = (x - self.timeline_x_start) / self.timeline_width
        relative_x = max(0.0, min(1.0, relative_x))  # Clamp to [0, 1]

        # Calculate target frame index
        target_idx = int(relative_x * (len(self.frames) - 1))
        self.current_idx = target_idx

    def display_frame(self):
        """Display current frame with annotations."""
        frame_path, frame_meta = self.get_current_frame_info()

        # Load image
        img = cv2.imread(frame_path)
        if img is None:
            print(f"Error loading: {frame_path}")
            return

        # Resize for display if too large
        h, w = img.shape[:2]
        max_size = 1200
        if w > max_size or h > max_size:
            scale = max_size / max(w, h)
            img = cv2.resize(img, None, fx=scale, fy=scale)

        # Store original dimensions before adding timeline
        original_h, original_w = img.shape[:2]

        # Get current label and stats
        label = self.labels.get(frame_path, "UNLABELED")
        timestamp = frame_meta['timestamp'] if frame_meta else 0.0

        # Format timestamp as MM:SS
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        timestamp_str = f"{minutes}:{seconds:02d}"

        # Get label statistics
        touching, not_touching, total, touching_pct, not_touching_pct = self.get_label_stats()

        # Determine balance status and color
        balance_status = "BALANCED"
        balance_color = (0, 255, 0)  # Green
        if total >= 10:  # Only show imbalance after 10+ labels
            ratio = max(touching, not_touching) / max(min(touching, not_touching), 1)
            if ratio > 2.0:
                balance_status = "IMBALANCED"
                balance_color = (0, 165, 255)  # Orange
            elif ratio > 3.0:
                balance_status = "VERY IMBALANCED"
                balance_color = (0, 0, 255)  # Red

        # Draw info
        info_lines = [
            f"Frame: {self.current_idx + 1}/{len(self.frames)}",
            f"Time: {timestamp_str}",
            f"Current label: {label}",
            "",
            f"Total labeled: {total}",
            f"  True (touching): {touching} ({touching_pct:.1f}%)",
            f"  False (not touching): {not_touching} ({not_touching_pct:.1f}%)",
            f"  Balance: {balance_status}",
            "",
            "T=true | F=false | S=skip | Q=quit",
            "Arrows=navigate | Click timeline to jump"
        ]

        y_offset = 30
        for i, line in enumerate(info_lines):
            # Use balance color for the balance status line
            if "Balance:" in line:
                color = balance_color
            else:
                color = (0, 255, 0)

            cv2.putText(img, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, color, 2)
            y_offset += 25

        # Add timeline scrubber at bottom
        timeline_bar_height = self.timeline_height
        padding = self.timeline_margin

        # Expand image to add timeline area
        new_h = original_h + timeline_bar_height + padding * 2
        timeline_img = np.zeros((new_h, original_w, 3), dtype=np.uint8)
        timeline_img[0:original_h, :] = img

        # Draw timeline background (dark gray)
        timeline_y = original_h + padding
        cv2.rectangle(timeline_img,
                     (padding, timeline_y),
                     (original_w - padding, timeline_y + timeline_bar_height),
                     (50, 50, 50), -1)

        # Draw timeline border
        cv2.rectangle(timeline_img,
                     (padding, timeline_y),
                     (original_w - padding, timeline_y + timeline_bar_height),
                     (200, 200, 200), 1)

        # Draw progress bar (represents labeled frames)
        if len(self.frames) > 0:
            labeled_ratio = len(self.labels) / len(self.frames)
            progress_width = int((original_w - 2 * padding) * labeled_ratio)
            if progress_width > 0:
                cv2.rectangle(timeline_img,
                            (padding, timeline_y),
                            (padding + progress_width, timeline_y + timeline_bar_height),
                            (40, 100, 40), -1)

        # Draw current position marker
        timeline_width = original_w - 2 * padding
        position_ratio = self.current_idx / max(len(self.frames) - 1, 1)
        marker_x = padding + int(timeline_width * position_ratio)

        # Draw marker as a vertical line
        cv2.line(timeline_img,
                (marker_x, timeline_y),
                (marker_x, timeline_y + timeline_bar_height),
                (0, 255, 255), 3)

        # Store timeline coordinates for mouse detection
        self.timeline_x_start = padding
        self.timeline_width = timeline_width
        self.timeline_y_start = timeline_y
        self.timeline_y_end = timeline_y + timeline_bar_height

        cv2.imshow(self.window_name, timeline_img)

    def save_labels(self):
        """Save labels to CSV."""
        if not self.labels:
            print("No labels to save")
            return

        # Prepare data
        data = []
        for frame_path, label in self.labels.items():
            frame_filename = os.path.basename(frame_path)

            # Find timestamp
            timestamp = None
            for f in self.metadata['frames']:
                if f['filename'] == frame_filename:
                    timestamp = f['timestamp']
                    break

            data.append({
                'frame_path': frame_path,
                'filename': frame_filename,
                'label': label,
                'timestamp': timestamp
            })

        # Create DataFrame and save
        df = pd.DataFrame(data)

        # Ensure output directory exists
        ensure_dir(os.path.dirname(self.labels_csv))

        df.to_csv(self.labels_csv, index=False)
        print(f"\nSaved {len(data)} labels to: {self.labels_csv}")

        # Print label distribution
        print("\nLabel distribution:")
        print(df['label'].value_counts())

    def run(self):
        """Run interactive labeling session."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n=== Frame Labeling Tool ===")
        print("Controls:")
        print("  T - Mark as True (positive class)")
        print("  F - Mark as False (negative class)")
        print("  S - Skip (unsure)")
        print("  Left Arrow - Previous frame")
        print("  Right Arrow - Next frame")
        print("  Q - Quit and save")
        print("  Timeline - Click or drag to jump to any frame")
        print()
        print("Tip: Aim for roughly 50/50 balance between true and false")
        print("     Balance status shown in display window")
        print()

        while True:
            self.display_frame()

            # Use waitKeyEx to get proper extended key codes for arrow keys
            key = cv2.waitKeyEx(0)

            frame_path, _ = self.get_current_frame_info()

            # IMPORTANT: Check arrow keys FIRST before checking character keys
            # Arrow keys on Linux/X11: Left=65361, Right=65363, Up=65362, Down=65364
            # When masked to 8-bit, these become Q, S, R, T which would trigger wrong actions!

            # Arrow keys (check BEFORE character keys)
            if key == 65361 or key == 2424832:  # Left arrow (X11 or other Linux)
                self.current_idx = max(self.current_idx - 1, 0)
                print(f"[{self.current_idx + 1}] <-- Previous frame")

            elif key == 65363 or key == 2555904:  # Right arrow (X11 or other Linux)
                self.current_idx = min(self.current_idx + 1, len(self.frames) - 1)
                print(f"[{self.current_idx + 1}] --> Next frame")

            elif key == 65362:  # Up arrow (X11) - currently unused
                pass

            elif key == 65364:  # Down arrow (X11) - currently unused
                pass

            # Now check character keys (letters, numbers, etc.)
            elif key == ord('q') or key == ord('Q'):
                # Quit
                self.save_labels()
                break

            elif key == ord('t') or key == ord('T'):
                # Label as touching (True - positive class)
                self.labels[frame_path] = 'touching'
                touching, not_touching, total, _, _ = self.get_label_stats()
                print(f"[{self.current_idx + 1}] TRUE (Balance: {touching}/{not_touching})")
                self.current_idx = min(self.current_idx + 1, len(self.frames) - 1)

            elif key == ord('f') or key == ord('F'):
                # Label as not touching (False - negative class)
                self.labels[frame_path] = 'not_touching'
                touching, not_touching, total, _, _ = self.get_label_stats()
                print(f"[{self.current_idx + 1}] FALSE (Balance: {touching}/{not_touching})")
                self.current_idx = min(self.current_idx + 1, len(self.frames) - 1)

            elif key == ord('s') or key == ord('S'):
                # Skip
                print(f"[{self.current_idx + 1}] SKIPPED")
                self.current_idx = min(self.current_idx + 1, len(self.frames) - 1)

            else:
                # Unknown key - helpful for debugging
                key_char = key & 0xFF
                if key > 255:  # Extended key we don't recognize
                    print(f"Unknown extended key: {key} (0x{key:X})")
                # Ignore other character keys silently

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Label extracted frames')
    parser.add_argument('--frames', required=True, help='Directory containing extracted frames')
    parser.add_argument('--output', help='Output CSV path (default: data/labels.csv)')

    args = parser.parse_args()

    # Determine output path
    output_csv = args.output if args.output else 'data/labels.csv'

    # Run labeler
    labeler = FrameLabeler(args.frames, output_csv)
    labeler.run()


if __name__ == '__main__':
    main()

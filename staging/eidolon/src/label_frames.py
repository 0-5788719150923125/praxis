#!/usr/bin/env python3
"""
Simple GUI tool for manually labeling video frames using mpv player.

Controls:
    T - Mark current frame as True (touching nose)
    F - Mark current frame as False (not touching)
    S - Save and quit
    ← / → Arrow Keys - Skip 3 frames (optimized scrubbing)
    , / . - Step backward/forward one frame (precise)
    Space - Pause/Play
    Mouse - Seek by clicking on timeline
    I - Refresh overlay display

Usage:
    python src/label_frames.py --video videos/my_video.mp4
"""

import os
import argparse
import pandas as pd
import tempfile
from pathlib import Path
from utils import ensure_dir
from video_frame_extractor import VideoFrameExtractor

try:
    import mpv
except ImportError:
    print("ERROR: python-mpv is not installed.")
    print("Please run: pip install python-mpv")
    print("Also ensure libmpv is installed: sudo pacman -S mpv")
    exit(1)


class FrameLabeler:
    """Interactive frame labeling tool using mpv player."""

    def __init__(self, video_path: str, labels_csv: str, target_fps: int = 5):
        self.video_path = os.path.abspath(video_path)
        self.labels_csv = labels_csv
        self.target_fps = target_fps

        # Create video frame extractor for on-demand single frame extraction
        self.extractor = VideoFrameExtractor(video_path, target_fps)
        self.video_info = self.extractor.get_video_metadata()

        print(f"Video: {Path(video_path).name}")
        print(f"Duration: {self.video_info['duration']:.1f}s")
        print(f"Resolution: {self.video_info['width']}x{self.video_info['height']}")
        print(f"FPS: {self.video_info['fps']:.2f}")

        # Load existing labels: {timestamp: label}
        # Note: We now store by timestamp only (not frame_idx) for simplicity
        self.labels = {}
        if os.path.exists(labels_csv):
            df = pd.read_csv(labels_csv)

            # Detect schema version
            if 'video_path' in df.columns:
                # NEW schema
                for _, row in df.iterrows():
                    if row['video_path'] == self.video_path:
                        timestamp = float(row['timestamp'])
                        self.labels[timestamp] = row['label']
            else:
                # OLD schema (backward compatibility)
                video_name = Path(self.video_path).stem
                for _, row in df.iterrows():
                    if video_name in row['frame_path']:
                        timestamp = float(row['timestamp'])
                        self.labels[timestamp] = row['label']

            print(f"Loaded {len(self.labels)} existing labels")

        # Get video FPS for frame skip information
        self.extractor = VideoFrameExtractor(video_path, target_fps)
        self.video_info = self.extractor.get_video_metadata()
        source_fps = self.video_info['fps']

        # Configure frame stepping (3 frames per arrow key press)
        # Reduced from 5 to minimize seeking lag while still being faster than 1-frame
        self.frames_to_skip = 3
        skip_time = self.frames_to_skip / source_fps

        # Create custom input configuration for arrow keys
        # Must use exact seeking for frame precision, optimized with demuxer cache
        input_conf_content = f"""# Custom key bindings for video labeling
# Exact seeking for frame precision - {self.frames_to_skip} frames per press
# Some lag is unavoidable with exact seeking in H.264/H.265 codecs
RIGHT repeatable no-osd seek {skip_time:.6f} exact
LEFT repeatable no-osd seek -{skip_time:.6f} exact
"""
        # Create temporary input.conf file
        self.input_conf_fd, self.input_conf_path = tempfile.mkstemp(suffix='.conf', text=True)
        with os.fdopen(self.input_conf_fd, 'w') as f:
            f.write(input_conf_content)

        # Initialize mpv player with custom input configuration
        self.player = mpv.MPV(
            input_default_bindings=True,  # Enable default mpv controls
            input_conf=self.input_conf_path,  # Use our custom arrow key bindings
            input_vo_keyboard=True,        # Enable keyboard input
            osc=True,                      # Show on-screen controller
            keep_open=True,                # Don't close when video ends
            pause=True,                    # Start paused for labeling
            video_sync='display-vdrop',    # Smooth playback
            audio='no',                    # Disable audio (labeling is visual only)
            demuxer_seekable_cache='yes',  # Cache seeking for smoother exact seeks
            cache='yes',                   # Enable cache
            demuxer_max_bytes='150M',      # Larger cache for better seek performance
        )

        # Setup custom keyboard bindings for labeling
        self.setup_keyboard_controls()

        # Setup property observers for persistent overlay updates
        @self.player.property_observer('time-pos')
        def on_time_change(_name, value):
            """Update overlay when playback position changes."""
            if value is not None:
                self.update_overlay()

        @self.player.property_observer('pause')
        def on_pause_change(_name, value):
            """Update overlay when pause state changes."""
            self.update_overlay()

        # Load video
        print("\nLoading video in mpv player...")
        self.player.play(self.video_path)

        try:
            self.player.wait_until_playing()
            print("✓ Video loaded")
            print(f"✓ Arrow keys configured: skip {self.frames_to_skip} frames per press (with caching)")
        except mpv.ShutdownError:
            print("\nmpv was closed before video loaded. Exiting.")
            self._cleanup_temp_file()
            raise SystemExit(0)

        # Show initial overlay
        self.update_overlay()

    def _cleanup_temp_file(self):
        """Clean up temporary input.conf file."""
        try:
            if hasattr(self, 'input_conf_path') and os.path.exists(self.input_conf_path):
                os.unlink(self.input_conf_path)
        except Exception:
            pass

    def setup_keyboard_controls(self):
        """Setup custom keyboard controls for labeling."""

        @self.player.on_key_press('t')
        def label_touching():
            """Mark current frame as touching (positive class)."""
            timestamp = self.get_current_timestamp()
            self.labels[timestamp] = 'touching'

            touching, not_touching = self.get_label_stats()
            print(f"✓ Labeled TOUCHING @ {timestamp:.2f}s (Balance: {touching}/{not_touching})")

            # Update overlay to reflect new label
            self.update_overlay()

            # Show brief confirmation
            self.player.show_text(
                f"✓ TOUCHING @ {self.format_timestamp(timestamp)}",
                duration=1500
            )

        @self.player.on_key_press('T')
        def label_touching_upper():
            """Same as lowercase t."""
            label_touching()

        @self.player.on_key_press('f')
        def label_not_touching():
            """Mark current frame as not touching (negative class)."""
            timestamp = self.get_current_timestamp()
            self.labels[timestamp] = 'not_touching'

            touching, not_touching = self.get_label_stats()
            print(f"✓ Labeled NOT TOUCHING @ {timestamp:.2f}s (Balance: {touching}/{not_touching})")

            # Update overlay to reflect new label
            self.update_overlay()

            # Show brief confirmation
            self.player.show_text(
                f"✓ NOT TOUCHING @ {self.format_timestamp(timestamp)}",
                duration=1500
            )

        @self.player.on_key_press('F')
        def label_not_touching_upper():
            """Same as lowercase f."""
            label_not_touching()

        @self.player.on_key_press('s')
        def save_and_quit():
            """Save and quit."""
            print("\nSaving and quitting...")
            self.save_labels()
            self.player.quit()

        @self.player.on_key_press('S')
        def save_and_quit_upper():
            """Same as lowercase s."""
            save_and_quit()

        @self.player.on_key_press('i')
        def show_stats():
            """Refresh overlay display."""
            self.update_overlay()
            print("Overlay refreshed")

        @self.player.on_key_press('I')
        def show_stats_upper():
            """Same as lowercase i."""
            show_stats()

    def get_current_timestamp(self):
        """Get current playback position in seconds."""
        return self.player.time_pos or 0.0

    def format_timestamp(self, timestamp):
        """Format timestamp as MM:SS."""
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        return f"{minutes}:{seconds:02d}"

    def frame_step_forward(self):
        """Step forward one frame."""
        try:
            self.player.command('frame-step')
            self.update_overlay()
            print(f"→ Frame forward @ {self.get_current_timestamp():.2f}s")
        except (mpv.ShutdownError, OSError):
            # Player was closed, ignore
            pass

    def frame_step_backward(self):
        """Step backward one frame."""
        try:
            self.player.command('frame-back-step')
            self.update_overlay()
            print(f"← Frame backward @ {self.get_current_timestamp():.2f}s")
        except (mpv.ShutdownError, OSError):
            # Player was closed, ignore
            pass

    def update_overlay(self):
        """Update persistent overlay with controls and stats."""
        try:
            timestamp = self.get_current_timestamp()
            touching, not_touching = self.get_label_stats()
            total = touching + not_touching

            # Check if current timestamp is labeled
            # Round to 2 decimals for comparison
            current_label = None
            for ts, label in self.labels.items():
                if abs(ts - timestamp) < 0.1:  # Within 0.1 seconds
                    current_label = label
                    break

            # Format current label status
            if current_label:
                label_status = f"Current: {current_label.upper()}"
            else:
                label_status = "Current: UNLABELED"

            # Calculate balance status
            if total == 0:
                balance_text = "No labels yet"
            else:
                touching_pct = (touching / total) * 100
                not_touching_pct = (not_touching / total) * 100
                ratio = max(touching, not_touching) / max(min(touching, not_touching), 1)

                if ratio <= 2.0:
                    balance_status = "BALANCED ✓"
                elif ratio <= 3.0:
                    balance_status = "IMBALANCED"
                else:
                    balance_status = "VERY IMBALANCED"

                balance_text = (
                    f"Total: {total} | Touching: {touching} ({touching_pct:.1f}%) | "
                    f"Not touching: {not_touching} ({not_touching_pct:.1f}%)\n"
                    f"Balance: {balance_status}"
                )

            # Build overlay text
            overlay_text = (
                f"=== Frame Labeling Tool ===\n"
                f"Time: {self.format_timestamp(timestamp)} | {label_status}\n"
                f"{balance_text}\n"
                f"\n"
                f"T=touching | F=not touching | S=save & quit\n"
                f"←/→=skip 3 frames | ,/.=step 1 frame | Space=pause/play"
            )

            # Show persistent overlay (10 minute duration = effectively permanent)
            self.player.show_text(overlay_text, duration=600000)
        except (mpv.ShutdownError, OSError):
            # Player was closed, ignore
            pass

    def get_label_stats(self):
        """Calculate current label distribution."""
        touching = sum(1 for label in self.labels.values() if label == 'touching')
        not_touching = sum(1 for label in self.labels.values() if label == 'not_touching')
        return touching, not_touching

    def save_labels(self):
        """Save labels to CSV, preserving labels from other videos."""
        if not self.labels:
            print("No labels to save")
            return

        # Load existing labels from CSV (all videos)
        existing_labels = []
        if os.path.exists(self.labels_csv):
            try:
                existing_df = pd.read_csv(self.labels_csv)

                # Keep labels from OTHER videos
                if 'video_path' in existing_df.columns:
                    # NEW schema - filter by video_path
                    other_videos_df = existing_df[existing_df['video_path'] != self.video_path]
                    existing_labels = other_videos_df.to_dict('records')
                else:
                    # OLD schema - filter by video name in frame_path
                    video_name = Path(self.video_path).stem
                    other_videos_df = existing_df[~existing_df['frame_path'].str.contains(video_name, regex=False)]
                    existing_labels = other_videos_df.to_dict('records')
            except Exception as e:
                print(f"Warning: Could not load existing labels: {e}")

        # Prepare data for current video in new schema format
        current_video_labels = []
        for timestamp, label in self.labels.items():
            # Calculate frame index for backward compatibility
            frame_idx = int(timestamp * self.target_fps)

            current_video_labels.append({
                'video_path': self.video_path,
                'frame_index': frame_idx,
                'label': label,
                'timestamp': timestamp
            })

        # Combine: labels from other videos + labels from current video
        all_labels = existing_labels + current_video_labels

        # Create DataFrame and save
        df = pd.DataFrame(all_labels)

        # Ensure output directory exists
        ensure_dir(os.path.dirname(self.labels_csv))

        df.to_csv(self.labels_csv, index=False)
        print(f"\n✓ Saved {len(current_video_labels)} labels for this video")
        print(f"✓ Total labels in file: {len(all_labels)} (across all videos)")

        # Print label distribution for current video
        current_df = pd.DataFrame(current_video_labels)
        print("\nLabel distribution for this video:")
        print(current_df['label'].value_counts())

    def run(self):
        """Run interactive labeling session."""
        print("\n=== Frame Labeling Tool (mpv) ===")
        print("Controls:")
        print("  T - Mark as TOUCHING (positive class)")
        print("  F - Mark as NOT TOUCHING (negative class)")
        print("  S - Save and quit")
        print("  I - Refresh overlay (updates automatically)")
        print("  Space - Pause/Play")
        print(f"  ← / → Arrow Keys - Skip {self.frames_to_skip} frames (optimized scrubbing)")
        print("  , / . - Step backward/forward one frame (precise)")
        print("  Left Click Timeline - Seek to position")
        print()
        print("Tip: Aim for roughly 50/50 balance between touching and not touching")
        print("The overlay shows stats automatically and updates as you label.")
        print()
        print("mpv player is now running. Press S when done to save and quit.")
        print()

        try:
            # Wait for player to close (blocks until user quits)
            self.player.wait_for_shutdown()
        finally:
            # Clean up temporary input.conf file
            self._cleanup_temp_file()


def main():
    parser = argparse.ArgumentParser(description='Label video frames using mpv player')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--output', help='Output CSV path (default: data/labels.csv)')
    parser.add_argument('--fps', type=int, default=5, help='Target FPS for frame indexing (default: 5)')

    args = parser.parse_args()

    # Determine output path
    output_csv = args.output if args.output else 'data/labels.csv'

    # Run labeler
    try:
        labeler = FrameLabeler(args.video, output_csv, args.fps)
        labeler.run()
    except SystemExit as e:
        # Clean exit (e.g., user closed window)
        exit(e.code if hasattr(e, 'code') else 0)
    except mpv.ShutdownError:
        # User closed mpv window - exit cleanly
        print("\nmpv window closed. Exiting without saving.")
        exit(0)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting without saving.")
        exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Simple GUI tool for manually labeling video frames using mpv player.

Controls:
    T - Mark current frame as True (positive class)
    F - Mark current frame as False (negative class)
    S - Save and quit
    → Arrow Key (click) - Skip forward 3 frames
    → Arrow Key (hold) - Play forward at 2x speed
    ← Arrow Key - Seek backward 3 frames (hold for continuous)
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
import threading
import time
from pathlib import Path
from utils import (
    ensure_dir,
    load_task_config,
    internal_to_display,
    display_to_internal,
    migrate_labels_to_internal
)
from video_frame_extractor import VideoFrameExtractor

try:
    import mpv
except ImportError:
    print("ERROR: python-mpv is not installed.")
    print("Please run: pip install python-mpv")
    print("Also ensure libmpv is installed: sudo pacman -S mpv")
    exit(1)

try:
    from pynput import keyboard
    from pynput.keyboard import Controller, Key
except ImportError:
    print("ERROR: pynput is not installed.")
    print("Please run: pip install pynput")
    exit(1)


class FrameLabeler:
    """Interactive frame labeling tool using mpv player."""

    def __init__(self, video_path: str, labels_csv: str, target_fps: int = 5,
                 debug: bool = False, task_config: dict = None):
        self.video_path = os.path.abspath(video_path)
        self.labels_csv = labels_csv
        self.target_fps = target_fps
        self.debug = debug

        # Load task configuration
        self.task_config = task_config or load_task_config()
        self.pos_label = self.task_config['labels']['positive']['display_name']
        self.neg_label = self.task_config['labels']['negative']['display_name']

        # Create video frame extractor for on-demand single frame extraction
        self.extractor = VideoFrameExtractor(video_path, target_fps)
        self.video_info = self.extractor.get_video_metadata()

        print(f"Video: {Path(video_path).name}")
        print(f"Duration: {self.video_info['duration']:.1f}s")
        print(f"Resolution: {self.video_info['width']}x{self.video_info['height']}")
        print(f"FPS: {self.video_info['fps']:.2f}")

        # Load existing labels: {timestamp: label}
        # Note: Labels stored internally as 'true'/'false' (task-agnostic)
        self.labels = {}
        if os.path.exists(labels_csv):
            df = pd.read_csv(labels_csv)

            # Migrate labels to internal format if needed
            df = migrate_labels_to_internal(df, backup=False)

            # Detect schema version
            # Check if video_path column exists AND has non-empty values
            if 'video_path' in df.columns and df['video_path'].notna().any():
                # NEW schema with populated video_path
                for _, row in df.iterrows():
                    # Handle both empty and populated video_path
                    if pd.notna(row['video_path']) and row['video_path'] == self.video_path:
                        timestamp = float(row['timestamp'])
                        # Labels already in internal format after migration
                        self.labels[timestamp] = row['label']

            # If no video_path data found, fall back to OLD schema (frame_path matching)
            if len(self.labels) == 0 and 'frame_path' in df.columns:
                video_name = Path(self.video_path).stem
                if self.debug:
                    print(f"DEBUG: Looking for video_name '{video_name}' in frame_path")
                    print(f"DEBUG: Sample frame_path: {df['frame_path'].iloc[0] if len(df) > 0 else 'N/A'}")

                for _, row in df.iterrows():
                    # Skip rows with NaN/empty frame_path
                    if pd.notna(row['frame_path']) and isinstance(row['frame_path'], str):
                        if video_name in row['frame_path']:
                            timestamp = float(row['timestamp'])
                            # Labels already in internal format after migration
                            self.labels[timestamp] = row['label']

            if len(self.labels) > 0:
                print(f"✓ Loaded {len(self.labels)} existing labels for this video")
            else:
                print(f"No existing labels found for this video")

        # Get video FPS for frame skip information
        self.extractor = VideoFrameExtractor(video_path, target_fps)
        self.video_info = self.extractor.get_video_metadata()
        source_fps = self.video_info['fps']

        # Configure playback speed for hold-to-scrub
        self.playback_speed = 2.0  # 2x speed for right arrow scrubbing

        # Click vs hold detection timing
        self.HOLD_THRESHOLD = 0.20  # 200ms - if held longer, it's a hold
        self.FRAMES_TO_SKIP = 3     # frames to skip on click

        # Calculate seek times (3 frames each direction)
        self.backward_frames = 3
        backward_seek_time = self.backward_frames / source_fps
        forward_seek_time = self.FRAMES_TO_SKIP / source_fps

        # Create custom input configuration for arrow keys
        # LEFT uses repeatable for continuous backward seeking
        # RIGHT uses single seek - hold detection switches to 2x playback mode
        input_conf_content = f"""# Custom key bindings for video labeling
# Right arrow: single forward seek - Python detects hold for 2x playback
RIGHT seek {forward_seek_time:.6f} exact
# Left arrow: backward seek (repeatable for continuous seeking)
LEFT repeatable seek -{backward_seek_time:.6f} exact
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

        # Pre-define the hold-mode input section (RIGHT=ignore) so we can quickly enable it
        # This overrides the repeatable seek when we want smooth 2x playback
        self.player.command('define-section', 'hold-mode', 'RIGHT ignore\n', 'force')

        # Setup custom keyboard bindings for labeling
        self.setup_keyboard_controls()

        # Setup keyboard listener for hold-to-scrub functionality
        self.setup_hold_to_scrub_listener()

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
            print(f"✓ Arrow keys: → click=skip {self.FRAMES_TO_SKIP} frames, hold=play {self.playback_speed}x | ← seek back {self.backward_frames} frames")
            if self.debug:
                print("✓ Debug mode enabled: timing diagnostics will be printed")
        except mpv.ShutdownError:
            print("\nmpv was closed before video loaded. Exiting.")
            self._cleanup_temp_file()
            raise SystemExit(0)

        # Show initial overlay
        self.update_overlay()

    def _cleanup_temp_file(self):
        """Clean up temporary input.conf file, keyboard listener, and timers."""
        # Cancel any pending hold timers
        try:
            for timer in getattr(self, 'hold_timers', {}).values():
                timer.cancel()
        except Exception:
            pass

        try:
            if hasattr(self, 'input_conf_path') and os.path.exists(self.input_conf_path):
                os.unlink(self.input_conf_path)
        except Exception:
            pass

        # Stop keyboard listener
        try:
            if hasattr(self, 'keyboard_listener'):
                self.keyboard_listener.stop()
        except Exception:
            pass

    def setup_keyboard_controls(self):
        """Setup custom keyboard controls for labeling."""

        @self.player.on_key_press('t')
        def label_positive():
            """Mark current frame as positive class."""
            timestamp = self.get_current_timestamp()
            self.labels[timestamp] = 'true'  # Store internal label

            positive_count, negative_count = self.get_label_stats()
            print(f"✓ Labeled {self.pos_label.upper()} @ {timestamp:.2f}s (Balance: {positive_count}/{negative_count})")

            # Update overlay to reflect new label
            self.update_overlay()

            # Show brief confirmation
            self.player.show_text(
                f"✓ {self.pos_label.upper()} @ {self.format_timestamp(timestamp)}",
                duration=1500
            )

        @self.player.on_key_press('T')
        def label_positive_upper():
            """Same as lowercase t."""
            label_positive()

        @self.player.on_key_press('f')
        def label_negative():
            """Mark current frame as negative class."""
            timestamp = self.get_current_timestamp()
            self.labels[timestamp] = 'false'  # Store internal label

            positive_count, negative_count = self.get_label_stats()
            print(f"✓ Labeled {self.neg_label.upper()} @ {timestamp:.2f}s (Balance: {positive_count}/{negative_count})")

            # Update overlay to reflect new label
            self.update_overlay()

            # Show brief confirmation
            self.player.show_text(
                f"✓ {self.neg_label.upper()} @ {self.format_timestamp(timestamp)}",
                duration=1500
            )

        @self.player.on_key_press('F')
        def label_negative_upper():
            """Same as lowercase f."""
            label_negative()

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

    def setup_hold_to_scrub_listener(self):
        """Setup hold-to-scrub using timer-based click vs hold detection."""
        # State tracking for click vs hold detection
        self.key_state_lock = threading.Lock()  # Protect against race conditions
        self.pressed_keys = set()      # Track physically pressed keys (filter auto-repeat)
        self.hold_timers = {}          # Pending hold detection timers
        self.is_holding = {}           # Whether key transitioned to hold state
        self.last_release_time = 0     # Track last release to detect click sequences
        self.CLICK_SEQUENCE_GAP = 0.4  # If press comes within 400ms of release, it's a click sequence

        def on_press(key):
            """Track key press timing for hold detection (mpv handles seeking)."""
            if key != keyboard.Key.right:
                return

            with self.key_state_lock:
                if key in self.pressed_keys:
                    return  # Filter auto-repeat events

                self.pressed_keys.add(key)
                self.is_holding[key] = False

                # Check if this press is part of a click sequence (recent release)
                time_since_release = time.time() - self.last_release_time
                in_click_sequence = time_since_release < self.CLICK_SEQUENCE_GAP

                # Only start hold timer if NOT in a click sequence
                if not in_click_sequence:
                    timer = threading.Timer(self.HOLD_THRESHOLD, self._on_hold_threshold, args=[key])
                    self.hold_timers[key] = timer
                    timer.start()
                    if self.debug:
                        print(f"[DEBUG] Hold timer started ({self.HOLD_THRESHOLD}s)")
                elif self.debug:
                    print(f"[DEBUG] Click sequence - hold suppressed")

        def on_release(key):
            """Handle key release - record time, pause if was holding."""
            if key != keyboard.Key.right:
                return

            with self.key_state_lock:
                if key not in self.pressed_keys:
                    return

                self.pressed_keys.discard(key)
                self.last_release_time = time.time()  # Track for click sequence detection

                # Cancel pending hold timer
                if key in self.hold_timers:
                    self.hold_timers[key].cancel()
                    del self.hold_timers[key]

                was_holding = self.is_holding.get(key, False)
                self.is_holding[key] = False

            # If was holding, pause playback and restore RIGHT key binding
            if was_holding:
                try:
                    self.player.pause = True
                    self.player.speed = 1.0
                    # Restore RIGHT key to normal seek behavior
                    self.player.command('disable-section', 'hold-mode')
                    if self.debug:
                        print(f"[DEBUG] Hold released - pausing")
                except (mpv.ShutdownError, OSError):
                    pass
            elif self.debug:
                print(f"[DEBUG] Released (click complete)")

        # Start keyboard listener
        self.keyboard_listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
            suppress=False
        )
        self.keyboard_listener.start()

    def _on_hold_threshold(self, key):
        """Called when hold threshold timer expires - start playback."""
        with self.key_state_lock:
            if key not in self.pressed_keys:
                return  # Key was released before threshold
            self.is_holding[key] = True

        # Perform actions outside the lock
        try:
            # Enable hold-mode section to stop mpv's repeatable seek
            self.player.command('enable-section', 'hold-mode', 'allow-vo-dragging')

            # Start smooth 2x playback
            self.player.speed = self.playback_speed
            self.player.pause = False
            if self.debug:
                print(f"[DEBUG] Hold threshold reached - playing at {self.playback_speed}x")
        except (mpv.ShutdownError, OSError):
            pass

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
            positive_count, negative_count = self.get_label_stats()
            total = positive_count + negative_count

            # Check if current timestamp is labeled
            # Round to 2 decimals for comparison
            current_label = None
            for ts, label in self.labels.items():
                if abs(ts - timestamp) < 0.1:  # Within 0.1 seconds
                    current_label = label
                    break

            # Format current label status (convert internal to display)
            if current_label:
                display_label = internal_to_display(current_label, self.task_config)
                label_status = f"Current: {display_label.upper()}"
            else:
                label_status = "Current: UNLABELED"

            # Calculate balance status
            if total == 0:
                balance_text = "No labels yet"
            else:
                positive_pct = (positive_count / total) * 100
                negative_pct = (negative_count / total) * 100
                ratio = max(positive_count, negative_count) / max(min(positive_count, negative_count), 1)

                if ratio <= 2.0:
                    balance_status = "BALANCED ✓"
                elif ratio <= 3.0:
                    balance_status = "IMBALANCED"
                else:
                    balance_status = "VERY IMBALANCED"

                balance_text = (
                    f"Total: {total} | {self.pos_label}: {positive_count} ({positive_pct:.1f}%) | "
                    f"{self.neg_label}: {negative_count} ({negative_pct:.1f}%)\n"
                    f"Balance: {balance_status}"
                )

            # Build overlay text
            overlay_text = (
                f"=== Frame Labeling Tool ===\n"
                f"Time: {self.format_timestamp(timestamp)} | {label_status}\n"
                f"{balance_text}\n"
                f"\n"
                f"T={self.pos_label} | F={self.neg_label} | S=save & quit\n"
                f"→=skip 3 (hold=2x) | ←=back 3 | ,/.=step 1 | Space=pause"
            )

            # Show persistent overlay (10 minute duration = effectively permanent)
            self.player.show_text(overlay_text, duration=600000)
        except (mpv.ShutdownError, OSError):
            # Player was closed, ignore
            pass

    def get_label_stats(self):
        """Calculate current label distribution."""
        positive = sum(1 for label in self.labels.values() if label == 'true')
        negative = sum(1 for label in self.labels.values() if label == 'false')
        return positive, negative

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
        print(f"  T - Mark as {self.pos_label.upper()} (positive class)")
        print(f"  F - Mark as {self.neg_label.upper()} (negative class)")
        print("  S - Save and quit")
        print("  I - Refresh overlay (updates automatically)")
        print("  Space - Pause/Play")
        print(f"  → Arrow (click) - Skip forward {self.FRAMES_TO_SKIP} frames")
        print(f"  → Arrow (HOLD) - Play forward at {self.playback_speed}x speed")
        print(f"  ← Arrow - Seek backward {self.backward_frames} frames (hold for continuous)")
        print("  , / . - Step backward/forward one frame (precise)")
        print("  Left Click Timeline - Seek to position")
        print()
        print(f"Tip: Aim for roughly 50/50 balance between {self.pos_label} and {self.neg_label}")
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
    parser.add_argument('--debug', action='store_true', help='Enable debug timing output')

    args = parser.parse_args()

    # Determine output path
    output_csv = args.output if args.output else 'data/labels.csv'

    # Load task config
    task_config = load_task_config()

    # Run labeler
    try:
        labeler = FrameLabeler(args.video, output_csv, args.fps,
                              debug=args.debug, task_config=task_config)
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

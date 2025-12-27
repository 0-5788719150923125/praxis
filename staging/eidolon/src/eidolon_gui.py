#!/usr/bin/env python3
"""
Eidolon - Unified GUI for the nose-touch detection pipeline.

Usage:
    python src/eidolon_gui.py
"""

import os
import sys
import json
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import pandas as pd
from utils import load_config, load_json, get_video_info


class EidolonGUI:
    """Main GUI application for Eidolon pipeline."""

    def __init__(self, root):
        self.root = root
        self.root.title("Eidolon - Nose Touch Detection Pipeline")

        # Calculate window size based on screen resolution
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Use 70% of screen dimensions, with sensible minimums
        window_width = max(1100, int(screen_width * 0.7))
        window_height = max(850, int(screen_height * 0.75))

        # Center the window
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

        # Load config
        try:
            self.config = load_config('config.yaml')
        except:
            messagebox.showerror("Error", "Could not load config.yaml")
            sys.exit(1)

        # State
        self.current_video = None
        self.video_info = None
        self.running_process = None

        # MLT Generation Parameters (initialized from config)
        self.threshold_var = None
        self.min_duration_var = None
        self.marker_buffer_var = None
        self.post_buffer_var = None
        self.mlt_mode_var = None
        self.mute_audio_var = None
        self.add_benny_hill_var = None

        # Create UI
        self.create_ui()

        # Update status on start
        self.refresh_overview()
        self.update_status()

    def create_ui(self):
        """Create the user interface."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)

        # === VIDEO SELECTION ===
        video_frame = ttk.LabelFrame(main_frame, text="Video Selection", padding="10")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        video_frame.columnconfigure(1, weight=1)

        ttk.Button(video_frame, text="Select Video", command=self.select_video).grid(
            row=0, column=0, sticky=tk.W, padx=(0, 10)
        )

        self.video_label = ttk.Label(video_frame, text="No video selected", foreground="gray")
        self.video_label.grid(row=0, column=1, sticky=(tk.W, tk.E))

        self.video_info_label = ttk.Label(video_frame, text="", foreground="gray", font=("", 9))
        self.video_info_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        # URL download section
        ttk.Label(video_frame, text="Or download from URL:", font=("", 9)).grid(
            row=2, column=0, sticky=tk.W, pady=(10, 0)
        )

        url_input_frame = ttk.Frame(video_frame)
        url_input_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        url_input_frame.columnconfigure(0, weight=1)

        self.download_url_var = tk.StringVar()
        self.url_entry = ttk.Entry(url_input_frame, textvariable=self.download_url_var)
        self.url_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))

        # Add right-click context menu for paste support
        self._create_url_entry_context_menu()

        ttk.Button(url_input_frame, text="Download Video", command=self.download_video).grid(
            row=0, column=1, sticky=tk.E
        )

        # Download status label
        self.download_status = ttk.Label(video_frame, text="", foreground="gray", font=("", 9))
        self.download_status.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        # === PROJECT OVERVIEW ===
        overview_frame = ttk.LabelFrame(main_frame, text="Project Overview", padding="10")
        overview_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        overview_frame.columnconfigure(0, weight=1)

        # Create Treeview for videos table
        columns = ('video', 'duration', 'frames', 'labeled', 'progress')

        # Configure row height
        style = ttk.Style()
        style.configure("Treeview", rowheight=25)

        self.videos_tree = ttk.Treeview(overview_frame, columns=columns, show='headings', height=5)

        self.videos_tree.heading('video', text='Video')
        self.videos_tree.heading('duration', text='Duration')
        self.videos_tree.heading('frames', text='Frames')
        self.videos_tree.heading('labeled', text='Labeled')
        self.videos_tree.heading('progress', text='Progress')

        self.videos_tree.column('video', width=200)
        self.videos_tree.column('duration', width=80)
        self.videos_tree.column('frames', width=80)
        self.videos_tree.column('labeled', width=80)
        self.videos_tree.column('progress', width=200)

        self.videos_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))

        # Add scrollbar
        tree_scroll = ttk.Scrollbar(overview_frame, orient=tk.VERTICAL, command=self.videos_tree.yview)
        tree_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.videos_tree.configure(yscrollcommand=tree_scroll.set)

        # Bind double-click to select video
        self.videos_tree.bind('<Double-1>', self.on_video_double_click)

        # Bind right-click to show context menu
        self.videos_tree.bind('<Button-3>', self.show_video_context_menu)
        self.videos_tree.bind('<Control-Button-1>', self.show_video_context_menu)  # Mac support

        # Project stats
        stats_frame = ttk.Frame(overview_frame)
        stats_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

        self.project_stats_label = ttk.Label(stats_frame, text="Total labels: 0 | Balance: N/A | Dataset: Not prepared", font=("", 9))
        self.project_stats_label.grid(row=0, column=0, sticky=tk.W)

        ttk.Button(stats_frame, text="Refresh", command=self.refresh_overview).grid(row=0, column=1, sticky=tk.E, padx=(10, 0))
        stats_frame.columnconfigure(0, weight=1)

        # === VIDEO ACTIONS (Single-Video Operations) ===
        video_actions_frame = ttk.LabelFrame(main_frame, text="Video Actions (Current Video)", padding="10")
        video_actions_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        video_actions_frame.columnconfigure(1, weight=1)

        row = 0

        ttk.Label(video_actions_frame, text="Work with the selected video for labeling:", font=("", 9, "italic"), foreground="gray").grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 5)
        )
        row += 1

        # Extract Frames
        ttk.Button(video_actions_frame, text="Extract Frames", command=self.extract_frames, width=20).grid(
            row=row, column=0, sticky=tk.W, pady=2
        )
        self.frames_status = ttk.Label(video_actions_frame, text="Not started", foreground="gray")
        self.frames_status.grid(row=row, column=1, sticky=tk.W, padx=(10, 0))
        row += 1

        # Label Frames
        ttk.Button(video_actions_frame, text="Label Frames", command=self.label_frames, width=20).grid(
            row=row, column=0, sticky=tk.W, pady=2
        )
        self.labels_status = ttk.Label(video_actions_frame, text="Not started", foreground="gray")
        self.labels_status.grid(row=row, column=1, sticky=tk.W, padx=(10, 0))
        row += 1

        # Clear Labels
        ttk.Button(video_actions_frame, text="Clear Labels", command=self.clear_labels, width=20).grid(
            row=row, column=0, sticky=tk.W, pady=2
        )
        row += 1

        # === PIPELINE ACTIONS (Multi-Video Operations) ===
        pipeline_frame = ttk.LabelFrame(main_frame, text="Pipeline Actions (All Videos)", padding="10")
        pipeline_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        pipeline_frame.columnconfigure(1, weight=1)

        row = 0

        ttk.Label(pipeline_frame, text="Train and use models with data from all labeled videos:", font=("", 9, "italic"), foreground="gray").grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 5)
        )
        row += 1

        # Prepare Dataset
        ttk.Button(pipeline_frame, text="1. Prepare Dataset", command=self.prepare_dataset, width=20).grid(
            row=row, column=0, sticky=tk.W, pady=2
        )
        self.dataset_status = ttk.Label(pipeline_frame, text="Not started", foreground="gray")
        self.dataset_status.grid(row=row, column=1, sticky=tk.W, padx=(10, 0))
        row += 1

        # Train Model
        ttk.Button(pipeline_frame, text="2. Train Model", command=self.train_model, width=20).grid(
            row=row, column=0, sticky=tk.W, pady=2
        )
        self.model_status = ttk.Label(pipeline_frame, text="Not started", foreground="gray")
        self.model_status.grid(row=row, column=1, sticky=tk.W, padx=(10, 0))
        row += 1

        # Separator
        ttk.Separator(pipeline_frame, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10
        )
        row += 1

        ttk.Label(pipeline_frame, text="Apply trained model to current video:", font=("", 9, "italic"), foreground="gray").grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 5)
        )
        row += 1

        # Run Inference
        ttk.Button(pipeline_frame, text="3. Run Inference", command=self.run_inference, width=20).grid(
            row=row, column=0, sticky=tk.W, pady=2
        )
        self.inference_status = ttk.Label(pipeline_frame, text="Not started", foreground="gray")
        self.inference_status.grid(row=row, column=1, sticky=tk.W, padx=(10, 0))
        row += 1

        # Process Events
        ttk.Button(pipeline_frame, text="4. Process Events", command=self.process_events, width=20).grid(
            row=row, column=0, sticky=tk.W, pady=2
        )
        self.events_status = ttk.Label(pipeline_frame, text="Not started", foreground="gray")
        self.events_status.grid(row=row, column=1, sticky=tk.W, padx=(10, 0))
        row += 1

        # Generate MLT from predictions
        ttk.Button(pipeline_frame, text="5. Generate Shotcut Project", command=self.generate_mlt, width=20).grid(
            row=row, column=0, sticky=tk.W, pady=2
        )
        self.mlt_status = ttk.Label(pipeline_frame, text="Not started", foreground="gray")
        self.mlt_status.grid(row=row, column=1, sticky=tk.W, padx=(10, 0))
        ttk.Button(pipeline_frame, text="Open in Shotcut", command=self.open_shotcut).grid(
            row=row, column=2, sticky=tk.E, padx=(10, 0)
        )
        row += 1

        # Generate MLT from labels (for preview/manual workflow)
        ttk.Button(pipeline_frame, text="   Generate from Labels", command=self.generate_mlt_from_labels, width=20).grid(
            row=row, column=0, sticky=tk.W, pady=2
        )
        ttk.Label(pipeline_frame, text="(Preview cuts from manual labels)", font=("", 8), foreground="gray").grid(
            row=row, column=1, columnspan=2, sticky=tk.W, padx=(10, 0)
        )
        row += 1

        # === MLT GENERATION SETTINGS ===
        settings_frame = ttk.LabelFrame(main_frame, text="Event Detection & MLT Settings", padding="10")
        settings_frame.grid(row=3, column=1, rowspan=1, sticky=(tk.W, tk.E, tk.N), pady=(0, 10), padx=(10, 0))
        settings_frame.columnconfigure(1, weight=1)

        # Configure main_frame to support 2-column layout
        main_frame.columnconfigure(1, weight=1)

        # Initialize parameter variables from config
        self.threshold_var = tk.DoubleVar(value=self.config['inference']['threshold'])
        self.min_duration_var = tk.DoubleVar(value=self.config['inference'].get('min_event_duration', 0.4))
        self.marker_buffer_var = tk.DoubleVar(value=self.config['mlt'].get('marker_buffer', 2.0))
        self.post_buffer_var = tk.DoubleVar(value=self.config['mlt'].get('post_buffer', 1.0))
        self.mlt_mode_var = tk.StringVar(value='cut_markers')
        self.mute_audio_var = tk.BooleanVar(value=self.config['mlt'].get('mute_audio', True))
        self.add_benny_hill_var = tk.BooleanVar(value=self.config['mlt'].get('add_benny_hill', False))

        row = 0

        # Threshold slider
        ttk.Label(settings_frame, text="Classification Threshold:", font=("", 9)).grid(
            row=row, column=0, sticky=tk.W, pady=(0, 5)
        )
        self.threshold_label = ttk.Label(settings_frame, text=f"{self.threshold_var.get():.2f}", font=("", 9, "bold"))
        self.threshold_label.grid(row=row, column=2, sticky=tk.W, padx=(5, 0), pady=(0, 5))
        row += 1

        threshold_scale = ttk.Scale(
            settings_frame, from_=0.0, to=1.0,
            variable=self.threshold_var, orient=tk.HORIZONTAL, length=200,
            command=lambda v: self.threshold_label.configure(text=f"{self.threshold_var.get():.2f}")
        )
        threshold_scale.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        row += 1

        # Min duration spinbox
        ttk.Label(settings_frame, text="Min Event Duration (sec):", font=("", 9)).grid(
            row=row, column=0, sticky=tk.W, pady=(0, 5)
        )
        ttk.Spinbox(
            settings_frame, from_=0.0, to=5.0, increment=0.1,
            textvariable=self.min_duration_var, width=10
        ).grid(row=row, column=1, sticky=tk.W, pady=(0, 5))
        row += 1

        # Marker buffer spinbox (pre-event buffer)
        ttk.Label(settings_frame, text="Pre-Event Buffer (sec):", font=("", 9)).grid(
            row=row, column=0, sticky=tk.W, pady=(0, 5)
        )
        ttk.Spinbox(
            settings_frame, from_=0.0, to=10.0, increment=0.5,
            textvariable=self.marker_buffer_var, width=10
        ).grid(row=row, column=1, sticky=tk.W, pady=(0, 5))
        row += 1

        # Post-event buffer spinbox (for extract mode)
        ttk.Label(settings_frame, text="Post-Event Buffer (sec):", font=("", 9)).grid(
            row=row, column=0, sticky=tk.W, pady=(0, 5)
        )
        self.post_buffer_spinbox = ttk.Spinbox(
            settings_frame, from_=0.0, to=10.0, increment=0.5,
            textvariable=self.post_buffer_var, width=10
        )
        self.post_buffer_spinbox.grid(row=row, column=1, sticky=tk.W, pady=(0, 5))
        row += 1

        # MLT Mode radio buttons
        ttk.Label(settings_frame, text="MLT Generation Mode:", font=("", 9, "bold")).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(10, 5)
        )
        row += 1

        ttk.Radiobutton(
            settings_frame, text="Full video with cut markers",
            variable=self.mlt_mode_var, value='cut_markers'
        ).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 2))
        row += 1

        ttk.Radiobutton(
            settings_frame, text="Extract event clips (montage)",
            variable=self.mlt_mode_var, value='extract_clips'
        ).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 2))
        row += 1

        # Mute audio checkbox
        ttk.Checkbutton(
            settings_frame, text="Mute source video audio",
            variable=self.mute_audio_var
        ).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))
        row += 1

        # Benny Hill theme checkbox
        ttk.Checkbutton(
            settings_frame, text="Add Benny Hill theme song",
            variable=self.add_benny_hill_var
        ).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        row += 1

        # Help text
        ttk.Label(
            settings_frame,
            text="Adjust these values and re-run steps 4-5\nto regenerate with different parameters",
            font=("", 8, "italic"), foreground="gray", justify=tk.LEFT
        ).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))

        # === OUTPUT LOG ===
        log_frame = ttk.LabelFrame(main_frame, text="Output Log", padding="10")
        log_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, state='disabled', wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Button(log_frame, text="Clear Log", command=self.clear_log).grid(
            row=1, column=0, sticky=tk.E, pady=(5, 0)
        )

    def _create_url_entry_context_menu(self):
        """Create right-click context menu for URL entry field with paste support."""
        menu = tk.Menu(self.url_entry, tearoff=0)

        menu.add_command(label="Cut", command=lambda: self.url_entry.event_generate("<<Cut>>"))
        menu.add_command(label="Copy", command=lambda: self.url_entry.event_generate("<<Copy>>"))
        menu.add_command(label="Paste", command=lambda: self.url_entry.event_generate("<<Paste>>"))
        menu.add_separator()
        menu.add_command(label="Select All", command=lambda: self.url_entry.select_range(0, tk.END))

        def show_menu(event):
            menu.post(event.x_root, event.y_root)

        # Bind right-click (Button-3 on Linux/Windows, Button-2 on Mac)
        self.url_entry.bind("<Button-3>", show_menu)
        # Also bind for Mac (Control-click)
        self.url_entry.bind("<Control-Button-1>", show_menu)

    def log(self, message, level="INFO"):
        """Add message to log."""
        self.log_text.config(state='normal')

        # Check if user is scrolled to bottom BEFORE inserting text
        # yview() returns (top, bottom) as fractions (0.0 to 1.0)
        yview = self.log_text.yview()
        at_bottom = yview[1] >= 0.99  # Consider "at bottom" if within 1%

        # Color coding
        tag = level.lower()
        self.log_text.tag_config("info", foreground="black")
        self.log_text.tag_config("success", foreground="green")
        self.log_text.tag_config("error", foreground="red")
        self.log_text.tag_config("warning", foreground="orange")

        self.log_text.insert(tk.END, f"[{level}] {message}\n", tag)

        # Only auto-scroll if user was already at the bottom
        if at_bottom:
            self.log_text.see(tk.END)

        self.log_text.config(state='disabled')

    def clear_log(self):
        """Clear the log."""
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')

    def refresh_overview(self):
        """Refresh the project overview with all videos and their status."""
        # Clear existing items
        for item in self.videos_tree.get_children():
            self.videos_tree.delete(item)

        # Get all video files from videos/ directory
        videos_base = self.config['paths']['videos']
        all_videos = {}  # video_name -> video_path
        if os.path.exists(videos_base):
            for ext in ['*.mp4', '*.mkv', '*.avi', '*.mov']:
                for video_path in Path(videos_base).glob(ext):
                    video_name = video_path.stem
                    all_videos[video_name] = str(video_path)

        # Get all frame directories (these represent videos with frames extracted)
        frames_base = self.config['paths']['frames']
        video_dirs = []
        if os.path.exists(frames_base):
            video_dirs = [d for d in os.listdir(frames_base)
                         if os.path.isdir(os.path.join(frames_base, d))]

        # Combine: all videos from videos/ directory + any orphaned frame directories
        all_video_names = set(all_videos.keys()) | set(video_dirs)

        # Load labels if available
        labels_data = {}
        labels_file = self.config['paths']['labels']
        if os.path.exists(labels_file):
            df = pd.read_csv(labels_file)
            # Group by video directory
            for _, row in df.iterrows():
                frame_path = row['frame_path']
                # Extract video directory name from frame path
                parts = Path(frame_path).parts
                if 'frames' in parts:
                    idx = parts.index('frames')
                    if idx + 1 < len(parts):
                        video_name = parts[idx + 1]
                        if video_name not in labels_data:
                            labels_data[video_name] = []
                        labels_data[video_name].append(row)

        # Populate table - show all videos (with or without frames)
        for video_name in sorted(all_video_names):
            frames_dir = os.path.join(frames_base, video_name)
            has_frames = os.path.exists(frames_dir)

            # Count frames
            frame_count = 0
            if has_frames:
                frame_count = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

            # Get video info
            duration = "Unknown"
            metadata_path = os.path.join(frames_dir, 'metadata.json')

            # Try to get duration from metadata if frames exist
            if has_frames and os.path.exists(metadata_path):
                metadata = load_json(metadata_path)
                if 'video_info' in metadata:
                    duration_sec = metadata['video_info'].get('duration', 0)
                    duration = f"{int(duration_sec // 60)}:{int(duration_sec % 60):02d}"
            # Otherwise try to get duration from video file directly
            elif video_name in all_videos:
                try:
                    video_info = get_video_info(all_videos[video_name])
                    duration_sec = video_info.get('duration', 0)
                    duration = f"{int(duration_sec // 60)}:{int(duration_sec % 60):02d}"
                except:
                    pass

            # Get label count and progress
            label_count = 0
            progress = "Not started" if not has_frames else "Not labeled"
            if video_name in labels_data:
                label_count = len(labels_data[video_name])
                # Find max timestamp to estimate progress
                max_timestamp = max(row['timestamp'] for row in labels_data[video_name])
                if metadata_path and os.path.exists(metadata_path):
                    metadata = load_json(metadata_path)
                    if 'video_info' in metadata:
                        video_duration = metadata['video_info'].get('duration', 0)
                        if video_duration > 0:
                            progress_pct = (max_timestamp / video_duration) * 100
                            progress = f"Last @ {int(max_timestamp // 60)}:{int(max_timestamp % 60):02d} ({progress_pct:.0f}%)"

            # Add to tree - store full video_name as iid for lookup
            display_name = video_name[:40] + '...' if len(video_name) > 40 else video_name
            self.videos_tree.insert('', 'end', iid=video_name, values=(
                display_name,
                duration,
                frame_count if has_frames else "-",
                label_count if label_count > 0 else "-",
                progress
            ))

        # Update project stats
        total_labels = sum(len(labels) for labels in labels_data.values())

        # Calculate balance
        balance_str = "N/A"
        if total_labels > 0 and os.path.exists(labels_file):
            df = pd.read_csv(labels_file)
            df = df[df['label'].isin(['touching', 'not_touching'])]
            if len(df) > 0:
                counts = df['label'].value_counts()
                touching = counts.get('touching', 0)
                not_touching = counts.get('not_touching', 0)
                if touching + not_touching > 0:
                    balance_str = f"{touching}/{not_touching}"

        # Check dataset
        dataset_status = "Not prepared"
        dataset_dir = self.config['paths']['dataset']
        if os.path.exists(os.path.join(dataset_dir, 'train')):
            dataset_status = "Ready"

        self.project_stats_label.config(
            text=f"Total labels: {total_labels} | Balance: {balance_str} | Dataset: {dataset_status}"
        )

    def on_video_double_click(self, event):
        """Handle double-click on video in overview to select it."""
        selection = self.videos_tree.selection()
        if not selection:
            return

        # Get the video name from the iid
        video_name = selection[0]
        video_path = None

        # First check if video exists in videos/ directory
        videos_base = self.config['paths']['videos']
        if os.path.exists(videos_base):
            for ext in ['.mp4', '.mkv', '.avi', '.mov']:
                candidate = os.path.join(videos_base, video_name + ext)
                if os.path.exists(candidate):
                    video_path = candidate
                    break

        # If not found, try to get from metadata (for videos with frames)
        if not video_path:
            frames_dir = os.path.join(self.config['paths']['frames'], video_name)
            metadata_path = os.path.join(frames_dir, 'metadata.json')

            if os.path.exists(metadata_path):
                try:
                    metadata = load_json(metadata_path)
                    video_path = metadata.get('source_video')
                except:
                    pass

        # Still not found
        if not video_path or not os.path.exists(video_path):
            messagebox.showwarning("Video Not Found",
                f"Could not find video file for: {video_name}\n\nCheck that the video file exists in the videos/ directory.")
            return

        # Set as current video
        self.current_video = video_path
        self.video_label.config(text=os.path.basename(video_path), foreground="black")

        # Get video info
        try:
            self.video_info = get_video_info(video_path)
            info_text = f"{self.video_info['width']}x{self.video_info['height']}, {self.video_info['fps']:.2f} fps, {self.video_info['duration']:.1f}s"
            self.video_info_label.config(text=info_text, foreground="black")
            self.log(f"Selected: {os.path.basename(video_path)}", "SUCCESS")
            self.update_status()
        except Exception as e:
            self.log(f"Error reading video info: {e}", "ERROR")

    def show_video_context_menu(self, event):
        """Show context menu for video in overview."""
        # Select the item under the cursor
        item = self.videos_tree.identify_row(event.y)
        if item:
            self.videos_tree.selection_set(item)

            # Create context menu
            menu = tk.Menu(self.videos_tree, tearoff=0)
            menu.add_command(label="Select Video", command=lambda: self.on_video_double_click(None))
            menu.add_separator()
            menu.add_command(label="Remove Video (Delete All Data)", command=self.remove_video)

            # Show menu
            menu.post(event.x_root, event.y_root)

    def remove_video(self):
        """Remove video and all associated data (frames, labels, predictions, events, MLT)."""
        selection = self.videos_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a video to remove")
            return

        video_name = selection[0]

        # Confirm deletion
        response = messagebox.askyesno(
            "Confirm Deletion",
            f"Remove '{video_name}'?\n\n"
            "This will delete:\n"
            "• Video file\n"
            "• Extracted frames\n"
            "• All labels for this video\n"
            "• Predictions and events\n"
            "• MLT projects\n\n"
            "This cannot be undone!",
            icon='warning'
        )

        if not response:
            return

        self.log(f"Removing video: {video_name}", "WARNING")

        deleted_items = []

        # 1. Delete video file from videos/
        videos_base = self.config['paths']['videos']
        if os.path.exists(videos_base):
            for ext in ['.mp4', '.mkv', '.avi', '.mov']:
                video_path = os.path.join(videos_base, video_name + ext)
                if os.path.exists(video_path):
                    os.remove(video_path)
                    deleted_items.append(f"Video file: {os.path.basename(video_path)}")
                    self.log(f"  Deleted video file", "INFO")
                    break

        # 2. Delete frames directory
        frames_dir = os.path.join(self.config['paths']['frames'], video_name)
        if os.path.exists(frames_dir):
            import shutil
            shutil.rmtree(frames_dir)
            deleted_items.append(f"Frames directory ({video_name})")
            self.log(f"  Deleted frames directory", "INFO")

        # 3. Remove labels from labels.csv
        labels_file = self.config['paths']['labels']
        if os.path.exists(labels_file):
            df = pd.read_csv(labels_file)
            original_count = len(df)

            # Filter out labels for this video
            frames_path_pattern = os.path.join(self.config['paths']['frames'], video_name)
            df = df[~df['frame_path'].str.contains(frames_path_pattern, na=False, regex=False)]

            removed_count = original_count - len(df)
            if removed_count > 0:
                df.to_csv(labels_file, index=False)
                deleted_items.append(f"{removed_count} labels")
                self.log(f"  Removed {removed_count} labels from labels.csv", "INFO")

        # 4. Delete predictions
        predictions_file = os.path.join(self.config['paths']['predictions'], f"{video_name}_predictions.json")
        if os.path.exists(predictions_file):
            os.remove(predictions_file)
            deleted_items.append("Predictions")
            self.log(f"  Deleted predictions", "INFO")

        # 5. Delete events
        events_file = os.path.join(self.config['paths']['events'], f"{video_name}_events.json")
        if os.path.exists(events_file):
            os.remove(events_file)
            deleted_items.append("Events")
            self.log(f"  Deleted events", "INFO")

        # 6. Delete MLT project
        mlt_file = os.path.join(self.config['paths']['mlt_projects'], f"{video_name}_project.mlt")
        if os.path.exists(mlt_file):
            os.remove(mlt_file)
            deleted_items.append("MLT project")
            self.log(f"  Deleted MLT project", "INFO")

        # Also check for "from_labels" MLT variant
        mlt_from_labels = os.path.join(self.config['paths']['mlt_projects'], f"{video_name}_from_labels.mlt")
        if os.path.exists(mlt_from_labels):
            os.remove(mlt_from_labels)
            self.log(f"  Deleted MLT (from labels)", "INFO")

        # If this was the current video, clear selection
        if self.current_video and Path(self.current_video).stem == video_name:
            self.current_video = None
            self.video_info = None
            self.video_label.config(text="No video selected", foreground="gray")
            self.video_info_label.config(text="")

        # Refresh UI
        self.refresh_overview()
        self.update_status()

        # Summary
        self.log(f"✓ Removed '{video_name}' and {len(deleted_items)} associated items", "SUCCESS")

    def select_video(self):
        """Select a video file."""
        filename = filedialog.askopenfilename(
            title="Select Video",
            initialdir="videos",
            filetypes=[
                ("Video files", "*.mp4 *.mkv *.avi *.mov"),
                ("All files", "*.*")
            ]
        )

        if filename:
            self.current_video = filename
            self.video_label.config(text=os.path.basename(filename), foreground="black")

            # Get video info
            try:
                self.video_info = get_video_info(filename)
                info_text = f"{self.video_info['width']}x{self.video_info['height']}, {self.video_info['fps']:.2f} fps, {self.video_info['duration']:.1f}s"
                self.video_info_label.config(text=info_text, foreground="black")
                self.log(f"Selected: {os.path.basename(filename)}", "SUCCESS")
                self.log(f"  {info_text}", "INFO")
            except Exception as e:
                self.log(f"Error reading video info: {e}", "ERROR")

            self.refresh_overview()
            self.update_status()

    def download_video(self):
        """Download video from URL using yt-dlp."""
        url = self.download_url_var.get().strip()

        if not url:
            messagebox.showwarning("No URL", "Please enter a video URL")
            return

        # Validate URL format (basic check)
        if not (url.startswith('http://') or url.startswith('https://')):
            messagebox.showwarning("Invalid URL", "URL must start with http:// or https://")
            return

        # Get output directory from config
        output_dir = self.config['paths']['videos']

        # Update status
        self.download_status.config(text="Starting download...", foreground="blue")
        self.log(f"Downloading video from: {url}", "INFO")

        # Build command
        cmd = [
            "python", "src/download_video.py",
            "--url", url,
            "--output", output_dir
        ]

        # Run download in background (uses existing threading pattern)
        self.run_command(cmd, on_complete=self._on_download_complete)

    def _on_download_complete(self):
        """Called after video download completes."""
        # Clear URL field
        self.download_url_var.set("")

        # Update status
        self.download_status.config(text="Download complete!", foreground="green")

        # Find and auto-select the most recently downloaded video
        videos_dir = self.config['paths']['videos']
        if os.path.exists(videos_dir):
            # Get all video files
            video_files = []
            for ext in ['*.mp4', '*.mkv', '*.avi', '*.mov']:
                video_files.extend(Path(videos_dir).glob(ext))

            if video_files:
                # Sort by modification time, get the newest
                newest_video = max(video_files, key=lambda p: p.stat().st_mtime)

                # Auto-select this video
                self.current_video = str(newest_video)
                self.video_label.config(text=newest_video.name, foreground="black")

                # Get and display video info
                try:
                    self.video_info = get_video_info(str(newest_video))
                    info_text = f"{self.video_info['width']}x{self.video_info['height']}, {self.video_info['fps']:.2f} fps, {self.video_info['duration']:.1f}s"
                    self.video_info_label.config(text=info_text, foreground="black")
                    self.log(f"Auto-selected: {newest_video.name}", "SUCCESS")
                    self.log(f"  {info_text}", "INFO")
                except Exception as e:
                    self.log(f"Error reading video info: {e}", "WARNING")

        # Refresh overview (will show in overview after frames are extracted)
        self.refresh_overview()

        # Update status for the newly selected video
        self.update_status()

        # Log success
        self.log("Video ready for processing - you can now extract frames", "SUCCESS")

        # Clear status after 3 seconds
        self.root.after(3000, lambda: self.download_status.config(text=""))

    def update_status(self):
        """Update status labels for all pipeline steps."""
        if not self.current_video:
            return

        video_name = Path(self.current_video).stem

        # Check frames
        frames_dir = os.path.join(self.config['paths']['frames'], video_name)
        if os.path.exists(frames_dir):
            frame_count = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
            self.frames_status.config(text=f"{frame_count} frames extracted", foreground="green")
        else:
            self.frames_status.config(text="Not started", foreground="gray")

        # Check labels (only for current video)
        labels_file = self.config['paths']['labels']
        if os.path.exists(labels_file):
            import pandas as pd
            try:
                df = pd.read_csv(labels_file)
                # Filter labels for current video only - match the frames directory path
                frames_path_pattern = os.path.join(self.config['paths']['frames'], video_name)
                video_labels = df[df['frame_path'].str.contains(frames_path_pattern, na=False, regex=False)]
                label_count = len(video_labels)
                if label_count > 0:
                    self.labels_status.config(text=f"{label_count} frames labeled", foreground="green")
                else:
                    self.labels_status.config(text="Not labeled yet", foreground="gray")
            except Exception as e:
                self.log(f"Error reading labels: {e}", "ERROR")
                self.labels_status.config(text="Error reading labels", foreground="orange")
        else:
            self.labels_status.config(text="Not started", foreground="gray")

        # Check dataset
        dataset_dir = self.config['paths']['dataset']
        if os.path.exists(os.path.join(dataset_dir, 'train')):
            self.dataset_status.config(text="Dataset prepared", foreground="green")
        else:
            self.dataset_status.config(text="Not started", foreground="gray")

        # Check model
        models_dir = self.config['paths']['models']
        model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))] if os.path.exists(models_dir) else []
        if model_dirs:
            self.model_status.config(text=f"Model trained ({model_dirs[0]})", foreground="green")
        else:
            self.model_status.config(text="Not started", foreground="gray")

        # Check inference (predictions)
        predictions_file = os.path.join(self.config['paths']['predictions'], f"{video_name}_predictions.json")
        if os.path.exists(predictions_file):
            data = load_json(predictions_file)
            pred_count = len(data)
            self.inference_status.config(text=f"{pred_count} predictions saved", foreground="green")
        else:
            self.inference_status.config(text="Not started", foreground="gray")

        # Check events processing
        events_file = os.path.join(self.config['paths']['events'], f"{video_name}_events.json")
        if os.path.exists(events_file):
            data = load_json(events_file)
            event_count = data['num_events']
            self.events_status.config(text=f"{event_count} events detected", foreground="green")
        else:
            self.events_status.config(text="Not started", foreground="gray")

        # Check MLT
        mlt_file = os.path.join(self.config['paths']['mlt_projects'], f"{video_name}_project.mlt")
        if os.path.exists(mlt_file):
            self.mlt_status.config(text="MLT project generated", foreground="green")
        else:
            self.mlt_status.config(text="Not started", foreground="gray")

    def run_command(self, cmd, on_complete=None):
        """Run a command in a separate thread."""
        def run():
            self.log(f"Running: {' '.join(cmd)}", "INFO")
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                # Stream output
                for line in process.stdout:
                    self.root.after(0, lambda l=line: self.log(l.strip(), "INFO"))

                process.wait()

                if process.returncode == 0:
                    self.root.after(0, lambda: self.log("✓ Completed successfully", "SUCCESS"))
                    self.root.after(0, self.refresh_overview)
                    if on_complete:
                        self.root.after(0, on_complete)
                else:
                    self.root.after(0, lambda: self.log(f"✗ Failed with code {process.returncode}", "ERROR"))

            except Exception as e:
                self.root.after(0, lambda: self.log(f"Error: {e}", "ERROR"))

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    def extract_frames(self):
        """Extract frames from video."""
        if not self.current_video:
            messagebox.showwarning("No Video", "Please select a video first")
            return

        cmd = ["python", "src/extract_frames.py", "--video", self.current_video]
        self.run_command(cmd, on_complete=self.update_status)

    def label_frames(self):
        """Launch labeling tool."""
        if not self.current_video:
            messagebox.showwarning("No Video", "Please select a video first")
            return

        video_name = Path(self.current_video).stem
        frames_dir = os.path.join(self.config['paths']['frames'], video_name)

        if not os.path.exists(frames_dir):
            messagebox.showwarning("No Frames", "Please extract frames first")
            return

        def delayed_update():
            """Update status after a short delay to ensure file is written."""
            self.root.after(500, self.update_status)

        cmd = ["python", "src/label_frames.py", "--frames", frames_dir]
        self.run_command(cmd, on_complete=delayed_update)

    def clear_labels(self):
        """Clear all labels for the current video."""
        if not self.current_video:
            messagebox.showwarning("No Video", "Please select a video first")
            return

        video_name = Path(self.current_video).stem
        labels_file = self.config['paths']['labels']

        if not os.path.exists(labels_file):
            messagebox.showinfo("No Labels", "No labels file exists yet.")
            return

        # Count how many labels will be deleted
        try:
            df = pd.read_csv(labels_file)
            frames_path_pattern = os.path.join(self.config['paths']['frames'], video_name)
            video_labels = df[df['frame_path'].str.contains(frames_path_pattern, na=False, regex=False)]
            label_count = len(video_labels)

            if label_count == 0:
                messagebox.showinfo("No Labels", f"No labels found for video: {video_name}")
                return

            # Confirm deletion
            response = messagebox.askyesno(
                "Clear Labels",
                f"Delete {label_count} labels for video:\n{video_name}\n\nThis cannot be undone!",
                icon='warning'
            )

            if not response:
                self.log("Label deletion cancelled", "INFO")
                return

            # Remove labels for this video
            remaining_labels = df[~df['frame_path'].str.contains(frames_path_pattern, na=False, regex=False)]

            # Save filtered labels
            if len(remaining_labels) > 0:
                remaining_labels.to_csv(labels_file, index=False)
                self.log(f"Deleted {label_count} labels for {video_name}", "SUCCESS")
                self.log(f"Remaining labels: {len(remaining_labels)}", "INFO")
            else:
                # No labels left, delete the file
                os.remove(labels_file)
                self.log(f"Deleted all {label_count} labels (file removed)", "SUCCESS")

            # Update displays
            self.refresh_overview()
            self.update_status()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear labels:\n{e}")
            self.log(f"Error clearing labels: {e}", "ERROR")

    def prepare_dataset(self):
        """Prepare train/val/test splits."""
        labels_file = self.config['paths']['labels']

        if not os.path.exists(labels_file):
            messagebox.showwarning("No Labels", "Please label frames first")
            return

        cmd = ["python", "src/prepare_dataset.py", "--labels", labels_file]
        self.run_command(cmd, on_complete=self.update_status)

    def train_model(self):
        """Train the model."""
        dataset_dir = self.config['paths']['dataset']

        if not os.path.exists(os.path.join(dataset_dir, 'train')):
            messagebox.showwarning("No Dataset", "Please prepare dataset first")
            return

        cmd = ["python", "src/train_classifier.py", "--dataset", dataset_dir, "--gui-mode"]
        self.run_command(cmd, on_complete=self.update_status)

    def run_inference(self):
        """Run raw inference on video (saves predictions only)."""
        if not self.current_video:
            messagebox.showwarning("No Video", "Please select a video first")
            return

        # Find trained model
        models_dir = self.config['paths']['models']
        if not os.path.exists(models_dir):
            messagebox.showwarning("No Model", "Please train a model first")
            return

        model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        if not model_dirs:
            messagebox.showwarning("No Model", "Please train a model first")
            return

        model_path = os.path.join(models_dir, model_dirs[0], "final")

        self.log(f"Running inference on {Path(self.current_video).name}...", "INFO")
        self.log("This will save raw predictions (probabilities only)", "INFO")

        cmd = ["python", "src/infer_video.py", "--video", self.current_video, "--model", model_path]
        self.run_command(cmd, on_complete=self.update_status)

    def process_events(self):
        """Process predictions to detect events with current threshold and min_duration settings."""
        if not self.current_video:
            messagebox.showwarning("No Video", "Please select a video first")
            return

        video_name = Path(self.current_video).stem
        predictions_file = os.path.join(self.config['paths']['predictions'], f"{video_name}_predictions.json")

        if not os.path.exists(predictions_file):
            messagebox.showwarning("No Predictions", "Please run inference first (step 3)")
            return

        threshold = self.threshold_var.get()
        min_duration = self.min_duration_var.get()

        self.log(f"Processing events with threshold={threshold:.2f}, min_duration={min_duration:.1f}s", "INFO")

        cmd = [
            "python", "src/process_events.py",
            "--predictions", predictions_file,
            "--threshold", str(threshold),
            "--min-duration", str(min_duration)
        ]
        self.run_command(cmd, on_complete=self.update_status)

    def generate_mlt(self):
        """Generate MLT project file from detected events with current settings."""
        if not self.current_video:
            messagebox.showwarning("No Video", "Please select a video first")
            return

        video_name = Path(self.current_video).stem
        events_file = os.path.join(self.config['paths']['events'], f"{video_name}_events.json")

        if not os.path.exists(events_file):
            messagebox.showwarning("No Events", "Please process events first (step 4)")
            return

        marker_buffer = self.marker_buffer_var.get()
        post_buffer = self.post_buffer_var.get()
        mode = self.mlt_mode_var.get()
        mute_audio = self.mute_audio_var.get()
        add_benny_hill = self.add_benny_hill_var.get()

        mode_desc = "cut markers" if mode == 'cut_markers' else "extracted clips montage"
        audio_desc = "muted" if mute_audio else "enabled"
        benny_desc = " + Benny Hill" if add_benny_hill else ""
        self.log(f"Generating MLT project with mode={mode_desc}, pre={marker_buffer:.1f}s, post={post_buffer:.1f}s, audio={audio_desc}{benny_desc}", "INFO")

        cmd = [
            "python", "src/generate_mlt.py",
            "--events", events_file,
            "--marker-buffer", str(marker_buffer),
            "--post-buffer", str(post_buffer),
            "--mode", mode
        ]

        # Explicitly pass mute audio flag
        if mute_audio:
            cmd.append("--mute-audio")
        else:
            cmd.append("--no-mute-audio")

        # Explicitly pass Benny Hill flag
        if add_benny_hill:
            cmd.append("--add-benny-hill")
        else:
            cmd.append("--no-benny-hill")
        self.run_command(cmd, on_complete=self.update_status)

    def generate_mlt_from_labels(self):
        """Generate MLT project file from manual labels (for preview)."""
        if not self.current_video:
            messagebox.showwarning("No Video", "Please select a video first")
            return

        # Check if labels exist
        labels_file = self.config['paths']['labels']
        if not os.path.exists(labels_file):
            messagebox.showwarning("No Labels", "Please label some frames first")
            return

        # Check if this video has labels
        try:
            import pandas as pd
            df = pd.read_csv(labels_file)
            video_name = Path(self.current_video).stem
            frames_path_pattern = os.path.join(self.config['paths']['frames'], video_name)
            video_labels = df[df['frame_path'].str.contains(frames_path_pattern, na=False, regex=False)]

            if len(video_labels) == 0:
                messagebox.showwarning("No Labels", f"No labels found for video: {video_name}")
                return

            # Count touching labels (labels are stored as strings "touching"/"not_touching")
            touching_count = (video_labels['label'] == 'touching').sum()
            if touching_count == 0:
                messagebox.showwarning("No Touching Labels", "No 'touching' labels found for this video.\n\nYou need at least one positive label to generate cuts.")
                return

        except Exception as e:
            messagebox.showerror("Error", f"Failed to read labels:\n{e}")
            return

        cmd = ["python", "src/labels_to_mlt.py", "--video", self.current_video]
        self.run_command(cmd, on_complete=self.update_status)

    def open_shotcut(self):
        """Open the MLT file in Shotcut."""
        if not self.current_video:
            messagebox.showwarning("No Video", "Please select a video first")
            return

        video_name = Path(self.current_video).stem
        output_dir = self.config['paths']['mlt_projects']

        # Check for both types of MLT files
        mlt_from_predictions = os.path.join(output_dir, f"{video_name}_project.mlt")
        mlt_from_labels = os.path.join(output_dir, f"{video_name}_from_labels.mlt")

        predictions_exists = os.path.exists(mlt_from_predictions)
        labels_exists = os.path.exists(mlt_from_labels)

        # Determine which file to open
        if not predictions_exists and not labels_exists:
            messagebox.showwarning("No MLT", "Please generate an MLT project first")
            return

        # If both exist, ask user which to open
        if predictions_exists and labels_exists:
            response = messagebox.askyesnocancel(
                "Multiple MLT Files",
                "Found two MLT files:\n\n"
                "• From predictions (inference results)\n"
                "• From labels (manual annotations)\n\n"
                "Open the one from predictions?\n\n"
                "(Yes = predictions, No = labels, Cancel = abort)"
            )
            if response is None:  # Cancel
                return
            mlt_file = mlt_from_predictions if response else mlt_from_labels
        else:
            # Only one exists
            mlt_file = mlt_from_predictions if predictions_exists else mlt_from_labels

        try:
            subprocess.Popen(["shotcut", mlt_file])
            self.log(f"Opened in Shotcut: {mlt_file}", "SUCCESS")
        except Exception as e:
            self.log(f"Error opening Shotcut: {e}", "ERROR")
            messagebox.showerror("Error", f"Could not open Shotcut:\n{e}")


def main():
    root = tk.Tk()
    app = EidolonGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()

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

        # Generate MLT
        ttk.Button(pipeline_frame, text="4. Generate Shotcut Project", command=self.generate_mlt, width=20).grid(
            row=row, column=0, sticky=tk.W, pady=2
        )
        self.mlt_status = ttk.Label(pipeline_frame, text="Not started", foreground="gray")
        self.mlt_status.grid(row=row, column=1, sticky=tk.W, padx=(10, 0))
        ttk.Button(pipeline_frame, text="Open in Shotcut", command=self.open_shotcut).grid(
            row=row, column=2, sticky=tk.E, padx=(10, 0)
        )
        row += 1

        # === OUTPUT LOG ===
        log_frame = ttk.LabelFrame(main_frame, text="Output Log", padding="10")
        log_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, state='disabled', wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Button(log_frame, text="Clear Log", command=self.clear_log).grid(
            row=1, column=0, sticky=tk.E, pady=(5, 0)
        )

    def log(self, message, level="INFO"):
        """Add message to log."""
        self.log_text.config(state='normal')

        # Color coding
        tag = level.lower()
        self.log_text.tag_config("info", foreground="black")
        self.log_text.tag_config("success", foreground="green")
        self.log_text.tag_config("error", foreground="red")
        self.log_text.tag_config("warning", foreground="orange")

        self.log_text.insert(tk.END, f"[{level}] {message}\n", tag)
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

        # Get all frame directories (these represent processed videos)
        frames_base = self.config['paths']['frames']
        video_dirs = []
        if os.path.exists(frames_base):
            video_dirs = [d for d in os.listdir(frames_base)
                         if os.path.isdir(os.path.join(frames_base, d))]

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

        # Populate table
        for video_dir in sorted(video_dirs):
            frames_dir = os.path.join(frames_base, video_dir)

            # Count frames
            frame_count = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

            # Get video info
            duration = "Unknown"
            metadata_path = os.path.join(frames_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                metadata = load_json(metadata_path)
                if 'video_info' in metadata:
                    duration_sec = metadata['video_info'].get('duration', 0)
                    duration = f"{int(duration_sec // 60)}:{int(duration_sec % 60):02d}"

            # Get label count and progress
            label_count = 0
            progress = "Not labeled"
            if video_dir in labels_data:
                label_count = len(labels_data[video_dir])
                # Find max timestamp to estimate progress
                max_timestamp = max(row['timestamp'] for row in labels_data[video_dir])
                if metadata_path and os.path.exists(metadata_path):
                    metadata = load_json(metadata_path)
                    if 'video_info' in metadata:
                        video_duration = metadata['video_info'].get('duration', 0)
                        if video_duration > 0:
                            progress_pct = (max_timestamp / video_duration) * 100
                            progress = f"Last @ {int(max_timestamp // 60)}:{int(max_timestamp % 60):02d} ({progress_pct:.0f}%)"

            # Add to tree - store full video_dir as iid for lookup
            display_name = video_dir[:40] + '...' if len(video_dir) > 40 else video_dir
            self.videos_tree.insert('', 'end', iid=video_dir, values=(
                display_name,
                duration,
                frame_count,
                label_count,
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

        # Get the full video_dir name from the iid
        video_dir = selection[0]

        # Get the source video path from metadata
        frames_dir = os.path.join(self.config['paths']['frames'], video_dir)
        metadata_path = os.path.join(frames_dir, 'metadata.json')

        if not os.path.exists(metadata_path):
            messagebox.showerror("Error", f"Metadata not found for: {video_dir}")
            return

        try:
            metadata = load_json(metadata_path)
            video_path = metadata.get('source_video')

            if not video_path or not os.path.exists(video_path):
                # Try to find video in videos directory
                videos_base = self.config['paths']['videos']
                if os.path.exists(videos_base):
                    for filename in os.listdir(videos_base):
                        if Path(filename).stem == video_dir or filename == video_dir:
                            video_path = os.path.join(videos_base, filename)
                            if os.path.isfile(video_path):
                                break
                    else:
                        messagebox.showwarning("Video Not Found",
                            f"Source video not found: {video_path}\n\nOriginal location may have moved.")
                        return
                else:
                    messagebox.showwarning("Video Not Found",
                        f"Source video not found: {video_path}")
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

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {e}")

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

        # Check inference
        events_file = os.path.join(self.config['paths']['events'], f"{video_name}_events.json")
        if os.path.exists(events_file):
            data = load_json(events_file)
            event_count = data['num_events']
            self.inference_status.config(text=f"{event_count} events detected", foreground="green")
        else:
            self.inference_status.config(text="Not started", foreground="gray")

        # Check MLT
        mlt_file = os.path.join(self.config['mlt']['output_dir'], f"{video_name}_project.mlt")
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
        """Run inference on video."""
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

        cmd = ["python", "src/infer_video.py", "--video", self.current_video, "--model", model_path]
        self.run_command(cmd, on_complete=self.update_status)

    def generate_mlt(self):
        """Generate MLT project file."""
        if not self.current_video:
            messagebox.showwarning("No Video", "Please select a video first")
            return

        video_name = Path(self.current_video).stem
        events_file = os.path.join(self.config['paths']['events'], f"{video_name}_events.json")

        if not os.path.exists(events_file):
            messagebox.showwarning("No Events", "Please run inference first")
            return

        cmd = ["python", "src/generate_mlt.py", "--events", events_file]
        self.run_command(cmd, on_complete=self.update_status)

    def open_shotcut(self):
        """Open the MLT file in Shotcut."""
        if not self.current_video:
            messagebox.showwarning("No Video", "Please select a video first")
            return

        video_name = Path(self.current_video).stem
        mlt_file = os.path.join(self.config['mlt']['output_dir'], f"{video_name}_project.mlt")

        if not os.path.exists(mlt_file):
            messagebox.showwarning("No MLT", "Please generate MLT project first")
            return

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

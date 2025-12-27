# Eidolon - Automated Nose-Touch Detection Pipeline

A machine learning pipeline for detecting nose-touching moments in videos and automatically generating Shotcut project files for video editing.

## Overview

Eidolon uses a fine-tuned vision transformer (DeiT-Small) to classify video frames as "touching" or "not touching" the nose, then generates a Shotcut MLT project file with markers at all detected moments.

**Key Features:**
- Frame-level binary classification using HuggingFace Transformers
- Interactive labeling tool for creating training data
- Automated event detection with configurable thresholds and buffers
- Direct Shotcut MLT project generation (no video re-encoding needed)
- GPU-accelerated inference
- Automated experiment naming for YouTube uploads
- Optional Benny Hill theme music integration

## Installation

1. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install Shotcut**: Download from https://shotcut.org/download/

## Quick Start - Use the GUI!

**We strongly recommend using the GUI** for the best experience:

```bash
cd staging/eidolon
python src/eidolon_gui.py
```

The GUI provides a unified interface for the entire workflow:
- **Extract frames** from videos for labeling
- **Label frames** interactively (with real-time class balance feedback)
- **Prepare dataset** splits (train/val/test)
- **Train models** with progress monitoring
- **Run inference** on target videos
- **Generate MLT projects** with multiple output modes (cut markers, montage, etc.)

All with real-time status updates, output logging, and no command line needed.

## Workflow Overview

### One-Time Setup (Training)

1. **Extract frames** from sample video (5 fps default)
2. **Label frames** manually (aim for 500-1000 frames, balanced classes)
   - Use GUI labeling tool with keyboard shortcuts (T/F/S)
   - Real-time balance indicator shows class distribution
3. **Prepare dataset** splits (70% train, 15% val, 15% test)
4. **Train classifier** (~10-30 minutes on GPU)

### Processing Videos (After Training)

5. **Run inference** on target video
6. **Generate MLT project** (instant XML generation)
7. **Open in Shotcut** and render final video

The GUI handles all of this through a simple tab-based interface.

## Configuration

Edit `config.yaml` to customize behavior:

```yaml
model:
  name: "facebook/deit-small-patch16-224"  # Or "facebook/convnext-tiny-224"

extraction:
  fps: 5  # Frame extraction rate

inference:
  threshold: 0.5  # Lower = more detections
  min_event_duration: 0.4  # Minimum event length (seconds)

mlt:
  marker_buffer: 2.0  # Seconds before each event
  post_buffer: 1.0  # Seconds after each event
  mute_audio: false  # Mute original video audio
  add_benny_hill: false  # Add Benny Hill theme music
```

## Output Modes

The MLT generator supports multiple modes:

- **cut_markers**: Full video with markers at detected moments (default)
- **montage**: Cut together only the detected segments
- **split_clips**: Individual clips for each event

## Troubleshooting

**No events detected:**
- Lower threshold: try 0.3
- Check video has actual nose-touching moments
- Label more diverse training data

**Too many false positives:**
- Raise threshold: try 0.7
- Label more negative examples (hand near face but not touching)

**CUDA out of memory:**
- Reduce batch size in config: `batch_size: 16` or `batch_size: 8`

**MLT file doesn't open:**
- Verify video path is absolute and correct
- Check Shotcut version (tested with 23.x+)

## Advanced: Command Line Usage

For batch processing or scripting, you can use the command line tools directly. The full pipeline can be run as:

```bash
python src/pipeline.py --video videos/my_video.mp4 --model models/deit-small-nose-touch/final
```

See individual script files for detailed command line options.

## Credits

Built using HuggingFace Transformers, PyTorch, OpenCV, and Shotcut/MLT Framework.

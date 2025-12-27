# Eidolon - Automated Nose-Touch Detection Pipeline

A machine learning pipeline for detecting nose-touching moments in videos and automatically generating Shotcut project files for video editing.

## Overview

Eidolon uses a fine-tuned vision transformer (DeiT-Small) to classify video frames as "touching" or "not touching" the nose, then generates a Shotcut MLT project file that references all detected moments. This allows for instant preview and editing without intermediate video processing.

**Key Features:**

- Frame-level binary classification using HuggingFace Transformers
- Interactive labeling tool for creating training data
- Automated event detection with configurable buffers
- Direct Shotcut MLT project generation (no video re-encoding needed)
- GPU-accelerated inference

## Requirements

### System Dependencies

- Python 3.8+
- FFmpeg (for video metadata extraction)
- CUDA (optional, for GPU acceleration)
- Shotcut (for final rendering)

### Python Dependencies

See `requirements.txt`

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

3. **Verify FFmpeg installation**:

```bash
ffmpeg -version
```

4. **Install Shotcut** (if not already installed):

- Download from: https://shotcut.org/download/
- Or via package manager: `sudo pacman -S shotcut`

## Project Structure

```
staging/eidolon/
├── config.yaml          # Main configuration file
├── PLAN.md             # Detailed implementation plan
├── README.md           # This file
│
├── videos/             # Source videos
├── data/               # Generated data (gitignored)
│   ├── frames/        # Extracted frames for labeling
│   ├── labels.csv     # Manual labels
│   └── dataset/       # Train/val/test splits
│
├── models/             # Trained models (gitignored)
├── outputs/            # Pipeline outputs (gitignored)
│   ├── predictions/   # Frame-level predictions
│   ├── events/        # Detected events
│   └── projects/      # Shotcut MLT files
│
└── src/                # Source code
    ├── extract_frames.py
    ├── label_frames.py
    ├── prepare_dataset.py
    ├── train_classifier.py
    ├── infer_video.py
    ├── generate_mlt.py
    └── utils.py
```

## Quick Start (GUI)

For the easiest experience, use the unified GUI:

```bash
cd staging/eidolon
python src/eidolon_gui.py
```

The GUI provides a clean interface for:
- Selecting videos
- Extracting frames
- Labeling frames
- Preparing datasets
- Training models
- Running inference
- Generating Shotcut projects

All with real-time status updates and output logging.

## Workflow (Command Line)

Alternatively, you can run the pipeline manually via command line. The pipeline consists of 4 main phases:

### Phase 1: Data Preparation (One-time setup)

#### Step 1.1: Extract frames from sample video

```bash
cd staging/eidolon
python src/extract_frames.py --video videos/your_video.mp4
```

This extracts frames at 5 fps (default) to `data/frames/your_video/`.

**Options:**

- `--fps 10` - Extract at different frame rate
- `--output path/to/output` - Custom output directory

#### Step 1.2: Label frames manually

```bash
python src/label_frames.py --frames data/frames/your_video
```

**Controls:**

- `T` - Mark as True (positive class - nose touch detected)
- `F` - Mark as False (negative class - no nose touch)
- `S` - Skip (unsure/ambiguous)
- `Left/Right Arrow` - Navigate frames
- `Q` - Quit and save

**Real-time Balance Display:**
The tool shows live statistics on-screen:

- Total frames labeled
- True/False counts and percentages
- Balance status with color coding:
  - 🟢 Green = Balanced (ratio < 2:1)
  - 🟠 Orange = Imbalanced (ratio 2:1-3:1)
  - 🔴 Red = Very imbalanced (ratio > 3:1)

**Tips:**

- Label 500-1000 frames for initial training
- Include edge cases: hand near face, scratching, adjusting glasses
- Aim for roughly balanced classes (50/50 true vs false)
- Watch the balance indicator and adjust labeling strategy accordingly
- You can resume labeling later - progress is saved to `data/labels.csv`

#### Step 1.3: Prepare dataset splits

```bash
python src/prepare_dataset.py --labels data/labels.csv
```

This splits labeled data into train (70%), validation (15%), and test (15%) sets.

Output: `data/dataset/` with subdirectories for each split.

### Phase 2: Model Training (One-time setup)

#### Step 2.1: Train the classifier

```bash
python src/train_classifier.py --dataset data/dataset
```

This fine-tunes DeiT-Small on your labeled data. Training takes ~10-30 minutes depending on:

- Number of labeled samples
- GPU availability
- Number of epochs

**Options:**

- `--model facebook/convnext-tiny-224` - Use different pretrained model
- `--output models/my-model` - Custom output directory

**Monitoring:**

- Training progress shown in terminal (loss, metrics every 10 steps)
- Validation metrics printed at end of each epoch

**Output:**

- Best model saved to: `models/deit-small-nose-touch/final/`
- Test set metrics printed at end
- Training logs saved to: `models/deit-small-nose-touch/logs/`

**Expected Performance:**

- Target: >90% recall, >80% precision on test set
- If performance is poor, label more data (especially hard examples)

### Phase 3: Inference on Videos

#### Step 3.1: Run inference on target video

```bash
python src/infer_video.py \
    --video videos/your_video.mp4 \
    --model models/facebook-deit-small-patch16-224-nose-touch/final
```

This:

1. Extracts frames at 5 fps
2. Classifies each frame (touching vs not_touching)
3. Groups positive frames into events
4. Applies 1-second buffers before/after each event
5. Saves results to JSON

**Output:**

- `outputs/predictions/your_video_predictions.json` - Frame-level predictions
- `outputs/events/your_video_events.json` - Detected events

**Options:**

- `--threshold 0.3` - Lower threshold = more detections (higher recall)
- `--threshold 0.7` - Higher threshold = fewer detections (higher precision)

**Performance:**

- ~3-5 minutes per hour of video on GPU
- ~15-30 minutes per hour on CPU

### Phase 4: Generate Shotcut Project

#### Step 4.1: Create MLT project file

```bash
python src/generate_mlt.py \
    --events outputs/events/your_video_events.json
```

This generates a Shotcut project file that references all detected segments.

**Output:**

- `outputs/projects/your_video_project.mlt`

**Options:**

- `--output my_project.mlt` - Custom output path

#### Step 4.2: Open in Shotcut and render

```bash
shotcut outputs/projects/your_video_project.mlt
```

Or open via Shotcut GUI: File → Open File → select `.mlt` file

**In Shotcut:**

1. Preview clips in timeline
2. Adjust boundaries if needed (drag clip edges)
3. Add transitions (optional): Properties → Dissolve
4. Export: File → Export → choose format
5. Render final video (this is the only video processing step!)

## Configuration

Edit `config.yaml` to customize behavior:

### Key Settings

```yaml
# Model selection
model:
  name: "facebook/deit-small-patch16-224" # Or "facebook/convnext-tiny-224"

# Frame extraction rate
extraction:
  fps: 5 # Lower = faster, higher = more accurate

# Classification threshold
inference:
  threshold: 0.5 # Lower = more detections, higher = fewer false positives
  min_event_duration: 0.4 # Minimum event length in seconds

# Event buffers
mlt:
  buffer_before: 1.0 # Seconds to add before each event
  buffer_after: 1.0 # Seconds to add after each event

# Training
training:
  batch_size: 32
  learning_rate: 5e-5
  epochs: 20
```

## Examples

### Complete Pipeline (First Time)

```bash
# 1. Extract frames from sample video
python src/extract_frames.py --video videos/sample.mp4

# 2. Label frames (interactive GUI)
python src/label_frames.py --frames data/frames/sample

# 3. Prepare dataset
python src/prepare_dataset.py --labels data/labels.csv

# 4. Train model
python src/train_classifier.py --dataset data/dataset

# 5. Run inference on new video
python src/infer_video.py \
    --video videos/target_video.mp4 \
    --model models/deit-small-nose-touch/final

# 6. Generate Shotcut project
python src/generate_mlt.py \
    --events outputs/events/target_video_events.json

# 7. Open in Shotcut and render
shotcut outputs/projects/target_video_project.mlt
```

### Process New Videos (After Training)

```bash
# Run inference
python src/infer_video.py \
    --video videos/new_video.mp4 \
    --model models/deit-small-nose-touch/final

# Generate project
python src/generate_mlt.py \
    --events outputs/events/new_video_events.json

# Open in Shotcut
shotcut outputs/projects/new_video_project.mlt
```

## Troubleshooting

### No events detected

- Lower the threshold: `--threshold 0.3`
- Check model performance on test set
- Verify video has actual nose-touching moments
- Label more training data with similar conditions

### Too many false positives

- Raise the threshold: `--threshold 0.7`
- Label more negative examples (hand near face but not touching)
- Increase `min_event_duration` in config

### Training fails with CUDA out of memory

- Reduce batch size in `config.yaml`: `batch_size: 16` or `batch_size: 8`
- Or train on CPU (slower): set `use_gpu: false` in config

### FFmpeg not found

- Install: `sudo apt install ffmpeg` (Ubuntu) or `brew install ffmpeg` (Mac)
- Or download from: https://ffmpeg.org/download.html

### Labeling tool window doesn't respond

- Try different OpenCV build: `pip install opencv-python-headless` then `pip install opencv-python`
- Check your display server supports GUI windows

### MLT file doesn't open in Shotcut

- Verify video path is absolute and correct
- Check Shotcut version (tested with 23.x+)
- Try opening video directly in Shotcut first

### Model not learning / poor accuracy

- Label more data (aim for 1000+ samples)
- Check class balance (should be roughly 50/50)
- Include diverse examples (different lighting, angles, backgrounds)
- Try different model: `--model facebook/convnext-tiny-224`

## Advanced Usage

### Adjusting Event Detection

Edit `config.yaml`:

```yaml
inference:
  threshold: 0.3 # Lower = more sensitive
  min_event_duration: 0.8 # Ignore brief touches
```

### Using Different Models

DeiT-Small (default - recommended):

```bash
python src/train_classifier.py --dataset data/dataset
```

ConvNeXt-Tiny (alternative):

```bash
python src/train_classifier.py \
    --dataset data/dataset \
    --model facebook/convnext-tiny-224
```

EfficientNet-B0 (fastest):

```bash
python src/train_classifier.py \
    --dataset data/dataset \
    --model google/efficientnet-b0
```

### Batch Processing Multiple Videos

Create a simple bash script:

```bash
#!/bin/bash
for video in videos/*.mp4; do
    echo "Processing: $video"
    python src/infer_video.py \
        --video "$video" \
        --model models/deit-small-nose-touch/final

    video_name=$(basename "$video" .mp4)
    python src/generate_mlt.py \
        --events "outputs/events/${video_name}_events.json"
done
```

### Active Learning (Iterative Improvement)

1. Train initial model on 500 frames
2. Run inference on multiple videos
3. Review false positives/negatives
4. Extract those frames and add to training set
5. Retrain model
6. Repeat until satisfied

## Performance Notes

### Frame Extraction

- 60-minute video @ 5 fps = ~18,000 frames
- ~2-3 minutes to extract

### Training

- 1000 samples, 20 epochs:
  - GPU (RTX 3080): ~15 minutes
  - CPU (16 cores): ~60 minutes

### Inference

- 60-minute video:
  - GPU: ~4 minutes
  - CPU: ~20 minutes

### MLT Generation

- Instant (just XML generation)

### Final Render (in Shotcut)

- Depends on:
  - Output format/codec
  - Number of clips
  - System hardware
  - Typically: 0.5-2x video duration

## Tips for Best Results

1. **Label carefully**: Quality > quantity

   - Include edge cases
   - Be consistent with labeling criteria
   - When in doubt, skip (don't guess)

2. **Start small, iterate**:

   - Label 500 frames → train → evaluate
   - Add hard examples → retrain
   - Repeat until satisfied

3. **Tune threshold based on use case**:

   - High recall (detect everything): threshold = 0.3
   - High precision (minimize false positives): threshold = 0.7
   - Balanced: threshold = 0.5

4. **Use GPU for training and inference**:

   - 5-10x faster than CPU
   - Check availability: `python -c "import torch; print(torch.cuda.is_available())"`

5. **Preview before rendering**:
   - MLT file allows instant preview in Shotcut
   - Adjust if needed before final render
   - Can manually edit clips in Shotcut timeline

## Future Enhancements

Potential extensions (not yet implemented):

- Automated transitions in MLT
- Multi-action detection (ear-touch, hand-wave, etc.)
- Web interface for labeling and pipeline management
- Real-time detection for live streams
- Active learning loop with confidence-based sampling

## License

See parent repository license.

## Credits

Built using:

- HuggingFace Transformers
- PyTorch
- OpenCV
- Shotcut / MLT Framework

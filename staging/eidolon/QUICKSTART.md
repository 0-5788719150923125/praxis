# Eidolon Quick Start Guide

A condensed guide to get started quickly.

## Initial Setup (One Time)

### 1. Install Dependencies

```bash
cd staging/eidolon
pip install -r requirements.txt
```

### 2. Extract & Label Training Frames

```bash
# Extract frames from your sample video
python src/extract_frames.py --video videos/your_video.mp4

# Label 500-1000 frames (watch balance indicator!)
python src/label_frames.py --frames data/frames/your_video
# Controls: T=true, F=false, S=skip, Q=quit, Arrows=navigate
```

### 3. Prepare Dataset & Train Model

```bash
# Create train/val/test splits
python src/prepare_dataset.py --labels data/labels.csv

# Train the classifier (10-30 min on GPU)
python src/train_classifier.py --dataset data/dataset
```

## Process Videos (After Training)

### One-Command Pipeline

```bash
python src/pipeline.py \
    --video videos/your_video.mp4 \
    --model models/deit-small-nose-touch/final
```

This will:
1. Run inference to detect nose-touching moments
2. Generate Shotcut project file

### Open in Shotcut

```bash
shotcut outputs/projects/your_video_project.mlt
```

Then: File → Export → Render

## Tuning

### More Detections (Higher Recall)

```bash
python src/pipeline.py \
    --video videos/your_video.mp4 \
    --model models/deit-small-nose-touch/final \
    --threshold 0.3
```

### Fewer False Positives (Higher Precision)

```bash
python src/pipeline.py \
    --video videos/your_video.mp4 \
    --model models/deit-small-nose-touch/final \
    --threshold 0.7
```

## Troubleshooting

**No events detected?**
- Lower threshold: `--threshold 0.3`
- Check video has nose-touching moments
- Verify model performance on test set

**Too many false positives?**
- Raise threshold: `--threshold 0.7`
- Label more negative examples
- Retrain model

**Training fails with CUDA error?**
- Reduce batch size in `config.yaml`: `batch_size: 16`

For more details, see `README.md`.

# 🎯 Implementation Plan: Nose-Touch Video Classifier Pipeline

## Executive Summary

We're building a **frame-level binary classifier** to detect nose-touching in videos, then generating a Shotcut project file (MLT XML) that references those moments. The pipeline consists of 4 distinct phases: **Data Preparation** → **Model Fine-tuning** → **Inference & Detection** → **Shotcut MLT Generation**.

---

## 🏗️ Architecture Overview

```
┌─────────────────┐
│  Source Videos  │
│   (30-60 min)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Frame Sampling         │  ← Extract frames @ 5-10 fps
│  (OpenCV/FFmpeg)        │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Binary Classifier      │  ← DeiT-Small (fine-tuned)
│  (HuggingFace)          │    Input: 224x224 frame
└────────┬────────────────┘    Output: [touch, no-touch]
         │
         ▼
┌─────────────────────────┐
│  Event Detection        │  ← Group positive frames into events
│  (Temporal Logic)       │    Apply 1s buffer before/after
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  MLT XML Generation     │  ← Generate Shotcut project file
│  (Direct References)    │    References source video segments
└─────────────────────────┘    No actual video processing!
         │
         ▼
  [Shotcut Project File]
         │
         ▼
  (Manual: Render in Shotcut when satisfied)
```

**Key Advantage:** No intermediate video processing! All detected segments are referenced in a single Shotcut project file. Video processing only happens once at the end when you render the final output.

---

## 🎨 Model Selection: DeiT-Small (Primary)

**Model ID:** `facebook/deit-small-patch16-224`

**Why this model:**
1. **Data efficiency**: Excels with hundreds to low thousands of labeled frames
2. **Spatial relationship detection**: Transformer self-attention captures "hand-near-nose" vs "hand-touching-nose" spatial nuance
3. **Transfer learning**: Distillation training provides strong priors for fine-grained tasks
4. **Speed**: 3-4x faster than ViT-Base, suitable for processing long videos
5. **Proven track record**: 79.9% ImageNet accuracy, excellent downstream performance

**Backup Option:** `facebook/convnext-tiny-224` (if CNN approach proves better with your specific data distribution)

---

## 📋 Phase 1: Data Preparation Pipeline

### 1.1 Frame Extraction
**Purpose:** Convert videos to labeled training frames

**Key Decisions:**
- **Sampling rate:** 5 fps (down from 60 fps source)
  - Rationale: Nose-touches last 0.5-2 seconds typically; 5fps = ~2.5-10 frames per event
  - Reduces 196k frames → 16k frames per 54-min video
  - Saves processing time, storage, and labeling effort
- **Resolution:** Resize to 224x224 (DeiT-Small input size)
- **Format:** JPEG at 95% quality (good compression, minimal artifacts)

**Tool:** `extract_frames.py`
- Input: Video file path
- Output: `frames/video_name/frame_NNNNNN.jpg`
- Metadata: `frames/video_name/metadata.json` (frame timestamps for MLT generation later)

### 1.2 Manual Labeling Workflow
**Purpose:** Create training dataset

**Approach:**
- Simple GUI tool (or even just manual file organization)
- Present frames one-by-one
- User presses key: `T` = touching, `N` = not touching, `S` = skip/unsure
- Output: `labels.csv` with columns: `frame_path, label, timestamp`

**Data Collection Strategy:**
1. Label ~500-1000 frames from sample video to start
2. Aim for ~50/50 class balance (undersample "not touching" if needed)
3. Include edge cases: hand near face, scratching, adjusting glasses, etc.

**Tool:** `label_frames.py` (simple OpenCV window + keyboard input)

### 1.3 Dataset Preparation
- Split into train/val/test (70/15/15)
- Create HuggingFace Dataset or PyTorch Dataset
- Data augmentation during training:
  - Random horizontal flip (person might touch either side)
  - Slight rotation (±5°)
  - Color jitter (lighting variations)
  - Random crop + resize (224x224)

---

## 🧠 Phase 2: Model Fine-tuning

### 2.1 Fine-tuning Setup
**Framework:** HuggingFace Transformers + Trainer API

**Architecture:**
```
DeiT-Small (pretrained on ImageNet)
    ↓
Replace classification head: 1000 classes → 2 classes
    ↓
Fine-tune entire model (or freeze early layers initially)
```

**Training Configuration:**
- **Loss:** Binary cross-entropy (or focal loss if class imbalance)
- **Optimizer:** AdamW
- **Learning rate:** 1e-4 to 5e-5 (lower for fine-tuning)
- **Batch size:** 16-32 (depends on GPU memory)
- **Epochs:** 10-20 (early stopping based on val loss)
- **Scheduler:** Cosine decay with warmup

**Key Considerations:**
- **Class balance:** If imbalanced, use class weights or focal loss
- **Metrics:** Accuracy, Precision, Recall, F1 (prioritize Recall to minimize false negatives)
- **Checkpointing:** Save best model based on val F1 score

**Tool:** `train_classifier.py`

### 2.2 Evaluation & Iteration
- Examine false positives/negatives on test set
- Iteratively add hard examples to training set
- Consider ensemble if single model isn't accurate enough

---

## 🔍 Phase 3: Inference & Event Detection

### 3.1 Frame-level Inference
**Purpose:** Classify all frames in target video

**Process:**
1. Extract frames from video @ 5 fps (same as training)
2. Load fine-tuned model
3. Batch inference (32-64 frames at a time)
4. Output: `predictions.json` with `[{frame_idx, timestamp, probability, predicted_class}, ...]`

**Optimization:**
- Use GPU if available
- Process in batches for speed
- Cache predictions to disk (in case of crash)

**Tool:** `infer_video.py`

### 3.2 Event Detection Logic
**Purpose:** Convert frame-level predictions to discrete "events"

**Algorithm:**
```
Events = []
current_event = None

for frame in predictions:
    if frame.predicted_class == "touching":
        if current_event is None:
            current_event = Event(start=frame.timestamp)
        current_event.end = frame.timestamp
    else:
        if current_event is not None:
            Events.append(current_event)
            current_event = None

# Add buffer
for event in Events:
    event.start -= 1.0  # 1 second buffer before
    event.end += 1.0    # 1 second buffer after
    event.start = max(0, event.start)  # Clamp to video bounds
```

**Refinements:**
- Optional: Require minimum event duration (e.g., 2 frames = 0.4s to filter flickering)
- Optional: Merge events closer than N seconds (but you said NO merging, so skip)
- **No temporal smoothing** (as requested - classifier has no memory)

**Output:** `events.json` with `[{event_id, start_time, end_time, num_frames}, ...]`

---

## 🎬 Phase 4: Shotcut MLT Project Generation

### 4.1 MLT XML Format Overview

Shotcut uses the **MLT (Media Lovin' Toolkit) XML format** for project files. This is a simple, well-documented XML structure that references video files and defines playlists, transitions, and effects.

**Key MLT Concepts:**
- **Producer:** References a source video file
- **Playlist:** Defines a sequence of clips (entries) from producers
- **Entry:** A segment of a producer with `in` and `out` points (frame numbers)
- **Tractor:** Combines multiple playlists (video tracks, audio tracks)

### 4.2 MLT Generation Strategy

**Input:** `events.json` from Phase 3

**Output:** `project.mlt` - Shotcut project file

**Structure:**
```xml
<?xml version="1.0" encoding="utf-8"?>
<mlt version="7.0.0">
  <!-- Define the source video as a producer -->
  <producer id="producer0">
    <property name="resource">/absolute/path/to/source_video.mp4</property>
    <property name="mlt_service">avformat-novalidate</property>
  </producer>

  <!-- Create a playlist with all detected segments -->
  <playlist id="playlist0">
    <!-- Event 1: frames 1500-1800 (timestamps 5.0s - 6.0s @ 60fps) -->
    <entry producer="producer0" in="300" out="360"/>  <!-- 5.0s - 6.0s -->

    <!-- Event 2: frames 4200-4500 (timestamps 14.0s - 15.0s @ 60fps) -->
    <entry producer="producer0" in="840" out="900"/>  <!-- 14.0s - 15.0s -->

    <!-- ... more entries ... -->
  </playlist>

  <!-- Tractor combines everything -->
  <tractor id="tractor0">
    <track producer="playlist0"/>
  </tractor>
</mlt>
```

**Key Implementation Details:**

1. **Frame Number Conversion:**
   - Events are in seconds (e.g., 5.0s - 6.0s)
   - MLT uses frame numbers
   - Need source video FPS: `frame_number = timestamp * fps`
   - Source video is 60fps, so 5.0s = frame 300

2. **Absolute Paths:**
   - MLT files use absolute paths to source videos
   - Need to resolve paths when generating XML

3. **No Re-encoding:**
   - Everything is just references to the source file
   - No video processing until final render

### 4.3 Tool: `generate_mlt.py`

**Functionality:**
```python
def generate_mlt_project(source_video_path, events, output_mlt_path):
    """
    Generate Shotcut MLT project file from detected events.

    Args:
        source_video_path: Absolute path to source video
        events: List of {start_time, end_time} dicts
        output_mlt_path: Where to save .mlt file
    """
    # 1. Get video FPS
    fps = get_video_fps(source_video_path)

    # 2. Convert event timestamps to frame numbers
    entries = []
    for event in events:
        in_frame = int(event['start_time'] * fps)
        out_frame = int(event['end_time'] * fps)
        entries.append({'in': in_frame, 'out': out_frame})

    # 3. Generate MLT XML
    xml = create_mlt_xml(source_video_path, entries, fps)

    # 4. Write to file
    with open(output_mlt_path, 'w') as f:
        f.write(xml)
```

**XML Generation:**
- Use Python's `xml.etree.ElementTree` or `lxml`
- Template-based or programmatic generation
- Include proper MLT headers and metadata

### 4.4 Future: Transitions & Effects

**Optional Enhancements (post-MVP):**
- Add cross-dissolve transitions between clips
- Add filters (color correction, etc.)
- Multiple video tracks (e.g., overlay text, graphics)

**MLT Transition Example:**
```xml
<transition id="transition0" in="360" out="380">
  <property name="mlt_service">mix</property>
  <property name="a_track">0</property>
  <property name="b_track">1</property>
</transition>
```

This can be added later once the basic pipeline works.

### 4.5 Workflow After MLT Generation

**User Workflow:**
1. Pipeline generates `project.mlt`
2. Open `project.mlt` in Shotcut
3. Preview all detected clips in timeline
4. **Manual refinement** (optional):
   - Adjust clip boundaries
   - Add transitions
   - Rearrange order
   - Add effects
5. **Render final video** (File → Export)
   - This is the ONLY video processing step
   - Shotcut handles all encoding

**Advantages:**
- ✅ Instant preview (no waiting for clip extraction)
- ✅ Easy iteration (adjust thresholds, re-run pipeline, regenerate MLT)
- ✅ Manual refinement possible (user can tweak in Shotcut)
- ✅ Fast workflow (no intermediate video files)

---

## 📁 Project Structure

```
staging/eidolon/
├── .gitignore              # (already exists)
├── videos/                 # (already exists)
│   └── Focus on YOUR Life...mp4
├── PLAN.md                 # This file
├── README.md               # Setup instructions
├── requirements.txt        # Python dependencies
├── config.yaml             # Configuration (sampling rate, buffer, model ID, etc.)
│
├── data/                   # Generated data (gitignored)
│   ├── frames/            # Extracted frames for labeling
│   │   └── video_name/
│   │       ├── frame_000001.jpg
│   │       ├── ...
│   │       └── metadata.json
│   ├── labels.csv         # Manual labels
│   └── dataset/           # Prepared train/val/test splits
│
├── models/                 # Saved models (gitignored)
│   └── deit-small-nose-touch/
│       ├── checkpoint-best/
│       └── training_logs/
│
├── outputs/                # Generated outputs (gitignored)
│   ├── predictions/       # Inference results
│   │   └── video_name_predictions.json
│   ├── events/            # Detected events
│   │   └── video_name_events.json
│   └── projects/          # Generated Shotcut project files
│       └── video_name_project.mlt
│
└── src/                    # Source code
    ├── __init__.py
    ├── extract_frames.py   # Phase 1: Frame extraction
    ├── label_frames.py     # Phase 1: Labeling GUI
    ├── prepare_dataset.py  # Phase 1: Train/val/test split
    ├── train_classifier.py # Phase 2: Fine-tuning
    ├── infer_video.py      # Phase 3: Inference + event detection
    ├── generate_mlt.py     # Phase 4: MLT project generation
    └── utils.py            # Shared utilities
```

---

## 🔧 Dependencies

**requirements.txt:**
```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.35.0
accelerate>=0.25.0
datasets>=2.14.0
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
pyyaml>=6.0
tqdm>=4.65.0
scikit-learn>=1.3.0
lxml>=4.9.0
```

**System Dependencies:**
- FFmpeg (for video metadata extraction)
- CUDA (optional, for GPU acceleration)
- Shotcut (for final rendering)

---

## ⚙️ Configuration File (config.yaml)

```yaml
# Model configuration
model:
  name: "facebook/deit-small-patch16-224"
  num_classes: 2
  image_size: 224

# Frame extraction
extraction:
  fps: 5  # Sample rate
  quality: 95  # JPEG quality

# Training
training:
  batch_size: 32
  learning_rate: 5e-5
  epochs: 20
  early_stopping_patience: 5
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15

# Inference
inference:
  batch_size: 64
  threshold: 0.5  # Classification threshold
  min_event_duration: 0.4  # Minimum duration in seconds (2 frames @ 5fps)

# MLT project generation
mlt:
  buffer_before: 1.0  # seconds
  buffer_after: 1.0   # seconds
  output_dir: "outputs/projects"
```

---

## 🎯 Implementation Workflow

### **Stage 1: Setup & Data Prep** (Days 1-2)
1. Create project structure
2. Install dependencies
3. Implement `extract_frames.py`
4. Extract frames from sample video
5. Implement `label_frames.py`
6. **Manual work:** Label 500-1000 frames

### **Stage 2: Training** (Day 3)
1. Implement `prepare_dataset.py`
2. Implement `train_classifier.py`
3. Fine-tune DeiT-Small on labeled data
4. Evaluate on test set
5. **Decision point:** If accuracy insufficient, label more data or try ConvNeXt-Tiny

### **Stage 3: Inference Pipeline** (Day 4)
1. Implement `infer_video.py` (both classification + event detection)
2. Run on sample video
3. Validate detected events manually (watch sample video, check timestamps)
4. Tune threshold / min_event_duration if needed

### **Stage 4: MLT Generation** (Day 5)
1. Research MLT XML format (or use existing Shotcut project as template)
2. Implement `generate_mlt.py`
3. Generate MLT file for sample video
4. **Manual validation:** Open in Shotcut, verify all clips are present and correctly timed
5. Iterate if timing issues

### **Stage 5: Testing & Iteration** (Days 6-7)
1. Run full pipeline on sample video end-to-end
2. Open generated MLT in Shotcut, preview clips
3. Identify failure modes (false positives/negatives)
4. Collect hard examples, add to training set
5. Retrain and evaluate improvements
6. Document workflow and create usage instructions

---

## 🚨 Critical Considerations & Risk Mitigation

### **Risk 1: Insufficient Training Data**
- **Mitigation:** Start with 500-1000 frames; DeiT-Small is data-efficient
- **Fallback:** Active learning - run inference, manually label misclassified frames, retrain

### **Risk 2: Class Imbalance**
- Most frames won't have nose-touching (90/10 split?)
- **Mitigation:** Undersample negatives, use class weights, or focal loss

### **Risk 3: False Negatives (Missing Touches)**
- You prioritized detecting every instance, even brief ones
- **Mitigation:** Tune classification threshold down (0.3 instead of 0.5) to favor recall over precision

### **Risk 4: False Positives**
- Hand near face, scratching, adjusting glasses
- **Mitigation:** Include these as negative examples during labeling

### **Risk 5: Processing Time**
- 54-min video @ 5fps = ~16k frames
- Inference: ~0.01s/frame (DeiT-Small on GPU) = ~160s = 3 minutes
- **Acceptable for batch processing**

### **Risk 6: Generalization to New Videos**
- Different people, lighting, backgrounds
- **Mitigation:** Include diverse examples in training set; data augmentation

### **Risk 7: MLT Format Compatibility**
- Shotcut MLT format might change between versions
- **Mitigation:** Test with current Shotcut version; use template from manually-created project

### **Risk 8: Frame Number Calculation**
- Variable frame rate videos (VFR) could cause sync issues
- **Mitigation:** Use FFmpeg to convert to constant frame rate (CFR) if needed

---

## 🎬 Success Criteria

**Minimum Viable Product (MVP):**
- ✅ Pipeline runs end-to-end on sample video
- ✅ Detects >90% of nose-touches (high recall)
- ✅ <20% false positive rate (precision >80%)
- ✅ Generated MLT file opens in Shotcut
- ✅ All clips have correct timing (±0.5s)
- ✅ Clips play correctly in Shotcut timeline

**Stretch Goals:**
- ✅ >95% recall, >90% precision
- ✅ Generalization to new videos without retraining
- ✅ Processing time <5 minutes per hour of video
- ✅ Automated transitions in MLT

---

## 🔄 Future Extensions (Post-MVP)

1. **Automated Transitions:** Add cross-dissolve between clips in MLT
2. **Multiple Actions:** Extend to detect other gestures (ear-touch, hand-wave, etc.)
3. **Web Interface:** Simple UI for uploading videos and downloading MLT projects
4. **Batch Processing:** Process multiple videos and generate combined MLT
5. **Active Learning:** Continuously improve model with new labeled data
6. **Export Presets:** Pre-configured Shotcut export settings for common formats

---

## 📊 Deliverables

By the end of implementation, you'll have:

1. **Trained Model:** Fine-tuned DeiT-Small checkpoint
2. **Inference Pipeline:** Command-line tool to process videos
3. **MLT Project File:** Shotcut project with all detected segments
4. **Documentation:** README with usage instructions
5. **Reproducible Workflow:** Config-driven pipeline for future videos

---

## 🎯 Example End-to-End Usage

**Command-line workflow:**

```bash
# 1. Extract frames for labeling
python src/extract_frames.py --video videos/my_video.mp4 --output data/frames

# 2. Label frames manually
python src/label_frames.py --frames data/frames/my_video

# 3. Prepare dataset
python src/prepare_dataset.py --labels data/labels.csv --output data/dataset

# 4. Train classifier
python src/train_classifier.py --config config.yaml --dataset data/dataset

# 5. Run inference on full video
python src/infer_video.py --video videos/my_video.mp4 --model models/deit-small-nose-touch

# 6. Generate Shotcut project
python src/generate_mlt.py --video videos/my_video.mp4 --events outputs/events/my_video_events.json

# 7. Open in Shotcut
shotcut outputs/projects/my_video_project.mlt

# 8. Render final video in Shotcut (manual step)
```

**Or as a single pipeline script:**

```bash
python src/pipeline.py --video videos/my_video.mp4 --model models/deit-small-nose-touch --output outputs/projects/my_video.mlt
```

---

## ❓ Open Questions Before Implementation

1. **MLT Template:** Should I create a sample Shotcut project manually first, then use it as a template for the XML structure?
2. **Labeling tool preference:** Simple keyboard-based OpenCV window? Or something fancier (web UI)?
3. **Class balance strategy:** Should we actively undersample negatives during labeling, or label naturally and handle imbalance in training?
4. **GPU availability:** Do you have a GPU for training/inference? (Will affect batch sizes and timing)

---

## 🎯 Recommendation: Proceed?

This plan balances your requirements:
- ✅ **Small:** Focused, single-purpose pipeline
- ✅ **Fast:** No intermediate video processing; DeiT-Small is efficient
- ✅ **Reliable:** Fine-tuned classifier > hand-crafted heuristics
- ✅ **Accurate:** Prioritizes recall (minimize false negatives)
- ✅ **Extensible:** Clean architecture for future enhancements
- ✅ **Smart:** Only process video once at the end (your suggestion!)

The MLT generation approach is significantly better than clip extraction because:
- No waiting for clip extraction
- Easy to iterate (re-run pipeline, regenerate MLT in seconds)
- Manual refinement possible in Shotcut GUI
- Single render step at the end

Ready to implement!

# Future Approaches for Video Event Detection

This document outlines potential improvements to the Eidolon pipeline for more robust video event detection.

## Current Limitations

The current system has several known limitations:

1. **Frame Independence**: Each frame is classified independently, ignoring temporal context
2. **Motion Blur**: Still frames often have motion blur, which degrades classification accuracy
3. **Temporal Noise**: Single-frame misclassifications create false positives
4. **Limited Context**: Model cannot learn motion patterns (e.g., hand approaching vs. retracting)
5. **Person Bias**: Models can overfit to specific individuals in training data

## Implemented Solutions

### Temporal Smoothing (Current)

We've implemented post-processing temporal smoothing with three methods:

- **Median filtering**: Replaces each prediction with the median of a surrounding window
  - Best for removing isolated spikes/false positives
  - Preserves sharp event boundaries
  - Window size 5-7 frames recommended

- **Moving average**: Smooths by averaging neighboring frames
  - Creates smoother transitions
  - Can blur true event boundaries
  - Good for reducing jitter

- **Consensus voting**: Requires multiple frames in window to agree
  - Most aggressive approach
  - Good for very noisy predictions
  - May miss brief events

**Configuration** (in `config.yaml`):
```yaml
inference:
  temporal_smoothing:
    enabled: true
    method: "median"  # or "moving_average", "consensus"
    window_size: 5
    consensus_threshold: 0.6
```

## Future Improvements

### 1. Clip-Based Classification (Medium Complexity)

Instead of classifying individual frames, classify short video clips (8-16 frames).

#### Approach A: Frame Stacking
**How it works:**
- Stack N consecutive frames as input channels
- Instead of 3 RGB channels, use 3×N channels
- Use existing DeiT/ViT architecture with modified input layer
- Label the center frame of each window

**Advantages:**
- Minimal architectural changes
- Captures temporal context through stacked frames
- Simple to implement and label

**Disadvantages:**
- Limited temporal modeling (just concatenation)
- Large input size (3×N channels)
- Doesn't explicitly model motion

**Implementation effort:** 1-2 days

**Labeling changes:** Minimal - extract clips centered on labeled frames

---

#### Approach B: 3D CNN (I3D, X3D, SlowFast)
**How it works:**
- Use models designed for video understanding
- Process spatial and temporal dimensions simultaneously
- 3D convolutions capture motion patterns
- Models available in `torchvision` or `pytorchvideo`

**Recommended models:**
- **X3D-S**: Lightweight, efficient, good for short clips
- **SlowFast-50**: Dual pathway (slow for spatial, fast for motion)
- **I3D**: Inflated 3D convolutions, proven architecture

**Advantages:**
- Explicitly models motion and temporal patterns
- Battle-tested architectures for action recognition
- Better at handling motion blur (uses it as motion information)
- Can distinguish approach vs. retract motions

**Disadvantages:**
- More compute-intensive than single-frame models
- Requires more training data
- Slightly more complex data pipeline

**Implementation effort:** 3-5 days

**Labeling changes:** Extract clips instead of frames (similar effort)

**Example code structure:**
```python
import torch
from pytorchvideo.models import x3d

model = x3d.create_x3d(
    model_num_class=2,
    input_clip_length=8,  # 8 frames per clip
    input_crop_size=224,
)
```

---

#### Approach C: Feature Averaging
**How it works:**
- Extract DeiT features from N consecutive frames
- Average features across frames
- Pass averaged features to classifier
- Quick experiment to test temporal aggregation

**Advantages:**
- Simple to implement
- Reuses existing trained model
- Reduces impact of single blurry frames
- Good proof-of-concept for temporal approaches

**Disadvantages:**
- Naive temporal modeling
- Loses fine-grained temporal structure
- Not as powerful as true video models

**Implementation effort:** Half day

---

### 2. Video Transformers (High Complexity)

For larger-scale deployments or when clip-based models aren't sufficient.

#### Models:
- **TimeSformer**: Pure attention across space and time
- **Video Swin Transformer**: Hierarchical video transformer
- **MViT (Multiscale Vision Transformer)**: Multiscale pooling for videos

**Advantages:**
- State-of-the-art performance on action recognition
- Can handle longer temporal context (16-32 frames)
- Learns complex spatiotemporal patterns
- Better generalization

**Disadvantages:**
- Very data-hungry (needs 1000s of examples)
- Computationally expensive
- Longer training time
- May be overkill for simple binary classification

**When to use:** Only if clip-based approaches (X3D/SlowFast) fail to achieve desired performance

**Implementation effort:** 1-2 weeks

---

### 3. Temporal Action Localization (High Complexity)

For variable-length events or when precise start/end timestamps matter.

#### Models:
- **Temporal Action Proposals (TAP)**
- **Boundary Matching Network (BMN)**
- **ActionFormer**: Transformer-based action localization

**How it works:**
- Model predicts start/end timestamps directly
- Can handle variable-length events
- Outputs confidence scores for event boundaries
- Common in sports analysis, surveillance

**Advantages:**
- Precise event boundaries
- Handles variable-length events naturally
- Learns event structure (start, middle, end)

**Disadvantages:**
- Requires segment-level labeling (much harder than frame labeling)
- More complex training pipeline
- Overkill for brief, discrete events like nose-taps

**When to use:** Only if events have significant duration variation or precise boundaries are critical

**Implementation effort:** 2-3 weeks

---

### 4. Data-Level Improvements

Improvements to training data to reduce person bias and improve generalization.

#### Class Balancing
**Problem:** Over-representation of certain individuals leads to person-specific models

**Solutions:**
- **Downsample over-represented individuals** during training
- **Upsample under-represented individuals** with augmentation
- **Stratified sampling** to ensure balanced batches
- **Per-person validation splits** to measure generalization

**Implementation effort:** 1 day

---

#### Strong Augmentation
**Current augmentation** (from Transformers library):
- Random resized crop
- Color jitter
- Horizontal flip

**Additional augmentation to try:**
- **Color jitter (stronger)**: Make model person/lighting invariant
- **Random perspective transforms**: Handle camera angle variations
- **Mixup/CutMix**: Regularization for better generalization
- **Random erasing**: Occlusion robustness
- **Temporal augmentation**: Speed up/slow down clips

**Implementation effort:** 1-2 days

---

#### Two-Stage Fine-Tuning
**Problem:** New people introduced late cause distribution shift

**Solution:**
1. Train model on original dataset
2. Freeze backbone (all layers except classifier)
3. Fine-tune only classifier on new people
4. Optionally unfreeze and fine-tune with low learning rate

**Advantages:**
- Prevents catastrophic forgetting
- Fast adaptation to new individuals
- Preserves learned features

**Implementation effort:** 1 day

---

## Recommended Implementation Roadmap

### Phase 1: Quick Wins (This Week)
✅ **Temporal smoothing** - Already implemented
- Test median vs. moving average vs. consensus
- Tune window size (try 5, 7, 11)
- Evaluate false positive reduction

### Phase 2: Clip-Based Models (Next Week)
Recommended: **X3D-S or SlowFast**
1. Implement clip extraction from labeled frames
2. Adapt training pipeline for clip input
3. Train X3D-S on existing labeled data
4. Compare performance to frame-based baseline

**Why this first:** Best performance-to-effort ratio. Video models are designed for exactly this problem.

### Phase 3: Data Improvements (Following Week)
1. Implement stratified sampling by person
2. Add stronger augmentation (color jitter, perspective)
3. Collect more diverse training data (if needed)
4. Per-person evaluation metrics

### Phase 4: Advanced Approaches (If Needed)
Only if Phase 2 & 3 don't achieve desired performance:
- Video transformers (MViT, TimeSformer)
- Temporal action localization
- Hybrid approaches (frame + clip models)

---

## Comparison Matrix

| Approach | Complexity | Data Needs | Accuracy Gain | Inference Speed | Implementation Time |
|----------|-----------|------------|---------------|-----------------|---------------------|
| Temporal Smoothing | Low | None | +5-15% | Same | ✅ Done |
| Frame Stacking | Low | None | +10-20% | -20% | 1-2 days |
| Feature Averaging | Low | None | +5-10% | -10% | 0.5 days |
| X3D / SlowFast | Medium | Same | +20-40% | -50% | 3-5 days |
| Video Transformers | High | 2-3x more | +30-50% | -70% | 1-2 weeks |
| Action Localization | Very High | Segment labels | +40-60% | -60% | 2-3 weeks |

---

## Evaluation Metrics

To properly evaluate improvements, track:

1. **Per-person accuracy**: Ensure model generalizes across individuals
2. **False positive rate**: Count spurious detections
3. **Event precision/recall**: Do detected events match ground truth?
4. **Temporal accuracy**: How close are event boundaries?
5. **Inference throughput**: Frames per second

**Recommended test set:** Videos of individuals NOT in training set to measure generalization.

---

## References

### Papers
- **I3D**: "Quo Vadis, Action Recognition?" (Carreira & Zisserman, 2017)
- **SlowFast**: "SlowFast Networks for Video Recognition" (Feichtenhofer et al., 2019)
- **X3D**: "X3D: Expanding Architectures for Efficient Video Recognition" (Feichtenhofer, 2020)
- **TimeSformer**: "Is Space-Time Attention All You Need?" (Bertasius et al., 2021)
- **Video Swin**: "Video Swin Transformer" (Liu et al., 2022)

### Libraries
- **PyTorchVideo**: https://pytorchvideo.org/ (recommended for X3D, SlowFast)
- **MMAction2**: https://github.com/open-mmlab/mmaction2 (comprehensive video understanding)
- **Torchvision**: Includes basic video models (R3D, MC3, S3D)

---

## Questions to Consider

Before investing in more complex approaches, ask:

1. **Is the current accuracy bottleneck the model or the data?**
   - If model: Try clip-based approaches
   - If data: Collect more diverse examples

2. **What's the cost of false positives vs. false negatives?**
   - High FP cost: Use aggressive temporal smoothing or consensus
   - High FN cost: Lower threshold, use ensemble models

3. **How much does motion matter for your task?**
   - Nose-taps involve clear motion → Video models will help significantly
   - Static detection (e.g., "glasses on/off") → Frame-based may suffice

4. **Do you have GPU resources for heavier models?**
   - Limited: Stick with X3D-S or frame stacking
   - Abundant: Try SlowFast or video transformers

---

## Conclusion

The most pragmatic path forward:

1. **Use temporal smoothing** (already implemented) to reduce false positives
2. **Switch to X3D or SlowFast** for true video understanding with motion
3. **Improve data diversity** to reduce person bias
4. **Only pursue video transformers** if the above don't achieve goals

Video models like X3D are the "sweet spot" - significantly better than frame-based approaches without the complexity of cutting-edge transformers.

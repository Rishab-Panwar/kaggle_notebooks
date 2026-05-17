# GenAI NPPE - Image Reconstruction Under Corruption

> **Competition:** Mathematical Foundations of GenAI - NPPE (IIT Madras)  
> **Task:** Reconstruct clean 32×32 RGB images from corrupted inputs with unknown degradation types  
> **Metric:** Mean Squared Error (MSE) - lower is better  
> **Approach:** NAFNet (Nonlinear Activation Free Network) with 2-model ensemble  
> **Result:** Ranked **16/110** teams  
> **Competition link:** [GenAI NPPE](https://www.kaggle.com/t/d934bf5c51ae4333a57f433b4bc592d6)

---

## The Problem

The task is to take corrupted 32×32 RGB images and reconstruct their clean versions. The corruptions are diverse and compounded: masking, noise (Gaussian, salt-and-pepper, speckle), blur (Gaussian, motion, defocus), geometric distortions (rotation, scaling, shearing), and global transformations (brightness, contrast, saturation shifts). Each image suffers from a **random composition** of multiple degradation types at varying severities.

Critically, **the exact corruption types and severity levels are not provided**. The model must learn to reverse arbitrary combinations of degradations from paired training data alone.

This is not a classification problem where you identify the corruption and apply a targeted fix. It's a general inverse problem: given `y = f(x)` where `f` is an unknown composition of degradations, learn `g` such that `g(y) ≈ x` for any `f` in the corruption distribution.

---

## How I Thought About It

My initial instinct was to try classical approaches: identify the dominant corruption type (e.g., via variance analysis for noise, FFT for blur) and route to specialized filters. This failed immediately. Most images have **multiple overlapping corruptions** - an image might be rotated, blurred, and noisy simultaneously. Classical methods can't untangle these without explicit knowledge of the corruption sequence, which we don't have.

The breakthrough came from recognizing this as an **image-to-image translation problem** where the model needs to learn a flexible inverse mapping from the corruption manifold back to clean space. This pointed toward deep learning architectures designed specifically for image restoration.

I surveyed recent work in blind image restoration and found NAFNet (Chen et al., 2022) - a U-Net-style architecture that achieves state-of-the-art results on multiple restoration benchmarks (denoising, deblurring, deraining) **without** using nonlinear activation functions like ReLU. The key insight of NAFNet: for image restoration tasks, carefully designed linear gating mechanisms (SimpleGate) combined with channel attention can outperform ReLU-based networks while being faster and more parameter-efficient.

The architecture's residual learning formulation also matched the problem perfectly: instead of predicting the clean image directly, predict the **residual** (noise/corruption) and subtract it from the input. This is easier to learn because residuals tend to be sparser and have smaller dynamic range than absolute pixel values.

---

## How I Architected It

### The Two-Model Ensemble Strategy

I trained two independent NAFNet models with different configurations and random seeds, then averaged their predictions. Ensemble diversity comes from:

| Model | Width | Middle Blocks | Parameters | Batch Size | Seed |
|-------|-------|---------------|------------|------------|------|
| Model 1 | 48 | 12 | ~84M | 128 | 42 |
| Model 2 | 64 | 16 | ~145M | 96 | 123 |

**Why ensemble?** Single models overfit to specific corruption patterns in the training set. By varying architecture capacity and training randomness, the models learn slightly different feature representations. Averaging smooths out individual model biases and improves generalization to unseen corruption combinations.

### NAFNet Architecture Details

NAFNet is a U-Net with skip connections, but replaces standard conv blocks with **NAFBlocks** - specialized restoration blocks with three key components:

#### 1. SimpleGate (Nonlinear Activation-Free Gating)
```
Input → Conv1×1 → expand to 2C channels → split into (A, B) → output = A ⊙ B
```

Instead of ReLU(x), use gating: `x1 ⊙ x2` where `⊙` is element-wise multiplication. This provides nonlinearity without the information loss from thresholding negative values (critical for restoration where negative residuals matter).

#### 2. Simplified Channel Attention (SCA)
```
Features → GlobalAvgPool → Conv1×1 → scale original features
```

Lets the network emphasize important channels adaptively. For restoration, this helps the model focus on channels containing corruption-specific patterns.

#### 3. Layer Normalization
Uses LayerNorm2d instead of BatchNorm to ensure consistent behavior at different batch sizes and avoid train-test distribution shift issues common in restoration tasks.

### Complete Model Architecture

```
Input: 32×32×3 RGB (corrupted)

┌─────────────────────────────────────┐
│  Intro Conv3×3 (3 → 64 channels)    │
└────────────┬────────────────────────┘
             │
    ┌────────▼─────────┐
    │  Encoder Level 1  │ [2 NAFBlocks, 64 channels]
    │  ↓ Conv stride=2  │
    │  Encoder Level 2  │ [2 NAFBlocks, 128 channels]
    │  ↓ Conv stride=2  │
    │  Encoder Level 3  │ [4 NAFBlocks, 256 channels]
    │  ↓ Conv stride=2  │
    │  Encoder Level 4  │ [8 NAFBlocks, 512 channels]
    │  ↓ Conv stride=2  │
    │                   │
    │  Bottleneck       │ [16 NAFBlocks, 512 channels]
    │                   │
    │  ↑ PixelShuffle   │
    │  Decoder Level 4  │ [2 NAFBlocks, 256 channels] + skip from Enc4
    │  ↑ PixelShuffle   │
    │  Decoder Level 3  │ [2 NAFBlocks, 128 channels] + skip from Enc3
    │  ↑ PixelShuffle   │
    │  Decoder Level 2  │ [2 NAFBlocks, 64 channels]  + skip from Enc2
    │  ↑ PixelShuffle   │
    │  Decoder Level 1  │ [2 NAFBlocks, 64 channels]  + skip from Enc1
    └────────┬──────────┘
             │
┌────────────▼────────────────────────┐
│  Ending Conv3×3 (64 → 3 channels)   │
└────────────┬────────────────────────┘
             │
         Residual
             │
Output = Input + Residual  (element-wise add, then clamp to [0,1])
```

**Key design decisions:**
- **Deeper bottleneck** (16 blocks) gives the model capacity to untangle complex corruption combinations
- **Asymmetric encoder-decoder** (more blocks in encoder) - corruption understanding matters more than upsampling
- **Skip connections** preserve spatial details lost during downsampling
- **Residual learning** (final output = input + predicted residual) makes optimization easier

---

## Specific Technical Decisions and Tradeoffs

### Two-Stage Training Strategy

**Stage 1 (50 epochs):** Train from scratch with combined L1 + MS-SSIM loss
- **L1 loss** optimizes pixel-wise accuracy (directly minimizes MSE in expectation)
- **MS-SSIM loss (weight=0.2)** preserves perceptual structure across scales
- **Rationale:** Pure L1 can produce blurry outputs; SSIM encourages crisp edges

**Stage 2 (8 epochs):** Fine-tune with L1 only at 20× lower learning rate
- **Rationale:** SSIM helps during feature learning but can hurt final MSE. Stage 2 directly optimizes the competition metric.

This gave ~3-5% MSE improvement over single-stage training.

### Exponential Moving Average (EMA) with decay=0.999

Instead of using final model weights, maintain a moving average of weights across training steps:
```
θ_ema(t) = 0.999 × θ_ema(t-1) + 0.001 × θ(t)
```

**Why it works:** EMA smooths out high-frequency weight oscillations during training, producing more stable and generalizable models. This is especially important for restoration where small weight changes can cause visible artifacts.

### Cosine Learning Rate Schedule with Warmup

- **Warmup (3% of steps):** Linear ramp from 0 → base_lr
- **Cosine decay:** Smooth decay to ~1e-6 over remaining steps

**Rationale:** Warmup stabilizes early training when gradients are large; cosine decay allows gradual convergence without abrupt drops that can destabilize restoration networks.

### Data Augmentation Strategy

Applied standard geometric augmentations during training:
- Horizontal flip (p=0.5)
- Vertical flip (p=0.5)
- 90° rotation (p=0.75, k ∈ {0,1,2,3})

**Critical:** Apply the **same** transformation to both corrupted and clean images to maintain correspondence.

**Why it helps:** The corruption types include geometric distortions, but the *orientation* of natural images in the clean set is arbitrary. Augmentation teaches the model that restoration should be orientation-invariant.

### Mixed Precision Training (float16)

Used PyTorch's automatic mixed precision (AMP) with gradient scaling.

**Benefits:**
- 1.5-2× training speedup
- 40% memory reduction → larger batch sizes

**Concern:** Restoration models are sensitive to numerical precision (small pixel errors accumulate). Solution: keep optimizer state in float32, only use float16 for forward/backward passes.

### Gradient Clipping (norm=1.0)

Clip gradients to max L2 norm of 1.0 before optimizer step.

**Rationale:** MS-SSIM loss can produce very large gradients on high-contrast edges, causing training instability. Clipping prevents gradient explosions while allowing the model to learn.

### Validation Split Strategy

Used 2000 images (4.2% of 48K training set) for validation. **Fixed split** (no cross-validation) because:
- 46K training examples is sufficient for this model capacity
- Need consistent validation signal to track progress across epochs
- Corruption distribution is continuous, so any random split is representative

### Test-Time Augmentation (TTA)

At inference, predict on 8 augmentations of each test image (identity + 3 rotations × 2 flips), then average after inverse-transforming predictions back to original orientation.

**Expected gain:** ~1-2% MSE improvement for ~8× compute cost. Used only for final submission, not intermediate checkpoints.

---

## What I Tried That Didn't Work

### Classical Preprocessing Pipelines

Tried building a corruption classifier (CNN trained on synthetically labeled corruptions) to route images to specialized denoisers/deblurrers. **Failed because:**
- Most images have 3-5 simultaneous corruptions - no single "type"
- Even with correct labels, classical filters interact poorly (e.g., denoising after deblurring amplifies blur)

### Transformer-Based Models (SwinIR)

Experimented with SwinIR, a Swin-Transformer architecture that achieves strong results on super-resolution.

**Validation MSE:** 89.3 (NAFNet: 76.8)

**Why it failed:** Transformers excel at long-range dependencies and global context. For 32×32 images with localized corruptions, the quadratic complexity of self-attention is pure overhead - convolutions capture local patterns more efficiently at this scale.

### Diffusion Models for Restoration

Tested DDPM-style denoising diffusion as a posterior sampler (corruption as forward diffusion, restoration as reverse).

**Problems:**
- Requires 100-1000 diffusion steps → 50-500× slower than feedforward
- Stochastic sampling → variance in predictions
- Designed for generation, not metric optimization

Abandoned early due to computational cost.

### GAN-Based Approaches (Pix2Pix, CycleGAN)

GAN discriminators can encourage photorealistic outputs, but:
- Adversarial loss optimizes perceptual quality, not MSE
- Training instability (mode collapse, oscillation) wastes experiments
- Final MSE was **higher** than direct supervised learning

**Lesson:** GANs optimize the wrong objective for this competition. MSE demands pixel-perfect accuracy, not perceptual realism.

### Single Large Model Instead of Ensemble

Trained a single 200M-parameter model (width=96, middle_blk=24).

**Validation MSE:** 74.2 (vs. 2-model ensemble: 71.6)

**Why ensemble wins:** The 200M model has more capacity but also overfits more aggressively. The ensemble's prediction variance (from architectural + seed diversity) acts as implicit regularization, improving generalization to unseen corruption combinations.

### Heavy Data Augmentation (MixUp, CutMix)

Tried mixing pairs of corrupted-clean examples during training.

**Result:** Validation MSE degraded by ~8%.

**Why it failed:** MixUp works for classification by teaching decision boundary smoothness. For restoration, it creates **unrealistic corruption patterns** (e.g., half of one corruption type blended with half of another) that confuse the model's understanding of the corruption manifold.

---

## Validation Methodology

Before final submission, I validated the approach using:

### Per-Corruption-Severity Analysis

Computed MSE vs. corruption severity (measured as distance from corrupted to clean in pixel space) on the validation set:

| Severity Percentile | MSE (Model 1) | MSE (Model 2) | MSE (Ensemble) |
|---------------------|---------------|---------------|----------------|
| p10 (mild) | 42.1 | 44.3 | 40.8 |
| p50 (medium) | 78.2 | 80.1 | 74.5 |
| p90 (severe) | 156.4 | 162.7 | 148.9 |

**Insight:** Ensemble gives consistent improvement across all corruption levels, with largest gains on severe corruptions (where individual models are least confident).

### Ablation Studies

| Configuration | Val MSE | Notes |
|---------------|---------|-------|
| Baseline (L1 only, no EMA) | 82.4 | - |
| + MS-SSIM (stage 1) | 79.1 | Preserves structure |
| + EMA | 77.3 | Smooths weights |
| + Cosine LR | 76.8 | Better convergence |
| + TTA (8-aug) | 75.2 | Inference-time gain |
| **2-model ensemble + TTA** | **71.6** | Final configuration |

### Training Curve Analysis

Monitored train/val MSE every 50 steps. Observations:
- Model 1 plateaus around epoch 40; Model 2 around epoch 35 (wider = faster convergence)
- Both models show overfitting after epoch 50 (train MSE decreases, val MSE increases)
- Stage 2 fine-tuning recovers ~2 MSE on validation

**Decision:** Use Stage 2 epoch 6 checkpoint (best val MSE) rather than final epoch.

---

## Training Details

### Model 1 (Width=48)

| Hyperparameter | Stage 1 | Stage 2 |
|----------------|---------|---------|
| Epochs | 60 | 10 |
| Learning rate | 2e-4 | 1e-5 |
| Batch size | 128 | 128 |
| Optimizer | AdamW (β=(0.9, 0.999), ε=1e-8) | AdamW |
| Weight decay | 1e-4 | 1e-4 |
| Loss | L1 + 0.2×MS-SSIM | L1 only |
| LR schedule | Cosine with 3% warmup | Cosine |
| Gradient clip | 1.0 | 1.0 |
| EMA decay | 0.999 | 0.999 |
| Augmentation | Flip + Rotate | Flip + Rotate |
| **Training time** | ~3.5 hours | ~35 min |

### Model 2 (Width=64)

| Hyperparameter | Stage 1 | Stage 2 |
|----------------|---------|---------|
| Epochs | 50 | 8 |
| Learning rate | 2e-4 | 1e-5 |
| Batch size | 96 | 96 |
| Optimizer | AdamW | AdamW |
| Weight decay | 1e-4 | 1e-4 |
| Loss | L1 + 0.2×MS-SSIM | L1 only |
| **Training time** | ~4 hours | ~30 min |

**Total training:** ~8.5 hours on Kaggle GPU (P100/T4)

---

## Stack

| Component | Library/Framework |
|-----------|-------------------|
| Deep learning | PyTorch 2.0+ |
| Mixed precision | torch.cuda.amp (autocast + GradScaler) |
| Data loading | torch.utils.data.DataLoader (4 workers, pin_memory) |
| Image I/O | PIL, NumPy |
| Loss functions | Custom MS-SSIM, PyTorch L1Loss |
| Optimizer | AdamW |
| Augmentation | Manual (NumPy flips/rotations) |
| Metrics | MSE (NumPy) |
| Checkpointing | torch.save/load |
| Hardware | Kaggle Notebook (P100 16GB / T4 16GB) |

---

## Key Takeaway

The highest-leverage insight in this project was recognizing that **blind image restoration with unknown corruption types is fundamentally an end-to-end learning problem**, not a decomposable pipeline of corruption detection + targeted filtering.

Classical approaches fail because:
1. **Corruption types overlap** - identifying a single type is impossible
2. **Classical filters interact poorly** - the order of operations matters but is unknown
3. **Corruption parameters vary continuously** - no discrete categories to classify

NAFNet succeeds because:
1. **Learns the inverse mapping directly** from paired data without explicit corruption modeling
2. **Residual formulation** (predict noise, not clean image) simplifies the learning task
3. **SimpleGate + attention** provides architectural bias toward restoration without information loss from ReLU thresholding
4. **Ensemble diversity** compensates for individual model biases

The two-model ensemble with different capacities and seeds proved more effective than a single larger model - **diversity beats capacity** when the corruption distribution is complex and training data is finite.

Finally, **two-stage training** (perceptual structure preservation → direct MSE optimization) reconciled the tension between producing visually coherent outputs and minimizing numerical error, a pattern that likely generalizes to other metric-driven restoration tasks.

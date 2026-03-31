# Fake Image Detection Pipeline
### IIT Mandi — Dual-Branch Deep Learning System

---

## Overview

This repository contains a complete PyTorch implementation of a dual-branch
fake image detection pipeline that combines:

- **Spatial Branch**: DINO-pretrained ViT-B/16 for semantic facial feature extraction
- **Frequency Branch**: FFT spectrum + ResNet-18 for spectral artefact detection
- **Cross-Attention Fusion**: Adaptive merging of both feature streams
- **Binary Classifier**: MLP head with label smoothing

---

## File Structure

```
fake_image_detection/
│
├── model.py          # Complete model architecture (ViT + ResNet18 + CrossAttn)
├── dataset.py        # Dataset loading, augmentation, DataLoader factory
├── train.py          # 3-phase training script with AMP, EMA, checkpointing
├── predict.py        # Inference on single image or folder
├── extract_frames.py # FF++ video → frame extraction utility
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python    | 3.9     | 3.10+       |
| PyTorch   | 2.0     | 2.1+        |
| CUDA      | 11.8    | 12.1+       |
| GPU VRAM  | 8 GB    | 16+ GB      |
| RAM       | 16 GB   | 32+ GB      |
| Storage   | 50 GB   | 200+ GB     |

> **CPU-only mode** is supported but training will be very slow.
> For a full FF++ training run, a GPU is strongly recommended.

---

## Step 1 — Installation

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# 2. Install PyTorch (visit https://pytorch.org for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## Step 2 — Prepare Your Dataset

### Option A: Your Own Folder of Real and Fake Images

Organise your images as follows:

```
dataset/
    train/
        real/       ← put real face images here
        fake/       ← put fake/deepfake images here
    val/
        real/
        fake/
    test/
        real/
        fake/
```

A rough 70/15/15 train/val/test split is recommended.

### Option B: FaceForensics++ (FF++)

1. Request access at: https://github.com/ondyari/FaceForensics
2. Download videos using their provided script
3. Extract frames:

```bash
python extract_frames.py \
    --data_root /path/to/FaceForensics++ \
    --output_dir ./dataset \
    --compression c23 \
    --every_n_frames 10 \
    --max_frames_per_video 50
```

This will automatically create train/val/test splits.

---

## Step 3 — Train the Model

### Custom dataset (real/fake folders):

```bash
python train.py \
    --data_root ./dataset \
    --batch_size 32 \
    --epochs 60 \
    --num_workers 4 \
    --test
```

### FaceForensics++ (with multiple fake methods):

```bash
python train.py \
    --data_root ./dataset \
    --real_subdir real \
    --fake_subdirs fake \
    --batch_size 32 \
    --epochs 60 \
    --test
```

### On a machine with limited VRAM (< 8 GB):

```bash
python train.py \
    --data_root ./dataset \
    --batch_size 16 \
    --grad_accum 4 \
    --epochs 60
```

### Resume interrupted training:

```bash
python train.py \
    --data_root ./dataset \
    --resume ./checkpoints/epoch_30.pth \
    --epochs 60
```

### Training phases (automatic):
- **Phase 1** (epochs 1–10): ViT frozen, only frequency branch + fusion + classifier trained
- **Phase 2** (epochs 11–30): Last 4 ViT blocks unfrozen
- **Phase 3** (epochs 31–60): Full fine-tuning, all parameters

### What to expect:
- Phase 1 is fast (~2 min/epoch on RTX 3080)
- Val AUC should reach ~0.85–0.90 by end of Phase 1
- Final Val AUC should reach ~0.95–0.98 on FF++ c23

---

## Step 4 — Run Inference

### Single image:

```bash
python predict.py \
    --checkpoint ./checkpoints/best_model.pth \
    --input ./test_image.jpg
```

### Single image with FFT visualisation:

```bash
python predict.py \
    --checkpoint ./checkpoints/best_model.pth \
    --input ./test_image.jpg \
    --visualise
```

### Folder of images:

```bash
python predict.py \
    --checkpoint ./checkpoints/best_model.pth \
    --input ./my_test_images/ \
    --output results.json
```

### Example output:
```
==================================================
  File:        ./test_image.jpg
  Prediction:  FAKE
  Probability: 0.9231
  Confidence:  92.3%
==================================================
```

---

## Step 5 — Understanding the Output

| Output | Meaning |
|--------|---------|
| Prediction: REAL  | Model believes the image is authentic |
| Prediction: FAKE  | Model believes the image is AI-generated or manipulated |
| Probability       | Raw probability of being fake (0 = definitely real, 1 = definitely fake) |
| Confidence        | How confident the model is in its prediction |

---

## Model Architecture Summary

```
Input Image (224x224x3)
        │
   RetinaFace Alignment
        │
   ┌────┴────┐
   │         │
ViT-B/16   FFT Layer
(DINO)    log(1+|F|)
   │         │
F_s∈R^768  ResNet-18
           Projection
              │
           F_f∈R^768
   │         │
   └────┬────┘
  Cross-Attention Fusion
  LN(F_s + Attn(F_s, F_f))
        │
    F_fusion∈R^768
        │
  Linear(768→512)
  BatchNorm + ReLU
  Dropout(0.4)
  Linear(512→1)
  Sigmoid
        │
   p ∈ (0,1)
   p>0.5 → FAKE
```

---

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Backbone | ViT-B/16 | DINO pretrained, ImageNet |
| Freq. CNN | ResNet-18 | ImageNet pretrained |
| Batch size | 32 | Increase if VRAM allows |
| Phase 1 LR | 1e-3 | For new modules only |
| Phase 2 LR | 1e-4 (ViT), 5e-4 (others) | Partial fine-tuning |
| Phase 3 LR | 1e-5 (ViT), 1e-4 (others) | Full fine-tuning |
| Weight decay | 0.05 | AdamW |
| Label smoothing | 0.1 | Prevents overconfidence |
| Dropout | 0.4 | In classifier |

---

## Troubleshooting

**CUDA out of memory**
→ Reduce `--batch_size` to 8 or 16, increase `--grad_accum` to 4 or 8

**DINO weights not downloading**
→ Ensure internet access. Weights are cached in `~/.cache/torch/hub/` after first download.
→ Manual download: `torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')`

**RetinaFace not finding faces**
→ The code falls back to centre-crop automatically. This is fine for FF++ frames.

**Low accuracy**
→ Check class balance in your dataset. The code applies weighted sampling automatically.
→ Try increasing Phase 1 epochs: `--phase1_epochs 20`

---

## Citation / References

If you use this code, please cite:

```
Caron et al., "Emerging Properties in Self-Supervised Vision Transformers", ICCV 2021
Rossler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images", ICCV 2019
Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021
He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
Frank et al., "Leveraging Frequency Analysis for Deep Fake Image Recognition", ICML 2020
```

---

*Developed for IIT Mandi — Fake Image Detection Research Project*

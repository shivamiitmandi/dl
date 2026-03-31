"""
dataset.py — Dataset Loading and Augmentation
==============================================
Supports:
  1. Custom folder (real/ and fake/ subfolders)
  2. FaceForensics++ (FF++) folder structure

Both paths produce aligned 224x224 face crops.
If RetinaFace alignment fails (no face found), the image is
centre-cropped as a fallback.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
import io

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ─────────────────────────────────────────────────────────────────────────────
# Face Alignment using RetinaFace
# ─────────────────────────────────────────────────────────────────────────────

# Canonical 5-point landmark positions in 224x224 output space
# (left eye, right eye, nose, left mouth, right mouth)
REFERENCE_LANDMARKS_224 = np.array([
    [74.98513,  86.52460],   # left eye
    [148.82915, 86.52460],   # right eye
    [111.90538, 128.78186],  # nose tip
    [83.50735,  159.97071],  # left mouth corner
    [140.30617, 159.97071],  # right mouth corner
], dtype=np.float32)


def align_face(img_bgr: np.ndarray,
               landmarks: np.ndarray,
               output_size: int = 224) -> np.ndarray:
    """
    Apply similarity transform to align face given 5 landmarks.

    Args:
        img_bgr: H x W x 3 BGR image (OpenCV format)
        landmarks: (5, 2) array of (x, y) landmark positions
        output_size: output square size in pixels

    Returns:
        aligned: output_size x output_size x 3 BGR image
    """
    ref = REFERENCE_LANDMARKS_224.copy()
    if output_size != 224:
        ref = ref * (output_size / 224.0)

    # Estimate similarity transform (rotation + scale + translation)
    # estimateAffinePartial2D finds the best-fit similarity transform
    tform, _ = cv2.estimateAffinePartial2D(
        landmarks.astype(np.float32),
        ref,
        method=cv2.LMEDS
    )
    if tform is None:
        # Fallback: simple centre crop
        h, w = img_bgr.shape[:2]
        s = min(h, w)
        y0 = (h - s) // 2; x0 = (w - s) // 2
        crop = img_bgr[y0:y0+s, x0:x0+s]
        return cv2.resize(crop, (output_size, output_size))

    aligned = cv2.warpAffine(
        img_bgr, tform, (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return aligned


def detect_and_align(img_bgr: np.ndarray,
                     detector,
                     output_size: int = 224) -> np.ndarray:
    """
    Detect face using RetinaFace and return aligned crop.
    Falls back to centre crop if no face detected.
    """
    try:
        from retinaface import RetinaFace
        faces = RetinaFace.detect_faces(img_bgr)
        if isinstance(faces, dict) and len(faces) > 0:
            # Pick face with highest confidence
            best = max(faces.values(), key=lambda f: f['score'])
            landmarks = np.array([
                best['landmarks']['left_eye'],
                best['landmarks']['right_eye'],
                best['landmarks']['nose'],
                best['landmarks']['mouth_left'],
                best['landmarks']['mouth_right'],
            ], dtype=np.float32)
            return align_face(img_bgr, landmarks, output_size)
    except Exception:
        pass

    # Fallback: centre crop
    h, w = img_bgr.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2; x0 = (w - s) // 2
    crop = img_bgr[y0:y0+s, x0:x0+s]
    return cv2.resize(crop, (output_size, output_size))


# ─────────────────────────────────────────────────────────────────────────────
# JPEG Compression Augmentation (simulates social media re-compression)
# ─────────────────────────────────────────────────────────────────────────────

class RandomJPEGCompression:
    """Randomly apply JPEG compression with quality in [lo, hi]."""

    def __init__(self, lo: int = 50, hi: int = 95, p: float = 0.5):
        self.lo = lo
        self.hi = hi
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if torch.rand(1).item() > self.p:
            return img
        quality = int(torch.randint(self.lo, self.hi + 1, (1,)).item())
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()


class RandomDownUpscale:
    """Simulate resizing artefacts by downscaling then upscaling."""

    def __init__(self, lo: float = 0.5, hi: float = 0.9, p: float = 0.3):
        self.lo = lo
        self.hi = hi
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if torch.rand(1).item() > self.p:
            return img
        factor = self.lo + torch.rand(1).item() * (self.hi - self.lo)
        w, h = img.size
        small = img.resize((int(w * factor), int(h * factor)), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


# ─────────────────────────────────────────────────────────────────────────────
# Transform Pipelines
# ─────────────────────────────────────────────────────────────────────────────

# ImageNet normalisation statistics (used because ViT was pretrained on ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform() -> T.Compose:
    """Full augmentation pipeline for training."""
    return T.Compose([
        T.Resize((224, 224)),
        # ── Spatial augmentations ──────────────────────────────────────────
        T.RandomHorizontalFlip(p=0.5),
        T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.05),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        # ── Compression augmentations ──────────────────────────────────────
        RandomJPEGCompression(lo=50, hi=95, p=0.5),
        RandomDownUpscale(lo=0.5, hi=0.9, p=0.3),
        # ── To tensor + normalise ──────────────────────────────────────────
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])


def get_val_transform() -> T.Compose:
    """Minimal transform for validation/test (no augmentation)."""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Class
# ─────────────────────────────────────────────────────────────────────────────

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


class FaceForensicsDataset(Dataset):
    """
    Generic dataset for fake image detection.

    Expects folder structure:
        root/
            real/   (or 'original' for FF++)
            fake/   (or 'DeepFakes', 'Face2Face', etc. for FF++)

    Args:
        root: path to split folder (e.g. dataset/train)
        transform: torchvision transform pipeline
        real_subdir: name of real image subdirectory
        fake_subdirs: list of fake subdirectory names
        max_per_class: cap samples per class (None = use all)
    """

    def __init__(
        self,
        root: str,
        transform=None,
        real_subdir: str = 'real',
        fake_subdirs: list = None,
        max_per_class: Optional[int] = None,
    ):
        self.root = Path(root)
        self.transform = transform or get_val_transform()

        if fake_subdirs is None:
            fake_subdirs = ['fake']

        # Collect real images
        real_dir = self.root / real_subdir
        self.samples = []  # (path, label)   label: 0=real, 1=fake

        real_paths = self._collect_images(real_dir)
        if max_per_class:
            real_paths = real_paths[:max_per_class]
        self.samples += [(p, 0) for p in real_paths]

        # Collect fake images from all fake subdirs
        for subdir in fake_subdirs:
            fake_dir = self.root / subdir
            if not fake_dir.exists():
                print(f"[Warning] Directory not found: {fake_dir}")
                continue
            fake_paths = self._collect_images(fake_dir)
            if max_per_class:
                fake_paths = fake_paths[:max_per_class]
            self.samples += [(p, 1) for p in fake_paths]

        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        print(f"[Dataset] {root}: {n_real} real | {n_fake} fake | total {len(self.samples)}")

    def _collect_images(self, directory: Path) -> list:
        paths = []
        if not directory.exists():
            return paths
        for f in sorted(directory.iterdir()):
            if f.suffix.lower() in VALID_EXTENSIONS:
                paths.append(f)
        return paths

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"[Warning] Could not load {path}: {e}. Using blank image.")
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for weighted loss.
        Returns tensor [w_real, w_fake] where higher weight = rarer class.
        """
        labels = [l for _, l in self.samples]
        n_real = labels.count(0)
        n_fake = labels.count(1)
        n_total = len(labels)
        w_real = n_total / (2 * n_real) if n_real > 0 else 1.0
        w_fake = n_total / (2 * n_fake) if n_fake > 0 else 1.0
        return torch.tensor([w_real, w_fake], dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler (balanced batches)."""
        class_weights = self.get_class_weights()
        return torch.tensor([class_weights[l] for _, l in self.samples])


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    train_dir: str,
    val_dir: str,
    test_dir: str = None,
    batch_size: int = 32,
    num_workers: int = 4,
    real_subdir: str = 'real',
    fake_subdirs: list = None,
    max_per_class: int = None,
    use_weighted_sampler: bool = True,
) -> dict:
    """
    Build train, val, (optionally test) DataLoaders.

    Returns dict with keys: 'train', 'val', ('test')
    """
    if fake_subdirs is None:
        fake_subdirs = ['fake']

    train_ds = FaceForensicsDataset(
        train_dir, get_train_transform(),
        real_subdir=real_subdir, fake_subdirs=fake_subdirs,
        max_per_class=max_per_class,
    )
    val_ds = FaceForensicsDataset(
        val_dir, get_val_transform(),
        real_subdir=real_subdir, fake_subdirs=fake_subdirs,
    )

    # Balanced sampling for training
    if use_weighted_sampler:
        sample_weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loaders = {
        'train': DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        ),
        'val': DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
    }

    if test_dir:
        test_ds = FaceForensicsDataset(
            test_dir, get_val_transform(),
            real_subdir=real_subdir, fake_subdirs=fake_subdirs,
        )
        loaders['test'] = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    return loaders


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        root = sys.argv[1]
        ds = FaceForensicsDataset(root + '/train', get_train_transform())
        img, label = ds[0]
        print(f"Sample shape: {img.shape}, label: {label}")
        w = ds.get_class_weights()
        print(f"Class weights: real={w[0]:.3f}, fake={w[1]:.3f}")
    else:
        print("Usage: python dataset.py <dataset_root>")

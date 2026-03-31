"""
predict.py — Inference Script
==============================
Run inference on a single image, a folder of images, or a video file.

Usage:
    # Single image
    python predict.py --checkpoint ./checkpoints/best_model.pth --input ./test_image.jpg

    # Folder of images
    python predict.py --checkpoint ./checkpoints/best_model.pth --input ./test_images/

    # With visualisation of FFT spectrum
    python predict.py --checkpoint ./checkpoints/best_model.pth --input ./test_image.jpg --visualise
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T

from model import FakeImageDetector
from dataset import get_val_transform, VALID_EXTENSIONS


# ─────────────────────────────────────────────────────────────────────────────
# Predictor Class
# ─────────────────────────────────────────────────────────────────────────────

class FakeImagePredictor:
    """
    High-level wrapper for fake image detection inference.

    Args:
        checkpoint_path: Path to saved .pth checkpoint
        device: 'cuda', 'cpu', or 'auto'
        threshold: Decision threshold (default 0.5)
    """

    def __init__(self, checkpoint_path: str, device: str = 'auto', threshold: float = 0.5):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"[Predictor] Loading model from {checkpoint_path}")
        print(f"[Predictor] Device: {self.device}")

        self.model = FakeImageDetector().to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()

        self.transform = get_val_transform()
        self.threshold = threshold

        print(f"[Predictor] Model loaded. Threshold: {threshold}")

    @torch.no_grad()
    def predict_image(self, image_path: str) -> Dict:
        """
        Predict a single image.

        Returns dict:
            - path: image path
            - probability: probability of being fake (0–1)
            - prediction: 'FAKE' or 'REAL'
            - confidence: confidence percentage
        """
        img = Image.open(image_path).convert('RGB')
        x = self.transform(img).unsqueeze(0).to(self.device)

        logit = self.model(x)
        prob  = torch.sigmoid(logit).item()
        label = 'FAKE' if prob > self.threshold else 'REAL'
        conf  = prob if label == 'FAKE' else 1.0 - prob

        return {
            'path':        str(image_path),
            'probability': round(prob, 4),
            'prediction':  label,
            'confidence':  round(conf * 100, 2),
        }

    @torch.no_grad()
    def predict_batch(self, image_paths: List[str], batch_size: int = 32) -> List[Dict]:
        """Predict a list of images in batches."""
        results = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            imgs = []
            valid_paths = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert('RGB')
                    imgs.append(self.transform(img))
                    valid_paths.append(p)
                except Exception as e:
                    print(f"[Warning] Could not load {p}: {e}")

            if not imgs:
                continue

            x = torch.stack(imgs).to(self.device)
            logits = self.model(x)
            probs  = torch.sigmoid(logits).cpu().numpy()

            for path, prob in zip(valid_paths, probs):
                label = 'FAKE' if prob > self.threshold else 'REAL'
                conf  = prob if label == 'FAKE' else 1.0 - prob
                results.append({
                    'path':        str(path),
                    'probability': round(float(prob), 4),
                    'prediction':  label,
                    'confidence':  round(float(conf) * 100, 2),
                })

        return results

    def predict_folder(self, folder_path: str, batch_size: int = 32) -> List[Dict]:
        """Predict all images in a folder."""
        folder = Path(folder_path)
        image_paths = sorted([
            str(f) for f in folder.rglob('*')
            if f.suffix.lower() in VALID_EXTENSIONS
        ])
        print(f"[Predictor] Found {len(image_paths)} images in {folder_path}")
        return self.predict_batch(image_paths, batch_size)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation (requires matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def visualise_prediction(image_path: str, result: Dict, predictor: FakeImagePredictor):
    """Show image alongside its FFT spectrum and prediction."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[Warning] matplotlib not installed. Skipping visualisation.")
        return

    img_pil  = Image.open(image_path).convert('RGB')
    x        = predictor.transform(img_pil)

    # Compute FFT for visualisation
    r, g, b = x[0:1], x[1:2], x[2:3]
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).squeeze()
    fft  = torch.fft.fftshift(torch.fft.fft2(gray))
    log_mag = torch.log1p(torch.abs(fft)).numpy()

    fig = plt.figure(figsize=(12, 5))
    gs  = gridspec.GridSpec(1, 3)

    # Original image
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img_pil)
    ax1.set_title('Input Image', fontsize=12)
    ax1.axis('off')

    # FFT Spectrum
    ax2 = fig.add_subplot(gs[1])
    im = ax2.imshow(log_mag, cmap='hot')
    ax2.set_title('FFT Log-Magnitude Spectrum', fontsize=12)
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046)

    # Prediction
    ax3 = fig.add_subplot(gs[2])
    color = '#d32f2f' if result['prediction'] == 'FAKE' else '#2e7d32'
    ax3.text(0.5, 0.55, result['prediction'],
             ha='center', va='center', fontsize=36, fontweight='bold', color=color,
             transform=ax3.transAxes)
    ax3.text(0.5, 0.35, f"Fake probability: {result['probability']:.3f}",
             ha='center', va='center', fontsize=13, transform=ax3.transAxes)
    ax3.text(0.5, 0.22, f"Confidence: {result['confidence']:.1f}%",
             ha='center', va='center', fontsize=13, transform=ax3.transAxes)
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_facecolor('#f5f5f5')

    plt.suptitle('Fake Image Detection Result', fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = str(image_path).rsplit('.', 1)[0] + '_prediction.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[Visualisation] Saved to {out_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fake Image Detection — Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--input',     type=str, required=True,
                        help='Path to image file or folder of images')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Decision threshold (default: 0.5)')
    parser.add_argument('--batch_size',type=int,   default=32)
    parser.add_argument('--output',    type=str,   default=None,
                        help='Save results to JSON file')
    parser.add_argument('--visualise', action='store_true',
                        help='Show FFT spectrum visualisation (single image only)')
    args = parser.parse_args()

    predictor = FakeImagePredictor(args.checkpoint, threshold=args.threshold)

    input_path = Path(args.input)
    if input_path.is_dir():
        results = predictor.predict_folder(str(input_path), args.batch_size)
        # Summary
        n_fake = sum(1 for r in results if r['prediction'] == 'FAKE')
        n_real = sum(1 for r in results if r['prediction'] == 'REAL')
        print(f"\n{'='*50}")
        print(f"Results: {n_real} REAL | {n_fake} FAKE | {len(results)} total")
        print(f"{'='*50}")
        for r in results:
            print(f"  [{r['prediction']:4s}] {r['confidence']:5.1f}%  {r['path']}")
    else:
        result = predictor.predict_image(str(input_path))
        print(f"\n{'='*50}")
        print(f"  File:        {result['path']}")
        print(f"  Prediction:  {result['prediction']}")
        print(f"  Probability: {result['probability']:.4f}")
        print(f"  Confidence:  {result['confidence']:.1f}%")
        print(f"{'='*50}")
        if args.visualise:
            visualise_prediction(str(input_path), result, predictor)
        results = [result]

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

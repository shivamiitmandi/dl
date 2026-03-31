"""
train.py — Three-Phase Training Script
========================================
Phase 1 (epochs 1-10):   ViT frozen,  train freq_branch + fusion + classifier
Phase 2 (epochs 11-30):  Unfreeze last 4 ViT blocks
Phase 3 (epochs 31-60):  Unfreeze all parameters (full fine-tuning)

Usage:
    # Custom folder
    python train.py --data_root ./dataset --batch_size 32 --epochs 60

    # With FF++ fake subdirs
    python train.py --data_root ./dataset --fake_subdirs DeepFakes Face2Face FaceSwap NeuralTextures

    # Resume training
    python train.py --data_root ./dataset --resume ./checkpoints/best_model.pth
"""

import os
import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

from model import FakeImageDetector
from dataset import get_dataloaders


# ─────────────────────────────────────────────────────────────────────────────
# Loss Function with Label Smoothing
# ─────────────────────────────────────────────────────────────────────────────

class SmoothedBCELoss(nn.Module):
    """
    Binary Cross-Entropy with label smoothing and optional class weighting.
    y_smooth = y * (1 - eps) + eps / 2
    """

    def __init__(self, smoothing: float = 0.1, pos_weight: torch.Tensor = None):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        # Apply label smoothing
        targets_smooth = targets * (1 - self.smoothing) + self.smoothing / 2
        return self.bce(logits, targets_smooth)


# ─────────────────────────────────────────────────────────────────────────────
# EMA (Exponential Moving Average) of model weights
# ─────────────────────────────────────────────────────────────────────────────

class ModelEMA:
    """
    Maintains an EMA copy of model weights for more stable inference.
    theta_ema = decay * theta_ema + (1 - decay) * theta_model
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name] = (self.decay * self.shadow[name] +
                                     (1 - self.decay) * p.data)

    def apply_shadow(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module, original: dict):
        for name, p in model.named_parameters():
            if name in original:
                p.data.copy_(original[name])


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Run one evaluation pass. Returns dict of metrics."""
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels.float())

        total_loss += loss.item() * len(labels)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds      = (all_probs > 0.5).astype(int)
    acc        = (preds == all_labels).mean()

    metrics = {
        'loss': total_loss / len(loader.dataset),
        'acc':  acc,
        'auc':  roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5,
        'ap':   average_precision_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5,
    }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Build Optimiser for a Given Phase
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model, phase, args):
    phase_configs = {
        1: dict(lr_vit=0.0,     lr_other=1e-3),   # ViT frozen
        2: dict(lr_vit=1e-4,    lr_other=5e-4),
        3: dict(lr_vit=1e-5,    lr_other=1e-4),
    }
    cfg = phase_configs[phase]
    param_groups = model.get_param_groups(cfg['lr_vit'], cfg['lr_other'])
    # Remove groups with no trainable params
    param_groups = [g for g in param_groups if len(g['params']) > 0]
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    return optimizer


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device,
                    grad_accum_steps=2, grad_clip=1.0, ema=None, epoch=0, total_epochs=1):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    for step, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels) / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)

        with torch.no_grad():
            probs = torch.sigmoid(logits.detach())
            preds = (probs > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total   += len(labels)
            total_loss += loss.item() * grad_accum_steps * len(labels)

        if step % 50 == 0:
            print(f"  Step [{step}/{len(loader)}]  "
                  f"loss={loss.item()*grad_accum_steps:.4f}  "
                  f"acc={correct/total:.4f}")

    return {'loss': total_loss / total, 'acc': correct / total}


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    # ── Setup ─────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    fake_subdirs = args.fake_subdirs if args.fake_subdirs else ['fake']
    loaders = get_dataloaders(
        train_dir   = os.path.join(args.data_root, 'train'),
        val_dir     = os.path.join(args.data_root, 'val'),
        test_dir    = os.path.join(args.data_root, 'test') if args.test else None,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        real_subdir = args.real_subdir,
        fake_subdirs= fake_subdirs,
        use_weighted_sampler=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FakeImageDetector().to(device)

    # Resume if checkpoint provided
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch - 1}")

    # ── Class-weighted loss ───────────────────────────────────────────────────
    class_weights = loaders['train'].dataset.get_class_weights().to(device)
    # pos_weight: weight for the fake (positive) class
    pos_weight = class_weights[1] / class_weights[0]
    criterion = SmoothedBCELoss(smoothing=0.1, pos_weight=pos_weight)

    # ── AMP scaler ────────────────────────────────────────────────────────────
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema = ModelEMA(model, decay=0.9999)

    # ── Training phases ───────────────────────────────────────────────────────
    phase_boundaries = {
        1: (1,              args.phase1_epochs),
        2: (args.phase1_epochs + 1,
            args.phase1_epochs + args.phase2_epochs),
        3: (args.phase1_epochs + args.phase2_epochs + 1,
            args.epochs),
    }

    current_phase = None
    optimizer = None
    scheduler = None
    best_auc  = 0.0
    history   = []

    total_epochs = args.epochs
    for epoch in range(start_epoch, total_epochs + 1):

        # ── Phase transition ──────────────────────────────────────────────────
        for phase, (start, end) in phase_boundaries.items():
            if start <= epoch <= end and phase != current_phase:
                current_phase = phase
                model.set_phase(phase)
                optimizer = build_optimizer(model, phase, args)
                n_steps = len(loaders['train']) * (end - epoch + 1)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=n_steps, eta_min=1e-7
                )
                print(f"\n{'='*60}")
                print(f"  Entering Phase {phase} at epoch {epoch}")
                print(f"{'='*60}")
                break

        # ── Train ─────────────────────────────────────────────────────────────
        t0 = time.time()
        print(f"\nEpoch [{epoch}/{total_epochs}]")
        train_metrics = train_one_epoch(
            model, loaders['train'], optimizer, criterion, scaler, device,
            grad_accum_steps=args.grad_accum,
            grad_clip=args.grad_clip,
            ema=ema, epoch=epoch, total_epochs=total_epochs,
        )
        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        val_metrics = evaluate(model, loaders['val'], criterion, device)
        elapsed = time.time() - t0

        print(f"  Train | loss={train_metrics['loss']:.4f}  acc={train_metrics['acc']:.4f}")
        print(f"  Val   | loss={val_metrics['loss']:.4f}  acc={val_metrics['acc']:.4f}  "
              f"auc={val_metrics['auc']:.4f}  ap={val_metrics['ap']:.4f}  "
              f"[{elapsed:.1f}s]")

        # ── Save history ──────────────────────────────────────────────────────
        history.append({'epoch': epoch, 'phase': current_phase,
                        **{f'train_{k}': v for k,v in train_metrics.items()},
                        **{f'val_{k}':   v for k,v in val_metrics.items()}})
        with open(os.path.join(args.log_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        # ── Save best checkpoint ──────────────────────────────────────────────
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_auc': best_auc,
                'args': vars(args),
            }, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"  *** New best AUC: {best_auc:.4f} — checkpoint saved ***")

        # Periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, os.path.join(args.checkpoint_dir, f'epoch_{epoch}.pth'))

    # ── Final test evaluation ─────────────────────────────────────────────────
    if 'test' in loaders:
        print("\n" + "="*60)
        print("FINAL TEST EVALUATION")
        print("="*60)
        best_ckpt = torch.load(os.path.join(args.checkpoint_dir, 'best_model.pth'),
                               map_location=device)
        model.load_state_dict(best_ckpt['model_state'])
        test_metrics = evaluate(model, loaders['test'], criterion, device)
        print(f"  Test | acc={test_metrics['acc']:.4f}  auc={test_metrics['auc']:.4f}  "
              f"ap={test_metrics['ap']:.4f}")
        with open(os.path.join(args.log_dir, 'test_results.json'), 'w') as f:
            json.dump(test_metrics, f, indent=2)

    print(f"\nTraining complete. Best Val AUC: {best_auc:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fake Image Detection — Training')

    # Data
    parser.add_argument('--data_root',    type=str, required=True,
                        help='Root dataset dir with train/val/test subfolders')
    parser.add_argument('--real_subdir',  type=str, default='real')
    parser.add_argument('--fake_subdirs', type=str, nargs='+', default=['fake'],
                        help='One or more fake subdirectory names')
    parser.add_argument('--test',         action='store_true',
                        help='Run test evaluation after training')

    # Training
    parser.add_argument('--epochs',       type=int, default=60)
    parser.add_argument('--phase1_epochs',type=int, default=10,
                        help='Epochs for Phase 1 (ViT frozen)')
    parser.add_argument('--phase2_epochs',type=int, default=20,
                        help='Epochs for Phase 2 (last 4 blocks)')
    parser.add_argument('--batch_size',   type=int, default=32)
    parser.add_argument('--num_workers',  type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--grad_clip',    type=float, default=1.0)
    parser.add_argument('--grad_accum',   type=int,   default=2,
                        help='Gradient accumulation steps')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir',         type=str, default='./logs')
    parser.add_argument('--resume',          type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()
    args.epochs = args.phase1_epochs + args.phase2_epochs + (args.epochs - args.phase1_epochs - args.phase2_epochs)
    main(args)

import torch
import torch.nn as nn
from tqdm import tqdm
from deepfake_utils import compute_metrics, bce_with_label_smoothing

def train_one_epoch(model, loader, optimizer, device, pos_weight, epoch, total):
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []
    pbar = tqdm(loader, desc=f"  Ep {epoch+1:03d}/{total} [Train]", leave=False, dynamic_ncols=True)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = bce_with_label_smoothing(preds, labels, 0.1, pos_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(preds.detach().cpu().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    m = compute_metrics(all_labels, all_probs)
    m["loss"] = total_loss / len(loader.dataset)
    return m

@torch.no_grad()
def evaluate(model, loader, device, pos_weight, epoch, total, split="Val"):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []
    pbar = tqdm(loader, desc=f"  Ep {epoch+1:03d}/{total} [{split}]  ", leave=False, dynamic_ncols=True)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        loss = bce_with_label_smoothing(preds, labels, 0.1, pos_weight)
        total_loss += loss.item() * imgs.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(preds.cpu().tolist())
    m = compute_metrics(all_labels, all_probs)
    m["loss"] = total_loss / len(loader.dataset)
    return m

def make_optimizer(model, phase, base_lr):
    if phase == 1:
        params = filter(lambda p: p.requires_grad, model.parameters())
        return torch.optim.AdamW(params, lr=base_lr, weight_decay=0.05)
    elif phase == 2:
        return torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": base_lr * 0.1},
            {"params": model.head.parameters(), "lr": base_lr},
        ], weight_decay=0.05)
    else:
        return torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": base_lr * 0.01},
            {"params": model.head.parameters(), "lr": base_lr * 0.1},
        ], weight_decay=0.05)

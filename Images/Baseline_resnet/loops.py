import torch
from tqdm import tqdm
from deepfake_utils import compute_metrics, bce_with_label_smoothing
def train_one_epoch(model, loader, optimizer, device, pos_weight):
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []
    for imgs, labels in tqdm(loader, desc="  Train", leave=False, dynamic_ncols=True):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = bce_with_label_smoothing(preds, labels, 0.0, pos_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(preds.detach().cpu().tolist())
    m = compute_metrics(all_labels, all_probs)
    m["loss"] = total_loss / len(loader.dataset)
    return m
@torch.no_grad()
def evaluate(model, loader, device, pos_weight):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []
    for imgs, labels in tqdm(loader, desc="  Val  ", leave=False, dynamic_ncols=True):
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        loss = bce_with_label_smoothing(preds, labels, 0.0, pos_weight)
        total_loss += loss.item() * imgs.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(preds.cpu().tolist())
    m = compute_metrics(all_labels, all_probs)
    m["loss"] = total_loss / len(loader.dataset)
    return m

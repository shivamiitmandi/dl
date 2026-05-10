import torch
import torch.nn as nn
from tqdm import tqdm
from deepfake_utils import (
    bce_with_label_smoothing,
    compute_metrics
)
def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    pos_weight,
    epoch,
    total_epochs,
):
    model.train()
    running_loss=0.0
    all_labels=[]
    all_probs=[]
    pbar=tqdm(
        loader,
        desc=f"Epoch {epoch+1}/{total_epochs} [Train]",
        leave=False,
        dynamic_ncols=True,
    )
    for imgs,labels in pbar:
        imgs=imgs.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        preds=model(imgs)
        loss=bce_with_label_smoothing(
            preds,
            labels,
            epsilon=0.1,
            pos_weight=pos_weight
        )
        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )
        optimizer.step()
        running_loss+=(
            loss.item()*imgs.size(0)
        )
        all_labels.extend(
            labels.cpu().tolist()
        )
        all_probs.extend(
            preds.detach()
            .cpu()
            .tolist()
        )
        live_acc=sum(
            1
            for p,l in zip(all_probs,all_labels)
            if (p>=0.5)==bool(l)
        )/len(all_labels)
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{live_acc:.4f}"
        )
    metrics=compute_metrics(
        all_labels,
        all_probs
    )
    metrics["loss"]=(
        running_loss/len(loader.dataset)
    )
    return metrics

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    pos_weight,
    epoch,
    total_epochs,
):
    model.eval()
    running_loss=0.0
    all_labels=[]
    all_probs=[]
    pbar=tqdm(
        loader,
        desc=f"Epoch {epoch+1}/{total_epochs} [Val]",
        leave=False,
        dynamic_ncols=True,
    )
    for imgs,labels in pbar:
        imgs=imgs.to(device)
        labels=labels.to(device)
        preds=model(imgs)
        loss=bce_with_label_smoothing(
            preds,
            labels,
            epsilon=0.1,
            pos_weight=pos_weight
        )
        running_loss+=(
            loss.item()*imgs.size(0)
        )
        all_labels.extend(
            labels.cpu().tolist()
        )
        all_probs.extend(
            preds.cpu().tolist()
        )
        pbar.set_postfix(
            loss=f"{loss.item():.4f}"
        )
    metrics=compute_metrics(
        all_labels,
        all_probs
    )
    metrics["loss"]=(
        running_loss/len(loader.dataset)
    )
    return metrics
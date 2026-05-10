import torch
import torch.nn as nn
from tqdm import tqdm
from utilities import compute_metrics

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
):
    model.train()
    running_loss=0.0
    all_labels=[]
    all_probs=[]
    for sequences,labels in tqdm(
        loader,
        desc="Train",
        leave=False,
        dynamic_ncols=True
    ):
        sequences=sequences.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits=model(sequences)
            loss=criterion(
                logits,
                labels
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()
        running_loss+=(
            loss.item()*sequences.size(0)
        )
        all_labels.extend(
            labels.cpu().tolist()
        )
        all_probs.extend(
            torch.sigmoid(logits)
            .detach()
            .cpu()
            .tolist()
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
    criterion,
    device,
):
    model.eval()
    running_loss=0.0
    all_labels=[]
    all_probs=[]
    for sequences,labels in tqdm(
        loader,
        desc="Val",
        leave=False,
        dynamic_ncols=True
    ):
        sequences=sequences.to(device)
        labels=labels.to(device)
        with torch.cuda.amp.autocast():
            logits=model(sequences)
            loss=criterion(
                logits,
                labels
            )
        running_loss+=(
            loss.item()*sequences.size(0)
        )
        all_labels.extend(
            labels.cpu().tolist()
        )
        all_probs.extend(
            torch.sigmoid(logits)
            .cpu()
            .tolist()
        )
    metrics=compute_metrics(
        all_labels,
        all_probs
    )
    metrics["loss"]=(
        running_loss/len(loader.dataset)
    )
    return metrics
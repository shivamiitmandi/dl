import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from deepfake_utils import (
    get_val_transform,
    compute_fft_spectrum,
    compute_metrics
)

@torch.no_grad()
def predict_single_image(model,image_path,device):
    transform=get_val_transform()
    image=Image.open(image_path).convert("RGB")
    image_tensor=transform(image)
    fft_tensor=compute_fft_spectrum(image_tensor)
    fft_batch=fft_tensor.unsqueeze(0).to(device)
    prob=model(fft_batch).item()
    verdict="FAKE" if prob>=0.5 else "REAL"
    print("\n"+"="*30)
    print(f"Image      : {image_path}")
    print(f"Prediction : {verdict}")
    print(f"Fake Prob  : {prob:.4f}")
    print("="*30+"\n")


@torch.no_grad()
def evaluate_dataset(model,loader,device):
    model.eval()
    all_labels=[]
    all_probs=[]
    for imgs,labels in tqdm(
        loader,
        desc="Testing"
    ):
        imgs=imgs.to(device)
        probs=model(imgs)
        all_labels.extend(
            labels.tolist()
        )
        all_probs.extend(
            probs.cpu().tolist()
        )
    metrics=compute_metrics(
        all_labels,
        all_probs
    )
    print("\n"+"="*30)
    print("Evaluation Results")
    print("="*30)
    print(
        f"Accuracy  : "
        f"{metrics['acc']:.4f}"
    )
    print(
        f"AUC       : "
        f"{metrics['auc']:.4f}"
    )
    print(
        f"F1 Score  : "
        f"{metrics['f1']:.4f}"
    )
    print(
        f"Precision : "
        f"{metrics['precision']:.4f}"
    )
    print(
        f"Recall    : "
        f"{metrics['recall']:.4f}"
    )
    print("="*30+"\n")
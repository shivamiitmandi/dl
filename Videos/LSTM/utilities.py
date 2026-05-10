import torch
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, roc_auc_score

def get_train_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def compute_metrics(labels, probs):
    preds = [1 if p > 0.5 else 0 for p in probs]
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0 
    return {"acc": acc, "auc": auc}

def save_checkpoint(state, filename):
    torch.save(state, filename)
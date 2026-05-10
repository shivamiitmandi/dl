import os
import glob
import random
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from deepfake_utils import get_train_transform, get_val_transform

def collect_samples(root, layout):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    samples = []
    if layout == "flat":
        for split in ("train", "valid", "test"):
            for label_str, label_int in [("real", 0), ("fake", 1)]:
                folder = os.path.join(root, split, label_str)
                if not os.path.isdir(folder):
                    continue
                for ext in exts:
                    for p in glob.glob(os.path.join(folder, ext)):
                        samples.append((p, label_int))
    elif layout == "nested":
        real_root = os.path.join(root, "real")
        fake_root = os.path.join(root, "fake")
        for folder in [real_root, fake_root]:
            label_int = 0 if folder == real_root else 1
            if not os.path.isdir(folder):
                print(f"  WARNING: {folder} does not exist, skipping.")
                continue
            for dirpath, _, files in os.walk(folder):
                for f in files:
                    if any(f.lower().endswith(e.replace("*", "")) for e in ["jpg", "jpeg", "png", "webp"]):
                        samples.append((os.path.join(dirpath, f), label_int))
    else:
        raise ValueError(f"Unknown layout: {layout}")
    return samples

def aggressive_collect_samples(root):
    samples = []
    real_root = os.path.join(root, "real")
    fake_root = os.path.join(root, "fake")
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif')
    for folder in [real_root, fake_root]:
        label_int = 0 if folder == real_root else 1
        if not os.path.isdir(folder):
            print(f"  WARNING: {folder} does not exist!")
            continue
        for dirpath, _, files in os.walk(folder, followlinks=True):
            for f in files:
                if f.lower().endswith(valid_exts):
                    samples.append((os.path.join(dirpath, f), label_int))
    return samples

def make_splits(samples, train_ratio=0.8, val_ratio=0.1, seed=42):
    random.seed(seed)
    real = [(p, l) for p, l in samples if l == 0]
    fake = [(p, l) for p, l in samples if l == 1]
    def split_list(lst):
        random.shuffle(lst)
        n = len(lst)
        n_tr = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return lst[:n_tr], lst[n_tr:n_tr + n_val], lst[n_tr + n_val:]
    r_tr, r_val, r_te = split_list(real)
    f_tr, f_val, f_te = split_list(fake)
    train = r_tr + f_tr
    val = r_val + f_val
    test = r_te + f_te
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test

def get_class_weights_from_samples(samples):
    n_real = sum(1 for _, l in samples if l == 0)
    n_fake = sum(1 for _, l in samples if l == 1)
    total = n_real + n_fake
    w_real = total / (2 * n_real + 1e-8)
    w_fake = total / (2 * n_fake + 1e-8)
    print(f"  Class weights → real={w_real:.3f}, fake={w_fake:.3f}")
    return torch.tensor([w_real, w_fake], dtype=torch.float32)

class DeepfakeDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        n_real = sum(1 for _, l in samples if l == 0)
        n_fake = sum(1 for _, l in samples if l == 1)
        print(f"  Dataset: {len(samples)} images  (real={n_real}, fake={n_fake})")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), 0)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)
    
def build_loaders(train_samples, val_samples, batch_size, num_workers):
    train_ds = DeepfakeDataset(train_samples, transform=get_train_transform())
    val_ds = DeepfakeDataset(val_samples, transform=get_val_transform())
    pf = 2 if num_workers > 0 else None
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0, prefetch_factor=pf,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0, prefetch_factor=pf,
    )
    return train_loader, val_loader

import os
import csv
import glob
import random
import numpy as np
import torch
import io
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score,average_precision_score

IMAGENET_MEAN=[0.485,0.456,0.406]
IMAGENET_STD=[0.229,0.224,0.225]

def get_train_transform(img_size=224):
    def jpeg_compress(img):
        quality=random.randint(30,100)
        buffer=io.BytesIO()
        img.save(buffer,format="JPEG",quality=quality)
        return Image.open(buffer)

    return transforms.Compose([
        transforms.RandomResizedCrop(img_size,scale=(0.8,1.0),ratio=(0.9,1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.2,0.2,0.2,0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.Lambda(jpeg_compress),
        transforms.GaussianBlur(kernel_size=3,sigma=(0.1,1.0)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x:x+0.01*torch.randn_like(x)),
        transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD),
    ])

def get_val_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD),
    ])

class FolderDeepfakeDataset(Dataset):

    def __init__(self,root,split="train",transform=None):

        self.transform=transform
        self.samples=[]

        for label_name,label in [("real",0),("fake",1)]:

            folder=os.path.join(root,split,label_name)

            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Missing folder: {folder}")

            extensions=("*.jpg","*.jpeg","*.png","*.webp")

            for ext in extensions:
                for path in glob.glob(os.path.join(folder,ext)):
                    self.samples.append((path,label))

        random.shuffle(self.samples)

        print(f"[{split}] Loaded {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):

        path,label=self.samples[idx]

        image=Image.open(path).convert("RGB")

        if self.transform:
            image=self.transform(image)

        return image,torch.tensor(label,dtype=torch.float32)

class CSVDeepfakeDataset(Dataset):

    def __init__(self,csv_path,img_root,transform=None):

        self.transform=transform
        self.samples=[]

        with open(csv_path,newline="") as f:

            reader=csv.DictReader(f)

            for row in reader:

                rel_path=row["path"].strip()

                label_name=row["label_str"].strip().lower()

                label=0 if label_name=="real" else 1

                full_path=os.path.join(img_root,rel_path)

                if os.path.isfile(full_path):
                    self.samples.append((full_path,label))

        print(f"Loaded {len(self.samples)} images from {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):

        path,label=self.samples[idx]

        image=Image.open(path).convert("RGB")

        if self.transform:
            image=self.transform(image)

        return image,torch.tensor(label,dtype=torch.float32)

def compute_fft_spectrum(img_tensor):

    gray=(0.299*img_tensor[0]+0.587*img_tensor[1]+0.114*img_tensor[2])

    fft=torch.fft.fft2(gray)

    fft=torch.fft.fftshift(fft)

    magnitude=torch.log1p(torch.abs(fft))

    magnitude=(magnitude-magnitude.min())/(magnitude.max()-magnitude.min()+1e-8)

    return magnitude.unsqueeze(0).repeat(3,1,1)

def compute_metrics(all_labels,all_probs):

    labels=np.array(all_labels)

    probs=np.array(all_probs)

    preds=(probs>=0.5).astype(int)

    acc=(preds==labels).mean()

    try:
        auc=roc_auc_score(labels,probs)
        ap=average_precision_score(labels,probs)

    except Exception:
        auc=0.0
        ap=0.0

    tp=((preds==1)&(labels==1)).sum()
    fp=((preds==1)&(labels==0)).sum()
    fn=((preds==0)&(labels==1)).sum()

    precision=tp/(tp+fp+1e-8)
    recall=tp/(tp+fn+1e-8)

    f1=(2*precision*recall)/(precision+recall+1e-8)

    return dict(
        acc=acc,
        auc=auc,
        ap=ap,
        precision=precision,
        recall=recall,
        f1=f1
    )

def save_checkpoint(state,path):

    os.makedirs(
        os.path.dirname(path)
        if os.path.dirname(path)
        else ".",
        exist_ok=True
    )

    torch.save(state,path)

    print(f"Checkpoint saved -> {path}")

def load_checkpoint(path,model,optimizer=None,scheduler=None):

    if not os.path.isfile(path):
        print(f"No checkpoint found at {path}")
        return 0,0.0

    checkpoint=torch.load(path,map_location="cpu")

    model.load_state_dict(checkpoint["model_state"])

    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if scheduler and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    start_epoch=checkpoint.get("epoch",0)+1

    best_auc=checkpoint.get("best_auc",0.0)

    print(f"Resumed from epoch {checkpoint.get('epoch',0)}")

    return start_epoch,best_auc

def get_class_weights(dataset):

    labels=[sample[1] for sample in dataset.samples]

    num_real=labels.count(0)
    num_fake=labels.count(1)

    total=num_real+num_fake

    real_weight=total/(2*num_real+1e-8)
    fake_weight=total/(2*num_fake+1e-8)

    print(f"Class weights -> real={real_weight:.3f}, fake={fake_weight:.3f}")

    return torch.tensor([real_weight,fake_weight],dtype=torch.float32)

def bce_with_label_smoothing(pred,target,epsilon=0.1,pos_weight=None):

    smooth_target=target*(1-epsilon)+epsilon/2

    if pos_weight is not None:

        weight=torch.where(
            target==1,
            pos_weight.to(target.device),
            torch.ones_like(target)
        )

        loss=torch.nn.functional.binary_cross_entropy(
            pred,
            smooth_target,
            weight=weight
        )

    else:

        loss=torch.nn.functional.binary_cross_entropy(
            pred,
            smooth_target
        )

    return loss
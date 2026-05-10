import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
sys.path.insert(0,os.path.dirname(__file__))
from data.dataset import FFTDataset
from models.model import FFTResNet18
from training.trainer import train_one_epoch,evaluate
from deepfake_utils import (
    FolderDeepfakeDataset,
    get_train_transform,
    get_val_transform,
    save_checkpoint,
    load_checkpoint,
    get_class_weights,
)

def build_arg_parser():
    parser=argparse.ArgumentParser(
        description="Train FFT + ResNet18 deepfake detector."
    )
    parser.add_argument(
        "--data_root",
        required=True
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4
    )
    parser.add_argument(
        "--ckpt_dir",
        default="checkpoints/fft_resnet18"
    )
    parser.add_argument(
        "--resume",
        action="store_true"
    )
    return parser


def main():
    args=build_arg_parser().parse_args()
    device=torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"\nUsing device: {device}")
    print(f"Dataset path: {args.data_root}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}\n")
    train_base=FolderDeepfakeDataset(
        args.data_root,
        "train",
        transform=get_train_transform()
    )
    val_base=FolderDeepfakeDataset(
        args.data_root,
        "valid",
        transform=get_val_transform()
    )
    train_dataset=FFTDataset(train_base)
    val_dataset=FFTDataset(val_base)
    train_loader=DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader=DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    weights=get_class_weights(train_base)
    pos_weight=weights[1].to(device)
    print(
        f"Positive class weight: "
        f"{pos_weight.item():.4f}"
    )
    model=FFTResNet18(
        pretrained=True
    ).to(device)
    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.05
    )
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    os.makedirs(
        args.ckpt_dir,
        exist_ok=True
    )
    latest_checkpoint=os.path.join(
        args.ckpt_dir,
        "latest.pt"
    )
    best_checkpoint=os.path.join(
        args.ckpt_dir,
        "best.pt"
    )
    start_epoch=0
    best_auc=0.0
    if args.resume:
        start_epoch,best_auc=load_checkpoint(
            latest_checkpoint,
            model,
            optimizer,
            scheduler
        )
    print("\n"+"="*60)
    print(
        f"FFT + ResNet18 | "
        f"epochs {start_epoch+1} -> {args.epochs}"
    )
    print("="*60+"\n")
    for epoch in range(start_epoch,args.epochs):
        train_metrics=train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            pos_weight,
            epoch,
            args.epochs
        )
        val_metrics=evaluate(
            model,
            val_loader,
            device,
            pos_weight,
            epoch,
            args.epochs
        )
        scheduler.step()
        print(
            f"Epoch [{epoch+1:03d}/{args.epochs}] | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['acc']:.4f} | "
            f"Train AUC: {train_metrics['auc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['acc']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )
        save_checkpoint(
            {
                "epoch":epoch,
                "model_state":model.state_dict(),
                "optimizer_state":optimizer.state_dict(),
                "scheduler_state":scheduler.state_dict(),
                "best_auc":best_auc,
                "val_metrics":val_metrics,
            },
            latest_checkpoint
        )
        if val_metrics["auc"]>best_auc:
            best_auc=val_metrics["auc"]
            save_checkpoint(
                {
                    "epoch":epoch,
                    "model_state":model.state_dict(),
                    "best_auc":best_auc,
                },
                best_checkpoint
            )
            print(
                f"New best AUC: "
                f"{best_auc:.4f}"
            )
    print(
        f"\nTraining complete. "
        f"Best validation AUC: {best_auc:.4f}"
    )
    print(
        f"Best model saved at: "
        f"{best_checkpoint}"
    )
if __name__=="__main__":
    main()
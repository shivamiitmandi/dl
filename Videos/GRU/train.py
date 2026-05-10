import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import VideoSequenceDataset
from models.model import DINOGRUClassifier
from training.trainer import train_one_epoch,evaluate
from utilities import save_checkpoint,get_train_transform,get_val_transform

sys.path.insert(0,os.path.dirname(__file__))

TRAINING_PHASES=[(1,10),(2,20)]

def build_arg_parser():
    parser=argparse.ArgumentParser(description="Train DINO-GRU deepfake detector.")
    parser.add_argument("--data_root",required=True)
    parser.add_argument("--seq_len",type=int,default=16)
    parser.add_argument("--batch_size",type=int,default=4)
    parser.add_argument("--base_lr",type=float,default=1e-4)
    parser.add_argument("--num_workers",type=int,default=4)
    parser.add_argument("--ckpt_dir",default="checkpoints/dino_gru")
    parser.add_argument("--resume",action="store_true")
    return parser

def build_phase2_optimizer(model,base_lr):
    return torch.optim.AdamW([
        {"params":model.backbone.parameters(),"lr":base_lr*0.1},
        {"params":model.gru.parameters(),"lr":base_lr},
        {"params":model.attention.parameters(),"lr":base_lr},
        {"params":model.classifier.parameters(),"lr":base_lr},
    ],weight_decay=0.05)

def main():
    args=build_arg_parser().parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"Dataset path: {args.data_root}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.base_lr}\n")
    train_dataset=VideoSequenceDataset(
        data_root=args.data_root,
        split="train",
        seq_len=args.seq_len,
        transform=get_train_transform(),
        is_train=True,
    )
    val_dataset=VideoSequenceDataset(
        data_root=args.data_root,
        split="valid",
        seq_len=args.seq_len,
        transform=get_val_transform(),
        is_train=False,
    )
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

    num_real=sum(1 for sample in train_dataset.samples if sample[1]==0.0)
    num_fake=sum(1 for sample in train_dataset.samples if sample[1]==1.0)
    pos_weight=torch.tensor([num_real/max(num_fake,1.0)],device=device)
    criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Real videos: {num_real} | Fake videos: {num_fake}")
    model=DINOGRUClassifier().to(device)
    scaler=torch.cuda.amp.GradScaler()
    os.makedirs(args.ckpt_dir,exist_ok=True)
    latest_checkpoint=os.path.join(args.ckpt_dir,"latest.pt")
    best_checkpoint=os.path.join(args.ckpt_dir,"best.pt")
    global_epoch=0
    best_auc=0.0

    if args.resume and os.path.isfile(latest_checkpoint):
        print(f"Loading checkpoint from {latest_checkpoint}")
        checkpoint=torch.load(latest_checkpoint,map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        global_epoch=checkpoint.get("global_epoch",0)
        best_auc=checkpoint.get("best_auc",0.0)
        if "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        print(f"Resumed from epoch {global_epoch}")

    phase_start=0
    for phase_id,total_epochs in TRAINING_PHASES:
        phase_end=phase_start+total_epochs
        if global_epoch>=phase_end:
            phase_start=phase_end
            continue
        print("\n"+"="*60)
        print(f"Phase {phase_id} ({phase_start+1} -> {phase_end})")
        print("="*60)

        if phase_id==1:
            model.freeze_backbone()
            optimizer=torch.optim.AdamW(
                filter(lambda p:p.requires_grad,model.parameters()),
                lr=args.base_lr
            )
        else:
            model.unfreeze_last_n_blocks(n=4)
            optimizer=build_phase2_optimizer(model,args.base_lr)
        remaining_epochs=phase_end-global_epoch

        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=remaining_epochs,
            eta_min=1e-6
        )

        for _ in range(remaining_epochs):
            current_epoch=global_epoch
            train_metrics=train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                scaler,
                device
            )
            val_metrics=evaluate(
                model,
                val_loader,
                criterion,
                device
            )
            scheduler.step()
            global_epoch+=1
            print(
                f"[Phase {phase_id}] Epoch {current_epoch+1:03d} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['acc']:.4f} | "
                f"Train AUC: {train_metrics['auc']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['acc']:.4f} | "
                f"Val AUC: {val_metrics['auc']:.4f}"
            )

            if val_metrics["auc"]>best_auc:
                best_auc=val_metrics["auc"]
                save_checkpoint(
                    {
                        "global_epoch":global_epoch,
                        "model_state":model.state_dict(),
                        "best_auc":best_auc,
                    },
                    best_checkpoint
                )

                print(f"New best AUC: {best_auc:.4f}")

            save_checkpoint(
                {
                    "global_epoch":global_epoch,
                    "phase":phase_id,
                    "model_state":model.state_dict(),
                    "optimizer_state":optimizer.state_dict(),
                    "scheduler_state":scheduler.state_dict(),
                    "scaler_state":scaler.state_dict(),
                    "best_auc":best_auc,
                },
                latest_checkpoint
            )
        phase_start=phase_end
    print(f"\nTraining complete. Best validation AUC: {best_auc:.4f}")
if __name__=="__main__":
    main()
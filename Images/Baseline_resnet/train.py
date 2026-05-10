import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import argparse
import torch
from deepfake_utils import get_class_weights, load_checkpoint
from Baseline_resnet.model import BaselineResNet50
from Baseline_resnet.data import build_loaders
from Baseline_resnet.phases import run_phases
def parse_args():
    p = argparse.ArgumentParser(description="Model 5 – Baseline ResNet-50")
    p.add_argument("--data_root", default=None)
    p.add_argument("--use_csv", action="store_true")
    p.add_argument("--train_csv", default=None)
    p.add_argument("--valid_csv", default=None)
    p.add_argument("--img_root", default=None)
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--lr_phase1", type=float, default=1e-4)
    p.add_argument("--lr_phase2", type=float, default=1e-5)
    p.add_argument("--lr_phase3", type=float, default=1e-5)
    p.add_argument("--epochs_phase1", type=int, default=32)
    p.add_argument("--epochs_phase2", type=int, default=32)
    p.add_argument("--epochs_phase3", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--ckpt_dir", default="checkpoints/model5_baseline")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--early_stop_patience", type=int, default=5)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_loader, val_loader, train_ds = build_loaders(args)
    weights = get_class_weights(train_ds)
    pos_weight = weights[1].to(device)
    model = BaselineResNet50().to(device)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, "latest.pt")
    best_path = os.path.join(args.ckpt_dir, "best.pt")
    global_epoch = 0
    best_auc = 0.0
    if args.resume and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        global_epoch = ckpt.get("global_epoch", 0)
        best_auc = ckpt.get("best_auc", 0.0)
        print(f"  Resumed from epoch {global_epoch}, best_auc={best_auc:.4f}")
    best_auc = run_phases(
        model, train_loader, val_loader, pos_weight,
        args, device, ckpt_path, best_path, global_epoch, best_auc,
    )
    print(f"\nTraining complete. Best Val AUC: {best_auc:.4f}")
    print(f"Best model saved at: {best_path}")
if __name__ == "__main__":
    main()

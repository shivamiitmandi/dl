import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import argparse
import torch
from torch.utils.data import DataLoader
from model import DINODeepfakeDetector
from data import collect_samples, make_splits, get_class_weights_from_samples, DeepfakeDataset, build_loaders
from phases import run_phases
from loops import evaluate
from deepfake_utils import get_val_transform
def parse_args():
    p = argparse.ArgumentParser(description="Model 4 (Revised) — DINO SSL ViT-B/16 deepfake detector")
    p.add_argument("--data_root", required=True)
    p.add_argument("--extra_root", default=None)
    p.add_argument("--layout", default="nested", choices=["flat", "nested"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--base_lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--ckpt_dir", default="checkpoints/model4_dino")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--phase1_epochs", type=int, default=10)
    p.add_argument("--phase2_epochs", type=int, default=20)
    p.add_argument("--phase3_epochs", type=int, default=30)
    return p.parse_args()
def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Layout : {args.layout}")
    print(f"Root   : {args.data_root}")
    print("\n  Scanning dataset...")
    all_samples = collect_samples(args.data_root, args.layout)
    if args.extra_root:
        print(f"\n  Scanning extra dataset: {args.extra_root}")
        extra = collect_samples(args.extra_root, args.layout)
        all_samples += extra
        print(f"  Combined total: {len(all_samples)} images")
    if args.layout == "flat":
        train_samples = [(p, l) for p, l in all_samples if "/train/" in p]
        val_samples = [(p, l) for p, l in all_samples if "/valid/" in p]
        test_samples = [(p, l) for p, l in all_samples if "/test/" in p]
        if not train_samples:
            train_samples, val_samples, test_samples = make_splits(all_samples)
    else:
        train_samples, val_samples, test_samples = make_splits(all_samples, train_ratio=0.8, val_ratio=0.1)
    print(f"\n  Train: {len(train_samples)}  |  Val: {len(val_samples)}  |  Test: {len(test_samples)}")
    train_loader, val_loader = build_loaders(train_samples, val_samples, args.batch_size, args.num_workers)
    weights = get_class_weights_from_samples(train_samples)
    pos_weight = weights[1].to(device)
    model = DINODeepfakeDetector().to(device)
    ckpt_path = os.path.join(args.ckpt_dir, "latest.pt")
    best_path = os.path.join(args.ckpt_dir, "best.pt")
    global_epoch = 0
    best_auc = 0.0
    if args.resume and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        global_epoch = ckpt.get("global_epoch", 0)
        best_auc = ckpt.get("best_auc", 0.0)
        print(f"\n  Resumed from epoch {global_epoch}  (best AUC so far: {best_auc:.4f})")
    best_auc, global_epoch = run_phases(
        model, train_loader, val_loader, pos_weight,
        args, device, ckpt_path, best_path, global_epoch, best_auc,
    )
    print(f"\n{'='*60}")
    print(f"  Training complete. Best Val AUC: {best_auc:.4f}")
    print(f"  Running final evaluation on test set...")
    print(f"{'='*60}")
    test_ds = DeepfakeDataset(test_samples, transform=get_val_transform())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    te = evaluate(model, test_loader, device, pos_weight, 0, 1, split="Test")
    print(f"\n  TEST RESULTS:")
    print(f"    Accuracy  : {te['acc']:.4f}")
    print(f"    AUC-ROC   : {te['auc']:.4f}")
    print(f"    F1        : {te['f1']:.4f}")
    print(f"    Precision : {te['precision']:.4f}")
    print(f"    Recall    : {te['recall']:.4f}")
    print(f"\n  Best model saved at: {best_path}")
if __name__ == "__main__":
    main()

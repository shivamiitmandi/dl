import os
import torch
from deepfake_utils import save_checkpoint, get_class_weights
from Baseline_resnet.loops import train_one_epoch, evaluate
def build_phase_schedule(args):
    return [
        (1, args.epochs_phase1, args.lr_phase1, lambda m: m.freeze_backbone()),
        (2, args.epochs_phase2, args.lr_phase2, lambda m: m.unfreeze_last_n_layers(30)),
        (3, args.epochs_phase3, args.lr_phase3, lambda m: m.unfreeze_all()),
    ]
def run_phases(model, train_loader, val_loader, pos_weight, args, device, ckpt_path, best_path, global_epoch, best_auc):
    schedule = build_phase_schedule(args)
    phase_start = 0
    for phase_id, n_epochs, lr, unfreeze_fn in schedule:
        phase_end = phase_start + n_epochs
        if global_epoch >= phase_end:
            phase_start = phase_end
            continue
        print(f"\n{'='*60}")
        print(f"  PHASE {phase_id}  (epochs {phase_start+1}→{phase_end}, lr={lr})")
        print(f"{'='*60}")
        unfreeze_fn(model)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.3, patience=8, min_lr=1e-6
        )
        patience_counter = 0
        best_val_loss = float("inf")
        for _ in range(phase_end - global_epoch):
            epoch = global_epoch
            tr = train_one_epoch(model, train_loader, optimizer, device, pos_weight)
            vl = evaluate(model, val_loader, device, pos_weight)
            scheduler.step(vl["acc"])
            global_epoch += 1
            print(
                f"[P{phase_id}] Ep {epoch+1:03d}  "
                f"Tr  loss={tr['loss']:.4f} acc={tr['acc']:.4f} auc={tr['auc']:.4f}  |  "
                f"Val loss={vl['loss']:.4f} acc={vl['acc']:.4f} auc={vl['auc']:.4f}  "
                f"f1={vl['f1']:.4f}"
            )
            save_checkpoint({
                "global_epoch": global_epoch,
                "phase": phase_id,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_auc": best_auc,
            }, ckpt_path)
            if vl["auc"] > best_auc:
                best_auc = vl["auc"]
                save_checkpoint({
                    "global_epoch": global_epoch,
                    "model_state": model.state_dict(),
                    "best_auc": best_auc,
                }, best_path)
                print(f"  New best AUC: {best_auc:.4f}")
            if phase_id >= 2:
                if vl["loss"] < best_val_loss:
                    best_val_loss = vl["loss"]
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= args.early_stop_patience:
                        print(f"  Early stopping triggered at epoch {epoch+1}.")
                        global_epoch = phase_end
                        break
        phase_start = phase_end
    return best_auc

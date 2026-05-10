import torch
from deepfake_utils import save_checkpoint
from loops import train_one_epoch, evaluate, make_optimizer

def run_phases(model, train_loader, val_loader, pos_weight, args, device, ckpt_path, best_path, global_epoch, best_auc):
    schedule = [
        (1, args.phase1_epochs),
        (2, args.phase2_epochs),
        (3, args.phase3_epochs),
    ]
    total_epochs = sum(e for _, e in schedule)
    phase_start = 0
    for phase_id, n_epochs in schedule:
        phase_end = phase_start + n_epochs
        if global_epoch >= phase_end:
            phase_start = phase_end
            continue
        print(f"\n{'='*60}")
        print(f"  PHASE {phase_id}  (epochs {phase_start+1}→{phase_end})")
        print(f"{'='*60}")
        if phase_id == 1:
            model.freeze_backbone()
        elif phase_id == 2:
            model.freeze_backbone()
            model.unfreeze_last_n_blocks(4)
        else:
            model.unfreeze_all()
        optimizer = make_optimizer(model, phase_id, args.base_lr)
        remaining = phase_end - global_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining, eta_min=1e-6
        )
        for _ in range(remaining):
            epoch = global_epoch
            tr = train_one_epoch(model, train_loader, optimizer, device, pos_weight, epoch, total_epochs)
            vl = evaluate(model, val_loader, device, pos_weight, epoch, total_epochs)
            scheduler.step()
            global_epoch += 1
            print(
                f"[P{phase_id}] Ep {epoch+1:03d}/{total_epochs}  "
                f"Tr  loss={tr['loss']:.4f} acc={tr['acc']:.4f} auc={tr['auc']:.4f}  |  "
                f"Val loss={vl['loss']:.4f} acc={vl['acc']:.4f} auc={vl['auc']:.4f}  f1={vl['f1']:.4f}"
            )
            save_checkpoint({
                "global_epoch": global_epoch,
                "phase": phase_id,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
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
        phase_start = phase_end
    return best_auc, global_epoch

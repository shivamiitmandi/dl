import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Images.DINO.model import DINODeepfakeDetector
from Images.DINO.data import aggressive_collect_samples, DeepfakeDataset
from deepfake_utils import get_val_transform, compute_metrics

def evaluate_wild_deepfake(
    data_root="/DATA/group11_datasets/shivam/Master_Video_Dataset",
    ckpt_path="/DATA/group11_datasets/shivam/dataset/fixes/model4_ssl_fft/checkpoints/model4_combined/best.pt",
    batch_size=32,
    num_workers=4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"\nScanning dataset at: {data_root} (Aggressive Mode)")
    test_samples = aggressive_collect_samples(data_root)
    if len(test_samples) == 0:
        print("No images found!")
        print("Run: find <data_root>/fake -type f | head -n 5")
        return
    print(f"Found {len(test_samples)} total images!")
    test_ds = DeepfakeDataset(test_samples, transform=get_val_transform())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print("\nInitializing DINO model...")
    model = DINODeepfakeDetector().to(device)
    print(f"Loading weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("\nStarting evaluation...")
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating", dynamic_ncols=True):
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(preds.cpu().tolist())
    metrics = compute_metrics(all_labels, all_probs)
    print(f"\n{'='*60}")
    print(f"  MASTER VIDEO DATASET - TEST RESULTS")
    print(f"{'='*60}")
    print(f"    Accuracy  : {metrics.get('acc', 0.0):.4f}")
    print(f"    AUC-ROC   : {metrics.get('auc', 0.0):.4f}")
    print(f"    F1 Score  : {metrics.get('f1', 0.0):.4f}")
    print(f"    Precision : {metrics.get('precision', 0.0):.4f}")
    print(f"    Recall    : {metrics.get('recall', 0.0):.4f}")
    print(f"{'='*60}")
    
if __name__ == "__main__":
    evaluate_wild_deepfake()

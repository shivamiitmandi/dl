import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import argparse
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from Images.DINO.model import DINODeepfakeDetector
from deepfake_utils import get_val_transform

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def analyze_image(image_path, ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Initializing MTCNN...")
    detector = MTCNN(margin=20, keep_all=False, post_process=False, device=device)
    print(f"Reading image: {image_path}")
    original_img = Image.open(image_path).convert("RGB")
    face_tensor = detector(original_img)
    if face_tensor is None:
        print("MTCNN could not detect a face in this image.")
        return
    face_img = Image.fromarray(face_tensor.permute(1, 2, 0).byte().cpu().numpy())
    transform = get_val_transform()
    input_tensor = transform(face_img).unsqueeze(0).to(device)
    print("Loading DINO Model...")
    model = DINODeepfakeDetector().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        score = model(input_tensor).item()
        pred_label = "FAKE" if score >= 0.5 else "REAL"
        confidence = score if pred_label == "FAKE" else 1 - score
    print(f"\nModel Prediction: {pred_label} (Confidence: {confidence:.2%})")
    target_layers = [model.backbone.blocks[-1].norm1]
    targets = [BinaryClassifierOutputTarget(1)]
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    print("Generating Grad-CAM heatmap...")
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    face_resized = np.array(face_img.resize((224, 224)))
    rgb_img = np.float32(face_resized) / 255.0
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cam_img_pil = Image.fromarray(cam_image)
    out_name = f"gradcam_{pred_label}_{os.path.basename(image_path)}"
    cam_img_pil.save(out_name)
    print(f"Saved visualization to: {out_name}")
    
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--ckpt", default="/DATA/group11_datasets/shivam/dataset/fixes/model4_ssl_fft/checkpoints/model4_combined/best.pt")
    return p.parse_args()
if __name__ == "__main__":
    args = parse_args()
    analyze_image(args.image, args.ckpt)

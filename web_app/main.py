import os
import sys
import tempfile
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import base64
from PIL import Image
import io
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from facenet_pytorch import MTCNN
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from Images.DINO.model import DINODeepfakeDetector
from Images.FFT.models.model import FFTResNet18
from Videos.GRU.models.model import DINOGRUClassifier
from Images.FFT.deepfake_utils import compute_fft_spectrum

app = FastAPI(title="Deepfake Detection UI")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {
    "DINO": None,
    "FFT": None,
    "GRU": None,
    "MTCNN": None
}

image_val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def load_models():
    if models["MTCNN"] is None:
        print("Loading MTCNN...")
        models["MTCNN"] = MTCNN(margin=20, keep_all=False, post_process=False, device=device)
    if models["DINO"] is None:
        print("Loading DINO model...")
        dino = DINODeepfakeDetector().to(device)
        checkpoint = torch.load(os.path.join(PROJECT_ROOT, "Images", "DINO", "best.pt"), map_location=device, weights_only=False)
        dino.load_state_dict(checkpoint["model_state"])
        dino.eval()
        models["DINO"] = dino
    if models["FFT"] is None:
        print("Loading FFT model...")
        fft = FFTResNet18(pretrained=False).to(device)
        checkpoint = torch.load(os.path.join(PROJECT_ROOT, "Images", "FFT", "best.pt"), map_location=device, weights_only=False)
        fft.load_state_dict(checkpoint["model_state"])
        fft.eval()
        models["FFT"] = fft
    if models["GRU"] is None:
        print("Loading GRU model...")
        gru = DINOGRUClassifier().to(device)
        checkpoint = torch.load(os.path.join(PROJECT_ROOT, "Videos", "GRU", "best.pt"), map_location=device, weights_only=False)
        gru.load_state_dict(checkpoint["model_state"])
        gru.eval()
        models["GRU"] = gru

@app.on_event("startup")
async def startup_event():
    pass

@app.get("/", response_class=HTMLResponse)
async def get_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...), model_choice: str = Form(...)):
    if model_choice not in ["DINO", "FFT"]:
        return JSONResponse({"error": "Invalid model choice"}, status_code=400)
    
    load_models()

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_tensor = image_val_transform(image)
    
    model = models[model_choice]
    
    with torch.no_grad():
        if model_choice == "DINO":
            batch = image_tensor.unsqueeze(0).to(device)
            prob = model(batch).item()
        elif model_choice == "FFT":
            fft_tensor = compute_fft_spectrum(image_tensor)
            batch = fft_tensor.unsqueeze(0).to(device)
            prob = model(batch).item()

    verdict = "FAKE" if prob >= 0.5 else "REAL"
    confidence = prob if prob >= 0.5 else 1.0 - prob
    
    return JSONResponse({
        "prediction": verdict,
        "probability": round(prob, 4),
        "confidence": round(confidence * 100, 2),
        "model": model_choice
    })

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def tensor_to_base64(img_array):
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def extract_faces_from_video(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        frame_indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
    frames, face_tensors, original_faces = [], [], []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        
        face_tensor = models["MTCNN"](img)
        if face_tensor is not None:
            face_img = Image.fromarray(face_tensor.permute(1, 2, 0).byte().cpu().numpy())
            original_faces.append(np.array(face_img.resize((224, 224))) / 255.0) 
            face_tensors.append(image_val_transform(face_img))
            
    cap.release()
    
    while len(face_tensors) < num_frames and len(face_tensors) > 0:
        face_tensors.append(face_tensors[-1])
        original_faces.append(original_faces[-1])
        
    if len(face_tensors) == 0:
        raise ValueError("No faces detected in the video.")
        
    return torch.stack(face_tensors).unsqueeze(0), original_faces 

@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    load_models()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        input_tensor, original_faces = extract_faces_from_video(tmp_path, num_frames=16)
    except ValueError as e:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        return JSONResponse({"error": f"Failed to extract frames: {str(e)}"}, status_code=500)
        
    input_tensor = input_tensor.to(device)
    model = models["GRU"]
    
    with torch.no_grad():
        logits, _ = model(input_tensor, return_attention=True)
        prob = torch.sigmoid(logits).item()
        
    verdict = "FAKE" if prob >= 0.5 else "REAL"
    confidence = prob if prob >= 0.5 else 1.0 - prob

    target_layers = [model.backbone.blocks[-1].norm1]
    flattened_tensor = input_tensor.view(-1, 3, 224, 224)
    
    flat_cam = GradCAM(model=model.backbone, target_layers=target_layers, reshape_transform=reshape_transform)
    grayscale_cams = flat_cam(input_tensor=flattened_tensor, targets=None)
    
    cam_results = []
    for i in range(16):
        vis = show_cam_on_image(original_faces[i], grayscale_cams[i, :], use_rgb=True)
        cam_results.append(tensor_to_base64(vis))
        
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    
    return JSONResponse({
        "prediction": verdict,
        "probability": round(prob, 4),
        "confidence": round(confidence * 100, 2),
        "model": "GRU",
        "frames": cam_results,
        "spatial_heatmaps": True
    })

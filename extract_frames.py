import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from pathlib import Path
from tqdm import tqdm

FF_ROOT=Path("/DATA/group11_datasets/FaceForensics_frames")
OUT_DIR=Path("/DATA/group11_datasets/shivam/FF_Cropped_Faces")

FAKE_DIRS=[
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures"
]
REAL_DIRS=["original"]

def crop_directory(source_dirs,label,detector):
    for folder_name in source_dirs:
        folder_path=FF_ROOT/folder_name
        if not folder_path.exists():
            print(
                f"Skipping {folder_name}, "
                f"not found."
            )
            continue
        video_folders=[
            folder
            for folder in folder_path.iterdir()
            if folder.is_dir()
        ]
        print(
            f"\nProcessing "
            f"{folder_name} "
            f"({len(video_folders)} videos)..."
        )
        for video_folder in tqdm(video_folders):
            video_name=video_folder.name
            output_video_dir=(
                OUT_DIR
                /label
                /folder_name
                /video_name
            )
            output_video_dir.mkdir(
                parents=True,
                exist_ok=True
            )
            frames=sorted(
                list(video_folder.glob("*.png"))
                +list(video_folder.glob("*.jpg"))
            )
            for index,frame_path in enumerate(frames):
                try:
                    image=Image.open(
                        frame_path
                    ).convert("RGB")
                    face_tensor=detector(image)
                    if face_tensor is not None:
                        face_image=Image.fromarray(
                            face_tensor
                            .permute(1,2,0)
                            .byte()
                            .cpu()
                            .numpy()
                        )
                        face_image.save(
                            output_video_dir
                            /f"frame_{index:04d}.jpg"
                        )
                except Exception:
                    pass
def main():
    device=torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    detector=MTCNN(
        margin=20,
        keep_all=False,
        post_process=False,
        device=device
    )
    crop_directory(
        REAL_DIRS,
        "real",
        detector
    )
    crop_directory(
        FAKE_DIRS,
        "fake",
        detector
    )
    print("\nAll faces cropped successfully.")
if __name__=="__main__":
    main()
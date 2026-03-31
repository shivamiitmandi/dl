"""
extract_frames.py — Extract frames from FaceForensics++ video files
====================================================================
Takes the FF++ video folder and extracts face-cropped frames,
saving them in the required train/val/test/real/fake structure.

Usage:
    python extract_frames.py \
        --data_root /path/to/FaceForensics++ \
        --output_dir ./dataset \
        --every_n_frames 10 \
        --compression c23

FF++ expected folder structure:
    FaceForensics++/
        original_sequences/youtube/c23/videos/
        manipulated_sequences/
            DeepFakes/c23/videos/
            Face2Face/c23/videos/
            FaceSwap/c23/videos/
            NeuralTextures/c23/videos/
"""

import os
import cv2
import argparse
import json
import random
from pathlib import Path
from tqdm import tqdm


SPLITS = {'train': 0.7, 'val': 0.15, 'test': 0.15}
FAKE_METHODS = ['DeepFakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']


def extract_frames_from_video(video_path: str, output_dir: str,
                               every_n: int = 10, max_frames: int = 50) -> int:
    """Extract every_n-th frame from a video, up to max_frames."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    saved = 0
    video_name = Path(video_path).stem

    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_n == 0:
            out_path = os.path.join(output_dir, f"{video_name}_f{frame_idx:05d}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1
        frame_idx += 1

    cap.release()
    return saved


def get_video_list(videos_dir: str):
    videos = []
    for ext in ['*.mp4', '*.avi', '*.mov']:
        videos.extend(Path(videos_dir).glob(ext))
    return sorted(videos)


def assign_splits(video_list, seed=42):
    """Randomly assign videos to train/val/test splits."""
    random.seed(seed)
    videos = [str(v) for v in video_list]
    random.shuffle(videos)
    n = len(videos)
    n_train = int(n * SPLITS['train'])
    n_val   = int(n * SPLITS['val'])
    return {
        'train': videos[:n_train],
        'val':   videos[n_train:n_train + n_val],
        'test':  videos[n_train + n_val:],
    }


def main(args):
    out = Path(args.output_dir)

    # ── Real frames ──────────────────────────────────────────────────────────
    real_dir = Path(args.data_root) / 'original_sequences' / 'youtube' / args.compression / 'videos'
    if real_dir.exists():
        print(f"Extracting REAL frames from {real_dir}")
        real_videos = get_video_list(str(real_dir))
        split_map   = assign_splits(real_videos)
        for split, vids in split_map.items():
            dest = out / split / 'real'
            dest.mkdir(parents=True, exist_ok=True)
            for vpath in tqdm(vids, desc=f'Real/{split}'):
                extract_frames_from_video(
                    vpath, str(dest),
                    every_n=args.every_n_frames,
                    max_frames=args.max_frames_per_video,
                )
    else:
        print(f"[Warning] Real video dir not found: {real_dir}")

    # ── Fake frames ───────────────────────────────────────────────────────────
    for method in FAKE_METHODS:
        fake_dir = (Path(args.data_root) / 'manipulated_sequences' /
                    method / args.compression / 'videos')
        if not fake_dir.exists():
            print(f"[Warning] Fake video dir not found: {fake_dir}")
            continue
        print(f"Extracting FAKE frames ({method}) from {fake_dir}")
        fake_videos = get_video_list(str(fake_dir))
        split_map   = assign_splits(fake_videos)
        for split, vids in split_map.items():
            dest = out / split / 'fake'
            dest.mkdir(parents=True, exist_ok=True)
            for vpath in tqdm(vids, desc=f'{method}/{split}'):
                extract_frames_from_video(
                    vpath, str(dest),
                    every_n=args.every_n_frames,
                    max_frames=args.max_frames_per_video,
                )

    # Print summary
    print("\nDataset Summary:")
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            d = out / split / label
            if d.exists():
                n = len(list(d.glob('*.jpg')))
                print(f"  {split:5s}/{label:4s}: {n:6d} images")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frames from FF++ videos')
    parser.add_argument('--data_root',   type=str, required=True)
    parser.add_argument('--output_dir',  type=str, default='./dataset')
    parser.add_argument('--compression', type=str, default='c23',
                        choices=['raw', 'c23', 'c40'])
    parser.add_argument('--every_n_frames', type=int, default=10)
    parser.add_argument('--max_frames_per_video', type=int, default=50)
    args = parser.parse_args()
    main(args)

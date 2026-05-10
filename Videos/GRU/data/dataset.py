import os
import glob
import random
import torch
from torch.utils.data import Dataset
from PIL import Image


class VideoSequenceDataset(Dataset):
    def __init__(self,
        data_root: str,
        split: str = "train",
        seq_len: int = 16,
        transform=None,
        is_train: bool = False,
    ):
        self.seq_len = seq_len
        self.transform = transform
        self.is_train = is_train
        all_samples = self._scan_videos(data_root)

        random.seed(42)
        random.shuffle(all_samples)

        split_idx = int(len(all_samples) * 0.8)
        if split == "train":
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]
        print(f"[Dataset] {split:>5s}: {len(self.samples)} videos loaded.")

    def _scan_videos(self, data_root: str):
        samples = []
        for label_name, label_value in [("real", 0.0), ("fake", 1.0)]:
            class_dir = os.path.join(data_root, label_name)
            if not os.path.isdir(class_dir):
                print(f"[Dataset] Warning: '{class_dir}' not found, skipping.")
                continue

            for video_id in os.listdir(class_dir):
                video_dir = os.path.join(class_dir, video_id)
                if not os.path.isdir(video_dir):
                    continue

                frames = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
                if len(frames) >= self.seq_len:
                    samples.append((frames, label_value))
        samples.sort(key=lambda x: x[0][0])
        return samples

    def _pick_indices(self, total_frames: int) -> list[int]:
        if self.is_train:
            start = random.randint(0, total_frames - self.seq_len)
            indices = list(range(start, start + self.seq_len))
            if random.random() < 0.5:
                indices = indices[::-1]  
        else:
            start = (total_frames - self.seq_len) // 2
            indices = list(range(start, start + self.seq_len))
        return indices

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        frame_paths, label = self.samples[idx]
        indices = self._pick_indices(len(frame_paths))

        frames = []
        for i in indices:
            img = Image.open(frame_paths[i]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        sequence = torch.stack(frames)
        return sequence, torch.tensor(label, dtype=torch.float32)

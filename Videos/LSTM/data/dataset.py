import os
import glob
import random
import torch
from torch.utils.data import Dataset
from PIL import Image

class VideoSequenceDataset(Dataset):
    def __init__(
        self,
        data_root,
        split="train",
        seq_len=10,
        transform=None,
        is_train=False,
    ):
        self.seq_len=seq_len
        self.transform=transform
        self.is_train=is_train
        all_samples=self._scan_videos(data_root)
        random.seed(42)
        random.shuffle(all_samples)
        split_idx=int(len(all_samples)*0.8)
        if split=="train":
            self.samples=all_samples[:split_idx]
        else:
            self.samples=all_samples[split_idx:]
        print(
            f"{split.capitalize()} dataset loaded "
            f"with {len(self.samples)} videos."
        )
    def _scan_videos(self,data_root):
        samples=[]
        for class_name,label in [
            ("real",0.0),
            ("fake",1.0),
        ]:
            class_dir=os.path.join(
                data_root,
                class_name
            )
            if not os.path.isdir(class_dir):
                print(f"Skipping missing folder: {class_dir}")
                continue
            for video_name in os.listdir(class_dir):
                video_dir=os.path.join(
                    class_dir,
                    video_name
                )
                if not os.path.isdir(video_dir):
                    continue
                frame_paths=sorted(
                    glob.glob(
                        os.path.join(video_dir,"*.jpg")
                    )
                )
                if len(frame_paths)>=self.seq_len:
                    samples.append(
                        (frame_paths,label)
                    )
        samples.sort(key=lambda x:x[0][0])
        return samples
    def _pick_indices(self,total_frames):
        if self.is_train:
            indices=sorted(
                random.sample(
                    range(total_frames),
                    self.seq_len
                )
            )
            if random.random()<0.5:
                indices=indices[::-1]
        else:
            indices=torch.linspace(
                0,
                total_frames-1,
                self.seq_len
            ).long().tolist()
        return indices
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,idx):
        frame_paths,label=self.samples[idx]
        selected_indices=self._pick_indices(
            len(frame_paths)
        )
        frames=[]
        for frame_index in selected_indices:
            image=Image.open(
                frame_paths[frame_index]
            ).convert("RGB")
            if self.transform:
                image=self.transform(image)
            frames.append(image)
        video_tensor=torch.stack(frames)
        return (
            video_tensor,
            torch.tensor(
                label,
                dtype=torch.float32
            ),
        )
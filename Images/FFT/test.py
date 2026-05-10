import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
sys.path.insert(0,os.path.dirname(__file__))
from data.dataset import FFTDataset
from models.model import FFTResNet18
from utils.inference import predict_single_image,evaluate_dataset
from deepfake_utils import FolderDeepfakeDataset,get_val_transform

def build_arg_parser():
    parser=argparse.ArgumentParser(
        description="Test FFT + ResNet18 deepfake detector."
    )
    parser.add_argument(
        "--model_path",
        required=True
    )
    parser.add_argument(
        "--image_path",
        default=None
    )
    parser.add_argument(
        "--data_root",
        default=None
    )
    parser.add_argument(
        "--split",
        default="valid",
        choices=["train","valid","test"]
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4
    )
    return parser


def load_model(model_path,device):
    model=FFTResNet18(
        pretrained=False
    ).to(device)
    checkpoint=torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(
        checkpoint["model_state"]
    )
    model.eval()
    print(f"Loaded model from {model_path}")
    return model


def main():
    args=build_arg_parser().parse_args()
    device=torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    model=load_model(
        args.model_path,
        device
    )
    if args.image_path is not None:
        predict_single_image(
            model,
            args.image_path,
            device
        )
    elif args.data_root is not None:
        base_dataset=FolderDeepfakeDataset(
            args.data_root,
            args.split,
            transform=get_val_transform()
        )
        fft_dataset=FFTDataset(
            base_dataset
        )
        loader=DataLoader(
            fft_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        evaluate_dataset(
            model,
            loader,
            device
        )
    else:
        print(
            "Provide either "
            "--image_path or --data_root"
        )
if __name__=="__main__":
    main()
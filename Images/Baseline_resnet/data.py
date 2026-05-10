from torch.utils.data import DataLoader
from deepfake_utils import (
    FolderDeepfakeDataset, CSVDeepfakeDataset,
    get_train_transform, get_val_transform,
    get_class_weights,
)
def build_loaders(args):
    if args.use_csv:
        assert args.train_csv and args.valid_csv and args.img_root, \
            "Pass --train_csv, --valid_csv, --img_root with --use_csv"
        train_ds = CSVDeepfakeDataset(args.train_csv, args.img_root, transform=get_train_transform())
        val_ds = CSVDeepfakeDataset(args.valid_csv, args.img_root, transform=get_val_transform())
    else:
        assert args.data_root, "Pass --data_root"
        train_ds = FolderDeepfakeDataset(args.data_root, "train", transform=get_train_transform())
        val_ds = FolderDeepfakeDataset(args.data_root, "valid", transform=get_val_transform())
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    return train_loader, val_loader, train_ds

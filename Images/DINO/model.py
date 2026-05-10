import os
import sys
import torch.nn as nn

class DINODeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        print("  Loading DINO ViT-B/16 (SSL pretrained on ImageNet)...")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        removed = project_root in sys.path
        if removed:
            sys.path.remove(project_root)
        try:
            import torch
            self.backbone = torch.hub.load(
                "facebookresearch/dino:main",
                "dino_vitb16",
                pretrained=True,
            )
        finally:
            if removed:
                sys.path.insert(0, project_root)
        self.backbone.head = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats).squeeze(1)
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("  Backbone frozen.")
    def unfreeze_last_n_blocks(self, n=4):
        blocks = self.backbone.blocks
        total = len(blocks)
        for blk in blocks[total - n:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.backbone.norm.parameters():
            p.requires_grad = True
        print(f"  Last {n} ViT blocks + norm unfrozen.")
        
    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
        print("  All parameters unfrozen.")

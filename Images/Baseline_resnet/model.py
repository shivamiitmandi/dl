import torch.nn as nn
from torchvision import models
class BaselineResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_feats = backbone.fc.in_features
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_feats, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.features(x)
        return self.head(x).squeeze(1)
    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False
    def unfreeze_last_n_layers(self, n=30):
        all_layers = list(self.features.children())
        for lyr in all_layers[-n:]:
            for p in lyr.parameters():
                p.requires_grad = True
    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

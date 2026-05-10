import torch.nn as nn
from torchvision import models

class FFTResNet18(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        weights=(
            models.ResNet18_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )
        backbone=models.resnet18(
            weights=weights
        )
        in_features=backbone.fc.in_features
        backbone.fc=nn.Sequential(
            nn.Linear(in_features,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256,1),
            nn.Sigmoid(),
        )
        self.net=backbone
    def forward(self,x):
        return self.net(x).squeeze(1)
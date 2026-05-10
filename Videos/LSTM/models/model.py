import torch
import torch.nn as nn

class DINOLSTMClassifier(nn.Module):
    def __init__(
        self,
        lstm_hidden=512,
        lstm_layers=1
    ):
        super().__init__()
        self.backbone=torch.hub.load(
            "facebookresearch/dino:main",
            "dino_vitb16",
            pretrained=True
        )
        self.backbone.head=nn.Identity()
        self.lstm=nn.LSTM(
            input_size=768,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.classifier=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(lstm_hidden,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,1),
        )
    def forward(self,x):
        B,T,C,H,W=x.shape
        frame_features=self.backbone(
            x.view(B*T,C,H,W)
        )
        frame_features=frame_features.view(B,T,-1)
        temporal_features,_=self.lstm(
            frame_features
        )
        video_representation=torch.mean(
            temporal_features,
            dim=1
        )
        output=self.classifier(
            video_representation
        )
        return output.squeeze(1)
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad=False
        for module in [self.lstm,self.classifier]:
            for param in module.parameters():
                param.requires_grad=True
        print("Backbone frozen.")
    def unfreeze_last_n_blocks(self,n=4):
        transformer_blocks=self.backbone.blocks
        for block in transformer_blocks[-n:]:
            for param in block.parameters():
                param.requires_grad=True
        for param in self.backbone.norm.parameters():
            param.requires_grad=True
        print(f"Unfroze last {n} transformer blocks.")
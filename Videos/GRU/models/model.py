import torch
import torch.nn as nn


class DINOGRUClassifier(nn.Module):
    def __init__(self, gru_hidden=512, gru_layers=1):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dino:main",
            "dino_vitb16",
            pretrained=True
        )
        self.backbone.head = nn.Identity()
        self.gru = nn.GRU(
            input_size=768,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True
        )
        self.attn = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(gru_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x, return_attention=False):
        B, T, C, H, W = x.shape
        frame_features = self.backbone(
            x.view(B * T, C, H, W)
        )
        frame_features = frame_features.view(B, T, -1)
        temporal_features, _ = self.gru(frame_features)
        attention_scores = self.attn(temporal_features)
        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )
        video_representation = torch.sum(
            attention_weights * temporal_features,
            dim=1
        )
        output = self.classifier(video_representation)
        if return_attention:
            return output.squeeze(1), attention_weights
        return output.squeeze(1)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for module in [self.gru,self.attn,self.classifier]:
            for param in module.parameters():
                param.requires_grad = True
        print("Backbone frozen.")

    def unfreeze_last_n_blocks(self, n=4):
        transformer_blocks = self.backbone.blocks
        for block in transformer_blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in self.backbone.norm.parameters():
            param.requires_grad = True
        print(f"Unfroze last {n} transformer blocks.")
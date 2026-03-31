"""
model.py — Dual-Branch Fake Image Detection Model
===================================================
Architecture:
  - Spatial Branch : DINO-pretrained ViT-B/16  → F_s ∈ R^768
  - Frequency Branch: ResNet-18 (on FFT spectrum) → F_f ∈ R^768
  - Cross-Attention Fusion → F_fusion ∈ R^768
  - Classification Head → p ∈ (0,1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Frequency Map Computation (FFT)
# ─────────────────────────────────────────────────────────────────────────────

class FFTLayer(nn.Module):
    """
    Converts a batch of RGB images to log-magnitude FFT spectrum.

    Steps:
      1. Convert RGB -> grayscale (weighted average)
      2. Compute 2-D FFT, shift zero-freq to centre
      3. Take log(1 + |F|) to compress dynamic range
      4. Replicate to 3 channels (ResNet expects 3-channel input)
      5. Normalise to [0, 1]
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W), values normalised with ImageNet stats
        # Convert to grayscale: ITU-R BT.601 luminance weights
        # We work in the normalised tensor space directly
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b  # (B, 1, H, W)
        gray = gray.squeeze(1)                          # (B, H, W)

        # 2-D FFT
        fft = torch.fft.fft2(gray)                     # complex (B, H, W)
        fft = torch.fft.fftshift(fft)                  # shift zero-freq to centre

        # Log magnitude spectrum
        magnitude = torch.abs(fft)                     # (B, H, W)
        log_mag = torch.log1p(magnitude)               # log(1 + |F|)

        # Normalise to [0, 1] per image
        B, H, W = log_mag.shape
        flat = log_mag.view(B, -1)
        min_v = flat.min(dim=1)[0].view(B, 1, 1)
        max_v = flat.max(dim=1)[0].view(B, 1, 1)
        log_mag = (log_mag - min_v) / (max_v - min_v + 1e-8)

        # Replicate to 3 channels so ResNet-18 can process it
        out = log_mag.unsqueeze(1).expand(-1, 3, -1, -1)  # (B, 3, H, W)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Frequency Branch (ResNet-18 on FFT map)
# ─────────────────────────────────────────────────────────────────────────────

class FrequencyBranch(nn.Module):
    """
    Processes the log-magnitude FFT spectrum through ResNet-18.
    Projects the 512-d output to 768-d to match ViT-B/16 dimension.
    """

    def __init__(self, out_dim: int = 768):
        super().__init__()
        self.fft = FFTLayer()

        # ResNet-18 pretrained on ImageNet
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        # Remove final FC layer; keep up to global average pooling
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 512, 1, 1)

        # Projection to match spatial branch dimension
        self.proj = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq_map = self.fft(x)                    # (B, 3, 224, 224)
        feat = self.encoder(freq_map)             # (B, 512, 1, 1)
        feat = feat.flatten(1)                    # (B, 512)
        feat = self.proj(feat)                    # (B, 768)
        return feat


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Spatial Branch (DINO ViT-B/16)
# ─────────────────────────────────────────────────────────────────────────────

class SpatialBranch(nn.Module):
    """
    DINO-pretrained ViT-B/16.
    Returns the [CLS] token as the global face representation (768-d).

    Weights are downloaded automatically from torch.hub on first use.
    They are cached in ~/.cache/torch/hub/
    """

    def __init__(self):
        super().__init__()
        # Load DINO ViT-B/16 pretrained weights
        self.vit = torch.hub.load(
            'facebookresearch/dino:main',
            'dino_vitb16',
            pretrained=True,
            verbose=False,
        )
        # Freeze all parameters initially; unfreezing is done in trainer
        for p in self.vit.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ViT forward pass returns CLS token features
        feat = self.vit(x)   # (B, 768)
        return feat

    def unfreeze_last_n_blocks(self, n: int):
        """Unfreeze last n transformer blocks for fine-tuning."""
        # Freeze everything first
        for p in self.vit.parameters():
            p.requires_grad = False
        # Unfreeze last n blocks
        blocks = list(self.vit.blocks)
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        # Always unfreeze norm layer
        for p in self.vit.norm.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze entire ViT backbone."""
        for p in self.vit.parameters():
            p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Cross-Attention Fusion Module
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Fuses spatial (F_s) and frequency (F_f) features via cross-attention.

    F_s acts as Query: "where is there suspicious structure?"
    F_f acts as Key/Value: "here are the spectral anomalies"

    Output: F_fusion = LayerNorm(F_s + CrossAttn(F_s, F_f))

    For two flat feature vectors, we treat each as a sequence of length 1.
    The cross-attention then degenerates to a gated fusion:
        A = softmax(Q * K^T / sqrt(d))  → scalar weight
        output = A * V
    We expand to multiple feature dimensions for richer interaction.
    """

    def __init__(self, dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_o = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        # Feed-forward after attention (for richer representation)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, F_s: torch.Tensor, F_f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            F_s: spatial features  (B, dim)
            F_f: frequency features (B, dim)
        Returns:
            fused: (B, dim)
        """
        B = F_s.shape[0]

        # Expand to sequence length 1: (B, 1, dim)
        F_s_seq = F_s.unsqueeze(1)
        F_f_seq = F_f.unsqueeze(1)

        # Project to Q, K, V
        Q = self.W_q(F_s_seq)  # (B, 1, dim)
        K = self.W_k(F_f_seq)  # (B, 1, dim)
        V = self.W_v(F_f_seq)  # (B, 1, dim)

        # Reshape for multi-head attention
        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, d_k)
        K = K.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, 1, 1)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Attended value
        out = torch.matmul(attn, V)                  # (B, H, 1, d_k)
        out = out.transpose(1, 2).contiguous()       # (B, 1, H, d_k)
        out = out.view(B, 1, self.dim)               # (B, 1, dim)
        out = self.W_o(out).squeeze(1)               # (B, dim)

        # Residual + LayerNorm (cross-attention residual)
        fused = self.norm(F_s + out)

        # Feed-forward + residual
        fused = self.norm2(fused + self.ffn(fused))

        return fused


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Classification Head
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Two-layer MLP classifier.
    F_fusion (768) → 512 → 1 → sigmoid → probability of fake
    """

    def __init__(self, in_dim: int = 768, hidden_dim: int = 512, dropout: float = 0.4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # raw logit (B, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Complete Model
# ─────────────────────────────────────────────────────────────────────────────

class FakeImageDetector(nn.Module):
    """
    Full dual-branch fake image detection model.

    Usage:
        model = FakeImageDetector()
        logit = model(x)           # x: (B, 3, 224, 224), normalised
        prob  = torch.sigmoid(logit)
        pred  = (prob > 0.5).long()  # 1 = fake, 0 = real
    """

    def __init__(self):
        super().__init__()
        self.spatial_branch  = SpatialBranch()
        self.freq_branch     = FrequencyBranch(out_dim=768)
        self.fusion          = CrossAttentionFusion(dim=768, num_heads=8)
        self.classifier      = ClassificationHead(in_dim=768, hidden_dim=512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        F_s     = self.spatial_branch(x)           # (B, 768)
        F_f     = self.freq_branch(x)              # (B, 768)
        F_fused = self.fusion(F_s, F_f)            # (B, 768)
        logit   = self.classifier(F_fused)         # (B, 1)
        return logit.squeeze(1)                    # (B,)

    def set_phase(self, phase: int):
        """
        Switch training phase:
          phase=1: freeze ViT entirely
          phase=2: unfreeze last 4 ViT blocks
          phase=3: unfreeze all ViT parameters
        """
        if phase == 1:
            print("[Phase 1] ViT fully frozen. Training: freq_branch + fusion + classifier")
            # ViT already frozen by default
        elif phase == 2:
            print("[Phase 2] Unfreezing last 4 ViT blocks")
            self.spatial_branch.unfreeze_last_n_blocks(4)
        elif phase == 3:
            print("[Phase 3] Full fine-tuning: all parameters unfrozen")
            self.spatial_branch.unfreeze_all()
        else:
            raise ValueError(f"Unknown phase: {phase}")

    def get_param_groups(self, lr_vit: float, lr_other: float):
        """
        Return parameter groups with differential learning rates.
        ViT parameters use a lower LR than newly initialised modules.
        """
        vit_params    = list(self.spatial_branch.parameters())
        other_params  = (list(self.freq_branch.parameters()) +
                         list(self.fusion.parameters()) +
                         list(self.classifier.parameters()))
        vit_ids    = {id(p) for p in vit_params}
        trainable_vit   = [p for p in vit_params   if p.requires_grad]
        trainable_other = [p for p in other_params if p.requires_grad]
        return [
            {'params': trainable_vit,   'lr': lr_vit},
            {'params': trainable_other, 'lr': lr_other},
        ]


if __name__ == '__main__':
    # Smoke test
    print("Building model...")
    model = FakeImageDetector()
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    print(f"Output shape: {out.shape}")   # Expected: torch.Size([2])
    print(f"Output values: {out}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params/1e6:.1f}M | Trainable: {trainable/1e6:.1f}M")

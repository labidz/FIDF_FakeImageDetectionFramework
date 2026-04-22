# models/FIDF.py


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from modules import LearnableDCT, SRMConv, DGCA, HyperGraphConv


class PretrainedRGBTrunk(nn.Module):
    """EfficientNet-B0 pretrained backbone for RGB stream."""
    
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Truncate at stage 5 (112 channels at H/16)
        self.features = nn.Sequential(*list(backbone.features.children())[:6])
        self.adapter = nn.Sequential(
            nn.Conv2d(112, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        return self.adapter(f)
    
    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False
            
    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and GELU."""
    
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Trunk(nn.Module):
    """Lightweight from-scratch trunk for DCT and noise streams."""
    
    def __init__(self, in_ch: int, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_ch, 64),
            ConvBlock(64, 64, stride=2),
            ConvBlock(64, 128),
            ConvBlock(128, 128, stride=2),
            ConvBlock(128, embed_dim),
            ConvBlock(embed_dim, embed_dim, stride=2),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyDecoder(nn.Module):
    """Decoder for masked reconstruction auxiliary task."""
    
    def __init__(self, dim: int = 256):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(dim, 128, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, 3, padding=1),
        )
        
    def forward(self, tokens: torch.Tensor, hw: int) -> torch.Tensor:
        B, N, D = tokens.shape
        x = tokens.transpose(1, 2).view(B, D, hw, hw)
        return self.up(x)


class TFHDN(nn.Module):
    """
    Tri-Forensic Hypergraph Discrepancy Network.
    
    Three complementary streams:
    1. RGB stream (pretrained EfficientNet-B0) - semantic features
    2. DCT stream (learnable DCT basis) - compression artifacts
    3. Noise stream (SRM-initialized) - local noise residuals
    
    Fusion via Discrepancy-Gated Cross-Attention (DGCA) and
    k-NN Hypergraph Convolution.
    """
    
    def __init__(self, dim: int = 256, heads: int = 4, k: int = 8,
                 num_classes: int = 2, pretrained_rgb: bool = True):
        super().__init__()
        
        # Forensic front-ends
        self.dct = LearnableDCT(block=8, channels=3)
        self.srm = SRMConv()
        
        # Trunks
        if pretrained_rgb:
            self.trunk_rgb = PretrainedRGBTrunk(embed_dim=dim)
        else:
            self.trunk_rgb = Trunk(3, dim)
            
        self.trunk_dct = Trunk(192, dim)
        self.trunk_noise = Trunk(3, dim)
        
        # Discrepancy-Gated Cross-Attention
        self.dgca_rgb = DGCA(dim, heads)
        self.dgca_dct = DGCA(dim, heads)
        self.dgca_noise = DGCA(dim, heads)
        
        # Hypergraph reasoning
        self.hgc1 = HyperGraphConv(dim, k=k)
        self.hgc2 = HyperGraphConv(dim, k=k)
        
        # Classification head
        self.cls = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(dim // 2, num_classes),
        )
        
        # Projection head for SupCon
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 128)
        )
        
        # Reconstruction decoder
        self.dec = TinyDecoder(dim)
        
        self._has_pretrained = pretrained_rgb
        
    def forward(self, x: torch.Tensor, return_aux: bool = False):
        # Pad for DCT
        x_dct = self._pad_to(x, 8)
        
        # Extract features from three streams
        f_rgb = self.trunk_rgb(x)
        f_dct_in = self.dct(x_dct)
        f_dct = self.trunk_dct(F.interpolate(f_dct_in, size=x.shape[-2:],
                                              mode='bilinear', align_corners=False))
        f_noise = self.trunk_noise(self.srm(x))
        
        # Harmonize spatial resolution
        target_hw = f_rgb.shape[-2:]
        if f_dct.shape[-2:] != target_hw:
            f_dct = F.adaptive_avg_pool2d(f_dct, target_hw)
        if f_noise.shape[-2:] != target_hw:
            f_noise = F.adaptive_avg_pool2d(f_noise, target_hw)
        
        # Tokenize
        B, D, H, W = f_rgb.shape
        t_rgb = f_rgb.flatten(2).transpose(1, 2)
        t_dct = f_dct.flatten(2).transpose(1, 2)
        t_noise = f_noise.flatten(2).transpose(1, 2)
        
        # DGCA: each stream attends to mean of other two
        t_rgb = self.dgca_rgb(t_rgb, (t_dct + t_noise) / 2)
        t_dct = self.dgca_dct(t_dct, (t_rgb + t_noise) / 2)
        t_noise = self.dgca_noise(t_noise, (t_rgb + t_dct) / 2)
        
        # Fuse tokens
        tokens = (t_rgb + t_dct + t_noise) / 3
        
        # Hypergraph reasoning
        tokens = self.hgc1(tokens)
        tokens = self.hgc2(tokens)
        
        # Pooled embedding
        pooled = tokens.mean(dim=1)
        logits = self.cls(pooled)
        
        if return_aux:
            recon = self.dec(tokens, hw=H)
            embed = F.normalize(self.proj(pooled), dim=-1)
            return logits, recon, embed
        
        return logits
    
    @staticmethod
    def _pad_to(x: torch.Tensor, mult: int) -> torch.Tensor:
        H, W = x.shape[-2:]
        pH = (mult - H % mult) % mult
        pW = (mult - W % mult) % mult
        if pH or pW:
            x = F.pad(x, (0, pW, 0, pH))
        return x
    
    def ortho_loss(self) -> torch.Tensor:
        return self.dct.ortho_reg()

# models/baselines/recomp_cnn.py
"""
Implementation of: "Deep Learning based Image Forgery Detection using 
Recompression Error Analysis"
Ali et al., 2022

Key aspects from paper:
- Input: Difference between original and recompressed image
- Architecture: 4 conv blocks with max pooling
- Recompression quality: JPEG quality 85 (as per paper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import io
from PIL import Image


class RecompressionCNN(nn.Module):
    """
    CNN trained on recompression difference features.
    
    Paper: Ali et al. (2022) - Deep Learning based Image Forgery Detection 
           using Recompression Error Analysis
    """
    
    def __init__(self, num_classes: int = 2, base_channels: int = 32):
        super().__init__()
        
        # Feature extraction blocks (4 conv blocks with max pooling)
        self.features = nn.Sequential(
            self._conv_block(3, base_channels),      # 32 channels
            nn.MaxPool2d(2),
            self._conv_block(base_channels, base_channels * 2),  # 64 channels
            nn.MaxPool2d(2),
            self._conv_block(base_channels * 2, base_channels * 4),  # 128 channels
            nn.MaxPool2d(2),
            self._conv_block(base_channels * 4, base_channels * 4),  # 128 channels
        )
        
        # Global pooling and classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # Recompression quality (fixed at 85 as per paper)
        self.recomp_quality = 85
        
    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        """Conv2d + BatchNorm + ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def compute_recompression_difference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute difference between original and recompressed image.
        
        Note: During training, we apply this as a differentiable approximation.
        For exact reproduction, precompute differences offline.
        """
        # Approximate JPEG compression via quantization simulation
        # This is a differentiable approximation of the process
        quality = self.recomp_quality / 100.0
        
        # Simulate quantization error
        quantized = torch.round(x * quality * 255) / (quality * 255 + 1e-8)
        difference = torch.abs(x - quantized)
        
        return difference
    
    def forward(self, x: torch.Tensor, use_recomp_diff: bool = True) -> torch.Tensor:
        if use_recomp_diff:
            x = self.compute_recompression_difference(x)
        
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
    
    @staticmethod
    def precompute_recompression(image_path: str, quality: int = 85) -> Image.Image:
        """
        Precompute exact recompression difference for a single image.
        Use this for offline dataset preparation.
        """
        from PIL import Image, ImageChops
        
        img = Image.open(image_path).convert('RGB')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        recompressed = Image.open(buffer)
        
        # Compute absolute difference
        diff = ImageChops.difference(img, recompressed)
        return diff

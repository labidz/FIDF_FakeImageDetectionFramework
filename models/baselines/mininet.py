# models/baselines/mininet.py
"""
Implementation of: "MiniNet: A Concise CNN for Image Forgery Detection"
Tyagi & Yadav, 2023

Key aspects from paper:
- Lightweight architecture: 3 conv layers
- Kernel sizes: 3x3 throughout
- Channels: 16 -> 32 -> 64
- Dropout: 0.3 after FC layers
"""

import torch
import torch.nn as nn


class MiniNet(nn.Module):
    """
    Concise CNN architecture for efficient forgery detection.
    
    Paper: Tyagi & Yadav (2023) - MiniNet: A Concise CNN for Image Forgery Detection
    Parameters: ~150K (lightweight)
    """
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        
        # Feature extraction (3 conv layers as per paper)
        self.features = nn.Sequential(
            # Block 1: 3 -> 16 channels
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/2, W/2
            
            # Block 2: 16 -> 32 channels
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/4, W/4
            
            # Block 3: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/8, W/8
        )
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier (paper: 64 -> 32 -> num_classes)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Kaiming initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before final classification."""
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        # Get penultimate layer features (before final linear)
        for layer in self.classifier[:-1]:
            x = layer(x)
        return x

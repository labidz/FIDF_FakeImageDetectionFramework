# models/baselines/resnet_tl.py
"""
Implementation of: "Image Forgery Detection using Transfer Learning"
Qazi et al., 2022

Key aspects from paper:
- ResNet50v2 pretrained on ImageNet
- Fine-tune only the final residual block (layer4)
- Classifier: 2048 -> 256 -> 2 with dropout 0.3
- Input size: 224x224 (we adapt to config)
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50v2TL(nn.Module):
    """
    Transfer learning baseline using ResNet50v2.
    
    Paper: Qazi et al. (2022) - Image Forgery Detection using Transfer Learning
    Architecture: ResNet50v2 (pretrained) + fine-tuned layer4 + custom head
    """
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        
        # Load pretrained ResNet50v2 (IMAGENET1K_V2)
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Freeze all layers initially
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze only the final residual block (layer4) as per paper
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
            
        # Replace classifier head (paper: 2048 -> 256 -> 2)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self._init_head()
        
    def _init_head(self):
        """Initialize classifier head with Xavier uniform."""
        for module in self.backbone.fc:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before final classification."""
        # Forward through all but the final fc layer
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Get features from penultimate layer
        for layer in self.backbone.fc[:-1]:
            x = layer(x)
        return x

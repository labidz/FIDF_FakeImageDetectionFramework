# modules/srm.py
"""
Spatial Rich Model (SRM) convolution module for noise residual extraction.
"""

import numpy as np
import torch
import torch.nn as nn


# SRM filter kernels (3 high-pass filters from steganalysis)
SRM_KERNELS = np.array([
    # 5x5 filter 1: edge detection
    [[0, 0, 0, 0, 0],
     [0, -1, 2, -1, 0],
     [0, 2, -4, 2, 0],
     [0, -1, 2, -1, 0],
     [0, 0, 0, 0, 0]],
    
    # 5x5 filter 2: spot detection
    [[-1, 2, -2, 2, -1],
     [2, -6, 8, -6, 2],
     [-2, 8, -12, 8, -2],
     [2, -6, 8, -6, 2],
     [-1, 2, -2, 2, -1]],
    
    # 5x5 filter 3: horizontal/vertical edges
    [[0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 1, -2, 1, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]],
], dtype=np.float32)

# Normalization factors
SRM_NORM = np.array([4.0, 12.0, 2.0]).reshape(3, 1, 1)
SRM_KERNELS = SRM_KERNELS / SRM_NORM


class SRMConv(nn.Module):
    """
    Learnable noise residual extractor initialized from SRM filters.
    Outputs 3-channel noise residual maps.
    """
    
    def __init__(self, num_filters: int = 3):
        super().__init__()
        
        # Initialize with SRM kernels (3 filters x 3 input channels)
        kernels = np.tile(SRM_KERNELS[:, None, :, :], (1, 3, 1, 1))
        
        self.conv = nn.Conv2d(3, num_filters, kernel_size=5, padding=2, bias=False)
        
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor(kernels))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

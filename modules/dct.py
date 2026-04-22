# modules/dct.py
"""
Learnable Discrete Cosine Transform (DCT) module.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableDCT(nn.Module):
    """
    8x8 block DCT with learnable basis Φ ∈ R^{64×64}.
    Initialized to true DCT-II basis with orthogonality regularization.
    """
    
    def __init__(self, block: int = 8, channels: int = 3):
        super().__init__()
        self.block = block
        self.channels = channels
        
        # Build true 2D DCT-II basis as initialization
        n = block
        x = np.arange(n)
        k = np.arange(n).reshape(-1, 1)
        D = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
        D[0, :] *= 1 / np.sqrt(2)
        D = D * np.sqrt(2 / n)  # 1D DCT
        
        # Kronecker product for 2D DCT basis (64x64)
        D2 = np.kron(D, D)
        
        self.basis = nn.Parameter(torch.tensor(D2, dtype=torch.float32))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable DCT to input tensor.
        
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            DCT coefficients (B, C*64, H/8, W/8)
        """
        B, C, H, W = x.shape
        b = self.block
        
        # Extract 8x8 blocks
        x = x.unfold(2, b, b).unfold(3, b, b)  # (B, C, H/b, W/b, b, b)
        Hb, Wb = x.shape[2], x.shape[3]
        x = x.contiguous().view(B, C, Hb, Wb, b * b)  # (B, C, Hb, Wb, 64)
        
        # Apply basis transformation
        y = torch.einsum('bchwn,kn->bchwk', x, self.basis)
        
        # Reshape to spatial grid
        y = y.permute(0, 1, 4, 2, 3).contiguous().view(B, C * b * b, Hb, Wb)
        
        return y
    
    def ortho_reg(self) -> torch.Tensor:
        """Orthogonality regularization loss."""
        I = torch.eye(self.basis.shape[0], device=self.basis.device)
        return ((self.basis @ self.basis.t() - I) ** 2).mean()

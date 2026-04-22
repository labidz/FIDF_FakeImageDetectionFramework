# modules/attention.py
"""
Discrepancy-Gated Cross-Attention (DGCA) module.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DGCA(nn.Module):
    """
    Discrepancy-Gated Cross-Attention.
    
    Each stream attends to the mean of other streams with a sigmoid gate
    trained on the magnitude of disagreement |q - k_other|.
    """
    
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        
        self.heads = heads
        self.dim_head = dim // heads
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
        # Discrepancy gate
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, q_tok: torch.Tensor, ctx_tok: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_tok: Query tokens from current stream (B, N, D)
            ctx_tok: Context tokens from other streams (B, N, D)
        Returns:
            Updated tokens after gated cross-attention (B, N, D)
        """
        B, N, D = q_tok.shape
        
        # Compute Q, K, V
        qkv_q = self.qkv(q_tok)
        qkv_ctx = self.qkv(ctx_tok)
        
        q = qkv_q[:, :, :D]
        k = qkv_ctx[:, :, D:2*D]
        v = qkv_ctx[:, :, 2*D:]
        
        # Reshape for multi-head attention
        q = q.view(B, N, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(B, N, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(B, N, self.heads, self.dim_head).transpose(1, 2)
        
        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dim_head)
        att = att.softmax(dim=-1)
        
        # Apply attention
        out = (att @ v).transpose(1, 2).contiguous().view(B, N, D)
        out = self.proj(out)
        
        # Discrepancy gate: emphasize where streams disagree most
        disagree = (q_tok - ctx_tok).abs()
        g = self.gate(disagree)
        
        # Residual connection with gating
        return self.norm(q_tok + g * out)

# modules/hypergraph.py
"""
k-NN Hypergraph Convolution module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperGraphConv(nn.Module):
    """
    Hypergraph Convolution via k-NN incidence matrix.
    
    Implements: D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X Θ
    Simplified via k-NN message passing.
    """
    
    def __init__(self, dim: int, k: int = 8):
        super().__init__()
        self.k = k
        self.theta = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Node features (B, N, D)
        Returns:
            Updated node features after hypergraph convolution (B, N, D)
        """
        B, N, D = X.shape
        
        # Compute cosine similarity between all nodes
        Xn = F.normalize(X, dim=-1)
        sim = Xn @ Xn.transpose(-2, -1)  # (B, N, N)
        
        # Get k nearest neighbors for each node
        topk = sim.topk(self.k, dim=-1).indices  # (B, N, k)
        
        # Gather features of k-NN
        idx = topk.unsqueeze(-1).expand(-1, -1, -1, D)  # (B, N, k, D)
        Xexp = X.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, D)
        gathered = torch.gather(Xexp, 2, idx)  # (B, N, k, D)
        
        # Message passing: average over hyperedge (k-NN set)
        msg = gathered.mean(dim=2)  # (B, N, D)
        
        # Transform and residual connection
        Y = self.theta(msg)
        
        return self.norm(X + Y)

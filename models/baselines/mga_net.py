# models/baselines/mga_net.py
"""
Implementation of: "MGA-Net: Multi-Graph Attention Network for Image Forgery Detection"
Chen et al., 2025

Key aspects from paper:
- Multi-scale feature extraction
- K parallel graph attention streams
- Learnable adjacency matrices per graph
- Fusion via attention-weighted aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) as per Veličković et al. 2018.
    Used as building block for MGA-Net.
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # Linear transformations
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(alpha)
        
    def forward(self, h: torch.Tensor, 
                adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            h: Node features (B, N, D_in)
            adj: Optional adjacency matrix (B, N, N). If None, compute fully connected.
        Returns:
            Updated node features (B, N, D_out)
        """
        B, N, _ = h.shape
        
        # Linear transformation
        Wh = self.W(h)  # (B, N, D_out)
        
        # Compute attention coefficients
        # Concatenate all pairs: (B, N, N, 2*D_out)
        Wh_i = Wh.unsqueeze(2).expand(B, N, N, -1)
        Wh_j = Wh.unsqueeze(1).expand(B, N, N, -1)
        concat = torch.cat([Wh_i, Wh_j], dim=-1)
        
        # Attention scores
        e = self.leaky_relu(self.a(concat).squeeze(-1))  # (B, N, N)
        
        # Apply adjacency mask if provided
        if adj is not None:
            e = e.masked_fill(adj == 0, float('-inf'))
        
        # Softmax over neighbors
        attention = F.softmax(e, dim=-1)
        attention = self.dropout_layer(attention)
        
        # Weighted aggregation
        h_prime = torch.bmm(attention, Wh)  # (B, N, D_out)
        
        return h_prime


class MultiScaleFeatureExtractor(nn.Module):
    """
    Multi-scale CNN backbone for MGA-Net.
    Extracts features at different resolutions.
    """
    
    def __init__(self, base_channels: int = 32):
        super().__init__()
        
        # Shared early layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Scale 1: 1x resolution
        self.scale1 = nn.Sequential(
            self._conv_block(base_channels, base_channels * 2),
            nn.MaxPool2d(2)
        )
        
        # Scale 2: 1/2 resolution
        self.scale2 = nn.Sequential(
            self._conv_block(base_channels * 2, base_channels * 4),
            nn.MaxPool2d(2)
        )
        
        # Scale 3: 1/4 resolution
        self.scale3 = nn.Sequential(
            self._conv_block(base_channels * 4, base_channels * 4),
        )
        
    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        
        f1 = self.scale1(x)  # H/2, W/2
        f2 = self.scale2(f1)  # H/4, W/4
        f3 = self.scale3(f2)  # H/4, W/4
        
        return f1, f2, f3


class MGANet(nn.Module):
    """
    Multi-Graph Attention Network for image forgery detection.
    
    Paper: Chen et al. (2025) - MGA-Net: Multi-Graph Attention Network 
           for Image Forgery Detection
    
    Key components:
    - Multi-scale feature extraction
    - K parallel graph attention streams
    - Cross-scale feature fusion
    """
    
    def __init__(self, num_classes: int = 2, num_graphs: int = 4,
                 hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        
        self.num_graphs = num_graphs
        self.hidden_dim = hidden_dim
        
        # Multi-scale backbone
        self.backbone = MultiScaleFeatureExtractor(base_channels=32)
        
        # Feature dimensions at each scale
        self.scale_dims = [64, 128, 128]  # After backbone
        
        # Project each scale to hidden dimension
        self.scale_projections = nn.ModuleList([
            nn.Conv2d(dim, hidden_dim, 1) for dim in self.scale_dims
        ])
        
        # K parallel graph attention streams
        self.graph_streams = nn.ModuleList([
            self._build_graph_stream(hidden_dim) for _ in range(num_graphs)
        ])
        
        # Learnable adjacency matrices per graph
        self.adj_matrices = nn.ParameterList([
            nn.Parameter(torch.randn(100, 100) * 0.1)  # Max 100 nodes
            for _ in range(num_graphs)
        ])
        
        # Cross-graph fusion attention
        self.fusion_attention = nn.Sequential(
            nn.Linear(hidden_dim * num_graphs, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_graphs),
            nn.Softmax(dim=-1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_graphs, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def _build_graph_stream(self, dim: int) -> nn.Module:
        """Build a single graph attention stream."""
        return nn.Sequential(
            GraphAttentionLayer(dim, dim, dropout=0.1),
            nn.LayerNorm(dim),
            GraphAttentionLayer(dim, dim, dropout=0.1),
            nn.LayerNorm(dim),
        )
    
    def _features_to_graph(self, features: torch.Tensor, 
                           num_nodes: int = 36) -> torch.Tensor:
        """
        Convert spatial features to graph nodes via adaptive pooling.
        
        Args:
            features: (B, C, H, W)
            num_nodes: Number of graph nodes (k)
        Returns:
            Node features: (B, num_nodes, C)
        """
        B, C, H, W = features.shape
        
        # Adaptive pooling to sqrt(num_nodes) x sqrt(num_nodes) grid
        grid_size = int(math.sqrt(num_nodes))
        pooled = F.adaptive_avg_pool2d(features, (grid_size, grid_size))
        
        # Flatten spatial dimensions to nodes
        nodes = pooled.flatten(2).transpose(1, 2)  # (B, num_nodes, C)
        
        return nodes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-scale feature extraction
        f1, f2, f3 = self.backbone(x)
        features = [f1, f2, f3]
        
        # Process each scale
        all_graph_outputs = []
        
        for scale_idx, feat in enumerate(features):
            # Project to hidden dimension
            proj = self.scale_projections[scale_idx](feat)
            
            # Convert to graph nodes
            nodes = self._features_to_graph(proj, num_nodes=36)
            
            # Process through K parallel graph streams
            stream_outputs = []
            for g in range(self.num_graphs):
                # Get learned adjacency for this graph
                adj = self._get_adjacency(nodes.size(1), g)
                
                # Graph attention
                out = self.graph_streams[g](nodes)
                stream_outputs.append(out.mean(dim=1))  # Global pooling
            
            # Concatenate all graph outputs for this scale
            scale_out = torch.cat(stream_outputs, dim=-1)  # (B, hidden_dim * K)
            all_graph_outputs.append(scale_out)
        
        # Fuse across scales (weighted sum)
        fused = torch.stack(all_graph_outputs, dim=0).mean(dim=0)
        
        # Cross-graph attention for final fusion
        att_weights = self.fusion_attention(fused)  # (B, K)
        
        # Weighted combination
        combined = fused.view(fused.size(0), self.num_graphs, -1)
        weighted = (combined * att_weights.unsqueeze(-1)).sum(dim=1)
        
        # Classification
        logits = self.classifier(weighted)
        
        return logits
    
    def _get_adjacency(self, num_nodes: int, graph_idx: int) -> torch.Tensor:
        """Get learned adjacency matrix for a specific graph."""
        adj = self.adj_matrices[graph_idx][:num_nodes, :num_nodes]
        # Apply sigmoid for [0,1] values
        adj = torch.sigmoid(adj)
        # Make symmetric (undirected graph)
        adj = (adj + adj.t()) / 2
        return adj

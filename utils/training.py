# utils/training.py
"""
Training utilities and classes.
"""

import os
import time
import math
from typing import Dict, Optional, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from .metrics import MetricsTracker


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    
    Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Normalized feature embeddings (B, D)
            labels: Class labels (B,)
        """
        features = features.float()
        B = features.size(0)
        
        # Compute similarity matrix
        sim = features @ features.t() / self.temperature
        
        # Mask out self-contrast
        neg_inf = torch.finfo(sim.dtype).min
        sim.fill_diagonal_(neg_inf)
        
        # Create positive mask
        mask = (labels.view(-1, 1) == labels.view(1, -1)).float()
        mask.fill_diagonal_(0)
        
        # Compute loss
        pos_count = mask.sum(1)
        if (pos_count > 0).sum() == 0:
            return sim.new_zeros(())
            
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        pos = (mask * log_prob).sum(1) / pos_count.clamp(min=1)
        pos = pos[pos_count > 0]
        
        return -pos.mean()


class Trainer:
    """
    Generic trainer for forgery detection models.
    """
    
    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, optimizer: torch.optim.Optimizer,
                 scheduler: Optional[Any] = None, device: str = 'cuda',
                 use_amp: bool = True, grad_accum_steps: int = 1):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp and device == 'cuda'
        self.grad_accum_steps = grad_accum_steps
        
        self.scaler = GradScaler(enabled=self.use_amp)
        self.metrics_tracker = MetricsTracker()
        
    def train_epoch(self, criterion: nn.Module,
                    aux_criterion: Optional[nn.Module] = None,
                    supcon_loss: Optional[SupConLoss] = None,
                    aux_weight: float = 0.1,
                    supcon_weight: float = 0.2) -> Dict[str, float]:
        
        self.model.train()
        self.metrics_tracker.reset()
        
        total_loss = 0.0
        self.optimizer.zero_grad()
        
        for step, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            with autocast(enabled=self.use_amp):
                # Forward pass
                if hasattr(self.model, 'forward') and 'return_aux' in self.model.forward.__code__.co_varnames:
                    logits, aux, features = self.model(images, return_aux=True)
                else:
                    logits = self.model(images)
                    aux = None
                    features = None
                
                # Main classification loss
                loss = criterion(logits, labels)
                
                # Auxiliary losses
                if aux is not None and aux_criterion is not None:
                    loss = loss + aux_weight * aux_criterion(aux, images)
                    
                if features is not None and supcon_loss is not None:
                    loss = loss + supcon_weight * supcon_loss(features, labels)
                
                loss = loss / self.grad_accum_steps
            
            # Backward
            self.scaler.scale(loss).backward()
            
            if (step + 

# train.py
"""
Main training script for TF-HDN and baseline models.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from models import (
    TFHDN,
    ResNet50v2TL,
    RecompressionCNN,
    MiniNet,
    MGANet,
    ELACNNXGB
)
from data import ForgeryDataset, get_transforms
from utils import (
    Trainer, 
    Evaluator, 
    set_seed, 
    load_config,
    get_optimizer,
    get_scheduler
)


MODEL_REGISTRY = {
    'tf_hdn': TFHDN,
    'resnet_tl': ResNet50v2TL,
    'recomp_cnn': RecompressionCNN,
    'mininet': MiniNet,
    'mga_net': MGANet,
    'ela_cnn_xgb': ELACNNXGB
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train forgery detection models')
    
    # Model selection
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model architecture to train')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    
    # Data
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., CASIA_v2)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory for datasets')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--wandb', action='store_true',
                        help='Log to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='tf-hdn')
    
    # Model-specific
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights (if available)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume from checkpoint')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    if args.wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
    
    # Load config
    config = load_config(args.config) if os.path.exists(args.config) else {}
    config.update(vars(args))
    
    # Create model
    model_class = MODEL_REGISTRY[args.model]
    model = model_class(**(config.get('model_kwargs', {})))
    model = model.to(device)
    
    # Data
    train_transform, val_transform = get_transforms(config.get('img_size', 224))
    
    train_dataset = ForgeryDataset(
        root=os.path.join(args.data_dir, args.dataset),
        split='train',
        transform=train_transform
    )
    
    val_dataset = ForgeryDataset(
        root=os.path.join(args.data_dir, args.dataset),
        split='val',
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Optimizer & Scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, len(train_loader))
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optim

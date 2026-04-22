# data/transforms.py
"""
Data augmentation and preprocessing transforms.
"""

import io
import random
from typing import Tuple, Optional

import torch
from torchvision import transforms
from PIL import Image, ImageChops


class RandomJPEG:
    """
    Random JPEG compression augmentation.
    Breaks compression-shortcut learning.
    """
    
    def __init__(self, p: float = 0.5, qlow: int = 50, qhigh: int = 95):
        self.p = p
        self.qlow = qlow
        self.qhigh = qhigh
        
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
            
        quality = random.randint(self.qlow, self.qhigh)
        buffer = io.BytesIO()
        img.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        
        return Image.open(buffer).convert('RGB')


class ELATransform:
    """
    Error Level Analysis (ELA) preprocessing.
    """
    
    def __init__(self, quality: int = 90):
        self.quality = quality
        
    def __call__(self, img: Image.Image) -> Image.Image:
        # Recompress
        buffer = io.BytesIO()
        img.save(buffer, 'JPEG', quality=self.quality)
        buffer.seek(0)
        recompressed = Image.open(buffer)
        
        # Compute difference
        diff = ImageChops.difference(img, recompressed)
        
        # Amplify
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema]) or 1
        scale = 255.0 / max_diff
        
        return diff.point(lambda p: int(p * scale))


def get_transforms(img_size: int = 224, augment: bool = True,
                   normalize: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get training and validation transforms.
    
    Args:
        img_size: Target image size
        augment: Whether to apply data augmentation
        normalize: Whether to apply ImageNet normalization
    
    Returns:
        train_transform, val_transform
    """
    
    # ImageNet statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Validation transform
    val_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]
    
    if normalize:
        val_transforms.append(transforms.Normalize(mean, std))
    
    val_transform = transforms.Compose(val_transforms)
    
    # Training transform
    if augment:
        train_transforms = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            RandomJPEG(p=0.5, qlow=50, qhigh=95),
            transforms.ToTensor(),
        ]
    else:
        train_transforms = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    
    if normalize:
        train_transforms.append(transforms.Normalize(mean, std))
    
    train_transform = transforms.Compose(train_transforms)
    
    return train_transform, val_transform

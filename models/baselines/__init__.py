# models/baselines/__init__.py
"""
Baseline model implementations for forgery detection.
"""

from .resnet_tl import ResNet50v2TL
from .recomp_cnn import RecompressionCNN
from .mininet import MiniNet
from .mga_net import MGANet
from .ela_cnn_xgb import ELACNNXGB

__all__ = [
    'ResNet50v2TL',
    'RecompressionCNN',
    'MiniNet',
    'MGANet',
    'ELACNNXGB'
]

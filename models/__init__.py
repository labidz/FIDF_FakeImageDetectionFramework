# models/__init__.py
"""
Model registry for forgery detection models.
"""

from .tf_hdn import TFHDN
from .baselines import (
    ResNet50v2TL,
    RecompressionCNN,
    MiniNet,
    MGANet,
    ELACNNXGB
)

__all__ = [
    'TFHDN',
    'ResNet50v2TL',
    'RecompressionCNN', 
    'MiniNet',
    'MGANet',
    'ELACNNXGB'
]

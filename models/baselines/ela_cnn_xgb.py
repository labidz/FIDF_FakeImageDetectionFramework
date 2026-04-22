# models/baselines/ela_cnn_xgb.py
"""
Implementation of: "Hybrid Deep Learning Framework for Image Forgery Detection 
using Error Level Analysis"
Kaur et al., 2024

Key aspects from paper:
- ELA preprocessing (quality=90)
- CNN feature extractor (4 conv blocks)
- XGBoost classifier (fitted on extracted features)
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageChops
import io
from typing import Optional, Tuple
import joblib
import xgboost as xgb


class ELAExtractor:
    """
    Error Level Analysis (ELA) preprocessing.
    Paper specifies JPEG quality = 90.
    """
    
    def __init__(self, quality: int = 90):
        self.quality = quality
        
    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Compute ELA difference map.
        
        Args:
            image: PIL Image in RGB mode
        Returns:
            ELA difference map as PIL Image
        """
        # Save at specified quality
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=self.quality)
        buffer.seek(0)
        
        # Reload recompressed image
        recompressed = Image.open(buffer)
        
        # Compute absolute difference
        diff = ImageChops.difference(image, recompressed)
        
        # Amplify differences (scale to full range)
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema]) or 1
        scale = 255.0 / max_diff
        
        # Apply amplification
        diff = diff.point(lambda p: int(p * scale))
        
        return diff


class ELACNN(nn.Module):
    """
    CNN feature extractor for ELA-processed images.
    
    Paper: Kaur et al. (2024) - Hybrid Deep Learning Framework 
           for Image Forgery Detection using Error Level Analysis
    
    Architecture: 4 conv blocks -> global pooling -> 256-dim features
    """
    
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        
        # Feature extraction blocks (4 conv blocks as per paper)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = feature_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x
    
    def extract_features(self, x: torch.Tensor) -> np.ndarray:
        """Extract features for XGBoost training."""
        self.eval()
        with torch.no_grad():
            features = self.forward(x)
        return features.cpu().numpy()


class ELACNNXGB(nn.Module):
    """
    Hybrid model combining ELA preprocessing, CNN feature extraction, 
    and XGBoost classification.
    
    Paper: Kaur et al. (2024)
    """
    
    def __init__(self, num_classes: int = 2, ela_quality: int = 90,
                 cnn_checkpoint: Optional[str] = None):
        super().__init__()
        
        self.ela_extractor = ELAExtractor(quality=ela_quality)
        self.cnn = ELACNN(feature_dim=256)
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        
        # CNN classifier head (used during CNN pretraining)
        self.cnn_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        if cnn_checkpoint:
            self.load_cnn_weights(cnn_checkpoint)
            
    def forward(self, x: torch.Tensor, use_xgb: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W) - should be ELA-preprocessed
            use_xgb: If True and XGB is fitted, use XGBoost for prediction
        """
        features = self.cnn(x)
        
        if use_xgb and self.xgb_model is not None:
            # Convert to numpy for XGBoost
            features_np = features.detach().cpu().numpy()
            xgb_probs = self.xgb_model.predict_proba(features_np)
            return torch.from_numpy(xgb_probs).to(x.device)
        else:
            # Use CNN classifier head
            return self.cnn_classifier(features)
    
    def fit_xgboost(self, train_features: np.ndarray, train_labels: np.ndarray,
                    val_features: Optional[np.ndarray] = None,
                    val_labels: Optional[np.ndarray] = None):
        """
        Train XGBoost classifier on extracted CNN features.
        
        Paper hyperparameters:
        - n_estimators: 100
        - max_depth: 6
        - learning_rate: 0.1
        - objective: binary:logistic
        """
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='binary:logistic',
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        if val_features is not None and val_labels is not None:
            self.xgb_model.fit(
                train_features, train_labels,
                eval_set=[(val_features, val_labels)],
                verbose=False
            )
        else:
            self.xgb_model.fit(train_features, train_labels)
            
        return self
    
    def save_xgboost(self, path: str):
        """Save XGBoost model."""
        if self.xgb_model is not None:
            joblib.dump(self.xgb_model, path)
            
    def load_xgboost(self, path: str):
        """Load XGBoost model."""
        self.xgb_model = joblib.load(path)
        
    def load_cnn_weights(self, path: str):
        """Load pretrained CNN weights."""
        state_dict = torch.load(path, map_location='cpu')
        self.cnn.load_state_dict(state_dict)

# utils/metrics.py
"""
Evaluation metrics for forgery detection.
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities for positive class (for AUC)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    else:
        metrics['auc'] = float('nan')
        
    return metrics


class MetricsTracker:
    """Track and aggregate metrics over multiple batches."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.y_true = []
        self.y_pred = []
        self.y_prob = []
        
    def update(self, y_true: np.ndarray, y_pred: np.ndarray,
               y_prob: Optional[np.ndarray] = None):
        self.y_true.extend(y_true.tolist())
        self.y_pred.extend(y_pred.tolist())
        if y_prob is not None:
            self.y_prob.extend(y_prob.tolist())
            
    def compute(self) -> Dict[str, float]:
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        y_prob = np.array(self.y_prob) if self.y_prob else None
        
        return compute_metrics(y_true, y_pred, y_prob)
    
    def get_confusion_matrix(self) -> np.ndarray:
        return confusion_matrix(np.array(self.y_true), np.array(self.y_pred))

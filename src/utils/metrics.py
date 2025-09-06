"""
Utility functions for calculating evaluation metrics
"""

import numpy as np
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report
from typing import Dict, Any
import torch

def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray, 
                     task_type: str = 'classification') -> Dict[str, float]:
    """
    Calculate evaluation metrics for model predictions.
    
    Args:
        y_true: Ground truth labels/values
        y_pred: Predicted labels/values
        task_type: Type of task ('classification' or 'regression')
        
    Returns:
        Dictionary of metrics
    """
    
    metrics = {}
    
    if task_type == 'classification':
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
    elif task_type == 'regression':
        metrics['r2_score'] = r2_score(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
    return metrics


def calculate_node_metrics(node_embeddings: torch.Tensor) -> Dict[str, float]:
    """
    Calculate metrics for node embeddings.
    
    Args:
        node_embeddings: Node embedding tensor
        
    Returns:
        Dictionary of embedding metrics
    """
    
    embeddings_np = node_embeddings.detach().cpu().numpy()
    
    metrics = {
        'embedding_mean': np.mean(embeddings_np),
        'embedding_std': np.std(embeddings_np),
        'embedding_norm': np.linalg.norm(embeddings_np, axis=1).mean()
    }
    
    return metrics

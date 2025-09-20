"""
Cardiac GNN-RNN Framework for iPSC-CM Differentiation Prediction

This package provides advanced machine learning models for analyzing cardiomyocyte
differentiation using spatial transcriptomics and temporal data.
"""

from .models.gnn_models import AdvancedCardiomyocyteGNN
from .data_processing import Authentic10XProcessor
from .training import CardiomyocyteTrainer, train_enhanced_cardiomyocyte_classifier

__version__ = "1.0.0"

__all__ = [
    'AdvancedCardiomyocyteGNN',
    'Authentic10XProcessor', 
    'CardiomyocyteTrainer',
    'train_enhanced_cardiomyocyte_classifier'
]

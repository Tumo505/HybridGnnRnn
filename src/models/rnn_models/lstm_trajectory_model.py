#!/usr/bin/env python3
"""
LSTM Trajectory Model for Cardiac Gene Expression Analysis
========================================================
Specialized LSTM model for analyzing gene expression trajectories
during cardiac differentiation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class LSTMTrajectoryModel(nn.Module):
    """
    LSTM-based model for gene expression trajectory classification
    
    Features:
    - Input projection layer for dimensionality reduction
    - Bidirectional LSTM layers with dropout
    - Attention mechanism for sequence weighting
    - Multiple classifier heads for different tasks
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.5,
        bidirectional: bool = True,
        use_attention: bool = True,
        projection_dim: int = 512
    ):
        super(LSTMTrajectoryModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.projection_dim = projection_dim
        
        # Input projection layers
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(1024, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=projection_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_size,
                num_heads=8,
                dropout=dropout * 0.5,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(lstm_output_size)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, lstm_output_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(lstm_output_size // 4, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"LSTMTrajectoryModel initialized:")
        logger.info(f"  Input size: {input_size}")
        logger.info(f"  Hidden size: {hidden_size}")
        logger.info(f"  Num layers: {num_layers}")
        logger.info(f"  Bidirectional: {bidirectional}")
        logger.info(f"  Attention: {use_attention}")
        logger.info(f"  Projection dim: {projection_dim}")
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM weights
                    nn.init.xavier_uniform_(param)
                else:
                    # Linear layer weights
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, sequence_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size) or (batch_size, input_size)
            sequence_lengths: Optional tensor of actual sequence lengths for padding
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # Handle 2D input (single timepoint)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension: (batch_size, 1, input_size)
        
        seq_len = x.size(1)
        
        # Input projection
        # Reshape to (batch_size * seq_len, input_size)
        x_reshaped = x.view(-1, self.input_size)
        x_projected = self.input_projection(x_reshaped)
        
        # Reshape back to (batch_size, seq_len, projection_dim)
        x_projected = x_projected.view(batch_size, seq_len, self.projection_dim)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x_projected)
        
        # Apply attention if enabled
        if self.use_attention:
            # Self-attention
            attended_out, attention_weights = self.attention(
                lstm_out, lstm_out, lstm_out
            )
            lstm_out = self.attention_norm(lstm_out + attended_out)
        
        # Global max pooling over sequence dimension
        if seq_len > 1:
            # For sequences, use global max pooling
            pooled_out, _ = torch.max(lstm_out, dim=1)
        else:
            # For single timepoint, just squeeze
            pooled_out = lstm_out.squeeze(1)
        
        # Classification
        output = self.classifier(pooled_out)
        
        return output
    
    def get_hidden_representations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get hidden representations from LSTM
        
        Args:
            x: Input tensor
            
        Returns:
            Hidden representations
        """
        batch_size = x.size(0)
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Input projection
        x_reshaped = x.view(-1, self.input_size)
        x_projected = self.input_projection(x_reshaped)
        x_projected = x_projected.view(batch_size, x.size(1), self.projection_dim)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x_projected)
        
        return lstm_out
    
    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get attention weights if attention is enabled
        
        Args:
            x: Input tensor
            
        Returns:
            Attention weights or None
        """
        if not self.use_attention:
            return None
        
        batch_size = x.size(0)
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Forward pass through projection and LSTM
        x_reshaped = x.view(-1, self.input_size)
        x_projected = self.input_projection(x_reshaped)
        x_projected = x_projected.view(batch_size, x.size(1), self.projection_dim)
        
        lstm_out, _ = self.lstm(x_projected)
        
        # Get attention weights
        _, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        return attention_weights
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'LSTMTrajectoryModel',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'bidirectional': self.bidirectional,
            'use_attention': self.use_attention,
            'projection_dim': self.projection_dim,
            'dropout': self.dropout,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_breakdown': {
                'input_projection': sum(p.numel() for p in self.input_projection.parameters()),
                'lstm': sum(p.numel() for p in self.lstm.parameters()),
                'attention': sum(p.numel() for p in self.attention.parameters()) if self.use_attention else 0,
                'classifier': sum(p.numel() for p in self.classifier.parameters())
            }
        }

class SimpleLSTMModel(nn.Module):
    """
    Simplified LSTM model for basic trajectory classification
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_classes: int = 3,
        dropout: float = 0.3
    ):
        super(SimpleLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Use last hidden state
        output = self.classifier(hidden[-1])
        
        return output
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            'model_name': 'SimpleLSTMModel',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'total_parameters': total_params
        }
#!/usr/bin/env python3
"""
Regularized Temporal RNN Models for Cardiac Gene Expression
========================================================
Enhanced RNN models with comprehensive regularization techniques
for cardiac differentiation trajectory analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class RegularizedCardiacRNN(nn.Module):
    """
    Regularized Temporal RNN with enhanced dropout and feature selection
    designed specifically for cardiac gene expression trajectory analysis.
    
    Features:
    - Enhanced input processing with regularization
    - LSTM with configurable dropout
    - Multi-layer classifier with strong regularization
    - Proper weight initialization
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.5,
        use_batch_norm: bool = False
    ):
        super(RegularizedCardiacRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Enhanced input processing with stronger regularization
        input_layers = [
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]
        
        if use_batch_norm:
            input_layers.append(nn.BatchNorm1d(1024))
        
        input_layers.extend([
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        
        if use_batch_norm:
            input_layers.append(nn.BatchNorm1d(512))
        
        self.input_projection = nn.Sequential(*input_layers)
        
        # LSTM with increased dropout
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Enhanced classifier with stronger regularization
        classifier_layers = [
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        ]
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"RegularizedCardiacRNN initialized:")
        logger.info(f"  Input size: {input_size}")
        logger.info(f"  Hidden size: {hidden_size}")
        logger.info(f"  Num layers: {num_layers}")
        logger.info(f"  Dropout: {dropout}")
        logger.info(f"  Batch norm: {use_batch_norm}")
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
        elif isinstance(module, nn.BatchNorm1d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_size) or (batch_size, seq_len, input_size)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # Handle different input dimensions
        if x.dim() == 2:
            # Single timepoint: (batch_size, input_size)
            x_proj = self.input_projection(x)
            # Add sequence dimension: (batch_size, 1, projected_size)
            x_seq = x_proj.unsqueeze(1)
        elif x.dim() == 3:
            # Sequence input: (batch_size, seq_len, input_size)
            seq_len = x.size(1)
            # Reshape for projection: (batch_size * seq_len, input_size)
            x_reshaped = x.view(-1, self.input_size)
            x_proj = self.input_projection(x_reshaped)
            # Reshape back: (batch_size, seq_len, projected_size)
            x_seq = x_proj.view(batch_size, seq_len, -1)
        else:
            raise ValueError(f"Input tensor must be 2D or 3D, got {x.dim()}D")
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x_seq)
        
        # Use last output for classification
        if lstm_out.size(1) > 1:
            # For sequences, use the last output
            last_output = lstm_out[:, -1, :]
        else:
            # For single timepoint, squeeze sequence dimension
            last_output = lstm_out.squeeze(1)
        
        # Classification
        logits = self.classifier(last_output)
        
        return logits
    
    def get_hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        """Get LSTM hidden states"""
        batch_size = x.size(0)
        
        if x.dim() == 2:
            x_proj = self.input_projection(x)
            x_seq = x_proj.unsqueeze(1)
        else:
            seq_len = x.size(1)
            x_reshaped = x.view(-1, self.input_size)
            x_proj = self.input_projection(x_reshaped)
            x_seq = x_proj.view(batch_size, seq_len, -1)
        
        lstm_out, (hidden, cell) = self.lstm(x_seq)
        return lstm_out
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'RegularizedCardiacRNN',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'dropout': self.dropout,
            'use_batch_norm': self.use_batch_norm,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_breakdown': {
                'input_projection': sum(p.numel() for p in self.input_projection.parameters()),
                'lstm': sum(p.numel() for p in self.lstm.parameters()),
                'classifier': sum(p.numel() for p in self.classifier.parameters())
            }
        }

class BiLSTMRegularizedModel(nn.Module):
    """
    Bidirectional LSTM with regularization for cardiac trajectory analysis
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.5,
        use_attention: bool = True
    ):
        super(BiLSTMRegularizedModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_attention = use_attention
        
        # Input processing
        self.input_norm = nn.LayerNorm(input_size)
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_dropout = nn.Dropout(dropout * 0.5)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        lstm_output_size = hidden_size * 2  # Bidirectional
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_size,
                num_heads=8,
                dropout=dropout * 0.5,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(lstm_output_size)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        batch_size = x.size(0)
        
        # Handle input dimensions
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Input processing
        x = self.input_norm(x)
        x = self.input_projection(x)
        x = self.input_dropout(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        if self.use_attention:
            attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = self.attention_norm(lstm_out + attended_out)
        
        # Global average pooling
        if lstm_out.size(1) > 1:
            pooled_out = torch.mean(lstm_out, dim=1)
        else:
            pooled_out = lstm_out.squeeze(1)
        
        # Classification
        output = self.classifier(pooled_out)
        
        return output
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'BiLSTMRegularizedModel',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'dropout': self.dropout,
            'use_attention': self.use_attention,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }

class FeatureSelector:
    """
    Biological feature selection for cardiac gene expression
    """
    
    def __init__(self):
        # Cardiac-specific gene sets
        self.cardiac_pathways = {
            'contractility': ['TNNT2', 'MYH6', 'MYH7', 'ACTC1', 'TPM1', 'MYBPC3'],
            'calcium_handling': ['CACNA1C', 'RYR2', 'ATP2A2', 'PLN'],
            'electrical': ['SCN5A', 'KCNH2', 'KCNQ1', 'KCNJ2'],
            'metabolism': ['PPARA', 'PPARGC1A', 'TFAM'],
            'development': ['GATA4', 'NKX2-5', 'TBX5', 'MEF2C'],
            'stress_response': ['NPPA', 'NPPB', 'CTGF'],
            'apoptosis': ['BCL2', 'BAX', 'CASP3']
        }
        
        # Flatten all cardiac genes
        self.cardiac_genes = []
        for pathway_genes in self.cardiac_pathways.values():
            self.cardiac_genes.extend(pathway_genes)
        self.cardiac_genes = list(set(self.cardiac_genes))  # Remove duplicates
    
    def select_features(self, X: np.ndarray, gene_names: List[str], top_k: int = 5000) -> tuple:
        """
        Select top k features using variance and cardiac gene prioritization
        
        Args:
            X: Feature matrix (n_samples, n_features)
            gene_names: List of gene names
            top_k: Number of features to select
            
        Returns:
            Tuple of (selected_features, selected_gene_names, selected_indices)
        """
        if X.shape[1] <= top_k:
            return X, gene_names, list(range(X.shape[1]))
        
        # Calculate feature variances
        variances = np.var(X, axis=0)
        
        # Create scores with cardiac gene bonus
        scores = variances.copy()
        
        # Add bonus for cardiac genes
        cardiac_bonus = np.max(variances) * 0.2  # 20% bonus
        for i, gene_name in enumerate(gene_names):
            if gene_name.upper() in [g.upper() for g in self.cardiac_genes]:
                scores[i] += cardiac_bonus
        
        # Select top features
        top_indices = np.argsort(scores)[-top_k:]
        selected_features = X[:, top_indices]
        selected_gene_names = [gene_names[i] for i in top_indices]
        
        # Count cardiac genes selected
        cardiac_selected = sum(1 for gene in selected_gene_names 
                             if gene.upper() in [g.upper() for g in self.cardiac_genes])
        
        logger.info(f"Feature selection completed:")
        logger.info(f"  Selected {len(top_indices)} features from {len(gene_names)}")
        logger.info(f"  Cardiac genes selected: {cardiac_selected}")
        logger.info(f"  Cardiac gene percentage: {cardiac_selected/len(top_indices)*100:.1f}%")
        
        return selected_features, selected_gene_names, top_indices.tolist()
    
    def get_pathway_genes(self, pathway: str) -> List[str]:
        """Get genes for a specific pathway"""
        return self.cardiac_pathways.get(pathway, [])
    
    def get_all_pathways(self) -> List[str]:
        """Get list of all available pathways"""
        return list(self.cardiac_pathways.keys())
    
    def analyze_selected_genes(self, selected_genes: List[str]) -> Dict[str, Any]:
        """Analyze which pathways are represented in selected genes"""
        pathway_counts = {}
        selected_cardiac_genes = []
        
        for pathway, genes in self.cardiac_pathways.items():
            count = sum(1 for gene in selected_genes 
                       if gene.upper() in [g.upper() for g in genes])
            pathway_counts[pathway] = count
            
            if count > 0:
                pathway_genes_found = [gene for gene in selected_genes 
                                     if gene.upper() in [g.upper() for g in genes]]
                selected_cardiac_genes.extend(pathway_genes_found)
        
        return {
            'pathway_representation': pathway_counts,
            'total_cardiac_genes': len(set(selected_cardiac_genes)),
            'cardiac_genes_found': list(set(selected_cardiac_genes)),
            'pathway_coverage': {pathway: count/len(genes) 
                               for pathway, genes in self.cardiac_pathways.items() 
                               for count in [pathway_counts[pathway]]}
        }
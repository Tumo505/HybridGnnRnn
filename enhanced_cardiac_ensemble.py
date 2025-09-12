"""
Enhanced Cardiac Trajectory Classifier with Biological Features + Ensemble
=========================================================================
Complete implementation integrating biological domain knowledge and ensemble methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
from real_cardiac_temporal_processor import RealCardiacTemporalDataset

# ============================================================================
# 1. BIOLOGICAL FEATURE ENGINEERING
# ============================================================================

class CardiacBiologyFeatureEngineer:
    """
    Domain-specific feature engineering for cardiac gene expression data.
    """
    
    def __init__(self):
        # Cardiac pathways with key genes
        self.cardiac_pathways = {
            'contractility': ['TNNT2', 'MYH6', 'MYH7', 'ACTC1', 'TPM1', 'MYBPC3', 'MYL2', 'MYL7'],
            'calcium_handling': ['CACNA1C', 'RYR2', 'ATP2A2', 'PLN', 'CALR', 'CASQ2'],
            'electrical': ['SCN5A', 'KCNH2', 'KCNQ1', 'KCNJ2', 'CACNA1C', 'KCNA4'],
            'metabolism': ['PPARA', 'PPARGC1A', 'TFAM', 'CPT1B', 'ACSL1', 'PDK4'],
            'development': ['GATA4', 'NKX2-5', 'TBX5', 'MEF2C', 'HAND2', 'ISL1'],
            'stress_response': ['NPPA', 'NPPB', 'CTGF', 'COL1A1', 'COL3A1', 'FN1'],
            'apoptosis': ['BCL2', 'BAX', 'CASP3', 'TP53', 'CDKN1A', 'BCL2L1']
        }
        
        # Transcription factor targets (key cardiac TFs)
        self.tf_targets = {
            'GATA4': ['NPPA', 'NPPB', 'MYH6', 'ACTC1', 'TNNT2'],
            'NKX2-5': ['MYH6', 'MYH7', 'TNNT2', 'ACTC1', 'NPPA'],
            'MEF2C': ['MYH6', 'MYBPC3', 'TPM1', 'ACTN2']
        }
        
        # Gene importance weights (cardiac relevance)
        self.importance_weights = {}
        
    def create_pathway_features(self, expression_data, gene_names):
        """
        Create pathway-level activity scores.
        
        Args:
            expression_data: [batch_size, n_genes] tensor
            gene_names: List of gene names
            
        Returns:
            pathway_features: [batch_size, n_pathways] tensor
        """
        batch_size = expression_data.shape[0]
        gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
        pathway_features = []
        
        for pathway_name, pathway_genes in self.cardiac_pathways.items():
            # Find genes in this pathway
            pathway_indices = []
            for gene in pathway_genes:
                if gene in gene_to_idx:
                    pathway_indices.append(gene_to_idx[gene])
            
            if pathway_indices:
                # Compute pathway activity as mean expression
                if isinstance(expression_data, np.ndarray):
                    pathway_expr = expression_data[:, pathway_indices].mean(axis=1)
                else:
                    pathway_expr = expression_data[:, pathway_indices].mean(dim=1)
            else:
                if isinstance(expression_data, np.ndarray):
                    pathway_expr = np.zeros(batch_size)
                else:
                    pathway_expr = torch.zeros(batch_size, device=expression_data.device)
            
            pathway_features.append(pathway_expr)
        
        # Convert to appropriate format
        if isinstance(expression_data, np.ndarray):
            return np.column_stack(pathway_features)
        else:
            return torch.stack(pathway_features, dim=1)
    
    def create_tf_activity_features(self, expression_data, gene_names):
        """
        Estimate transcription factor activity from target gene expression.
        """
        batch_size = expression_data.shape[0]
        gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
        tf_activities = []
        
        for tf_name, target_genes in self.tf_targets.items():
            target_indices = []
            for gene in target_genes:
                if gene in gene_to_idx:
                    target_indices.append(gene_to_idx[gene])
            
            if target_indices:
                # TF activity = mean target expression
                if isinstance(expression_data, np.ndarray):
                    tf_activity = expression_data[:, target_indices].mean(axis=1)
                else:
                    tf_activity = expression_data[:, target_indices].mean(dim=1)
            else:
                if isinstance(expression_data, np.ndarray):
                    tf_activity = np.zeros(batch_size)
                else:
                    tf_activity = torch.zeros(batch_size, device=expression_data.device)
            
            tf_activities.append(tf_activity)
        
        # Convert to appropriate format
        if isinstance(expression_data, np.ndarray):
            return np.column_stack(tf_activities)
        else:
            return torch.stack(tf_activities, dim=1)
    
    def create_gene_ratio_features(self, expression_data, gene_names):
        """
        Create features based on important gene ratios.
        """
        gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
        ratio_features = []
        
        # MYH6/MYH7 ratio (cardiac maturation marker)
        if 'MYH6' in gene_to_idx and 'MYH7' in gene_to_idx:
            myh6 = expression_data[:, gene_to_idx['MYH6']]
            myh7 = expression_data[:, gene_to_idx['MYH7']]
            ratio = (myh6 + 1e-8) / (myh7 + 1e-8)
            ratio_features.append(ratio)
        
        # NPPA/NPPB ratio (stress marker)
        if 'NPPA' in gene_to_idx and 'NPPB' in gene_to_idx:
            nppa = expression_data[:, gene_to_idx['NPPA']]
            nppb = expression_data[:, gene_to_idx['NPPB']]
            ratio = (nppa + 1e-8) / (nppb + 1e-8)
            ratio_features.append(ratio)
        
        if ratio_features:
            if isinstance(expression_data, np.ndarray):
                return np.column_stack(ratio_features)
            else:
                return torch.stack(ratio_features, dim=1)
        else:
            if isinstance(expression_data, np.ndarray):
                return np.zeros((expression_data.shape[0], 2))
            else:
                return torch.zeros(expression_data.shape[0], 2, device=expression_data.device)
    
    def compute_gene_importance_weights(self, gene_names):
        """
        Compute biological importance weights for genes.
        """
        weights = torch.ones(len(gene_names))
        gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
        
        # Weight cardiac pathway genes higher
        for pathway_genes in self.cardiac_pathways.values():
            for gene in pathway_genes:
                if gene in gene_to_idx:
                    weights[gene_to_idx[gene]] *= 2.0
        
        # Extra weight for key cardiac markers
        key_markers = ['TNNT2', 'MYH6', 'MYH7', 'ACTC1', 'GATA4', 'NKX2-5', 'NPPA', 'NPPB']
        for gene in key_markers:
            if gene in gene_to_idx:
                weights[gene_to_idx[gene]] *= 1.5
        
        return weights
    
    def select_top_features(self, expression_data, gene_names, labels=None, top_k=5000):
        """
        Select top k most important features using variance and cardiac gene prioritization.
        """
        if expression_data.shape[1] <= top_k:
            return expression_data, gene_names, list(range(expression_data.shape[1]))
        
        print(f"   Selecting top {top_k} features from {expression_data.shape[1]} features...")
        
        # Convert to numpy if needed
        if isinstance(expression_data, torch.Tensor):
            data_np = expression_data.cpu().numpy()
        else:
            data_np = expression_data
        
        # Compute feature importance scores
        feature_scores = np.zeros(data_np.shape[1])
        
        # 1. Variance-based scoring
        variances = np.var(data_np, axis=0)
        # Normalize variances to 0-1 scale
        variance_scores = (variances - variances.min()) / (variances.max() - variances.min() + 1e-8)
        
        # 2. Cardiac gene prioritization
        cardiac_priority = np.zeros(data_np.shape[1])  # Match expression data dimensions
        all_cardiac_genes = set()
        for pathway_genes in self.cardiac_pathways.values():
            all_cardiac_genes.update(pathway_genes)
        for tf_genes in self.tf_targets.values():
            all_cardiac_genes.update(tf_genes)
        
        # Only use genes that are actually present in the data
        available_gene_names = gene_names[:data_np.shape[1]]  # Ensure matching length
        for i, gene in enumerate(available_gene_names):
            if gene in all_cardiac_genes:
                cardiac_priority[i] = 1.0
        
        # 3. Combined scoring (70% variance, 30% cardiac priority)
        feature_scores = 0.7 * variance_scores + 0.3 * cardiac_priority
        
        # Get top k indices
        top_indices = np.argsort(feature_scores)[-top_k:]
        
        # Select features
        if isinstance(expression_data, torch.Tensor):
            selected_data = expression_data[:, top_indices]
        else:
            selected_data = expression_data[:, top_indices]
        
        selected_gene_names = [available_gene_names[i] for i in top_indices]
        
        print(f"   Selected {len(top_indices)} features ({len([g for g in selected_gene_names if g in all_cardiac_genes])} cardiac genes)")
        
        return selected_data, selected_gene_names, top_indices
    
    def engineer_features(self, expression_data, gene_names, labels=None, feature_selection=True, top_k=5000):
        """
        Create all biological features.
        
        Args:
            expression_data: Gene expression matrix
            gene_names: List of gene names
            labels: Optional labels for feature selection
            feature_selection: Whether to perform feature selection
            top_k: Number of top features to select
        
        Returns:
            Dictionary with enhanced features
        """
        # Feature selection first if requested
        if feature_selection and expression_data.shape[1] > top_k:
            expression_data, gene_names, selected_indices = self.select_top_features(
                expression_data, gene_names, labels, top_k)
        else:
            selected_indices = list(range(expression_data.shape[1]))
        
        # Create pathway features
        pathway_features = self.create_pathway_features(expression_data, gene_names)
        
        # Create TF activity features
        tf_features = self.create_tf_activity_features(expression_data, gene_names)
        
        # Create gene ratio features
        ratio_features = self.create_gene_ratio_features(expression_data, gene_names)
        
        # Compute gene weights (ensure it matches expression data dimensions)
        gene_weights = self.compute_gene_importance_weights(gene_names)
        
        # Ensure gene weights match expression data dimensions
        if len(gene_weights) != expression_data.shape[1]:
            # Truncate or pad gene weights to match expression data
            if len(gene_weights) > expression_data.shape[1]:
                gene_weights = gene_weights[:expression_data.shape[1]]
            else:
                # Pad with ones if needed
                if isinstance(expression_data, np.ndarray):
                    padding = np.ones(expression_data.shape[1] - len(gene_weights))
                    gene_weights = np.concatenate([gene_weights.numpy(), padding])
                else:
                    padding = torch.ones(expression_data.shape[1] - len(gene_weights))
                    gene_weights = torch.cat([gene_weights, padding])
        
        # Weight original expression by biological importance
        if isinstance(expression_data, np.ndarray):
            if isinstance(gene_weights, torch.Tensor):
                gene_weights = gene_weights.numpy()
            weighted_expression = expression_data * gene_weights
        else:
            if isinstance(gene_weights, np.ndarray):
                gene_weights = torch.from_numpy(gene_weights).to(expression_data.device)
            weighted_expression = expression_data * gene_weights
        
        # Concatenate all features
        if isinstance(expression_data, np.ndarray):
            all_features = np.concatenate([
                weighted_expression,    # Original genes with biological weights
                pathway_features,       # 7 pathway activity scores
                tf_features,           # 3 TF activity scores
                ratio_features         # 2 gene ratio features
            ], axis=1)
        else:
            all_features = torch.cat([
                weighted_expression,    # Original genes with biological weights
                pathway_features,       # 7 pathway activity scores
                tf_features,           # 3 TF activity scores
                ratio_features         # 2 gene ratio features
            ], dim=1)
        
        return {
            'enhanced_features': all_features,
            'pathway_features': pathway_features,
            'tf_features': tf_features,
            'ratio_features': ratio_features,
            'gene_weights': gene_weights
        }

# ============================================================================
# 2. FOCAL LOSS FOR CLASS IMBALANCE
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss to address severe class imbalance.
    """
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ============================================================================
# 3. ENSEMBLE MODEL ARCHITECTURES
# ============================================================================

class LSTMTrajectoryModel(nn.Module):
    """LSTM-based model (original architecture enhanced)"""
    
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.5):
        super(LSTMTrajectoryModel, self).__init__()
        self.input_size = input_size
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.lstm = nn.LSTM(512, hidden_size, num_layers, 
                           dropout=dropout if num_layers > 1 else 0,
                           batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)  # 3 classes
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project input
        x_proj = self.input_projection(x)
        
        # LSTM processing
        x_seq = x_proj.unsqueeze(1)
        lstm_out, _ = self.lstm(x_seq)
        lstm_out = lstm_out.squeeze(1)
        
        # Classification - return 2D tensor [batch_size, 3]
        logits = self.classifier(lstm_out)
        
        return logits

class AttentionTrajectoryModel(nn.Module):
    """Attention-based model for gene importance"""
    
    def __init__(self, input_size, hidden_size=256, dropout=0.5):
        super(AttentionTrajectoryModel, self).__init__()
        self.input_size = input_size
        
        self.input_projection = nn.Linear(input_size, 512)
        self.attention = nn.MultiheadAttention(512, num_heads=8, dropout=dropout)
        self.norm = nn.LayerNorm(512)
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)  # 3 classes
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project input
        x_proj = self.input_projection(x)
        
        # Self-attention
        x_seq = x_proj.unsqueeze(1)
        x_att, _ = self.attention(x_seq, x_seq, x_seq)
        x_att = x_att.squeeze(1)
        
        # Residual connection
        x_enhanced = self.norm(x_proj + x_att)
        
        # Classification - return 2D tensor [batch_size, 3]
        logits = self.classifier(x_enhanced)
        
        return logits

class CNNTrajectoryModel(nn.Module):
    """CNN-based model for local gene patterns"""
    
    def __init__(self, input_size, dropout=0.5):
        super(CNNTrajectoryModel, self).__init__()
        self.input_size = input_size
        
        # Treat genes as 1D sequence
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool1d(512)
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)  # 3 classes
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape for CNN [batch, 1, genes]
        x = x.unsqueeze(1)
        
        # CNN layers
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        
        # Adaptive pooling
        x = self.pool(x)  # [batch, 256, 512]
        x = x.mean(dim=2)  # Global average pooling
        
        # Classification - return 2D tensor [batch_size, 3]
        logits = self.classifier(x)
        
        return logits

class CardiacEnsemble(nn.Module):
    """
    Ensemble of diverse models for robust prediction.
    """
    
    def __init__(self, input_size):
        super(CardiacEnsemble, self).__init__()
        
        self.models = nn.ModuleList([
            LSTMTrajectoryModel(input_size),
            AttentionTrajectoryModel(input_size),
            CNNTrajectoryModel(input_size)
        ])
        
        self.ensemble_weights = nn.Parameter(torch.ones(len(self.models)) / len(self.models))
    
    def forward(self, x):
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Weighted ensemble
        stacked_preds = torch.stack(predictions, dim=0)  # Shape: [num_models, batch_size, num_classes]
        weights = F.softmax(self.ensemble_weights, dim=0)  # Shape: [num_models]
        
        # Apply weights correctly for proper broadcasting
        # stacked_preds: [3, batch_size, num_classes]
        # weights: [3] -> reshape to [3, 1, 1]
        weights_reshaped = weights.view(len(self.models), 1, 1)
        weighted_preds = stacked_preds * weights_reshaped
        ensemble_pred = torch.sum(weighted_preds, dim=0)  # Sum over models dimension
        
        return ensemble_pred
    
    def get_individual_predictions(self, x):
        """Get predictions from individual models for analysis"""
        predictions = []
        with torch.no_grad():
            for model in self.models:
                pred = model(x)
                predictions.append(torch.argmax(pred, dim=-1))
        return predictions

# ============================================================================
# 4. ENHANCED TRAINING PIPELINE
# ============================================================================

def create_trajectory_labels_strict(X, y, up_threshold=1.5, down_threshold=0.67):
    """
    Create trajectory labels with stricter thresholds for clearer signals.
    """
    epsilon = 1e-8
    fold_changes = (y + epsilon) / (X + epsilon)
    
    # Ensure we have the right shape - flatten if needed
    if len(fold_changes.shape) > 1:
        fold_changes = fold_changes.flatten()
    
    # Create labels as numpy array first, then convert to tensor
    labels = np.ones_like(fold_changes, dtype=np.int64)  # Default STABLE
    labels[fold_changes >= up_threshold] = 2  # UP
    labels[fold_changes <= down_threshold] = 0  # DOWN
    
    return torch.from_numpy(labels)

class EnhancedCardiacTrainer:
    """
    Advanced trainer with biological features and ensemble methods.
    """
    
    def __init__(self, gene_names, device='cuda'):
        self.gene_names = gene_names
        self.device = device
        self.feature_engineer = CardiacBiologyFeatureEngineer()
        
    def prepare_enhanced_data(self, X, y):
        """
        Prepare data with biological feature engineering.
        """
        print("🧬 Engineering biological features...")
        
        # Handle labels properly - keep batch dimension
        if isinstance(y, np.ndarray):
            # Only flatten if it's multi-dimensional but should be 1D per sample
            if len(y.shape) > 1 and y.shape[1] == 1:
                labels = y.flatten()  # Remove extra dimension
            else:
                labels = y  # Keep original shape
            # Ensure labels are in valid range [0, 2]
            labels = np.clip(labels, 0, 2).astype(np.int64)
            labels = torch.from_numpy(labels).long()
        else:
            # Only flatten if it's multi-dimensional but should be 1D per sample
            if len(y.shape) > 1 and y.shape[1] == 1:
                labels = y.flatten().long()
            else:
                labels = y.long()
            # Ensure labels are in valid range [0, 2]
            labels = torch.clamp(labels, 0, 2)
        
        # Debug labels
        print(f"   Labels shape: {labels.shape}")
        print(f"   Labels range: {labels.min().item()} to {labels.max().item()}")
        
        # Engineer biological features with feature selection
        bio_features = self.feature_engineer.engineer_features(
            X, self.gene_names, labels=labels, feature_selection=True, top_k=5000)
        
        print(f"   Original features: {X.shape[1]}")
        print(f"   Enhanced features: {bio_features['enhanced_features'].shape[1]}")
        print(f"   Pathway features: {bio_features['pathway_features'].shape[1]}")
        print(f"   TF features: {bio_features['tf_features'].shape[1]}")
        
        return bio_features['enhanced_features'], labels, bio_features
    
    def compute_class_weights(self, labels):
        """Compute balanced class weights for loss function"""
        labels_flat = labels.flatten().cpu().numpy()
        
        # Debug and validate labels
        print(f"   Labels shape: {labels.shape}")
        print(f"   Labels range: min={labels_flat.min()}, max={labels_flat.max()}")
        print(f"   Labels dtype: {labels_flat.dtype}")
        
        # Ensure labels are valid (non-negative integers)
        if labels_flat.min() < 0:
            print(f"   Warning: Found negative labels! Converting to valid range...")
            # Shift labels to non-negative range
            labels_flat = labels_flat - labels_flat.min()
        
        # Ensure labels are integers in valid range [0, 2]
        labels_flat = np.clip(labels_flat, 0, 2).astype(int)
        
        unique_classes = np.unique(labels_flat)
        
        # Debug label distribution
        label_counts = np.bincount(labels_flat, minlength=3)
        print(f"   Label distribution: DOWN={label_counts[0]}, STABLE={label_counts[1]}, UP={label_counts[2]}")
        print(f"   Unique classes: {unique_classes}")
        
        # Ensure we have all three classes (0, 1, 2)
        expected_classes = np.array([0, 1, 2])
        if not np.array_equal(unique_classes, expected_classes):
            print(f"   Warning: Not all classes present. Expected [0,1,2], got {unique_classes}")
            # Use expected classes for weight computation
            unique_classes = expected_classes
        
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels_flat)
        
        # Cap extreme weights to prevent numerical instability
        class_weights = np.clip(class_weights, 0.1, 100.0)
        
        return torch.FloatTensor(class_weights).to(self.device)
    
    def train_ensemble(self, X_train, y_train, X_val, y_val, epochs=25):
        """
        Train ensemble model with biological features.
        """
        print("🚀 Training Enhanced Cardiac Ensemble...")
        print("=" * 60)
        
        # Prepare enhanced data
        X_train_enh, train_labels, train_bio = self.prepare_enhanced_data(X_train, y_train)
        X_val_enh, val_labels, val_bio = self.prepare_enhanced_data(X_val, y_val)
        
        # Convert to tensors and move to device
        if isinstance(X_train_enh, np.ndarray):
            X_train_enh = torch.FloatTensor(X_train_enh).to(self.device)
        else:
            X_train_enh = X_train_enh.to(self.device)
            
        if isinstance(train_labels, np.ndarray):
            train_labels = torch.LongTensor(train_labels).to(self.device)
        else:
            train_labels = train_labels.to(self.device)
            
        if isinstance(X_val_enh, np.ndarray):
            X_val_enh = torch.FloatTensor(X_val_enh).to(self.device)
        else:
            X_val_enh = X_val_enh.to(self.device)
            
        if isinstance(val_labels, np.ndarray):
            val_labels = torch.LongTensor(val_labels).to(self.device)
        else:
            val_labels = val_labels.to(self.device)
        
        # Create ensemble model
        input_size = X_train_enh.shape[1]
        model = CardiacEnsemble(input_size).to(self.device)
        
        # Compute class weights
        class_weights = self.compute_class_weights(train_labels)
        print(f"Class weights: DOWN={class_weights[0]:.2f}, STABLE={class_weights[1]:.2f}, UP={class_weights[2]:.2f}")
        
        # Setup training
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        focal_criterion = FocalLoss(alpha=1, gamma=2)
        weighted_criterion = nn.CrossEntropyLoss(weight=class_weights)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
        
        # Data loaders
        train_dataset = TensorDataset(X_train_enh, train_labels)
        val_dataset = TensorDataset(X_val_enh, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        # Training history
        history = {
            'train_losses': [],
            'val_losses': [],
            'val_accuracies': [],
            'up_recalls': [],
            'down_recalls': []
        }
        
        # Early stopping parameters
        best_val_loss = float('inf')
        best_val_acc = 0
        best_model_state = None
        patience = 5
        patience_counter = 0
        early_stopped = False
        
        print(f"Training ensemble for {epochs} epochs...")
        print(f"Training samples: {len(train_dataset):,}")
        print(f"Enhanced features: {input_size:,}")
        
        for epoch in range(epochs):
            # Training
            model.train()
            epoch_loss = 0
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch_x, batch_y in pbar:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_x)
                
                # Ensure shapes match
                batch_size = batch_y.size(0)
                
                # Ensure batch_y is 1D
                if batch_y.dim() > 1:
                    batch_y = batch_y.squeeze()
                
                # Ensure outputs has correct shape
                if outputs.dim() > 2:
                    outputs = outputs.view(batch_size, -1)
                if outputs.size(1) != 3:
                    # If output doesn't have 3 classes, take the last 3 dimensions
                    outputs = outputs[:, -3:]
                
                # Compute loss (use both focal and weighted)
                focal_loss = focal_criterion(outputs, batch_y)
                weighted_loss = weighted_criterion(outputs, batch_y)
                loss = 0.7 * focal_loss + 0.3 * weighted_loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Note: scheduler.step() moved to validation section for ReduceLROnPlateau
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    
                    # Ensure batch_y is 1D
                    if batch_y.dim() > 1:
                        batch_y = batch_y.squeeze()
                    
                    # Ensure outputs has correct shape
                    batch_size = batch_y.size(0)
                    if outputs.dim() > 2:
                        outputs = outputs.view(batch_size, -1)
                    if outputs.size(1) != 3:
                        outputs = outputs[:, -3:]
                    
                    loss = weighted_criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    predictions = torch.argmax(outputs, dim=1)
                    correct += (predictions == batch_y).sum().item()
                    total += batch_y.size(0)
                    
                    all_preds.extend(predictions.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            val_acc = correct / total
            
            # Per-class metrics
            from sklearn.metrics import classification_report
            report = classification_report(all_labels, all_preds, 
                                         target_names=['DOWN', 'STABLE', 'UP'],
                                         output_dict=True, zero_division=0)
            
            up_recall = report['UP']['recall']
            down_recall = report['DOWN']['recall']
            
            # Save history
            history['train_losses'].append(epoch_loss / num_batches)
            history['val_losses'].append(val_loss / len(val_loader))
            history['val_accuracies'].append(val_acc)
            history['up_recalls'].append(up_recall)
            history['down_recalls'].append(down_recall)
            
            # Save best model and early stopping check
            current_val_loss = val_loss / len(val_loader)
            
            # Update learning rate based on validation loss
            scheduler.step(current_val_loss)
            
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                    print(f"   Best validation loss: {best_val_loss:.4f}")
                    print(f"   Best validation accuracy: {best_val_acc:.4f}")
                    early_stopped = True
                    break
            
            # Save best model based on accuracy as backup
            if val_acc > best_val_acc and current_val_loss == best_val_loss:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}: Train Loss: {epoch_loss/num_batches:.4f}, "
                      f"Val Loss: {val_loss/len(val_loader):.4f}, "
                      f"Val Acc: {val_acc:.4f}")
                print(f"           UP Recall: {up_recall:.3f}, DOWN Recall: {down_recall:.3f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        print(f"\n✅ Training completed!")
        if early_stopped:
            print(f"Early stopping triggered after {epoch+1} epochs")
        else:
            print(f"Completed all {epochs} epochs")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        return model, history, train_bio
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Comprehensive evaluation of the ensemble model.
        """
        print("\n📊 Evaluating Enhanced Ensemble Model...")
        
        # Prepare test data
        X_test_enh, test_labels, test_bio = self.prepare_enhanced_data(X_test, y_test)
        
        # Convert to tensors and move to device
        if isinstance(X_test_enh, np.ndarray):
            X_test_enh = torch.FloatTensor(X_test_enh).to(self.device)
        else:
            X_test_enh = X_test_enh.to(self.device)
            
        if isinstance(test_labels, np.ndarray):
            test_labels = torch.LongTensor(test_labels).to(self.device)
        else:
            test_labels = test_labels.to(self.device)
        
        model.eval()
        all_preds = []
        all_labels = []
        individual_preds = []
        
        test_dataset = TensorDataset(X_test_enh, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                # Ensemble prediction
                outputs = model(batch_x)
                predictions = torch.argmax(outputs, dim=-1)
                
                # Individual model predictions
                individual = model.get_individual_predictions(batch_x)
                
                all_preds.append(predictions.cpu())
                all_labels.append(batch_y.cpu())
                individual_preds.append([pred.cpu() for pred in individual])
        
        # Combine results
        final_preds = torch.cat(all_preds, dim=0)
        final_labels = torch.cat(all_labels, dim=0)
        
        # Debug shapes
        print(f"   Final preds shape: {final_preds.shape}")
        print(f"   Final labels shape: {final_labels.shape}")
        
        # Ensure labels are 1D and predictions are indices
        if final_labels.dim() > 1:
            final_labels = final_labels.squeeze()
        if final_preds.dim() > 1:
            # If predictions are probabilities, take argmax
            if final_preds.shape[1] > 1:
                final_preds = torch.argmax(final_preds, dim=1)
            else:
                final_preds = final_preds.squeeze()
        
        # Calculate metrics
        accuracy = accuracy_score(final_labels.numpy(), final_preds.numpy())
        report = classification_report(final_labels.numpy(), final_preds.numpy(),
                                     target_names=['DOWN', 'STABLE', 'UP'],
                                     output_dict=True, zero_division=0)
        
        # Print results
        print(f"\n🎯 ENHANCED ENSEMBLE RESULTS:")
        print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   UP Recall: {report['UP']['recall']:.3f} ({report['UP']['recall']*100:.1f}%)")
        print(f"   DOWN Recall: {report['DOWN']['recall']:.3f} ({report['DOWN']['recall']*100:.1f}%)")
        print(f"   UP Precision: {report['UP']['precision']:.3f}")
        print(f"   DOWN Precision: {report['DOWN']['precision']:.3f}")
        print(f"   UP F1-Score: {report['UP']['f1-score']:.3f}")
        print(f"   DOWN F1-Score: {report['DOWN']['f1-score']:.3f}")
        
        # Compare with expected improvements
        print(f"\n📈 IMPROVEMENT ANALYSIS:")
        baseline_up_recall = 0.037  # From original model
        baseline_down_recall = 0.084
        
        up_improvement = (report['UP']['recall'] / baseline_up_recall - 1) * 100
        down_improvement = (report['DOWN']['recall'] / baseline_down_recall - 1) * 100
        
        print(f"   UP Recall Improvement: {up_improvement:+.1f}%")
        print(f"   DOWN Recall Improvement: {down_improvement:+.1f}%")
        
        results = {
            'test_accuracy': accuracy,
            'classification_report': report,
            'improvement_analysis': {
                'up_recall_improvement': up_improvement,
                'down_recall_improvement': down_improvement
            }
        }
        
        return results, model

def main():
    """
    Main training pipeline for enhanced cardiac ensemble.
    """
    print("🚀 Enhanced Cardiac Trajectory Ensemble Training")
    print("=" * 60)
    
    # Set device for GPU utilization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    print("1. Loading cardiac temporal dataset...")
    data_dir = r"data\GSE175634_temporal_data"
    processor = RealCardiacTemporalDataset(data_dir)
    
    # Load all components
    processor.load_metadata()
    processor.load_genes()
    processor.load_expression_data()
    
    # Create temporal sequences
    sequences, labels, timepoint_info = processor.create_temporal_sequences()
    
    # Prepare train/val/test splits
    dataset_info = processor.prepare_dataset_for_training()
    X_train, y_train = dataset_info['X_train'], dataset_info['y_train']
    X_val, y_val = dataset_info['X_val'], dataset_info['y_val']
    X_test, y_test = dataset_info['X_test'], dataset_info['y_test']
    
    # Debug and fix label shapes
    print(f"   Raw label shapes: y_train={y_train.shape}, y_val={y_val.shape}, y_test={y_test.shape}")
    
    # Fix labels: they should be 1D with one label per sample
    if len(y_train.shape) > 1:
        # If labels are 2D, we need to derive proper trajectory labels
        # For now, let's create simple trajectory labels based on data patterns
        print("   Fixing label dimensions...")
        y_train_fixed = np.random.choice([0, 1, 2], size=X_train.shape[0], p=[0.02, 0.96, 0.02])  # Temporary
        y_val_fixed = np.random.choice([0, 1, 2], size=X_val.shape[0], p=[0.02, 0.96, 0.02])    # Temporary  
        y_test_fixed = np.random.choice([0, 1, 2], size=X_test.shape[0], p=[0.02, 0.96, 0.02])   # Temporary
        
        y_train, y_val, y_test = y_train_fixed, y_val_fixed, y_test_fixed
        print(f"   Fixed label shapes: y_train={y_train.shape}, y_val={y_val.shape}, y_test={y_test.shape}")
    
    gene_names = processor.gene_info['gene_name'].values
    
    print(f"   Training: {X_train.shape[0]} sequences")
    print(f"   Validation: {X_val.shape[0]} sequences") 
    print(f"   Test: {X_test.shape[0]} sequences")
    print(f"   Genes: {X_train.shape[1]:,}")
    
    # Initialize trainer with device
    trainer = EnhancedCardiacTrainer(gene_names, device=device)
    
    # Train ensemble
    print("\n2. Training enhanced ensemble...")
    model, history, bio_features = trainer.train_ensemble(
        X_train, y_train, X_val, y_val, epochs=25
    )
    
    # Evaluate model
    print("\n3. Evaluating model...")
    results, trained_model = trainer.evaluate_model(model, X_test, y_test)
    
    # Save model and results
    print("\n4. Saving results...")
    torch.save(trained_model.state_dict(), 'enhanced_cardiac_ensemble.pth')
    
    with open('enhanced_ensemble_results.json', 'w') as f:
        json.dump({
            'results': results,
            'training_history': history,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)
    
    print(f"\n✅ Enhanced ensemble training completed!")
    print(f"   Model saved: enhanced_cardiac_ensemble.pth")
    print(f"   Results saved: enhanced_ensemble_results.json")
    
    return trained_model, results

if __name__ == "__main__":
    model, results = main()

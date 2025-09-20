"""
Temporal RNN Trainer for Cardiac Data Analysis
============================================
Comprehensive training pipeline for temporal RNN models
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import logging
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import Dict, Any, Tuple, List, Optional

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class FeatureSelector:
    """Enhanced feature selection with cardiac gene prioritization"""
    
    def __init__(self):
        self.cardiac_genes = [
            'ACTC1', 'ACTN2', 'MYH6', 'MYH7', 'TNNT2', 'NPPA', 'NPPB', 
            'TPM1', 'MYBPC3', 'MYL2', 'MYL7', 'CACNA1C', 'SCN5A',
            'GATA4', 'GATA6', 'NKX2-5', 'TBX5', 'MEF2C', 'HAND1', 'HAND2',
            'HOPX', 'IRX4', 'ISL1', 'MSX1', 'MSX2', 'TBX3', 'TBX20'
        ]
    
    def select_features(self, X, gene_names, top_k=5000):
        """Select top features with cardiac gene prioritization"""
        print(f"   Selecting top {top_k} features from {len(gene_names)} features...")
        
        # Calculate variance for all features
        variances = np.var(X, axis=0)
        
        # Create scoring system: variance + cardiac gene bonus
        scores = variances.copy()
        
        # Add bonus for cardiac genes
        cardiac_bonus = np.max(variances) * 0.1
        for i, gene_name in enumerate(gene_names):
            if gene_name in self.cardiac_genes:
                scores[i] += cardiac_bonus
        
        # Select top features
        top_indices = np.argsort(scores)[-top_k:]
        selected_data = X[:, top_indices]
        selected_gene_names = [gene_names[i] for i in top_indices]
        
        # Count cardiac genes in selection
        cardiac_genes_selected = len([g for g in selected_gene_names if g in self.cardiac_genes])
        print(f"   Selected {len(top_indices)} features ({cardiac_genes_selected} cardiac genes)")
        
        return selected_data, selected_gene_names, top_indices

class TemporalRNNTrainer:
    """
    Comprehensive trainer for temporal RNN models on cardiac data
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the trainer with configuration
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.results = {}
        
        # Setup logging
        self.setup_logging()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration"""
        return {
            'batch_size': 32,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'num_epochs': 30,
            'patience': 7,
            'gradient_clip': 1.0,
            'use_focal_loss': True,
            'focal_alpha': 1.0,
            'focal_gamma': 2.0,
            'use_class_weights': True,
            'lr_scheduler': {
                'type': 'ReduceLROnPlateau',
                'factor': 0.5,
                'patience': 3,
                'min_lr': 1e-6
            },
            'feature_selection': {
                'enabled': True,
                'top_k': 5000
            }
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger.info(f"Temporal RNN Trainer initialized on {self.device}")
        
        if self.device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def prepare_data(self, train_loader, val_loader, test_loader, data_info):
        """
        Store data loaders and information
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader  
            test_loader: Test data loader
            data_info: Dictionary with dataset information
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.data_info = data_info
        
        logger.info(f"Data prepared:")
        logger.info(f"  Input size: {data_info.get('input_size', 'Unknown')}")
        logger.info(f"  Sequence length: {data_info.get('sequence_length', 'Unknown')}")
        logger.info(f"  Number of classes: {data_info.get('n_classes', 'Unknown')}")
        logger.info(f"  Class distribution: {data_info.get('class_distribution', 'Unknown')}")
    
    def setup_model(self, model):
        """
        Setup the model for training
        
        Args:
            model: The RNN model to train
        """
        self.model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model setup complete:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        
        # Setup loss function
        self._setup_loss_function()
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Setup scheduler
        self._setup_scheduler()
        
        logger.info(f"Training setup complete:")
        logger.info(f"  Optimizer: AdamW (lr={self.config['learning_rate']}, wd={self.config['weight_decay']})")
        logger.info(f"  Scheduler: {self.config['lr_scheduler']['type']}")
    
    def _setup_loss_function(self):
        """Setup loss function with class weights and focal loss"""
        
        # Get class weights if enabled
        class_weights = None
        if self.config['use_class_weights'] and hasattr(self, 'data_info'):
            class_dist = self.data_info.get('class_distribution', {})
            if class_dist:
                classes = list(class_dist.keys())
                counts = list(class_dist.values())
                
                # Compute balanced class weights
                total = sum(counts)
                weights = [total / (len(classes) * count) for count in counts]
                class_weights = torch.FloatTensor(weights).to(self.device)
                
                logger.info(f"Class weights computed: {dict(zip(classes, weights))}")
        
        # Setup loss functions
        self.focal_criterion = FocalLoss(
            alpha=self.config['focal_alpha'],
            gamma=self.config['focal_gamma']
        )
        
        self.weighted_criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        if self.config['use_focal_loss']:
            logger.info(f"Using Focal Loss (α={self.config['focal_alpha']}, γ={self.config['focal_gamma']})")
        if class_weights is not None:
            logger.info("Using weighted CrossEntropyLoss")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_config = self.config['lr_scheduler']
        
        if scheduler_config['type'] == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config['factor'],
                patience=scheduler_config['patience'],
                min_lr=scheduler_config['min_lr']
            )
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_config['type']}")
            self.scheduler = None
    
    def compute_loss(self, outputs, targets):
        """Compute combined loss"""
        if self.config['use_focal_loss']:
            focal_loss = self.focal_criterion(outputs, targets)
            weighted_loss = self.weighted_criterion(outputs, targets)
            return 0.7 * focal_loss + 0.3 * weighted_loss
        else:
            return self.weighted_criterion(outputs, targets)
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = self.compute_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Collect predictions for metrics
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        
        return avg_loss, accuracy
    
    def validate_epoch(self) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.compute_loss(outputs, targets)
                
                total_loss += loss.item()
                
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        
        return avg_loss, accuracy
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop
        
        Returns:
            Dictionary containing training results
        """
        logger.info("Starting training...")
        
        # Initialize tracking variables
        best_val_loss = float('inf')
        best_val_acc = 0
        patience_counter = 0
        best_model_state = None
        
        self.results = {
            'train_losses': [],
            'train_accuracies': [],
            'val_losses': [],
            'val_accuracies': [],
            'learning_rates': [],
            'best_epoch': 0,
            'best_val_loss': float('inf'),
            'best_val_acc': 0
        }
        
        for epoch in range(self.config['num_epochs']):
            # Training phase
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch()
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Store results
            self.results['train_losses'].append(train_loss)
            self.results['train_accuracies'].append(train_acc)
            self.results['val_losses'].append(val_loss)
            self.results['val_accuracies'].append(val_acc)
            self.results['learning_rates'].append(current_lr)
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                self.results['best_epoch'] = epoch
                self.results['best_val_loss'] = best_val_loss
                self.results['best_val_acc'] = best_val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                logger.info(f"Epoch {epoch+1}: New best model (val_loss={val_loss:.4f}, val_acc={val_acc:.4f})")
            else:
                patience_counter += 1
            
            logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']}: "
                       f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                       f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, lr={current_lr:.2e}")
            
            # Early stopping
            if patience_counter >= self.config['patience']:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Loaded best model from epoch {self.results['best_epoch']+1}")
        
        return self.results
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on test data
        
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Evaluating model on test data...")
        
        self.model.eval()
        all_preds = []
        all_targets = []
        test_loss = 0
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.compute_loss(outputs, targets)
                test_loss += loss.item()
                
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        test_loss /= len(self.test_loader)
        test_acc = accuracy_score(all_targets, all_preds)
        
        # Generate classification report
        class_report = classification_report(
            all_targets, all_preds, 
            output_dict=True, zero_division=0
        )
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(all_targets, all_preds)
        
        evaluation_results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': all_preds,
            'targets': all_targets
        }
        
        logger.info(f"Test Results:")
        logger.info(f"  Test Loss: {test_loss:.4f}")
        logger.info(f"  Test Accuracy: {test_acc:.4f}")
        
        return evaluation_results
    
    def save_results(self, filepath: str):
        """Save training and evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_to_save = {
            'timestamp': timestamp,
            'config': self.config,
            'training_results': self.results,
            'model_info': {
                'total_params': sum(p.numel() for p in self.model.parameters()),
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        if self.model is None:
            return {}
        
        return {
            'model_type': type(self.model).__name__,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'device': str(self.device),
            'config': self.config
        }
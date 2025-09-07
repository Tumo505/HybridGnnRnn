"""
Comprehensive Hyperparameter Optimization for GNN Model
"""

import torch
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
import logging
from typing import Dict, Any, Tuple
import json
import os
from datetime import datetime

# Import our modules
import sys
sys.path.append('/Users/tumokgabeng/Projects/HybridGnnRnn/src')

from data.enhanced_cardiac_loader import create_enhanced_cardiac_loaders
from models.enhanced_spatial_gnn import EnhancedSpatialGNN
from models.spatial_gnn import SpatialGNN

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """Comprehensive hyperparameter optimization using Optuna"""
    
    def __init__(self, 
                 data_path: str = '/Users/tumokgabeng/Projects/HybridGnnRnn/data',
                 n_trials: int = 100,
                 cv_folds: int = 3,
                 device: str = None):
        
        self.data_path = data_path
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load data once
        self.train_loader, self.val_loader, self.test_loader, self.dataset_info = \
            create_enhanced_cardiac_loaders(data_path, batch_size=4, num_workers=0)
        
        logger.info(f"Dataset info: {self.dataset_info}")
        logger.info(f"Using device: {self.device}")
    
    def create_model(self, trial: optuna.trial.Trial, model_type: str = 'enhanced') -> torch.nn.Module:
        """Create model with trial-suggested hyperparameters"""
        
        # Common hyperparameters
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        num_layers = trial.suggest_int('num_layers', 2, 5)
        
        if model_type == 'enhanced':
            # Enhanced model specific parameters
            num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
            use_residual = trial.suggest_categorical('use_residual', [True, False])
            use_attention = trial.suggest_categorical('use_attention', [True, False])
            attention_dropout = trial.suggest_float('attention_dropout', 0.1, 0.3)
            
            model = EnhancedSpatialGNN(
                input_dim=self.dataset_info['num_features'],
                hidden_dim=hidden_dim,
                output_dim=self.dataset_info['num_classes'],
                num_layers=num_layers,
                dropout=dropout,
                num_heads=num_heads,
                use_residual=use_residual,
                use_attention=use_attention,
                attention_dropout=attention_dropout
            )
        else:
            # Standard model
            model = SpatialGNN(
                input_dim=self.dataset_info['num_features'],
                hidden_dim=hidden_dim,
                output_dim=self.dataset_info['num_classes'],
                num_layers=num_layers,
                dropout=dropout
            )
        
        return model.to(self.device)
    
    def create_optimizer(self, trial: optuna.trial.Trial, model: torch.nn.Module):
        """Create optimizer with trial-suggested hyperparameters"""
        
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            beta1 = trial.suggest_float('beta1', 0.8, 0.95)
            beta2 = trial.suggest_float('beta2', 0.9, 0.999)
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay,
                betas=(beta1, beta2)
            )
        else:  # SGD
            momentum = trial.suggest_float('momentum', 0.8, 0.95)
            optimizer = torch.optim.SGD(
                model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay, 
                momentum=momentum
            )
        
        return optimizer
    
    def create_scheduler(self, trial: optuna.trial.Trial, optimizer):
        """Create learning rate scheduler with trial-suggested hyperparameters"""
        
        scheduler_name = trial.suggest_categorical('scheduler', ['cosine', 'step', 'exponential', 'none'])
        
        if scheduler_name == 'cosine':
            T_max = trial.suggest_int('T_max', 10, 100)
            eta_min = trial.suggest_float('eta_min', 1e-7, 1e-4, log=True)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=eta_min
            )
        elif scheduler_name == 'step':
            step_size = trial.suggest_int('step_size', 10, 50)
            gamma = trial.suggest_float('gamma', 0.1, 0.9)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_name == 'exponential':
            gamma = trial.suggest_float('exp_gamma', 0.9, 0.99)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        else:
            scheduler = None
        
        return scheduler
    
    def create_loss_function(self, trial: optuna.trial.Trial):
        """Create loss function with trial-suggested hyperparameters"""
        
        loss_name = trial.suggest_categorical('loss', ['focal', 'label_smoothing', 'cross_entropy'])
        
        if loss_name == 'focal':
            alpha = trial.suggest_float('focal_alpha', 0.5, 2.0)
            gamma = trial.suggest_float('focal_gamma', 1.0, 3.0)
            criterion = FocalLoss(alpha=alpha, gamma=gamma)
        elif loss_name == 'label_smoothing':
            smoothing = trial.suggest_float('label_smoothing', 0.05, 0.2)
            criterion = LabelSmoothingLoss(smoothing=smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        
        return criterion
    
    def train_and_evaluate(self, trial: optuna.trial.Trial, model_type: str = 'enhanced') -> float:
        """Train model and return validation accuracy"""
        
        try:
            # Create model components
            model = self.create_model(trial, model_type)
            optimizer = self.create_optimizer(trial, model)
            scheduler = self.create_scheduler(trial, optimizer)
            criterion = self.create_loss_function(trial)
            
            # Training hyperparameters
            max_epochs = trial.suggest_int('max_epochs', 20, 100)
            early_stopping_patience = trial.suggest_int('early_stopping_patience', 5, 15)
            grad_clip = trial.suggest_float('grad_clip', 0.1, 2.0)
            
            # Training loop
            best_val_acc = 0.0
            patience_counter = 0
            
            for epoch in range(max_epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch in self.train_loader:
                    batch = batch.to(self.device)
                    
                    optimizer.zero_grad()
                    out = model(batch)
                    loss = criterion(out, batch.y)
                    
                    # Check for NaN
                    if torch.isnan(loss):
                        logger.warning(f"NaN loss detected in trial {trial.number}")
                        return 0.0  # Return poor score for NaN trials
                    
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(out.data, 1)
                    train_total += batch.y.size(0)
                    train_correct += (predicted == batch.y).sum().item()
                
                # Validation phase
                model.eval()
                val_correct = 0
                val_total = 0
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch in self.val_loader:
                        batch = batch.to(self.device)
                        out = model(batch)
                        loss = criterion(out, batch.y)
                        
                        if torch.isnan(loss):
                            continue
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(out.data, 1)
                        val_total += batch.y.size(0)
                        val_correct += (predicted == batch.y).sum().item()
                
                val_acc = val_correct / val_total if val_total > 0 else 0.0
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        break
                
                # Learning rate scheduling
                if scheduler is not None:
                    scheduler.step()
                
                # Report intermediate result for pruning
                trial.report(val_acc, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            return best_val_acc
            
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {str(e)}")
            return 0.0
    
    def optimize(self, model_type: str = 'enhanced') -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        
        # Create study
        study_name = f"gnn_optimization_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)
        )
        
        # Optimize
        logger.info(f"Starting optimization with {self.n_trials} trials...")
        study.optimize(
            lambda trial: self.train_and_evaluate(trial, model_type),
            n_trials=self.n_trials,
            callbacks=[self._trial_callback]
        )
        
        # Results
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Best validation accuracy: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Save results
        results = {
            'best_params': best_params,
            'best_value': best_value,
            'study_name': study_name,
            'model_type': model_type,
            'n_trials': len(study.trials),
            'optimization_time': datetime.now().isoformat()
        }
        
        # Save to file
        results_dir = '/Users/tumokgabeng/Projects/HybridGnnRnn/optimization_results'
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f'{study_name}_results.json')
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        
        return results
    
    def _trial_callback(self, study, trial):
        """Callback function for trial completion"""
        if trial.value is not None:
            logger.info(f"Trial {trial.number} completed with value: {trial.value:.4f}")

# Loss functions for optimization
class FocalLoss(torch.nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(torch.nn.Module):
    """Label Smoothing Loss"""
    
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        loss = -torch.sum(torch.log_softmax(pred, dim=1) * smooth_one_hot, dim=1)
        return loss.mean()

def main():
    """Main optimization function"""
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(
        n_trials=50,  # Reduced for initial testing
        cv_folds=3
    )
    
    # Optimize both model types
    logger.info("Starting hyperparameter optimization...")
    
    # Standard model optimization
    logger.info("Optimizing standard spatial GNN...")
    standard_results = optimizer.optimize(model_type='standard')
    
    # Enhanced model optimization
    logger.info("Optimizing enhanced spatial GNN...")
    enhanced_results = optimizer.optimize(model_type='enhanced')
    
    # Compare results
    logger.info("=== OPTIMIZATION RESULTS ===")
    logger.info(f"Standard GNN best accuracy: {standard_results['best_value']:.4f}")
    logger.info(f"Enhanced GNN best accuracy: {enhanced_results['best_value']:.4f}")
    
    if enhanced_results['best_value'] > standard_results['best_value']:
        logger.info("Enhanced GNN performs better!")
        best_model = 'enhanced'
        best_results = enhanced_results
    else:
        logger.info("Standard GNN performs better!")
        best_model = 'standard'
        best_results = standard_results
    
    logger.info(f"Best model type: {best_model}")
    logger.info(f"Best parameters: {best_results['best_params']}")
    
    return best_results

if __name__ == "__main__":
    main()

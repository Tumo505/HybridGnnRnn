"""
Training module for Spatial GNN models
Handles training loop, validation, and model evaluation for cardiomyocyte differentiation prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from pathlib import Path
import logging
import time
from typing import Dict, List, Optional, Tuple
import json
import wandb

from ..models.spatial_gnn import SpatialGNN
from ..utils.metrics import calculate_metrics

class GNNTrainer:
    """
    Trainer class for Spatial GNN models.
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(self,
                 model: SpatialGNN,
                 device: str = 'auto',
                 experiment_name: str = 'spatial_gnn_experiment',
                 log_dir: str = 'experiments',
                 use_wandb: bool = True,
                 wandb_project: str = 'hybrid-gnn-rnn',
                 wandb_config: dict = None):
        """
        Initialize the GNN trainer.
        
        Args:
            model: SpatialGNN model to train
            device: Device to use ('cuda', 'cpu', or 'auto')
            experiment_name: Name for the experiment
            log_dir: Directory for logging and checkpoints
            use_wandb: Whether to use Weights & Biases for tracking
            wandb_project: WandB project name
            wandb_config: Configuration dict for WandB
        """
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model = model.to(self.device)
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        
        # Initialize WandB if requested
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=experiment_name,
                config=wandb_config or {},
                reinit=True
            )
            wandb.watch(self.model)
        
        # Setup logging directories
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.log_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Setup tensorboard
        self.writer = SummaryWriter(self.log_dir / 'tensorboard')
        
        # Training history with additional metrics
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'train_r2': [],
            'val_loss': [],
            'val_acc': [],
            'val_r2': [],
            'learning_rate': [],
            'train_class_loss': [],  # Separate classification loss
            'train_reg_loss': [],    # Separate regression loss
            'val_class_loss': [],
            'val_reg_loss': []
        }
        
        # Early stopping parameters
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stop = False
        
        self.logger.info(f"Trainer initialized on {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if self.use_wandb:
            self.logger.info(f"WandB tracking enabled for project: {wandb_project}")
        
    def setup_training(self,
                      learning_rate: float = 1e-3,
                      weight_decay: float = 5e-4,  # Increased L2 regularization
                      classification_weight: float = 0.7,  # Adjusted weights
                      regression_weight: float = 0.3,
                      use_focal_loss: bool = True):  # Added focal loss option
        """
        Setup optimizers and loss functions with enhanced regularization.
        
        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for L2 regularization (increased)
            classification_weight: Weight for classification loss
            regression_weight: Weight for regression loss
            use_focal_loss: Whether to use focal loss for imbalanced classes
        """
        
        # Enhanced optimizer with stronger regularization
        self.optimizer = optim.AdamW(  # Changed to AdamW for better regularization
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Enhanced loss functions
        if use_focal_loss:
            # Focal loss for handling class imbalance
            self.classification_criterion = self._focal_loss
        else:
            # Add label smoothing for regularization
            self.classification_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        self.regression_criterion = nn.SmoothL1Loss()  # More robust than MSE
        
        # Loss weights
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        
        # Enhanced learning rate scheduler with early stopping
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,  # Less aggressive reduction
            patience=5,   # More responsive
            min_lr=1e-6,
            verbose=True
        )
        
        # Add cosine annealing for better convergence
        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        self.logger.info(f"Enhanced training setup complete - LR: {learning_rate}, WD: {weight_decay}")
        self.logger.info(f"Using AdamW optimizer with stronger regularization")
        
    def _focal_loss(self, pred, target, alpha=1.0, gamma=2.0):
        """Focal loss for handling class imbalance."""
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
        
    def compute_loss(self, 
                    classification_pred: torch.Tensor,
                    regression_pred: torch.Tensor,
                    classification_target: torch.Tensor,
                    regression_target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss for multi-task learning.
        
        Args:
            classification_pred: Classification predictions
            regression_pred: Regression predictions  
            classification_target: Classification targets
            regression_target: Regression targets
            
        Returns:
            Combined loss and individual loss components
        """
        
        # Classification loss - handle shape mismatch
        if classification_pred.size(0) > 0 and classification_target.size(0) > 0:
            # If we have node-level predictions but graph-level targets, aggregate
            if classification_pred.size(0) != classification_target.size(0):
                if hasattr(data, 'batch'):
                    # Use batch pooling to aggregate node predictions to graph level
                    from torch_geometric.utils import global_mean_pool
                    classification_pred = global_mean_pool(classification_pred, data.batch)
                else:
                    # For single graph, use mean pooling
                    classification_pred = classification_pred.mean(dim=0, keepdim=True)
                    
            class_loss = self.classification_criterion(classification_pred, classification_target)
        else:
            class_loss = torch.tensor(0.0, device=self.device)
            
        # Regression loss
        if regression_pred.size(0) > 0 and regression_target.size(0) > 0:
            reg_loss = self.regression_criterion(regression_pred.squeeze(), regression_target.float())
        else:
            reg_loss = torch.tensor(0.0, device=self.device)
            
        # Combined loss
        total_loss = (self.classification_weight * class_loss + 
                     self.regression_weight * reg_loss)
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'classification_loss': class_loss.item(),
            'regression_loss': reg_loss.item()
        }
        
        return total_loss, loss_dict
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Training metrics for the epoch
        """
        
        self.model.train()
        total_loss = 0.0
        total_class_loss = 0.0
        total_reg_loss = 0.0
        
        all_class_preds = []
        all_class_targets = []
        all_reg_preds = []
        all_reg_targets = []
        
        num_batches = len(train_loader)
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            classification_pred, regression_pred = self.model(data)
            
            # Create targets from batch data
            if hasattr(data, 'y_class') and hasattr(data, 'y_reg'):
                # Use enhanced targets from our data loader
                classification_target = data.y_class
                regression_target = data.y_reg
            elif hasattr(data, 'y') and data.y is not None:
                # Use the first element as graph-level target
                if data.y.dim() > 0 and data.y.numel() > 1:
                    classification_target = data.y[:1]  # Take first element as graph label
                else:
                    classification_target = data.y
                    
                # Create regression target as random efficiency scores for now
                regression_target = torch.rand(1, device=self.device)
            else:
                # Graph-level classification target (1 target per graph)
                classification_target = torch.randint(0, self.model.num_classes, 
                                                    (1,), 
                                                    device=self.device)
                # Graph-level regression target (1 target per graph)
                regression_target = torch.rand(1, device=self.device)
            
            # Compute loss
            loss, loss_dict = self.compute_loss(
                classification_pred, regression_pred,
                classification_target, regression_target
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss_dict['total_loss']
            total_class_loss += loss_dict['classification_loss']
            total_reg_loss += loss_dict['regression_loss']
            
            # Store predictions for metrics calculation
            all_class_preds.append(torch.argmax(classification_pred, dim=1).cpu())
            all_class_targets.append(classification_target.cpu())
            all_reg_preds.append(regression_pred.squeeze().cpu())
            all_reg_targets.append(regression_target.cpu())
            
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        avg_class_loss = total_class_loss / num_batches
        avg_reg_loss = total_reg_loss / num_batches
        
        # Concatenate all predictions
        all_class_preds = torch.cat(all_class_preds)
        all_class_targets = torch.cat(all_class_targets)
        
        # Handle regression predictions (ensure they are at least 1D)
        reg_preds_list = []
        reg_targets_list = []
        for pred, target in zip(all_reg_preds, all_reg_targets):
            if pred.dim() == 0:
                pred = pred.unsqueeze(0)
            if target.dim() == 0:
                target = target.unsqueeze(0)
            reg_preds_list.append(pred)
            reg_targets_list.append(target)
            
        all_reg_preds = torch.cat(reg_preds_list)
        all_reg_targets = torch.cat(reg_targets_list)
        
        # Calculate accuracy and R2
        accuracy = accuracy_score(all_class_targets.numpy(), all_class_preds.numpy())
        r2 = r2_score(all_reg_targets.numpy(), all_reg_preds.detach().numpy())
        
        metrics = {
            'loss': avg_loss,
            'classification_loss': avg_class_loss,
            'regression_loss': avg_reg_loss,
            'accuracy': accuracy,
            'r2_score': r2
        }
        
        return metrics
        
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Validation metrics for the epoch
        """
        
        self.model.eval()
        total_loss = 0.0
        total_class_loss = 0.0
        total_reg_loss = 0.0
        
        all_class_preds = []
        all_class_targets = []
        all_reg_preds = []
        all_reg_targets = []
        
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                
                # Forward pass
                classification_pred, regression_pred = self.model(data)
                
                # Create targets from batch data
                if hasattr(data, 'y_class') and hasattr(data, 'y_reg'):
                    # Use enhanced targets from our data loader
                    classification_target = data.y_class
                    regression_target = data.y_reg
                elif hasattr(data, 'y') and data.y is not None:
                    # Use the first element as graph-level target
                    if data.y.dim() > 0 and data.y.numel() > 1:
                        classification_target = data.y[:1]  # Take first element as graph label
                    else:
                        classification_target = data.y
                        
                    # Create regression target as random efficiency scores for now
                    regression_target = torch.rand(1, device=self.device)
                else:
                    # Graph-level classification target (1 target per graph)
                    classification_target = torch.randint(0, self.model.num_classes, 
                                                        (1,), 
                                                        device=self.device)
                    # Graph-level regression target (1 target per graph)
                    regression_target = torch.rand(1, device=self.device)
                
                # Compute loss
                loss, loss_dict = self.compute_loss(
                    classification_pred, regression_pred,
                    classification_target, regression_target
                )
                
                # Accumulate metrics
                total_loss += loss_dict['total_loss']
                total_class_loss += loss_dict['classification_loss'] 
                total_reg_loss += loss_dict['regression_loss']
                
                # Store predictions
                all_class_preds.append(torch.argmax(classification_pred, dim=1).cpu())
                all_class_targets.append(classification_target.cpu())
                all_reg_preds.append(regression_pred.squeeze().cpu())
                all_reg_targets.append(regression_target.cpu())
        
        # Calculate metrics
        avg_loss = total_loss / num_batches
        avg_class_loss = total_class_loss / num_batches
        avg_reg_loss = total_reg_loss / num_batches
        
        # Concatenate predictions
        all_class_preds = torch.cat(all_class_preds)
        all_class_targets = torch.cat(all_class_targets)
        
        # Handle regression predictions (ensure they are at least 1D)
        reg_preds_list = []
        reg_targets_list = []
        for pred, target in zip(all_reg_preds, all_reg_targets):
            if pred.dim() == 0:
                pred = pred.unsqueeze(0)
            if target.dim() == 0:
                target = target.unsqueeze(0)
            reg_preds_list.append(pred)
            reg_targets_list.append(target)
            
        all_reg_preds = torch.cat(reg_preds_list)
        all_reg_targets = torch.cat(reg_targets_list)
        
        # Calculate accuracy and R2
        accuracy = accuracy_score(all_class_targets.numpy(), all_class_preds.numpy())
        r2 = r2_score(all_reg_targets.numpy(), all_reg_preds.numpy())
        
        metrics = {
            'loss': avg_loss,
            'classification_loss': avg_class_loss,
            'regression_loss': avg_reg_loss,
            'accuracy': accuracy,
            'r2_score': r2
        }
        
        return metrics
        
    def train(self,
             train_loader: DataLoader,
             val_loader: Optional[DataLoader] = None,
             num_epochs: int = 100,
             save_every: int = 10,
             early_stopping_patience: int = 15):  # Reduced patience
        """
        Enhanced training loop with early stopping and regularization.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of training epochs
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Early stopping patience (reduced)
        """
        
        self.logger.info(f"Starting enhanced training for {num_epochs} epochs...")
        self.logger.info(f"Early stopping patience: {early_stopping_patience}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader)
                val_loss = val_metrics['loss']
                
                # Learning rate scheduling (both schedulers)
                self.scheduler.step(val_loss)
                self.cosine_scheduler.step()
                
                # Enhanced early stopping check
                if val_loss < self.best_val_loss:
                    improvement = (self.best_val_loss - val_loss) / self.best_val_loss
                    if improvement > 0.01:  # Require 1% improvement
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        self.save_checkpoint(epoch, is_best=True)
                        self.logger.info(f"New best validation loss: {val_loss:.6f} (improvement: {improvement:.2%})")
                    else:
                        self.patience_counter += 1
                else:
                    self.patience_counter += 1
                    
                # Check for early stopping
                if self.patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
                    self.early_stop = True
                    break
                    
            else:
                val_metrics = None
                val_loss = train_metrics['loss']
                self.scheduler.step(val_loss)
                self.cosine_scheduler.step()
            
            # Log metrics
            self._log_epoch_metrics(epoch, train_metrics, val_metrics)
            
            # Save regular checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch)
                
            epoch_time = time.time() - start_time
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
            
        self.logger.info("Training completed!")
        self.writer.close()
        
    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Optional[Dict]):
        """Log metrics for the current epoch."""
        
        # Update history
        self.train_history['epoch'].append(epoch)
        self.train_history['train_loss'].append(train_metrics['loss'])
        self.train_history['train_acc'].append(train_metrics['accuracy'])
        self.train_history['train_r2'].append(train_metrics['r2_score'])
        self.train_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        
        # Tensorboard logging
        self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
        self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
        self.writer.add_scalar('R2/Train', train_metrics['r2_score'], epoch)
        self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
        
        # WandB logging
        if self.use_wandb:
            wandb_log = {
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'train/r2_score': train_metrics['r2_score'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
        
        log_msg = f"Epoch {epoch+1} - Train Loss: {train_metrics['loss']:.4f}, " \
                 f"Train Acc: {train_metrics['accuracy']:.3f}, Train R2: {train_metrics['r2_score']:.3f}"
        
        if val_metrics is not None:
            self.train_history['val_loss'].append(val_metrics['loss'])
            self.train_history['val_acc'].append(val_metrics['accuracy'])
            self.train_history['val_r2'].append(val_metrics['r2_score'])
            
            self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/Val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('R2/Val', val_metrics['r2_score'], epoch)
            
            # Add validation metrics to WandB log
            if self.use_wandb:
                wandb_log.update({
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/r2_score': val_metrics['r2_score']
                })
            
            log_msg += f", Val Loss: {val_metrics['loss']:.4f}, " \
                      f"Val Acc: {val_metrics['accuracy']:.3f}, Val R2: {val_metrics['r2_score']:.3f}"
        
        # Log to WandB
        if self.use_wandb:
            wandb.log(wandb_log)
        
        self.logger.info(log_msg)
        
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'model_config': {
                'input_dim': self.model.input_dim,
                'hidden_dims': self.model.hidden_dims,
                'output_dim': self.model.output_dim,
                'num_classes': self.model.num_classes,
                'conv_type': self.model.conv_type,
                'use_attention': self.model.use_attention,
                'dropout': self.model.dropout,
                'use_batch_norm': self.model.use_batch_norm
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at epoch {epoch+1}")
            
        # Save training history as JSON
        history_path = self.log_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
            
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint and return the epoch number."""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_history = checkpoint['train_history']
        
        epoch = checkpoint['epoch']
        self.logger.info(f"Loaded checkpoint from epoch {epoch+1}")
        
        return epoch


def create_data_loaders(data: Data, 
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       batch_size: int = 1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/validation/test data loaders from a single graph.
    
    Args:
        data: PyTorch Geometric Data object
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        batch_size: Batch size (usually 1 for single large graphs)
        
    Returns:
        Train, validation, and test data loaders
    """
    
    # For now, we'll create data loaders with the same graph
    # In practice, you would split the graph or use different graphs
    train_loader = DataLoader([data], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader([data], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader([data], batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the training module
    print("Testing GNN training module...")
    
    from ..models.spatial_gnn import create_spatial_gnn
    from torch_geometric.data import Data
    
    # Create test model and data
    config = {
        'input_dim': 100,
        'hidden_dims': [64, 32],
        'output_dim': 16,
        'num_classes': 3,
        'conv_type': 'GCN'
    }
    
    model = create_spatial_gnn(config)
    
    # Create test data
    num_nodes = 50
    x = torch.randn(num_nodes, 100)
    edge_index = torch.randint(0, num_nodes, (2, 100))
    y = torch.randint(0, 3, (1,))  # Graph-level label
    
    data = Data(x=x, edge_index=edge_index, y=y)
    train_loader, val_loader, _ = create_data_loaders(data, batch_size=1)
    
    # Test trainer
    trainer = GNNTrainer(model, device='cpu', experiment_name='test_run')
    trainer.setup_training(learning_rate=1e-3)
    
    try:
        # Short training test
        trainer.train(train_loader, val_loader, num_epochs=2, save_every=1)
        print("✓ Training test successful!")
        
    except Exception as e:
        print(f"✗ Training test failed: {e}")
        import traceback
        traceback.print_exc()

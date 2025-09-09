"""
Full-Scale Advanced RNN Training Script
Optimized for NVIDIA RTX 5070 with 12GB VRAM

This script trains the advanced temporal RNN model for cardiomyocyte differentiation prediction
using the complete GSE175634 temporal dataset with full GPU utilization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from pathlib import Path
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
import sys
sys.path.append(str(Path(__file__).parent))

from src.models.advanced_temporal_rnn import AdvancedTemporalRNN, TemporalRNNLoss, create_advanced_rnn_model
from src.data_processing.temporal_data_loader import create_temporal_dataloaders, collate_temporal_batch


class AdvancedRNNTrainer:
    """
    Advanced trainer for temporal RNN with full GPU optimization.
    Features mixed precision training, advanced scheduling, and comprehensive monitoring.
    """
    
    def __init__(
        self,
        model: AdvancedTemporalRNN,
        train_loader,
        val_loader,
        test_loader,
        config: Dict,
        device: torch.device
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        # Initialize training components
        self._setup_training()
        self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_r2 = -float('inf')
        self.early_stopping_counter = 0
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'val_r2': [],
            'train_diff_loss': [], 'train_cell_loss': [],
            'val_diff_loss': [], 'val_cell_loss': []
        }
        
    def _setup_training(self):
        """Initialize optimizers, schedulers, and loss functions."""
        # Loss function
        self.criterion = TemporalRNNLoss(
            differentiation_weight=self.config['loss_weights']['differentiation'],
            cell_type_weight=self.config['loss_weights']['cell_type'],
            smoothness_weight=self.config['loss_weights']['smoothness']
        )
        
        # Optimizer with different learning rates for different components
        embedding_params = []
        lstm_params = []
        attention_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'embedding' in name:
                embedding_params.append(param)
            elif 'lstm' in name:
                lstm_params.append(param)
            elif 'attention' in name:
                attention_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = []
        if embedding_params:
            param_groups.append({
                'params': embedding_params,
                'lr': self.config['learning_rate'] * 0.5,
                'weight_decay': self.config['weight_decay'] * 2
            })
        if lstm_params:
            param_groups.append({
                'params': lstm_params,
                'lr': self.config['learning_rate'],
                'weight_decay': self.config['weight_decay']
            })
        if attention_params:
            param_groups.append({
                'params': attention_params,
                'lr': self.config['learning_rate'] * 1.5,
                'weight_decay': self.config['weight_decay'] * 0.5
            })
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.config['learning_rate'],
                'weight_decay': self.config['weight_decay']
            })
        
        # Fallback to all parameters if grouping failed
        if not param_groups:
            param_groups = [{'params': self.model.parameters()}]
        
        if self.config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                param_groups,
                betas=self.config['betas'],
                eps=self.config['eps']
            )
        elif self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(param_groups)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']}")
        
        # Learning rate scheduler
        if self.config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config['scheduler_params']['T_0'],
                T_mult=self.config['scheduler_params']['T_mult'],
                eta_min=self.config['scheduler_params']['eta_min']
            )
        elif self.config['scheduler'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config['scheduler_params']['factor'],
                patience=self.config['scheduler_params']['patience'],
                verbose=True
            )
        else:
            self.scheduler = None
        
        # Mixed precision scaler for RTX 5070 optimization
        self.scaler = GradScaler() if self.config['mixed_precision'] else None
        
        # Gradient clipping
        self.max_grad_norm = self.config['max_grad_norm']
        
    def _setup_logging(self):
        """Setup logging and monitoring."""
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"advanced_rnn_{timestamp}"
        self.log_dir = Path(self.config['log_dir']) / self.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Tensorboard logging
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Save configuration
        with open(self.log_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Experiment: {self.experiment_name}")
        print(f"Log directory: {self.log_dir}")
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_diff_loss = 0
        total_cell_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            sequences = batch['sequences'].to(self.device)
            diff_targets = batch['differentiation_efficiency'].to(self.device)
            cell_targets = batch['cell_type'].to(self.device)
            lengths = batch['lengths'].to(self.device) if 'lengths' in batch else None
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    predictions = self.model(sequences, lengths)
                    
                    targets = {
                        'differentiation_efficiency': diff_targets,
                        'cell_type': cell_targets.squeeze()
                    }
                    
                    losses = self.criterion(predictions, targets)
                    loss = losses['total_loss']
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(sequences, lengths)
                
                targets = {
                    'differentiation_efficiency': diff_targets,
                    'cell_type': cell_targets.squeeze()
                }
                
                losses = self.criterion(predictions, targets)
                loss = losses['total_loss']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            total_diff_loss += losses['differentiation_loss'].item()
            total_cell_loss += losses['cell_type_loss'].item()
            num_batches += 1
            
            # Log batch statistics
            if batch_idx % self.config['log_interval'] == 0:
                print(f'Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}: '
                      f'Loss={loss.item():.4f}, '
                      f'Diff={losses["differentiation_loss"].item():.4f}, '
                      f'Cell={losses["cell_type_loss"].item():.4f}')
        
        # Calculate average losses
        avg_loss = total_loss / num_batches
        avg_diff_loss = total_diff_loss / num_batches
        avg_cell_loss = total_cell_loss / num_batches
        
        return {
            'loss': avg_loss,
            'diff_loss': avg_diff_loss,
            'cell_loss': avg_cell_loss
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_diff_loss = 0
        total_cell_loss = 0
        all_predictions = []
        all_targets = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                sequences = batch['sequences'].to(self.device)
                diff_targets = batch['differentiation_efficiency'].to(self.device)
                cell_targets = batch['cell_type'].to(self.device)
                lengths = batch['lengths'].to(self.device) if 'lengths' in batch else None
                
                # Forward pass
                predictions = self.model(sequences, lengths)
                
                targets = {
                    'differentiation_efficiency': diff_targets,
                    'cell_type': cell_targets.squeeze()
                }
                
                losses = self.criterion(predictions, targets)
                
                # Update statistics
                total_loss += losses['total_loss'].item()
                total_diff_loss += losses['differentiation_loss'].item()
                total_cell_loss += losses['cell_type_loss'].item()
                num_batches += 1
                
                # Store predictions for R² calculation
                all_predictions.append(predictions['differentiation_efficiency'].cpu())
                all_targets.append(diff_targets.cpu())
        
        # Calculate average losses
        avg_loss = total_loss / num_batches
        avg_diff_loss = total_diff_loss / num_batches
        avg_cell_loss = total_cell_loss / num_batches
        
        # Calculate R² score
        predictions_concat = torch.cat(all_predictions, dim=0).numpy()
        targets_concat = torch.cat(all_targets, dim=0).numpy()
        
        # R² calculation
        ss_res = np.sum((targets_concat - predictions_concat) ** 2)
        ss_tot = np.sum((targets_concat - np.mean(targets_concat)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'loss': avg_loss,
            'diff_loss': avg_diff_loss,
            'cell_loss': avg_cell_loss,
            'r2': r2_score
        }
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config['num_epochs']} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training on: {self.device}")
        
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_r2'].append(val_metrics['r2'])
            self.training_history['train_diff_loss'].append(train_metrics['diff_loss'])
            self.training_history['train_cell_loss'].append(train_metrics['cell_loss'])
            self.training_history['val_diff_loss'].append(val_metrics['diff_loss'])
            self.training_history['val_cell_loss'].append(val_metrics['cell_loss'])
            
            # Logging
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start_time
            
            print(f"Epoch {epoch+1}/{self.config['num_epochs']} ({epoch_time:.1f}s):")
            print(f"  Train Loss: {train_metrics['loss']:.4f} "
                  f"(Diff: {train_metrics['diff_loss']:.4f}, Cell: {train_metrics['cell_loss']:.4f})")
            print(f"  Val Loss: {val_metrics['loss']:.4f} "
                  f"(Diff: {val_metrics['diff_loss']:.4f}, Cell: {val_metrics['cell_loss']:.4f})")
            print(f"  Val R²: {val_metrics['r2']:.4f}")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # Tensorboard logging
            self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Train/DiffLoss', train_metrics['diff_loss'], epoch)
            self.writer.add_scalar('Train/CellLoss', train_metrics['cell_loss'], epoch)
            self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Val/DiffLoss', val_metrics['diff_loss'], epoch)
            self.writer.add_scalar('Val/CellLoss', val_metrics['cell_loss'], epoch)
            self.writer.add_scalar('Val/R2', val_metrics['r2'], epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Save best model
            if val_metrics['r2'] > self.best_val_r2:
                self.best_val_r2 = val_metrics['r2']
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model.pth', epoch, val_metrics)
                self.early_stopping_counter = 0
                print(f"  *** New best model saved (R² = {val_metrics['r2']:.4f}) ***")
            else:
                self.early_stopping_counter += 1
            
            # Early stopping
            if (self.config['early_stopping_patience'] > 0 and 
                self.early_stopping_counter >= self.config['early_stopping_patience']):
                print(f"Early stopping after {epoch+1} epochs")
                break
            
            # Save regular checkpoint
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_metrics)
        
        # Final evaluation
        print("\\nTraining completed!")
        print(f"Total training time: {(time.time() - start_time) / 3600:.2f} hours")
        print(f"Best validation R²: {self.best_val_r2:.4f}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Test evaluation
        self.load_checkpoint('best_model.pth')
        test_metrics = self.test()
        print(f"Test R²: {test_metrics['r2']:.4f}")
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        
        # Save training history
        self.save_training_history()
        self.writer.close()
        
    def test(self) -> Dict[str, float]:
        """Test the model on the test set."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                sequences = batch['sequences'].to(self.device)
                diff_targets = batch['differentiation_efficiency'].to(self.device)
                cell_targets = batch['cell_type'].to(self.device)
                lengths = batch['lengths'].to(self.device) if 'lengths' in batch else None
                
                predictions = self.model(sequences, lengths)
                
                targets = {
                    'differentiation_efficiency': diff_targets,
                    'cell_type': cell_targets.squeeze()
                }
                
                losses = self.criterion(predictions, targets)
                total_loss += losses['total_loss'].item()
                num_batches += 1
                
                all_predictions.append(predictions['differentiation_efficiency'].cpu())
                all_targets.append(diff_targets.cpu())
        
        # Calculate metrics
        avg_loss = total_loss / num_batches
        predictions_concat = torch.cat(all_predictions, dim=0).numpy()
        targets_concat = torch.cat(all_targets, dim=0).numpy()
        
        ss_res = np.sum((targets_concat - predictions_concat) ** 2)
        ss_tot = np.sum((targets_concat - np.mean(targets_concat)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {'loss': avg_loss, 'r2': r2_score}
    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics,
            'config': self.config,
            'training_history': self.training_history
        }
        torch.save(checkpoint, self.log_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.log_dir / filename, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    def save_training_history(self):
        """Save training history to CSV."""
        df = pd.DataFrame(self.training_history)
        df.to_csv(self.log_dir / 'training_history.csv', index=False)


def get_default_config() -> Dict:
    """Get default training configuration optimized for RTX 5070."""
    return {
        # Model architecture
        'model': {
            'hidden_dim': 1024,
            'num_layers': 4,
            'num_attention_heads': 16,
            'embedding_dim': 512,
            'dropout': 0.15,
            'bidirectional': True,
            'use_attention': True,
            'use_residual': True
        },
        
        # Training parameters
        'num_epochs': 200,
        'batch_size': 64,  # Optimized for RTX 5070
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'optimizer': 'adamw',
        
        # Scheduler
        'scheduler': 'cosine',
        'scheduler_params': {
            'T_0': 20,
            'T_mult': 2,
            'eta_min': 1e-6,
            'factor': 0.5,
            'patience': 10
        },
        
        # Loss weights
        'loss_weights': {
            'differentiation': 1.0,
            'cell_type': 0.3,
            'smoothness': 0.1
        },
        
        # Regularization
        'max_grad_norm': 1.0,
        'mixed_precision': True,  # Enable for RTX 5070
        'early_stopping_patience': 30,
        
        # Data
        'n_top_genes': 3000,
        'num_workers': 8,  # Optimize for your CPU
        'pin_memory': True,
        
        # Logging
        'log_interval': 50,
        'save_interval': 10,
        'log_dir': 'experiments'
    }


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Advanced Temporal RNN')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to GSE175634 temporal data')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to custom config JSON file')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    
    args = parser.parse_args()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load configuration
    if args.config_path:
        with open(args.config_path, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    print("Configuration:")
    print(json.dumps(config, indent=2))
    
    # Create data loaders
    print("\\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_temporal_dataloaders(
        data_path=args.data_path,
        batch_size=config['batch_size'],
        num_workers=0,  # Set to 0 to avoid Windows multiprocessing issues
        pin_memory=False,  # Disable pin_memory to avoid Windows issues
        n_top_genes=config['n_top_genes']
    )
    
    # Get input dimension from first batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch['sequences'].shape[-1]
    print(f"Input dimension: {input_dim}")
    
    # Create model
    print("\\nCreating model...")
    model = create_advanced_rnn_model(input_dim=input_dim, config=config['model'])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = AdvancedRNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

"""
Comprehensive training pipeline for the Hybrid GNN-RNN model.
Includes data loading, training loops, validation, and optimization.
"""

import os
import sys
import json
import yaml
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import wandb

# Import our models and utilities
sys.path.append('/Users/tumokgabeng/Projects/HybridGnnRnn/src')
from models.hybrid_model import HybridGNNRNN, LightweightHybridGNNRNN
from utils.memory_utils import MemoryMonitor
from data.preprocessing import CardiomyocyteDataPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    
    # Model architecture
    model_type: str = "lightweight"  # "full" or "lightweight"
    node_feature_dim: int = 2000
    gnn_hidden_dim: int = 128
    rnn_hidden_dim: int = 128
    fusion_dim: int = 256
    num_gnn_layers: int = 2
    num_rnn_layers: int = 1
    dropout: float = 0.2
    
    # Training parameters
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-4
    
    # Data parameters
    sequence_length: int = 7
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Optimization
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    gradient_clipping: float = 1.0
    accumulation_steps: int = 1
    
    # Regularization
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    
    # Loss weights
    efficiency_weight: float = 1.0
    maturation_weight: float = 1.0
    uncertainty_weight: float = 0.1
    
    # Memory optimization
    use_amp: bool = True  # Automatic Mixed Precision
    use_checkpoint: bool = True
    max_memory_gb: float = 16.0
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    
    # Logging and checkpointing
    log_interval: int = 10
    save_interval: int = 50
    use_wandb: bool = False  # Disabled by default
    project_name: str = "HybridGNNRNN-Cardiomyocyte"
    experiment_name: str = None
    
    # Paths
    data_dir: str = "/Users/tumokgabeng/Projects/HybridGnnRnn/data/processed"
    output_dir: str = "/Users/tumokgabeng/Projects/HybridGnnRnn/experiments"
    checkpoint_dir: str = None
    
    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = f"hybrid_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if self.checkpoint_dir is None:
            self.checkpoint_dir = os.path.join(self.output_dir, self.experiment_name, "checkpoints")
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


class CardiomyocyteDataset(Dataset):
    """Dataset class for cardiomyocyte differentiation data."""
    
    def __init__(
        self,
        spatial_data: List[Data],
        temporal_data: Dict[str, torch.Tensor],
        config: TrainingConfig,
        split: str = "train"
    ):
        self.spatial_data = spatial_data
        self.temporal_sequences = temporal_data['sequences']
        self.temporal_masks = temporal_data['masks']
        self.temporal_labels = temporal_data['labels']
        self.config = config
        self.split = split
        
        # Ensure data lengths match
        min_len = min(len(self.spatial_data), len(self.temporal_sequences))
        self.spatial_data = self.spatial_data[:min_len]
        self.temporal_sequences = self.temporal_sequences[:min_len]
        self.temporal_masks = self.temporal_masks[:min_len]
        self.temporal_labels = self.temporal_labels[:min_len]
        
        logger.info(f"{split} dataset: {len(self)} samples")
    
    def __len__(self):
        return len(self.spatial_data)
    
    def __getitem__(self, idx):
        spatial = self.spatial_data[idx]
        temporal_seq = self.temporal_sequences[idx]
        temporal_mask = self.temporal_masks[idx]
        temporal_label = self.temporal_labels[idx]
        
        # Create targets
        efficiency_target = temporal_label  # Normalized progress
        maturation_target = torch.clamp(torch.round(temporal_label * 4), 0, 4).long()  # 5 classes
        
        return {
            'spatial': spatial,
            'temporal_sequence': temporal_seq,
            'temporal_mask': temporal_mask,
            'efficiency_target': efficiency_target,
            'maturation_target': maturation_target
        }


class HybridModelTrainer:
    """Comprehensive trainer for the Hybrid GNN-RNN model."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(max_memory_gb=config.max_memory_gb)
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize loss functions
        self.criterion_efficiency = nn.MSELoss()
        self.criterion_maturation = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        
        # Initialize AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp and torch.cuda.is_available() else None
        
        # Initialize logging
        self.writer = SummaryWriter(log_dir=os.path.join(config.output_dir, config.experiment_name, "tensorboard"))
        
        if config.use_wandb:
            wandb.init(
                project=config.project_name,
                name=config.experiment_name,
                config=asdict(config)
            )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_efficiency_loss': [],
            'train_maturation_loss': [],
            'val_efficiency_loss': [],
            'val_maturation_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
    
    def _create_model(self):
        """Create the hybrid model based on configuration."""
        if self.config.model_type == "lightweight":
            model = LightweightHybridGNNRNN(
                node_feature_dim=self.config.node_feature_dim,
                gnn_hidden_dim=self.config.gnn_hidden_dim,
                rnn_hidden_dim=self.config.rnn_hidden_dim,
                fusion_dim=self.config.fusion_dim,
                num_gnn_layers=self.config.num_gnn_layers,
                num_rnn_layers=self.config.num_rnn_layers,
                dropout=self.config.dropout,
                prediction_tasks=['multitask'],
                use_uncertainty=True,
                memory_efficient=True,
                use_checkpoint=self.config.use_checkpoint,
                use_positional_encoding=False  # Disable to fix dimension mismatch
            )
        else:
            model = HybridGNNRNN(
                node_feature_dim=self.config.node_feature_dim,
                gnn_hidden_dim=self.config.gnn_hidden_dim,
                rnn_hidden_dim=self.config.rnn_hidden_dim,
                fusion_dim=self.config.fusion_dim,
                num_gnn_layers=self.config.num_gnn_layers,
                num_rnn_layers=self.config.num_rnn_layers,
                dropout=self.config.dropout,
                prediction_tasks=['multitask'],
                use_uncertainty=True,
                memory_efficient=True,
                use_checkpoint=self.config.use_checkpoint,
                use_positional_encoding=False  # Disable to fix dimension mismatch
            )
        
        summary = model.get_model_summary()
        logger.info(f"Created model with {summary['total_parameters']:,} parameters")
        
        return model
    
    def _create_optimizer(self):
        """Create optimizer based on configuration."""
        if self.config.optimizer == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler based on configuration."""
        if self.config.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.num_epochs // 4,
                gamma=0.5
            )
        elif self.config.scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        else:
            scheduler = None
        
        return scheduler
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and split the data."""
        logger.info("Loading processed data...")
        
        # Load processed data
        data_path = os.path.join(self.config.data_dir, "cardiomyocyte_datasets.pt")
        if not os.path.exists(data_path):
            logger.info("Processed data not found. Processing raw data...")
            from data.preprocessing import process_all_datasets
            processed_data = process_all_datasets()
        else:
            processed_data = torch.load(data_path, weights_only=False)
        
        # Extract spatial and temporal data
        spatial_data = []
        for key, value in processed_data.items():
            if key.startswith('spatial_') and isinstance(value, Data):
                spatial_data.append(value)
        
        temporal_data = processed_data.get('temporal_sequences', {})
        
        if not spatial_data or not temporal_data:
            raise ValueError("No suitable data found for training")
        
        # Split data
        total_samples = min(len(spatial_data), len(temporal_data.get('sequences', [])))
        indices = np.random.permutation(total_samples)
        
        train_end = int(self.config.train_split * total_samples)
        val_end = train_end + int(self.config.val_split * total_samples)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Create datasets
        train_spatial = [spatial_data[i] for i in train_indices]
        val_spatial = [spatial_data[i] for i in val_indices]
        test_spatial = [spatial_data[i] for i in test_indices]
        
        train_temporal = {
            'sequences': temporal_data['sequences'][train_indices],
            'masks': temporal_data['masks'][train_indices],
            'labels': temporal_data['labels'][train_indices]
        }
        val_temporal = {
            'sequences': temporal_data['sequences'][val_indices],
            'masks': temporal_data['masks'][val_indices],
            'labels': temporal_data['labels'][val_indices]
        }
        test_temporal = {
            'sequences': temporal_data['sequences'][test_indices],
            'masks': temporal_data['masks'][test_indices],
            'labels': temporal_data['labels'][test_indices]
        }
        
        # Create datasets
        train_dataset = CardiomyocyteDataset(train_spatial, train_temporal, self.config, "train")
        val_dataset = CardiomyocyteDataset(val_spatial, val_temporal, self.config, "val")
        test_dataset = CardiomyocyteDataset(test_spatial, test_temporal, self.config, "test")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=0,  # Set to 0 for MPS compatibility
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=0,
            pin_memory=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=0,
            pin_memory=False
        )
        
        return train_loader, val_loader, test_loader
    
    def _collate_fn(self, batch):
        """Custom collate function for batching."""
        spatial_batch = [item['spatial'] for item in batch]
        spatial_batch = Batch.from_data_list(spatial_batch)
        
        temporal_sequences = torch.stack([item['temporal_sequence'] for item in batch])
        temporal_masks = torch.stack([item['temporal_mask'] for item in batch])
        efficiency_targets = torch.stack([item['efficiency_target'] for item in batch])
        maturation_targets = torch.stack([item['maturation_target'] for item in batch])
        
        return {
            'spatial': spatial_batch,
            'temporal_sequence': temporal_sequences,
            'temporal_mask': temporal_masks,
            'efficiency_target': efficiency_targets,
            'maturation_target': maturation_targets
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_efficiency_loss = 0.0
        total_maturation_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass with AMP if available
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    loss, losses = self._compute_loss(batch)
            else:
                loss, losses = self._compute_loss(batch)
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            total_efficiency_loss += losses['efficiency_loss']
            total_maturation_loss += losses['maturation_loss']
            num_batches += 1
            
            # Log progress
            if batch_idx % self.config.log_interval == 0:
                logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Eff: {losses['efficiency_loss']:.4f}, "
                    f"Mat: {losses['maturation_loss']:.4f}"
                )
            
            # Memory management
            if batch_idx % 50 == 0:
                self.memory_monitor.check_memory(f"Epoch {self.current_epoch}, Batch {batch_idx}")
        
        return {
            'total_loss': total_loss / num_batches,
            'efficiency_loss': total_efficiency_loss / num_batches,
            'maturation_loss': total_maturation_loss / num_batches
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_efficiency_loss = 0.0
        total_maturation_loss = 0.0
        all_maturation_preds = []
        all_maturation_targets = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._move_batch_to_device(batch)
                
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss, losses = self._compute_loss(batch)
                        outputs = self._get_model_outputs(batch)
                else:
                    loss, losses = self._compute_loss(batch)
                    outputs = self._get_model_outputs(batch)
                
                total_loss += loss.item()
                total_efficiency_loss += losses['efficiency_loss']
                total_maturation_loss += losses['maturation_loss']
                
                # Collect predictions for accuracy calculation
                maturation_preds = torch.argmax(outputs['multitask_maturation_probs'], dim=1)
                all_maturation_preds.extend(maturation_preds.cpu().numpy())
                all_maturation_targets.extend(batch['maturation_target'].cpu().numpy())
                
                num_batches += 1
        
        # Calculate accuracy
        accuracy = accuracy_score(all_maturation_targets, all_maturation_preds)
        
        return {
            'total_loss': total_loss / num_batches,
            'efficiency_loss': total_efficiency_loss / num_batches,
            'maturation_loss': total_maturation_loss / num_batches,
            'accuracy': accuracy
        }
    
    def _move_batch_to_device(self, batch):
        """Move batch to the appropriate device."""
        batch['spatial'] = batch['spatial'].to(self.device)
        batch['temporal_sequence'] = batch['temporal_sequence'].to(self.device)
        batch['temporal_mask'] = batch['temporal_mask'].to(self.device)
        batch['efficiency_target'] = batch['efficiency_target'].to(self.device)
        batch['maturation_target'] = batch['maturation_target'].to(self.device)
        return batch
    
    def _get_model_outputs(self, batch):
        """Get model outputs."""
        return self.model(
            node_features=batch['spatial'].x,
            edge_index=batch['spatial'].edge_index,
            pos=batch['spatial'].pos,
            batch=batch['spatial'].batch,
            temporal_features=batch['temporal_sequence'],
            temporal_mask=batch['temporal_mask']
        )
    
    def _compute_loss(self, batch):
        """Compute the total loss."""
        outputs = self._get_model_outputs(batch)
        
        # Efficiency loss
        efficiency_pred = outputs['multitask_differentiation_efficiency']
        efficiency_loss = self.criterion_efficiency(efficiency_pred, batch['efficiency_target'])
        
        # Maturation loss
        maturation_logits = outputs['multitask_maturation_logits']
        maturation_loss = self.criterion_maturation(maturation_logits, batch['maturation_target'])
        
        # Total loss
        total_loss = (
            self.config.efficiency_weight * efficiency_loss +
            self.config.maturation_weight * maturation_loss
        )
        
        # Add uncertainty loss if available
        if 'multitask_efficiency_uncertainty' in outputs:
            uncertainty_loss = outputs['multitask_efficiency_uncertainty'].mean()
            total_loss += self.config.uncertainty_weight * uncertainty_loss
        
        losses = {
            'efficiency_loss': efficiency_loss.item(),
            'maturation_loss': maturation_loss.item()
        }
        
        return total_loss, losses
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config),
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"checkpoint_epoch_{self.current_epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Load data
        train_loader, val_loader, test_loader = self.load_data()
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total_loss'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['total_loss']:.4f}, "
                f"Val Loss: {val_metrics['total_loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_metrics['total_loss'], epoch)
            self.writer.add_scalar('Loss/Val', val_metrics['total_loss'], epoch)
            self.writer.add_scalar('Accuracy/Val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Log to wandb
            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['total_loss'],
                    'val_loss': val_metrics['total_loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'learning_rate': current_lr
                })
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['total_loss'])
            self.training_history['val_loss'].append(val_metrics['total_loss'])
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            self.training_history['learning_rate'].append(current_lr)
            
            # Check for improvement
            is_best = val_metrics['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total_loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if epoch % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info("Training completed!")
        
        # Final evaluation on test set
        if test_loader:
            test_metrics = self.validate(test_loader)
            logger.info(f"Test Loss: {test_metrics['total_loss']:.4f}, Test Acc: {test_metrics['accuracy']:.4f}")
            
            if self.config.use_wandb:
                wandb.log({
                    'test_loss': test_metrics['total_loss'],
                    'test_accuracy': test_metrics['accuracy']
                })
        
        # Return model and training history
        return self.model, self.training_history


def main():
    """Main training function."""
    # Load configuration
    config = TrainingConfig()
    
    # Create trainer
    trainer = HybridModelTrainer(config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

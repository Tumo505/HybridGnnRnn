"""
Training Script for Hybrid GNN-RNN Framework

This script orchestrates the complete training pipeline for the hybrid model,
including data loading, model training, validation, and checkpointing.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import wandb
from datetime import datetime
import json

# Import our modules
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from models.hybrid_model import HybridGNNRNN, LightweightHybridGNNRNN
from data.loaders import SpatialDataset, TemporalDataset, HybridDataset, create_dataloaders
from training.utils import Trainer, MetricsCalculator, MultiTaskLoss
from utils.memory_utils import MemoryMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Hybrid GNN-RNN Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the datasets')
    parser.add_argument('--spatial_data_path', type=str, 
                        default='data/Spatially resolved multiomics of human cardiac niches',
                        help='Path to spatial data')
    parser.add_argument('--temporal_data_path', type=str,
                        default='data/GSE175634_temporal',
                        help='Path to temporal data')
    parser.add_argument('--cache_dir', type=str, default='cache',
                        help='Directory for caching preprocessed data')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='hybrid',
                        choices=['hybrid', 'lightweight'],
                        help='Type of model to train')
    parser.add_argument('--gnn_type', type=str, default='GraphSAGE',
                        choices=['GraphSAGE', 'GAT', 'GCN'],
                        help='Type of GNN encoder')
    parser.add_argument('--rnn_type', type=str, default='LSTM',
                        choices=['LSTM', 'GRU'],
                        help='Type of RNN encoder')
    parser.add_argument('--fusion_type', type=str, default='attention',
                        choices=['attention', 'gated', 'adaptive'],
                        help='Type of fusion mechanism')
    
    # Architecture arguments
    parser.add_argument('--node_feature_dim', type=int, default=100,
                        help='Dimension of node features')
    parser.add_argument('--gnn_hidden_dim', type=int, default=256,
                        help='Hidden dimension for GNN')
    parser.add_argument('--rnn_hidden_dim', type=int, default=256,
                        help='Hidden dimension for RNN')
    parser.add_argument('--fusion_dim', type=int, default=512,
                        help='Fusion layer dimension')
    parser.add_argument('--num_gnn_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--num_rnn_layers', type=int, default=2,
                        help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                        help='Number of batches to accumulate gradients')
    
    # Scheduler arguments
    parser.add_argument('--scheduler', type=str, default='reduce_lr_on_plateau',
                        choices=['reduce_lr_on_plateau', 'cosine_annealing', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--factor', type=float, default=0.5,
                        help='Factor for ReduceLROnPlateau scheduler')
    
    # Memory optimization
    parser.add_argument('--use_checkpoint', action='store_true',
                        help='Use gradient checkpointing')
    parser.add_argument('--memory_efficient', action='store_true',
                        help='Enable memory efficient mode')
    parser.add_argument('--max_memory_gb', type=float, default=12.0,
                        help='Maximum memory usage in GB')
    
    # Data loading arguments
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--pin_memory', action='store_true',
                        help='Pin memory for data loading')
    parser.add_argument('--subsample_rate', type=float, default=0.1,
                        help='Rate for subsampling large datasets')
    
    # Experiment arguments
    parser.add_argument('--experiment_name', type=str, default='hybrid_gnn_rnn',
                        help='Name of the experiment')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='cardiomyocyte-differentiation',
                        help='Wandb project name')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda, mps)')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup the appropriate device for training."""
    if device_arg == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using MPS (Metal Performance Shaders) device")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info("Using CUDA device")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU device")
    else:
        device = torch.device(device_arg)
        logger.info(f"Using specified device: {device}")
    
    return device


def create_model(args: argparse.Namespace) -> nn.Module:
    """Create the hybrid model based on arguments."""
    model_kwargs = {
        'node_feature_dim': args.node_feature_dim,
        'gnn_hidden_dim': args.gnn_hidden_dim,
        'rnn_hidden_dim': args.rnn_hidden_dim,
        'fusion_dim': args.fusion_dim,
        'gnn_type': args.gnn_type,
        'rnn_type': args.rnn_type,
        'fusion_type': args.fusion_type,
        'num_gnn_layers': args.num_gnn_layers,
        'num_rnn_layers': args.num_rnn_layers,
        'dropout': args.dropout,
        'prediction_tasks': ['multitask'],
        'use_uncertainty': True,
        'use_checkpoint': args.use_checkpoint,
        'memory_efficient': args.memory_efficient
    }
    
    if args.model_type == 'lightweight':
        model = LightweightHybridGNNRNN(**model_kwargs)
    else:
        model = HybridGNNRNN(**model_kwargs)
    
    logger.info(f"Created {args.model_type} model")
    
    # Print model summary
    summary = model.get_model_summary()
    logger.info(f"Model parameters: {summary['total_parameters']:,}")
    logger.info(f"Trainable parameters: {summary['trainable_parameters']:,}")
    
    return model


def load_datasets(args: argparse.Namespace) -> Dict[str, Any]:
    """Load and prepare datasets."""
    logger.info("Loading datasets...")
    
    # Initialize memory monitor
    memory_monitor = MemoryMonitor(max_memory_gb=args.max_memory_gb)
    
    datasets = {}
    
    try:
        # For now, create dummy datasets for testing
        # In production, you would load real data here
        logger.info("Creating dummy datasets for testing...")
        
        # Create synthetic spatial data
        n_samples = 1000
        spatial_features = torch.randn(n_samples, args.node_feature_dim)
        spatial_coords = torch.randn(n_samples, 2)
        spatial_targets = torch.rand(n_samples)
        
        # Mock AnnData object structure
        class MockAnnData:
            def __init__(self, X, obs):
                self.X = X if isinstance(X, np.ndarray) else X.numpy()
                self.obs = obs.copy() if hasattr(obs, 'copy') else obs
                self.obsm = {'spatial': spatial_coords.numpy()}
                self.var = pd.DataFrame({'highly_variable': [True] * self.X.shape[1]}, 
                                      index=[f'gene_{i}' for i in range(self.X.shape[1])])
                self.var_names = self.var.index
            
            def __len__(self):
                return self.X.shape[0]
            
            def __getitem__(self, key):
                if isinstance(key, tuple) and len(key) == 2:
                    row_key, col_key = key
                    # Handle row indexing
                    if isinstance(row_key, slice):
                        X_subset = self.X[row_key]
                        obs_subset = self.obs.iloc[row_key]
                    elif isinstance(row_key, (list, np.ndarray)):
                        X_subset = self.X[row_key]
                        obs_subset = self.obs.iloc[row_key]
                    else:
                        X_subset = self.X[row_key:row_key+1]
                        obs_subset = self.obs.iloc[row_key:row_key+1]
                    
                    # Handle column indexing
                    if isinstance(col_key, np.ndarray) and col_key.dtype == bool:
                        X_subset = X_subset[:, col_key]
                    elif isinstance(col_key, (list, np.ndarray)):
                        X_subset = X_subset[:, col_key]
                    elif isinstance(col_key, slice):
                        X_subset = X_subset[:, col_key]
                    
                    return MockAnnData(X_subset, obs_subset)
                else:
                    # Single key indexing (rows only)
                    if isinstance(key, slice):
                        X_subset = self.X[key]
                        obs_subset = self.obs.iloc[key]
                    elif isinstance(key, (list, np.ndarray)):
                        X_subset = self.X[key]
                        obs_subset = self.obs.iloc[key]
                    else:
                        X_subset = self.X[key:key+1]
                        obs_subset = self.obs.iloc[key:key+1]
                    
                    return MockAnnData(X_subset, obs_subset)
            
            def copy(self):
                return MockAnnData(self.X.copy(), self.obs.copy())
        
        # Create mock observations
        obs_data = pd.DataFrame({
            'cell_type': np.random.choice(['CM', 'CF', 'EC'], n_samples),
            'differentiation_efficiency': spatial_targets.numpy()
        })
        
        mock_adata = MockAnnData(spatial_features.numpy(), obs_data)
        
        # Create spatial dataset
        spatial_dataset = SpatialDataset(
            adata=mock_adata,
            k_neighbors=10,
            max_nodes_per_graph=1000,
            subsample_rate=args.subsample_rate,
            cache_dir=args.cache_dir
        )
        
        # Create temporal datasets (mock)
        temporal_datasets = []
        timepoints = [0, 1, 3, 7, 14, 21, 28]  # Days
        
        for t in timepoints:
            # Create time-dependent features
            temporal_features = spatial_features + 0.1 * t * torch.randn_like(spatial_features)
            temp_obs = obs_data.copy()
            temp_obs['timepoint'] = t
            
            temp_adata = MockAnnData(temporal_features.numpy(), temp_obs)
            temporal_datasets.append(temp_adata)
        
        temporal_dataset = TemporalDataset(
            adata_list=temporal_datasets,
            timepoints=timepoints,
            sequence_length=len(timepoints),
            max_cells_per_timepoint=100,  # Reduced for testing
            min_cells_per_timepoint=10,   # Reduced for testing
            subsample_rate=1.0,           # Don't subsample further
            align_cells=False             # Use unaligned for simplicity
        )
        
        # Create hybrid dataset
        hybrid_dataset = HybridDataset(
            spatial_dataset=spatial_dataset,
            temporal_dataset=temporal_dataset,
            match_samples=True
        )
        
        # Split datasets
        dataset_size = len(hybrid_dataset)
        train_size = int(0.7 * dataset_size)
        val_size = int(0.15 * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            hybrid_dataset, [train_size, val_size, test_size]
        )
        
        datasets = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
        
        logger.info(f"Dataset sizes - Train: {len(train_dataset)}, "
                   f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Memory check
        memory_monitor.check_memory("After dataset loading")
        
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise
    
    return datasets


def create_optimizer_and_scheduler(model: nn.Module, args: argparse.Namespace):
    """Create optimizer and learning rate scheduler."""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    if args.scheduler == 'reduce_lr_on_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.factor,
            patience=args.patience,
            verbose=True
        )
    elif args.scheduler == 'cosine_annealing':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.learning_rate * 0.01
        )
    else:
        scheduler = None
    
    return optimizer, scheduler


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(args.save_dir) / f"{args.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(experiment_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info(f"Starting experiment: {args.experiment_name}")
    logger.info(f"Experiment directory: {experiment_dir}")
    
    # Setup device
    device = setup_device(args.device)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.experiment_name}_{timestamp}",
            config=vars(args)
        )
    
    try:
        # Load datasets
        datasets = load_datasets(args)
        
        # Create data loaders
        dataloaders = create_dataloaders(
            datasets,
            batch_sizes={
                'train': args.batch_size,
                'val': args.batch_size * 2,
                'test': args.batch_size * 2
            },
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
        
        # Create model
        model = create_model(args)
        
        # Create optimizer and scheduler
        optimizer, scheduler = create_optimizer_and_scheduler(model, args)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            use_wandb=args.use_wandb,
            project_name=args.wandb_project,
            gradient_clip_val=args.gradient_clip_val,
            accumulate_grad_batches=args.accumulate_grad_batches
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_epoch = 0
        
        logger.info("Starting training...")
        
        for epoch in range(args.epochs):
            # Train epoch
            train_metrics = trainer.train_epoch(dataloaders['train'], epoch)
            
            # Validate epoch
            val_metrics = trainer.validate(dataloaders['val'], epoch)
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}/{args.epochs}")
            logger.info(f"Train Loss: {train_metrics.get('avg_total_loss', 'N/A'):.4f}")
            logger.info(f"Val Loss: {val_metrics.get('avg_total_loss', 'N/A'):.4f}")
            
            # Check for best model
            val_loss = val_metrics.get('avg_total_loss', float('inf'))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                
                # Save best model
                best_model_path = experiment_dir / 'best_model.pt'
                trainer.save_checkpoint(str(best_model_path), epoch, val_metrics)
                logger.info(f"New best model saved at epoch {epoch+1}")
            
            # Save checkpoint periodically
            if (epoch + 1) % args.save_every == 0:
                checkpoint_path = experiment_dir / f'checkpoint_epoch_{epoch+1}.pt'
                trainer.save_checkpoint(str(checkpoint_path), epoch, val_metrics)
            
            # Scheduler step (for ReduceLROnPlateau)
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            
            # Memory cleanup
            gc.collect()
            if device.type == 'mps':
                torch.mps.empty_cache()
            elif device.type == 'cuda':
                torch.cuda.empty_cache()
        
        logger.info(f"Training completed. Best epoch: {best_epoch+1}, Best val loss: {best_val_loss:.4f}")
        
        # Final evaluation on test set
        logger.info("Evaluating on test set...")
        test_metrics = trainer.validate(dataloaders['test'], epoch)
        logger.info(f"Test Loss: {test_metrics.get('avg_total_loss', 'N/A'):.4f}")
        
        # Save final model
        final_model_path = experiment_dir / 'final_model.pt'
        trainer.save_checkpoint(str(final_model_path), args.epochs, test_metrics)
        
        # Save training summary
        summary = {
            'best_epoch': best_epoch + 1,
            'best_val_loss': best_val_loss,
            'final_test_loss': test_metrics.get('avg_total_loss', 'N/A'),
            'total_epochs': args.epochs,
            'model_parameters': model.get_model_summary()['total_parameters']
        }
        
        with open(experiment_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        if args.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()

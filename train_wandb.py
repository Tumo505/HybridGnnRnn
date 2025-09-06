#!/usr/bin/env python3
"""
Training script with WandB integration for Spatial GNN model
"""

import argparse
import yaml
import torch
import numpy as np
import random
from pathlib import Path
import logging
import sys
import wandb

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data_processing.spatial_loader import load_and_process_heart_data, SpatialDataProcessor
from src.models.spatial_gnn import create_spatial_gnn
from src.training.trainer import GNNTrainer, create_data_loaders

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description='Train Spatial GNN with WandB tracking')
    parser.add_argument('--project', default='hybrid-gnn-cardiac',
                       help='WandB project name')
    parser.add_argument('--experiment_name', default='spatial_gnn_wandb_v1',
                       help='Experiment name')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[512, 256, 128],
                       help='Hidden dimensions')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--device', default='auto',
                       help='Device to use')
    parser.add_argument('--data_path', default='data/processed_visium_heart.h5ad',
                       help='Path to data file')
    parser.add_argument('--n_neighbors', type=int, default=6,
                       help='Number of spatial neighbors')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable WandB tracking')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Set random seed
    set_seed(42)
    logger.info("Set random seed to 42")
    
    # WandB configuration
    wandb_config = {
        'model_type': 'GCN',
        'hidden_dims': args.hidden_dims,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'n_neighbors': args.n_neighbors,
        'seed': 42,
        'architecture': 'SpatialGNN',
        'task': 'cardiomyocyte_differentiation'
    }
    
    use_wandb = not args.no_wandb
    
    try:
        # Load and process data
        logger.info("Loading and processing spatial transcriptomics data...")
        processor = SpatialDataProcessor(n_neighbors=args.n_neighbors)
        data = processor.process_dataset(args.data_path)
        
        logger.info(f"Loaded dataset: {data.x.shape[0]} nodes, {data.x.shape[1]} features")
        
        # Create model
        logger.info("Creating Spatial GNN model...")
        model_config = {
            'input_dim': data.x.shape[1],
            'hidden_dims': args.hidden_dims,
            'num_classes': 10,
            'conv_type': 'GCN',
            'dropout': 0.2,
            'use_attention': True
        }
        model = create_spatial_gnn(model_config)
        
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            data,
            train_ratio=0.7,
            val_ratio=0.2,
            batch_size=args.batch_size
        )
        
        # Create trainer with WandB
        logger.info("Setting up trainer with WandB tracking...")
        trainer = GNNTrainer(
            model=model,
            device=args.device,
            experiment_name=args.experiment_name,
            log_dir='experiments',
            use_wandb=use_wandb,
            wandb_project=args.project,
            wandb_config=wandb_config
        )
        
        # Setup training parameters
        trainer.setup_training(
            learning_rate=args.learning_rate,
            weight_decay=0.0001,
            classification_weight=1.0,
            regression_weight=0.1
        )
        
        # Start training
        logger.info("Starting training with WandB tracking...")
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            save_every=5
        )
        
        # Save final model
        final_model_path = Path('experiments') / args.experiment_name / 'final_model.pth'
        torch.save(model.state_dict(), final_model_path)
        
        # Save model artifact to WandB
        if use_wandb:
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(str(final_model_path))
            wandb.log_artifact(artifact)
        
        logger.info("Training completed successfully!")
        logger.info(f"Final model saved to {final_model_path}")
        
        if use_wandb:
            wandb.finish()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if use_wandb:
            wandb.finish(exit_code=1)
        raise

if __name__ == '__main__':
    main()

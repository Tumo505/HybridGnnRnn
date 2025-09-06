#!/usr/bin/env python3
"""
Main training script for Spatial GNN model
Trains a Graph Neural Network on spatial transcriptomics data for 
cardiomyocyte differentiation prediction.
"""

import argparse
import yaml
import torch
import numpy as np
import random
from pathlib import Path
import logging
import sys

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

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='Train Spatial GNN for Cardiac Analysis')
    parser.add_argument('--config', '-c', default='configs/gnn_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', default='data',
                       help='Path to data directory')
    parser.add_argument('--experiment_name', default=None,
                       help='Experiment name (overrides config)')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--device', default=None,
                       help='Device to use (overrides config)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with smaller dataset')
    parser.add_argument('--use_wandb', action='store_true', default=True,
                       help='Use Weights & Biases for tracking (default: True)')
    parser.add_argument('--wandb_project', default='hybrid-gnn-rnn',
                       help='WandB project name')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable WandB tracking')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.experiment_name:
        config['experiment']['name'] = args.experiment_name
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.device:
        config['hardware']['device'] = args.device
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Set random seed
    set_seed(config['experiment']['seed'])
    logger.info(f"Set random seed to {config['experiment']['seed']}")
    
    # Print configuration
    logger.info("Configuration:")
    logger.info(f"  Experiment: {config['experiment']['name']}")
    logger.info(f"  Device: {config['hardware']['device']}")
    logger.info(f"  Epochs: {config['training']['num_epochs']}")
    logger.info(f"  Learning Rate: {config['training']['learning_rate']}")
    logger.info(f"  Model: {config['model']['conv_type']} with {config['model']['hidden_dims']}")
    
    try:
        # Load and process data
        logger.info("Loading and processing spatial transcriptomics data...")
        
        if args.debug:
            logger.info("Debug mode: Using smaller dataset")
            # Create a smaller test dataset for debugging
            processor = SpatialDataProcessor(
                n_neighbors=config['data']['n_neighbors'],
                n_top_genes=min(500, config['data']['n_top_genes']),  # Smaller for debug
                normalize=config['data']['normalize'],
                log_transform=config['data']['log_transform']
            )
            data_path = Path(config['data']['data_dir']) / config['data']['processed_file']
            data = processor.process_dataset(str(data_path))
            
            # Subset the data for faster debugging
            subset_size = min(1000, data.num_nodes)
            indices = torch.randperm(data.num_nodes)[:subset_size]
            data.x = data.x[indices]
            data.pos = data.pos[indices]
            data.y = data.y[:1] if hasattr(data, 'y') else torch.zeros(1, dtype=torch.long)
            data.num_nodes = subset_size
            
            # Recreate edge index for subset
            processor_subset = SpatialDataProcessor(n_neighbors=min(6, subset_size-1))
            import scanpy as sc
            import anndata as ad
            
            # Create temporary anndata for subset
            adata_subset = ad.AnnData(X=data.x.numpy())
            adata_subset.obsm['spatial'] = data.pos.numpy()
            data.edge_index = processor_subset.create_spatial_graph(adata_subset)
            
            logger.info(f"Debug dataset: {data.num_nodes} nodes, {data.x.shape[1]} features")
            
        else:
            # Load full dataset
            data = load_and_process_heart_data(config['data']['data_dir'])
            logger.info(f"Loaded dataset: {data.num_nodes} nodes, {data.x.shape[1]} features")
        
        # Update model input dimension based on actual data
        config['model']['input_dim'] = data.x.shape[1]
        
        # Create model
        logger.info("Creating Spatial GNN model...")
        model = create_spatial_gnn(config['model'])
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            data,
            train_ratio=config['data']['train_ratio'],
            val_ratio=config['data']['val_ratio'],
            batch_size=config['training']['batch_size']
        )
        
        # Determine WandB usage
        use_wandb = args.use_wandb and not args.no_wandb
        
        # Prepare WandB config
        wandb_config = {
            'model_type': config['model']['conv_type'],
            'hidden_dims': config['model']['hidden_dims'],
            'num_layers': config['model']['num_layers'],
            'learning_rate': config['training']['learning_rate'],
            'weight_decay': config['training']['weight_decay'],
            'batch_size': config['training']['batch_size'],
            'num_epochs': config['training']['num_epochs'],
            'n_neighbors': config['data']['n_neighbors'],
            'train_ratio': config['data']['train_ratio'],
            'val_ratio': config['data']['val_ratio'],
            'classification_weight': config['training']['classification_weight'],
            'regression_weight': config['training']['regression_weight'],
            'seed': config['experiment']['seed']
        }
        
        # Create trainer
        logger.info("Setting up trainer...")
        trainer = GNNTrainer(
            model=model,
            device=config['hardware']['device'],
            experiment_name=config['experiment']['name'],
            log_dir=config['experiment']['log_dir'],
            use_wandb=use_wandb,
            wandb_project=args.wandb_project,
            wandb_config=wandb_config
        )
        
        # Setup training parameters
        trainer.setup_training(
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            classification_weight=config['training']['classification_weight'],
            regression_weight=config['training']['regression_weight']
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['num_epochs'],
            save_every=config['training']['save_every'],
            early_stopping_patience=config['training']['early_stopping_patience']
        )
        
        logger.info("Training completed successfully!")
        
        # Save final model
        final_model_path = Path(trainer.log_dir) / 'final_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'data_info': {
                'num_nodes': data.num_nodes,
                'num_features': data.x.shape[1],
                'num_edges': data.edge_index.shape[1]
            }
        }, final_model_path)
        
        logger.info(f"Final model saved to {final_model_path}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

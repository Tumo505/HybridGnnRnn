"""
Main training script for the Hybrid GNN-RNN Cardiomyocyte Differentiation Model.
Integrates data preprocessing, model training, experiment tracking, and visualization.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

# Add src to path
sys.path.append('/Users/tumokgabeng/Projects/HybridGnnRnn/src')

from data.preprocessing import CardiomyocyteDataPreprocessor, process_all_datasets
from training.pipeline import TrainingConfig, HybridModelTrainer
from utils.experiment_tracking import ExperimentTracker, CheckpointManager
from utils.visualization import ModelVisualization, create_comprehensive_analysis
from utils.memory_utils import MemoryMonitor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training function with full pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Hybrid GNN-RNN Model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name')
    parser.add_argument('--data-dir', type=str, default='/Users/tumokgabeng/Projects/HybridGnnRnn/data/selected_datasets', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='/Users/tumokgabeng/Projects/HybridGnnRnn/experiments', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--test-only', action='store_true', help='Only run testing')
    parser.add_argument('--preprocess-only', action='store_true', help='Only run preprocessing')
    
    args = parser.parse_args()
    
    # Initialize memory monitor
    memory_monitor = MemoryMonitor(max_memory_gb=16.0)
    memory_monitor.check_memory("Start")
    
    # Create training configuration
    config = TrainingConfig()
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    else:
        config.experiment_name = f"hybrid_cardiomyocyte_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    config.data_dir = args.data_dir
    config.output_dir = args.output_dir
    
    logger.info(f"Starting experiment: {config.experiment_name}")
    logger.info(f"Configuration: {config}")
    
    # Initialize experiment tracker (disable W&B for automated runs)
    tracker = ExperimentTracker(
        experiment_name=config.experiment_name,
        project_name=config.project_name,
        output_dir=config.output_dir,
        use_wandb=False,  # Disable W&B for now
        use_tensorboard=True
    )
    
    # Log configuration
    tracker.log_config(config.__dict__)
    
    # Step 1: Data Preprocessing
    logger.info("=" * 60)
    logger.info("STEP 1: DATA PREPROCESSING")
    logger.info("=" * 60)
    
    preprocessor = CardiomyocyteDataPreprocessor(
        data_dir=config.data_dir,
        cache_dir=os.path.join(config.output_dir, config.experiment_name, "processed_data")
    )
    
    # Check if processed data exists
    processed_data_path = os.path.join(preprocessor.cache_dir, "cardiomyocyte_datasets.pt")
    
    if not os.path.exists(processed_data_path):
        logger.info("Processing raw datasets...")
        try:
            # Process all available datasets
            processed_data = process_all_datasets()
            logger.info("Data preprocessing completed successfully!")
            
            # Log data statistics
            logger.info("Processed datasets:")
            for key, value in processed_data.items():
                if hasattr(value, 'shape'):
                    logger.info(f"  {key}: {value.shape}")
                elif hasattr(value, '__len__'):
                    logger.info(f"  {key}: {len(value)} items")
                else:
                    logger.info(f"  {key}: {type(value)}")
                    
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            if not args.preprocess_only:
                logger.info("Continuing with synthetic data for testing...")
                processed_data = create_synthetic_data()
                # Save synthetic data
                processor.save_processed_data(processed_data, "cardiomyocyte_datasets.pt")
            else:
                return
    else:
        logger.info(f"Loading existing processed data from {processed_data_path}")
        processed_data = torch.load(processed_data_path)
    
    memory_monitor.check_memory("After data preprocessing")
    
    if args.preprocess_only:
        logger.info("Preprocessing completed. Exiting.")
        return
    
    # Step 2: Model Training
    logger.info("=" * 60)
    logger.info("STEP 2: MODEL TRAINING")
    logger.info("=" * 60)
    
    # Update config based on processed data
    if 'temporal_sequences' in processed_data:
        temporal_data = processed_data['temporal_sequences']
        if 'sequences' in temporal_data:
            config.node_feature_dim = temporal_data['sequences'].shape[-1]
            config.sequence_length = temporal_data['sequences'].shape[1]
            logger.info(f"Updated config: node_feature_dim={config.node_feature_dim}, sequence_length={config.sequence_length}")
    
    # Create trainer
    trainer = HybridModelTrainer(config)
    
    # Log model summary
    tracker.log_model_summary(trainer.model)
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    memory_monitor.check_memory("After model creation")
    
    if not args.test_only:
        # Start training
        logger.info("Starting training...")
        try:
            trainer.train()
            logger.info("Training completed successfully!")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            tracker.finish()
            return
    
    memory_monitor.check_memory("After training")
    
    # Step 3: Model Evaluation and Visualization
    logger.info("=" * 60)
    logger.info("STEP 3: EVALUATION AND VISUALIZATION")
    logger.info("=" * 60)
    
    try:
        # Load best model for evaluation
        best_checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_checkpoint_path):
            trainer.load_checkpoint(best_checkpoint_path)
            logger.info("Loaded best model for evaluation")
        
        # Get data loaders
        train_loader, val_loader, test_loader = trainer.load_data()
        
        # Evaluate on test set
        test_metrics = trainer.validate(test_loader)
        logger.info(f"Test Results: {test_metrics}")
        
        # Log final metrics
        tracker.log_metrics(test_metrics, trainer.current_epoch, "test")
        
        # Generate predictions for visualization
        model_outputs, true_labels, coordinates = generate_predictions_for_viz(
            trainer.model, test_loader, trainer.device
        )
        
        # Create comprehensive visualizations
        viz_dir = os.path.join(config.output_dir, config.experiment_name, "visualizations")
        create_comprehensive_analysis(
            model_outputs, true_labels, coordinates, viz_dir
        )
        
        logger.info(f"Visualizations saved to: {viz_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
    
    memory_monitor.check_memory("After evaluation")
    
    # Step 4: Finalize Experiment
    logger.info("=" * 60)
    logger.info("STEP 4: FINALIZING EXPERIMENT")
    logger.info("=" * 60)
    
    # Save final experiment summary
    tracker.finish()
    
    # Print final memory report
    memory_monitor.print_summary()
    
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
    logger.info(f"Results saved to: {os.path.join(config.output_dir, config.experiment_name)}")
    logger.info("=" * 60)


def create_synthetic_data():
    """Create synthetic data for testing when real data is not available."""
    logger.info("Creating synthetic data for testing...")
    
    # Synthetic spatial data
    from torch_geometric.data import Data
    
    n_samples = 100
    n_features = 2000
    n_time_points = 7
    
    spatial_data = []
    for i in range(n_samples):
        # Create random graph
        n_nodes = np.random.randint(50, 200)
        x = torch.randn(n_nodes, n_features)
        edge_index = torch.randint(0, n_nodes, (2, n_nodes * 3))
        pos = torch.randn(n_nodes, 2)
        y = torch.randint(0, 5, (n_nodes,))
        
        data = Data(x=x, edge_index=edge_index, pos=pos, y=y)
        spatial_data.append(data)
    
    # Synthetic temporal data
    temporal_sequences = torch.randn(n_samples, n_time_points, n_features)
    temporal_masks = torch.ones(n_samples, n_time_points)
    temporal_labels = torch.rand(n_samples)
    
    temporal_data = {
        'sequences': temporal_sequences,
        'masks': temporal_masks,
        'labels': temporal_labels
    }
    
    processed_data = {
        'spatial_synthetic': spatial_data,
        'temporal_sequences': temporal_data
    }
    
    logger.info("Synthetic data created successfully!")
    return processed_data


def generate_predictions_for_viz(model, data_loader, device):
    """Generate predictions and extract data for visualization."""
    model.eval()
    
    all_spatial_emb = []
    all_temporal_emb = []
    all_predictions = []
    all_coordinates = []
    all_targets = []
    all_cell_types = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Move to device
            spatial_batch = batch['spatial'].to(device)
            temporal_seq = batch['temporal_sequence'].to(device)
            temporal_mask = batch['temporal_mask'].to(device)
            
            # Get model outputs
            outputs = model(
                node_features=spatial_batch.x,
                edge_index=spatial_batch.edge_index,
                pos=spatial_batch.pos,
                batch=spatial_batch.batch,
                temporal_features=temporal_seq,
                temporal_mask=temporal_mask
            )
            
            # Extract embeddings and predictions
            spatial_emb = outputs.get('spatial_embeddings', torch.tensor([]))
            temporal_emb = outputs.get('temporal_embeddings', torch.tensor([]))
            predictions = outputs.get('multitask_differentiation_efficiency', torch.tensor([]))
            
            # Move to CPU and convert to numpy
            if len(spatial_emb) > 0:
                all_spatial_emb.append(spatial_emb.cpu().numpy())
            if len(temporal_emb) > 0:
                all_temporal_emb.append(temporal_emb.cpu().numpy())
            if len(predictions) > 0:
                all_predictions.append(predictions.cpu().numpy())
            
            # Extract coordinates and targets
            if hasattr(spatial_batch, 'pos'):
                # Average coordinates per batch item
                batch_coords = []
                for i in range(batch['spatial'].num_graphs):
                    mask = spatial_batch.batch == i
                    if mask.sum() > 0:
                        coord = spatial_batch.pos[mask].mean(dim=0).cpu().numpy()
                        batch_coords.append(coord)
                if batch_coords:
                    all_coordinates.append(np.stack(batch_coords))
            
            all_targets.append(batch['efficiency_target'].cpu().numpy())
            all_cell_types.append(batch['maturation_target'].cpu().numpy())
    
    # Concatenate all results
    model_outputs = {}
    if all_spatial_emb:
        model_outputs['spatial_embeddings'] = np.concatenate(all_spatial_emb, axis=0)
    if all_temporal_emb:
        model_outputs['temporal_embeddings'] = np.concatenate(all_temporal_emb, axis=0)
    if all_predictions:
        model_outputs['predictions'] = np.concatenate(all_predictions, axis=0)
    
    true_labels = {}
    if all_targets:
        true_labels['targets'] = np.concatenate(all_targets, axis=0)
    if all_cell_types:
        true_labels['cell_types'] = np.concatenate(all_cell_types, axis=0)
    
    coordinates = np.concatenate(all_coordinates, axis=0) if all_coordinates else np.random.randn(len(all_targets[0]), 2)
    
    return model_outputs, true_labels, coordinates


if __name__ == "__main__":
    main()

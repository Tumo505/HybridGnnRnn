"""
Complete training pipeline with optimized data handling for large spatial transcriptomics files.
"""

import logging
import sys
from pathlib import Path
sys.path.append('src')

from data.robust_data_strategy import RobustDataStrategy
from data.memory_efficient_processing import MemoryEfficientProcessor, SpatialDataOptimizer
from training.pipeline import HybridModelTrainer, TrainingConfig
from utils.experiment_tracking import ExperimentTracker
from utils.visualization import ModelVisualization
import torch
import scanpy as sc

logger = logging.getLogger(__name__)

class OptimizedTrainingPipeline:
    """
    Complete training pipeline optimized for large spatial transcriptomics data.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_strategy = RobustDataStrategy()
        self.processor = MemoryEfficientProcessor()
        self.optimizer = SpatialDataOptimizer()
        
        # Set up experiment tracking
        self.tracker = ExperimentTracker(
            project_name="HybridGnnRnn_OptimizedTraining",
            config=config.__dict__
        )
        
        logger.info("Initialized optimized training pipeline")
    
    def load_and_process_data(self, use_cached: bool = True):
        """Load and process data with memory optimization."""
        
        # Check if processed data exists
        processed_dir = Path("processed_training_data")
        spatial_file = processed_dir / "spatial_training_data.h5ad"
        temporal_file = processed_dir / "temporal_training_data.h5ad"
        
        if use_cached and spatial_file.exists() and temporal_file.exists():
            logger.info("Loading cached processed data...")
            
            try:
                spatial_data = sc.read_h5ad(spatial_file)
                temporal_data = sc.read_h5ad(temporal_file)
                
                logger.info(f"Loaded cached data - Spatial: {spatial_data.shape}, Temporal: {temporal_data.shape}")
                return spatial_data, temporal_data
                
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}")
        
        # Create new processed data
        logger.info("Creating new processed training data...")
        datasets = self.data_strategy.create_training_dataset(
            use_synthetic=True,  # Use synthetic spatial due to corruption
            temporal_data_dir="data/selected_datasets/temporal_data",
            max_cells=self.config.max_cells_per_dataset,
            max_genes=self.config.max_genes
        )
        
        # Save processed data
        self.data_strategy.save_training_dataset(datasets, "processed_training_data")
        
        return datasets.get('spatial'), datasets.get('temporal')
    
    def prepare_training_data(self, spatial_data, temporal_data):
        """Prepare data for training with memory-efficient processing."""
        
        # Process spatial data
        if spatial_data is not None:
            logger.info("Processing spatial data...")
            spatial_data = self.optimizer.optimize_for_training(
                spatial_data,
                max_cells=self.config.max_cells_per_dataset,
                max_genes=self.config.max_genes
            )
        
        # Process temporal data  
        if temporal_data is not None:
            logger.info("Processing temporal data...")
            temporal_data = self.processor.process_with_gene_filtering(
                temporal_data,
                max_genes=self.config.max_genes,
                target_memory_gb=4.0
            )
        
        return spatial_data, temporal_data
    
    def run_training(self, spatial_data=None, temporal_data=None):
        """Run the complete training pipeline."""
        
        logger.info("Starting optimized training pipeline...")
        
        # Load and process data
        if spatial_data is None or temporal_data is None:
            spatial_data, temporal_data = self.load_and_process_data()
        
        # Prepare data for training
        spatial_data, temporal_data = self.prepare_training_data(spatial_data, temporal_data)
        
        # Log data info
        if spatial_data is not None:
            self.tracker.log_metrics({
                'spatial_cells': spatial_data.n_obs,
                'spatial_genes': spatial_data.n_vars
            })
        
        if temporal_data is not None:
            self.tracker.log_metrics({
                'temporal_cells': temporal_data.n_obs,
                'temporal_genes': temporal_data.n_vars
            })
        
        # Initialize trainer
        trainer = HybridModelTrainer(self.config, self.tracker)
        
        # Train model
        logger.info("Starting model training...")
        model, training_history = trainer.train(
            spatial_data=spatial_data,
            temporal_data=temporal_data
        )
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        visualizer = ModelVisualization(model, self.config.device)
        
        if spatial_data is not None:
            visualizer.plot_spatial_embeddings(spatial_data, save_path="outputs/spatial_embeddings.png")
        
        if temporal_data is not None:
            visualizer.plot_temporal_trajectories(temporal_data, save_path="outputs/temporal_trajectories.png")
        
        # Create dashboard
        dashboard_path = visualizer.create_interactive_dashboard(
            spatial_data=spatial_data,
            temporal_data=temporal_data,
            training_history=training_history
        )
        
        logger.info(f"Training complete! Dashboard available at: {dashboard_path}")
        
        return model, training_history


def run_optimized_training():
    """Run the complete optimized training pipeline."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Configure training with memory-optimized settings
    config = TrainingConfig(
        # Model settings
        spatial_dim=2500,
        temporal_dim=2500,
        hidden_dim=256,
        attention_heads=8,
        num_layers=4,
        
        # Training settings  
        batch_size=32,  # Smaller batch size for memory efficiency
        learning_rate=1e-4,
        num_epochs=100,
        
        # Data settings
        max_cells_per_dataset=12000,  # Limit dataset size
        max_genes=2500,
        
        # Memory optimization
        use_amp=True,
        gradient_clip_val=1.0,
        
        # Device
        device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    )
    
    # Initialize and run pipeline
    pipeline = OptimizedTrainingPipeline(config)
    
    try:
        model, history = pipeline.run_training()
        
        print("\\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"✅ Model trained successfully")
        print(f"✅ Training history: {len(history)} epochs")
        print(f"✅ Outputs saved to: outputs/")
        print(f"✅ Logs saved to: training.log")
        print("="*60)
        
        return model, history
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Run the optimized training
    model, history = run_optimized_training()

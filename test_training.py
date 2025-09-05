"""
Quick test to verify the training pipeline works correctly.
"""

import logging
import sys
import os

# Add src to path
sys.path.append('src')

from training.pipeline import HybridModelTrainer, TrainingConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Test training pipeline."""
    logger.info("🧪 Testing training pipeline...")
    
    # Create test configuration
    config = TrainingConfig(
        batch_size=4,
        num_epochs=2,  # Just 2 epochs for testing
        node_feature_dim=2000,
        use_wandb=False,  # Disable wandb for testing
        patience=10
    )
    
    # Create trainer
    trainer = HybridModelTrainer(config)
    
    # Test training
    try:
        model, training_history = trainer.train()
        logger.info("✅ Training pipeline test successful!")
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Training history keys: {list(training_history.keys()) if training_history else 'None'}")
        return True
    except Exception as e:
        logger.error(f"❌ Training pipeline test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

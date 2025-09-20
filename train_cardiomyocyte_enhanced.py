#!/usr/bin/env python3
"""
Entry point for training the Enhanced Cardiomyocyte Subtype Classifier
"""
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training import train_enhanced_cardiomyocyte_classifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    logger.info("-> Starting Enhanced Cardiomyocyte Classifier Training")
    logger.info("-> Using organized package structure from src/")
    
    # Optional: Custom configuration
    config = {
        'device': 'cpu',
        'hidden_dim': 128,
        'dropout': 0.4,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'classifier_lr': 0.005,
        'classifier_wd': 0.005,
        'max_epochs': 400,
        'patience': 60,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'grad_clip': 0.5,
        'l2_reg': 0.0001
    }
    
    # Run training
    results = train_enhanced_cardiomyocyte_classifier(config)
    
    if results:
        print(f"\nENHANCED CARDIOMYOCYTE SUBTYPE CLASSIFICATION RESULTS:")
        print(f"  Test Accuracy: {results['test_accuracy']*100:.2f}%")
        print(f"  Best Val Accuracy: {results['best_val_accuracy']*100:.2f}%")
        
        print(f"\n-> Per-Class Performance:")
        for subtype, metrics in results['per_class_metrics'].items():
            print(f"  {subtype}: F1={metrics['f1']*100:.1f}% | P={metrics['precision']*100:.1f}% | R={metrics['recall']*100:.1f}% | {metrics['support']} cells")
        
        print(f"\nENHANCED CARDIOMYOCYTE CLASSIFICATION READY!")
        print(f"  Model can distinguish {results['dataset_info']['num_classes']} cardiomyocyte subtypes")
        print(f"  Advanced architecture with skip connections and feature fusion")
        print(f"  Suitable for cardiomyocyte differentiation state analysis")
  
    else:
        print("❌ Training failed - check cache file exists.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
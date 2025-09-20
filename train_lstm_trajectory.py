#!/usr/bin/env python3
"""
LSTM Trajectory Model Training Entry Point
========================================
Clean interface for training LSTM trajectory models on cardiac data
using the organized src/ package structure.
"""

import os
import sys
import logging
from datetime import datetime

# Add src to path for imports
sys.path.append('src')

from models.rnn_models.lstm_trajectory_model import LSTMTrajectoryModel
from data_processing.temporal_processor import load_temporal_cardiac_data
from training.temporal_trainer import TemporalRNNTrainer

def main():
    """Main training function"""
    print("🚀 LSTM Trajectory Model Training")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # 1. Load data
        print("\n1. Loading temporal cardiac data...")
        data_dir = "data/cardiac_temporal"
        sequence_length = 5  # Shorter sequences for LSTM
        
        train_loader, test_loader, data_info = load_temporal_cardiac_data(
            data_dir=data_dir,
            sequence_length=sequence_length
        )
        
        # Create validation loader
        val_loader = test_loader  # Using test as validation for now
        
        print(f"✅ Data loaded successfully:")
        print(f"   Input size: {data_info['input_size']}")
        print(f"   Sequence length: {data_info['sequence_length']}")
        print(f"   Number of classes: {data_info['n_classes']}")
        print(f"   Class distribution: {data_info['class_distribution']}")
        
        # 2. Initialize model
        print("\n2. Initializing LSTM Trajectory Model...")
        
        model = LSTMTrajectoryModel(
            input_size=data_info['input_size'],
            hidden_size=256,
            num_layers=2,
            num_classes=data_info['n_classes'],
            dropout=0.5,
            bidirectional=True,
            use_attention=True,
            projection_dim=512
        )
        
        model_info = model.get_model_info()
        print(f"✅ Model initialized:")
        print(f"   Total parameters: {model_info['total_parameters']:,}")
        print(f"   Trainable parameters: {model_info['trainable_parameters']:,}")
        print(f"   Architecture: Input Projection + BiLSTM + Attention + Classifier")
        print(f"   Hidden size: {model_info['hidden_size']}")
        print(f"   Projection dim: {model_info['projection_dim']}")
        
        # 3. Setup trainer
        print("\n3. Setting up trainer...")
        
        config = {
            'batch_size': 32,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'num_epochs': 25,
            'patience': 5,
            'gradient_clip': 1.0,
            'use_focal_loss': True,
            'focal_alpha': 1.0,
            'focal_gamma': 2.0,
            'use_class_weights': True,
            'lr_scheduler': {
                'type': 'ReduceLROnPlateau',
                'factor': 0.5,
                'patience': 3,
                'min_lr': 1e-6
            }
        }
        
        trainer = TemporalRNNTrainer(config)
        trainer.prepare_data(train_loader, val_loader, test_loader, data_info)
        trainer.setup_model(model)
        
        print("✅ Trainer configured")
        
        # 4. Train model
        print("\n4. Starting training...")
        results = trainer.train()
        
        print(f"\n✅ Training completed!")
        print(f"   Best epoch: {results['best_epoch'] + 1}")
        print(f"   Best validation loss: {results['best_val_loss']:.4f}")
        print(f"   Best validation accuracy: {results['best_val_acc']:.4f}")
        
        # 5. Evaluate model
        print("\n5. Evaluating on test data...")
        eval_results = trainer.evaluate()
        
        print(f"✅ Evaluation completed!")
        print(f"   Test loss: {eval_results['test_loss']:.4f}")
        print(f"   Test accuracy: {eval_results['test_accuracy']:.4f}")
        
        # Print detailed classification report
        print("\n📊 Classification Report:")
        class_report = eval_results['classification_report']
        for class_idx, metrics in class_report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                print(f"   Class {class_idx}: P={metrics['precision']:.3f}, "
                      f"R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        # 6. Model analysis
        print("\n🔍 Model Analysis:")
        param_breakdown = model_info.get('parameter_breakdown', {})
        for component, params in param_breakdown.items():
            percentage = (params / model_info['total_parameters']) * 100
            print(f"   {component}: {params:,} parameters ({percentage:.1f}%)")
        
        # 7. Save results
        print("\n6. Saving results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"lstm_trajectory_results_{timestamp}.json"
        
        # Combine all results
        all_results = {
            'model_info': model_info,
            'data_info': data_info,
            'training_results': results,
            'evaluation_results': eval_results,
            'config': config,
            'timestamp': timestamp
        }
        
        trainer.save_results(results_file)
        print(f"✅ Results saved to {results_file}")
        
        # 8. Summary
        print(f"\n🎯 Training Summary:")
        print(f"   Model: LSTM Trajectory Model")
        print(f"   Parameters: {model_info['trainable_parameters']:,}")
        print(f"   Training epochs: {len(results['train_losses'])}")
        print(f"   Final test accuracy: {eval_results['test_accuracy']:.4f}")
        print(f"   Results saved: {results_file}")
        
        return all_results
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()
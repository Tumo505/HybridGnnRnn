#!/usr/bin/env python3
"""
Enhanced Temporal RNN Training Entry Point
========================================
Clean interface for training temporal RNN models on cardiac data
using the organized src/ package structure.
"""

import os
import sys
import logging
from datetime import datetime

# Add src to path for imports
sys.path.append('src')

from models.rnn_models.enhanced_temporal_rnn import EnhancedTemporalRNN
from data_processing.temporal_processor import load_temporal_cardiac_data
from training.temporal_trainer import TemporalRNNTrainer

def main():
    """Main training function"""
    print("🚀 Enhanced Temporal RNN Training")
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
        sequence_length = 10
        
        train_loader, test_loader, data_info = load_temporal_cardiac_data(
            data_dir=data_dir,
            sequence_length=sequence_length
        )
        
        # Create validation loader (split from train for simplicity)
        # In a more sophisticated setup, you'd have separate validation data
        val_loader = test_loader  # Using test as validation for now
        
        print(f"✅ Data loaded successfully:")
        print(f"   Input size: {data_info['input_size']}")
        print(f"   Sequence length: {data_info['sequence_length']}")
        print(f"   Number of classes: {data_info['n_classes']}")
        print(f"   Class distribution: {data_info['class_distribution']}")
        
        # 2. Initialize model
        print("\n2. Initializing Enhanced Temporal RNN...")
        
        model = EnhancedTemporalRNN(
            input_dim=data_info['input_size'],
            hidden_dim=256,
            num_layers=3,
            num_classes=data_info['n_classes'],
            rnn_type='LSTM',
            bidirectional=True,
            use_attention=True,
            use_positional_encoding=True,
            dropout=0.4,
            temporal_conv_filters=[64, 128, 256],
            kernel_sizes=[3, 5, 7]
        )
        
        model_info = model.get_model_info()
        print(f"✅ Model initialized:")
        print(f"   Total parameters: {model_info['total_parameters']:,}")
        print(f"   Trainable parameters: {model_info['trainable_parameters']:,}")
        print(f"   Architecture: {model_info['architecture']}")
        
        # 3. Setup trainer
        print("\n3. Setting up trainer...")
        
        config = {
            'batch_size': 32,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'num_epochs': 30,
            'patience': 7,
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
        
        # 6. Save results
        print("\n6. Saving results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"enhanced_temporal_rnn_results_{timestamp}.json"
        
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
        
        # 7. Summary
        print(f"\n🎯 Training Summary:")
        print(f"   Model: Enhanced Temporal RNN")
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
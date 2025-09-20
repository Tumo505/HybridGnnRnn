#!/usr/bin/env python3
"""
Temporal Cardiac RNN Training Entry Point with Wandb Integration
===============================================================
Clean interface for training temporal RNN models on cardiac data
using the organized src/ package structure with comprehensive tracking.
"""

import os
import sys
import logging
from datetime import datetime
import wandb
import torch

# Add src to path for imports
sys.path.append('src')

from models.rnn_models.temporal_cardiac_rnn import TemporalCardiacRNN, create_temporal_cardiac_rnn
from data_processing.temporal_processor import load_temporal_cardiac_data
from training.temporal_trainer import TemporalRNNTrainer

def main():
    """Main training function with Wandb tracking"""
    print("🚀 Temporal Cardiac RNN Training with Wandb")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Initialize Wandb
    wandb.init(
        project="temporal-cardiac-rnn",
        name=f"temporal_cardiac_rnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags=["temporal", "cardiac", "rnn", "cardiomyocyte", "differentiation"],
        notes="Training Temporal Cardiac RNN for cardiomyocyte differentiation prediction"
    )
    
    try:
        # 1. Load data
        print("\n1. Loading temporal cardiac data...")
        data_dir = "data/GSE175634_temporal_data"
        sequence_length = 10  # Temporal sequences for the model
        
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
        
        # Log data info to Wandb
        wandb.config.update({
            "input_size": data_info['input_size'],
            "sequence_length": data_info['sequence_length'],
            "n_classes": data_info['n_classes'],
            "class_distribution": data_info['class_distribution']
        })
        
        # 2. Initialize model
        print("\n2. Initializing Temporal Cardiac RNN...")
        
        model = create_temporal_cardiac_rnn(
            input_size=data_info['input_size'],
            num_classes=data_info['n_classes'],
            hidden_size=256,
            num_layers=3,
            dropout=0.5
        )
        
        model_info = model.get_model_info()
        print(f"✅ Model initialized:")
        print(f"   Model type: {model_info['model_name']}")
        print(f"   Total parameters: {model_info['total_parameters']:,}")
        print(f"   Trainable parameters: {model_info['trainable_parameters']:,}")
        print(f"   Architecture: {model_info['architecture']}")
        print(f"   Input size: {model_info['input_size']}")
        print(f"   Memory usage: {model_info['memory_usage_mb']:.2f} MB")
        
        # Log model info to Wandb
        wandb.config.update({
            "model_name": model_info['model_name'],
            "total_parameters": model_info['total_parameters'],
            "trainable_parameters": model_info['trainable_parameters'],
            "architecture": model_info['architecture'],
            "input_size": model_info['input_size'],
            "memory_usage_mb": model_info['memory_usage_mb']
        })
        
        # 3. Setup trainer configuration
        print("\n3. Setting up trainer with Wandb integration...")
        
        config = {
            'batch_size': 16,  # Smaller batch size for CPU training
            'learning_rate': 1e-3,
            'weight_decay': 1e-3,
            'num_epochs': 30,  # Reasonable number of epochs
            'patience': 8,
            'gradient_clip': 0.5,
            'use_focal_loss': True,
            'focal_alpha': 1.0,
            'focal_gamma': 3.0,
            'use_class_weights': True,
            'lr_scheduler': {
                'type': 'ReduceLROnPlateau',
                'factor': 0.3,
                'patience': 4,
                'min_lr': 1e-7
            },
            'device': 'cpu'  # Force CPU training
        }
        
        # Log training config to Wandb
        wandb.config.update(config)
        
        trainer = TemporalRNNTrainer(config)
        
        # Force CPU device due to CUDA kernel compatibility issues with RTX 5070 Ti
        trainer.device = torch.device("cpu")
        print(f"   Forcing CPU training due to CUDA compatibility issues")
        
        trainer.prepare_data(train_loader, val_loader, test_loader, data_info)
        trainer.setup_model(model)
        
        print("✅ Trainer configured with Wandb integration")
        
        # 4. Train model with Wandb logging
        print("\n4. Starting temporal training with Wandb tracking...")
        results = trainer.train()
        
        # Log training results to Wandb
        wandb.log({
            "best_epoch": results['best_epoch'],
            "best_val_loss": results['best_val_loss'],
            "best_val_acc": results['best_val_acc'],
            "total_epochs": len(results['train_losses'])
        })
        
        # Log training curves
        for epoch, (train_loss, val_loss, train_acc, val_acc) in enumerate(zip(
            results['train_losses'], results['val_losses'], 
            results['train_accuracies'], results['val_accuracies']
        )):
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "learning_rate": results.get('learning_rates', [config['learning_rate']])[epoch] if epoch < len(results.get('learning_rates', [])) else config['learning_rate']
            })
        
        print(f"\n✅ Training completed!")
        print(f"   Best epoch: {results['best_epoch'] + 1}")
        print(f"   Best validation loss: {results['best_val_loss']:.4f}")
        print(f"   Best validation accuracy: {results['best_val_acc']:.4f}")
        
        # 5. Evaluate model
        print("\n5. Evaluating on test data...")
        eval_results = trainer.evaluate()
        
        # Log evaluation results to Wandb
        wandb.log({
            "test_loss": eval_results['test_loss'],
            "test_accuracy": eval_results['test_accuracy']
        })
        
        print(f"✅ Evaluation completed!")
        print(f"   Test loss: {eval_results['test_loss']:.4f}")
        print(f"   Test accuracy: {eval_results['test_accuracy']:.4f}")
        
        # Print detailed classification report
        print("\n📊 Classification Report:")
        class_report = eval_results['classification_report']
        class_metrics = {}
        for class_idx, metrics in class_report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                print(f"   Class {class_idx}: P={metrics['precision']:.3f}, "
                      f"R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
                class_metrics[f"class_{class_idx}_precision"] = metrics['precision']
                class_metrics[f"class_{class_idx}_recall"] = metrics['recall']
                class_metrics[f"class_{class_idx}_f1"] = metrics['f1-score']
        
        # Log class-wise metrics to Wandb
        wandb.log(class_metrics)
        
        # 6. Model analysis
        print("\n🔧 Model Analysis:")
        print(f"   Total parameters: {model_info['total_parameters']:,}")
        print(f"   Model complexity: {model_info['total_parameters'] / 1e6:.1f}M parameters")
        
        # Check for overfitting
        train_losses = results['train_losses']
        val_losses = results['val_losses']
        final_gap = abs(train_losses[-1] - val_losses[-1])
        print(f"   Final train-val gap: {final_gap:.4f}")
        
        overfitting_status = "good" if final_gap < 0.1 else ("moderate" if final_gap < 0.2 else "high")
        wandb.log({
            "final_train_val_gap": final_gap,
            "overfitting_status": overfitting_status
        })
        
        if final_gap < 0.1:
            print("   ✅ Good generalization (low train-val gap)")
        elif final_gap < 0.2:
            print("   ⚠️  Moderate overfitting")
        else:
            print("   ❌ High overfitting detected")
        
        # 7. Save results
        print("\n7. Saving results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"temporal_cardiac_rnn_results_{timestamp}.json"
        
        # Combine all results
        all_results = {
            'model_info': model_info,
            'data_info': data_info,
            'training_results': results,
            'evaluation_results': eval_results,
            'config': config,
            'analysis': {
                'final_train_val_gap': final_gap,
                'overfitting_status': overfitting_status,
                'weight_decay': config['weight_decay'],
                'gradient_clip': config['gradient_clip']
            },
            'timestamp': timestamp
        }
        
        trainer.save_results(results_file)
        print(f"✅ Results saved to {results_file}")
        
        # Log results to Wandb (skip file upload due to Windows permissions)
        print(f"   Training metrics logged to Wandb: {wandb.run.url}")
        
        # 8. Summary
        print(f"\n🎯 Training Summary:")
        print(f"   Model: {model_info['model_name']}")
        print(f"   Parameters: {model_info['trainable_parameters']:,}")
        print(f"   Training epochs: {len(results['train_losses'])}")
        print(f"   Final test accuracy: {eval_results['test_accuracy']:.4f}")
        print(f"   Wandb run: {wandb.run.url}")
        print(f"   Results saved: {results_file}")
        
        # Final summary to Wandb
        wandb.summary.update({
            "final_test_accuracy": eval_results['test_accuracy'],
            "final_test_loss": eval_results['test_loss'],
            "best_val_accuracy": results['best_val_acc'],
            "model_parameters": model_info['trainable_parameters'],
            "training_epochs": len(results['train_losses']),
            "overfitting_gap": final_gap
        })
        
        return all_results
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        wandb.finish(exit_code=1)
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    results = main()
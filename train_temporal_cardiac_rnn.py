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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from models.rnn_models.temporal_cardiac_rnn import TemporalCardiacRNN, create_temporal_cardiac_rnn
from data_processing.temporal_processor import load_temporal_cardiac_data
from training.temporal_trainer import TemporalRNNTrainer

def create_training_curves(results, output_dir):
    """Create training curves visualization"""
    print("📊 Creating training curves...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Colors and styling
    colors = {
        'train': '#2E86AB',
        'val': '#A23B72',
        'accent': '#F18F01',
        'success': '#C73E1D'
    }
    
    epochs = range(1, len(results['train_losses']) + 1)
    
    # Loss curves
    ax1.plot(epochs, results['train_losses'], color=colors['train'], 
             linewidth=2.5, label='Training Loss', marker='o', markersize=4)
    ax1.plot(epochs, results['val_losses'], color=colors['val'], 
             linewidth=2.5, label='Validation Loss', marker='s', markersize=4)
    
    best_epoch = results.get('best_epoch', np.argmin(results['val_losses']))
    ax1.axvline(x=best_epoch + 1, color=colors['accent'], linestyle='--', 
                alpha=0.8, linewidth=2, label=f'Best Epoch ({best_epoch + 1})')
    
    ax1.set_title('🔥 Training & Validation Loss', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')
    
    # Accuracy curves
    ax2.plot(epochs, [acc * 100 for acc in results['train_accuracies']], 
             color=colors['train'], linewidth=2.5, label='Training Accuracy', 
             marker='o', markersize=4)
    ax2.plot(epochs, [acc * 100 for acc in results['val_accuracies']], 
             color=colors['val'], linewidth=2.5, label='Validation Accuracy', 
             marker='s', markersize=4)
    
    ax2.axvline(x=best_epoch + 1, color=colors['accent'], linestyle='--', 
                alpha=0.8, linewidth=2, label=f'Best Epoch ({best_epoch + 1})')
    
    ax2.set_title('🎯 Training & Validation Accuracy', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')
    
    # Learning rate (if available)
    if 'learning_rates' in results and results['learning_rates']:
        ax3.plot(epochs, results['learning_rates'], color=colors['accent'], 
                 linewidth=2.5, marker='o', markersize=3)
        ax3.set_title('📈 Learning Rate Schedule', fontsize=14, fontweight='bold', pad=20)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Learning Rate', fontsize=12)
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        ax3.set_facecolor('#f8f9fa')
    else:
        ax3.text(0.5, 0.5, '🔍 Learning Rate\nData Not Available', 
                ha='center', va='center', transform=ax3.transAxes,
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
        ax3.set_facecolor('#f8f9fa')
    
    # Training progress analysis
    train_val_gap = [abs(t - v) for t, v in zip(results['train_losses'], results['val_losses'])]
    ax4.plot(epochs, train_val_gap, color=colors['success'], linewidth=2.5, 
             marker='o', markersize=4, label='Train-Val Gap')
    ax4.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Overfitting Threshold')
    
    ax4.set_title('🧠 Generalization Analysis', fontsize=14, fontweight='bold', pad=20)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Loss Gap', fontsize=12)
    ax4.legend(frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    # Save the figure
    filepath = output_dir / "01_training_curves.png"
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return filepath

def create_confusion_matrix(eval_results, output_dir):
    """Create confusion matrix visualization"""
    print("📊 Creating confusion matrix...")
    
    if eval_results and 'confusion_matrix' in eval_results:
        cm = eval_results['confusion_matrix']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar_kws={'label': 'Count'}, square=True)
        
        ax.set_title('🎯 Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('True Class', fontsize=12)
        
        plt.tight_layout()
    else:
        # Create placeholder
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, '⚠️ Evaluation Required\n\nRun model evaluation to\ngenerate confusion matrix',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))
        ax.set_title('📊 Confusion Matrix (Placeholder)', fontsize=16, fontweight='bold')
        ax.set_facecolor('#f8f9fa')
    
    # Save the figure
    filepath = output_dir / "02_confusion_matrix.png"
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return filepath

def create_class_performance(eval_results, output_dir):
    """Create class performance analysis"""
    print("📊 Creating class performance analysis...")
    
    if eval_results and 'classification_report' in eval_results:
        class_report = eval_results['classification_report']
        
        # Extract metrics for visualization
        classes = []
        precision = []
        recall = []
        f1_score = []
        
        for class_idx, metrics in class_report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                classes.append(f'Class {class_idx}')
                precision.append(metrics['precision'])
                recall.append(metrics['recall'])
                f1_score.append(metrics['f1-score'])
        
        if classes:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            x = np.arange(len(classes))
            width = 0.25
            
            # Precision, Recall, F1 comparison
            ax1.bar(x - width, precision, width, label='Precision', color='#2E86AB', alpha=0.8)
            ax1.bar(x, recall, width, label='Recall', color='#A23B72', alpha=0.8)
            ax1.bar(x + width, f1_score, width, label='F1-Score', color='#F18F01', alpha=0.8)
            
            ax1.set_title('📊 Per-Class Performance Metrics', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Classes', fontsize=12)
            ax1.set_ylabel('Score', fontsize=12)
            ax1.set_xticks(x)
            ax1.set_xticklabels(classes, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1.1)
            
            # Individual metric plots
            for idx, (ax, metric, values, color, title) in enumerate([
                (ax2, 'Precision', precision, '#2E86AB', 'Precision by Class'),
                (ax3, 'Recall', recall, '#A23B72', 'Recall by Class'),
                (ax4, 'F1-Score', f1_score, '#F18F01', 'F1-Score by Class')
            ]):
                bars = ax.bar(classes, values, color=color, alpha=0.8)
                ax.set_title(f'📈 {title}', fontsize=12, fontweight='bold')
                ax.set_ylabel(metric, fontsize=10)
                ax.set_ylim(0, 1.1)
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
                
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
        else:
            # No valid class data
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, '📊 No Class Performance Data\n\nRun evaluation with classification report',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=14, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.7))
            ax.set_title('📊 Class Performance Analysis', fontsize=16, fontweight='bold')
            ax.set_facecolor('#f8f9fa')
    else:
        # Create placeholder
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, '⚠️ Evaluation Required\n\nRun model evaluation to\ngenerate class performance metrics',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.7))
        ax.set_title('📊 Class Performance Analysis (Placeholder)', fontsize=16, fontweight='bold')
        ax.set_facecolor('#f8f9fa')
    
    # Save the figure
    filepath = output_dir / "03_class_performance.png"
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return filepath

def create_temporal_analysis(results, data_info, output_dir):
    """Create temporal-specific analysis"""
    print("📊 Creating temporal analysis...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Model convergence analysis
    epochs = range(1, len(results['train_losses']) + 1)
    smoothed_train = pd.Series(results['train_losses']).rolling(window=3, center=True).mean()
    smoothed_val = pd.Series(results['val_losses']).rolling(window=3, center=True).mean()
    
    ax1.plot(epochs, results['train_losses'], alpha=0.5, color='#2E86AB', linewidth=1)
    ax1.plot(epochs, smoothed_train, color='#2E86AB', linewidth=2.5, label='Train (Smoothed)')
    ax1.plot(epochs, results['val_losses'], alpha=0.5, color='#A23B72', linewidth=1)
    ax1.plot(epochs, smoothed_val, color='#A23B72', linewidth=2.5, label='Val (Smoothed)')
    
    ax1.set_title('🧠 Loss Convergence Analysis', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Sequence length analysis
    seq_len = data_info.get('sequence_length', 10)
    ax2.bar(['Sequence Length'], [seq_len], color='#F18F01', alpha=0.8)
    ax2.set_title('📏 Temporal Sequence Configuration', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Length', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add text annotation
    ax2.text(0, seq_len/2, f'{seq_len} time steps', ha='center', va='center', 
             fontsize=12, fontweight='bold', color='white')
    
    # Training stability
    loss_std = pd.Series(results['train_losses']).rolling(window=5).std()
    epochs_std = range(5, len(results['train_losses']) + 1)
    
    ax3.plot(epochs_std, loss_std[4:], color='#C73E1D', linewidth=2.5, marker='o', markersize=3)
    ax3.set_title('📊 Training Stability (Loss Std Dev)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss Standard Deviation', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Model complexity vs performance
    total_params = data_info.get('total_parameters', 0)
    best_val_acc = results.get('best_val_acc', 0)
    
    ax4.scatter([total_params/1e6], [best_val_acc*100], s=200, color='#2E86AB', alpha=0.8)
    ax4.set_title('🎯 Model Efficiency', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Parameters (Millions)', fontsize=12)
    ax4.set_ylabel('Best Validation Accuracy (%)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # Add annotation
    ax4.annotate(f'Current Model\n{total_params/1e6:.1f}M params\n{best_val_acc*100:.1f}% acc',
                xy=(total_params/1e6, best_val_acc*100), xytext=(10, 10),
                textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    # Save the figure
    filepath = output_dir / "04_temporal_analysis.png"
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return filepath

def create_performance_summary(results, eval_results, model_info, data_info, output_dir):
    """Create comprehensive performance summary"""
    summary_path = output_dir / "performance_summary.txt"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("TEMPORAL CARDIAC RNN PERFORMANCE SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Performance metrics
        f.write("KEY PERFORMANCE METRICS:\n")
        f.write("-" * 25 + "\n")
        
        training_results = results
        
        # Test metrics
        if eval_results:
            test_acc = eval_results.get('test_accuracy', 'N/A')
            test_loss = eval_results.get('test_loss', 'N/A')
            
            if isinstance(test_acc, (int, float)):
                acc_str = f"{test_acc:.4f}"
            else:
                acc_str = str(test_acc)
            f.write(f"Test Accuracy: {acc_str}\n")
            
            if isinstance(test_loss, (int, float)):
                loss_str = f"{test_loss:.4f}"
            else:
                loss_str = str(test_loss)
            f.write(f"Test Loss: {loss_str}\n")
        else:
            f.write("Test Accuracy: N/A (evaluation not run)\n")
            f.write("Test Loss: N/A (evaluation not run)\n")
        
        # Training metrics
        best_val_acc = training_results.get('best_val_acc', 'N/A')
        best_epoch = training_results.get('best_epoch', None)
        
        if isinstance(best_val_acc, (int, float)):
            val_acc_str = f"{best_val_acc:.4f}"
        else:
            val_acc_str = str(best_val_acc)
        f.write(f"Best Validation Accuracy: {val_acc_str}\n")
        f.write(f"Best Epoch: {best_epoch + 1 if best_epoch is not None else 'N/A'}\n")
        f.write(f"Total Training Epochs: {len(training_results.get('train_losses', []))}\n\n")
        
        # Model information
        f.write("MODEL INFORMATION:\n")
        f.write("-" * 18 + "\n")
        f.write(f"Architecture: {model_info.get('architecture', 'N/A')}\n")
        
        total_params = model_info.get('total_parameters', 'N/A')
        if isinstance(total_params, (int, float)):
            params_str = f"{total_params:,}"
        else:
            params_str = str(total_params)
        f.write(f"Total Parameters: {params_str}\n")
        f.write(f"Memory Usage: {model_info.get('memory_usage_mb', 'N/A')} MB\n\n")
        
        # Generalization assessment
        f.write("GENERALIZATION ASSESSMENT:\n")
        f.write("-" * 26 + "\n")
        train_losses = training_results.get('train_losses', [])
        val_losses = training_results.get('val_losses', [])
        
        if train_losses and val_losses:
            gap = abs(train_losses[-1] - val_losses[-1])
            if gap < 0.1:
                f.write("✅ Excellent generalization (train-val gap < 0.1)\n")
            elif gap < 0.2:
                f.write("⚠️  Moderate overfitting (train-val gap < 0.2)\n")
            else:
                f.write("❌ High overfitting detected (train-val gap >= 0.2)\n")
        else:
            f.write("❓ Unable to assess generalization\n")
    
    return summary_path

def create_visualizations_after_training(results, eval_results, model_info, data_info):
    """Create all visualizations after training completes"""
    print("\n🎨 CREATING PERFORMANCE VISUALIZATIONS")
    print("=" * 55)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"temporal_rnn_visualizations_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Create all visualizations
    try:
        figures = []
        
        # 1. Training curves
        filepath1 = create_training_curves(results, output_dir)
        figures.append(('Training Curves', filepath1))
        print(f"✅ Saved: {filepath1}")
        
        # 2. Confusion matrix
        filepath2 = create_confusion_matrix(eval_results, output_dir)
        figures.append(('Confusion Matrix', filepath2))
        print(f"✅ Saved: {filepath2}")
        
        # 3. Class performance
        filepath3 = create_class_performance(eval_results, output_dir)
        figures.append(('Class Performance', filepath3))
        print(f"✅ Saved: {filepath3}")
        
        # 4. Temporal analysis
        filepath4 = create_temporal_analysis(results, data_info, output_dir)
        figures.append(('Temporal Analysis', filepath4))
        print(f"✅ Saved: {filepath4}")
        
        # 5. Performance summary
        summary_path = create_performance_summary(results, eval_results, model_info, data_info, output_dir)
        print(f"✅ Saved: {summary_path}")
        
        print(f"\n🎉 VISUALIZATION COMPLETE!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📄 Files created: {len(figures) + 1}")
        
        # Success message with key highlights
        print(f"\n🏆 PERFORMANCE HIGHLIGHTS:")
        best_val = results.get('best_val_acc', 'N/A')
        if isinstance(best_val, (int, float)):
            val_str = f"{best_val:.2%}"
        else:
            val_str = str(best_val)
        print(f"   Best Validation Accuracy: {val_str}")
        
        if eval_results and eval_results.get('test_accuracy'):
            test_acc = eval_results['test_accuracy']
            if isinstance(test_acc, (int, float)):
                test_str = f"{test_acc:.2%}"
            else:
                test_str = str(test_acc)
            print(f"   Test Accuracy: {test_str}")
        else:
            print("   Note: Run evaluation to get test metrics")
        
        # Format model parameters
        total_params = model_info.get('total_parameters', 'N/A')
        if isinstance(total_params, (int, float)):
            params_str = f"{total_params:,}"
        else:
            params_str = str(total_params)
        print(f"   Model Size: {params_str} parameters")
        print(f"   Training Status: {'Completed' if results.get('train_losses') else 'Incomplete'}")
        
        return output_dir, figures
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, []

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
        
        # 8. Create comprehensive visualizations
        print(f"\n🎨 Creating performance visualizations...")
        visualization_dir, viz_files = create_visualizations_after_training(
            results, eval_results, model_info, data_info
        )
        
        if visualization_dir:
            print(f"✅ Visualizations saved to: {visualization_dir}")
            # Log visualization info to Wandb
            wandb.log({
                "visualization_files_created": len(viz_files),
                "visualization_directory": str(visualization_dir)
            })
        
        # 9. Summary
        print(f"\n🎯 Training Summary:")
        print(f"   Model: {model_info['model_name']}")
        print(f"   Parameters: {model_info['trainable_parameters']:,}")
        print(f"   Training epochs: {len(results['train_losses'])}")
        print(f"   Final test accuracy: {eval_results['test_accuracy']:.4f}")
        print(f"   Wandb run: {wandb.run.url}")
        print(f"   Results saved: {results_file}")
        if visualization_dir:
            print(f"   Visualizations: {visualization_dir}")
        
        # Final summary to Wandb
        wandb.summary.update({
            "final_test_accuracy": eval_results['test_accuracy'],
            "final_test_loss": eval_results['test_loss'],
            "best_val_accuracy": results['best_val_acc'],
            "model_parameters": model_info['trainable_parameters'],
            "training_epochs": len(results['train_losses']),
            "overfitting_gap": final_gap,
            "visualizations_created": len(viz_files) if viz_files else 0
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
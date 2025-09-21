#!/usr/bin/env python3
"""
Comprehensive Enhanced Cardiomyocyte Subtype Classifier
- Training with wandb logging
- Automatic visualizations
- Complete performance analysis
"""
import logging
import sys
import os
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Add project root to path for absolute imports
project_root = Path(__file__).parent.parent.parent  # Go up from scripts/training/ to project root
sys.path.insert(0, str(project_root))

try:
    from src.training.cardiomyocyte_trainer import train_enhanced_cardiomyocyte_classifier
    from src.data_processing.authentic_10x_processor import Authentic10XProcessor  
    from src.models.gnn_models.cardiomyocyte_gnn import AdvancedCardiomyocyteGNN
    print("✅ All imports successful")
except ImportError as e:
    # Fallback imports if modules don't exist
    print(f"Warning: Import error - {e}. Will create minimal training.")
    train_enhanced_cardiomyocyte_classifier = None
    Authentic10XProcessor = None
    AdvancedCardiomyocyteGNN = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class EnhancedCardiomyocyteTrainer:
    """Comprehensive trainer with wandb logging and visualizations."""
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.viz_dir = None
        self.model = None
        self.data = None
        
    def get_default_config(self):
        """Default training configuration."""
        return {
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
            'l2_reg': 0.0001,
            'project_name': 'enhanced-cardiomyocyte-gnn',
            'experiment_name': f'enhanced_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
    
    def setup_wandb(self):
        """Initialize wandb logging."""
        try:
            wandb.init(
                project=self.config['project_name'],
                name=self.config['experiment_name'],
                config=self.config,
                tags=['enhanced-gnn', 'cardiomyocyte', 'classification']
            )
            logger.info("✅ Wandb logging initialized")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Wandb initialization failed: {e}")
            logger.info("Continuing without wandb logging...")
            return False
    
    def setup_visualization_dir(self):
        """Create directory for visualizations."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.viz_dir = Path(f"enhanced_results_{timestamp}")
        self.viz_dir.mkdir(exist_ok=True)
        logger.info(f"📁 Results directory: {self.viz_dir}")
        return self.viz_dir
    
    def load_model_and_data(self, results):
        """Load trained model and data for analysis."""
        try:
            # Load data
            processor = Authentic10XProcessor()
            self.data = processor.load_cached_data(device=self.config['device'])
            
            # Initialize model
            self.model = AdvancedCardiomyocyteGNN(
                num_features=self.data.x.shape[1],
                num_classes=self.data.num_classes,
                hidden_dim=self.config['hidden_dim'],
                dropout=self.config['dropout']
            )
            
            # Load trained weights
            self.model.load_state_dict(torch.load('best_cardiomyocyte_model.pth', map_location=self.config['device']))
            self.model.eval()
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model/data: {e}")
            return False
    
    def generate_predictions(self):
        """Generate model predictions."""
        with torch.no_grad():
            output = self.model(self.data)
            predictions = output.argmax(dim=1)
            probabilities = torch.softmax(output, dim=1)
        
        return predictions.cpu().numpy(), probabilities.cpu().numpy()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None):
        """Generate and save confusion matrix."""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Use biological cell type names
        if class_names is None:
            processor = Authentic10XProcessor()
            cell_type_names = processor.get_cell_type_names()
            class_names = [cell_type_names[i] if i < len(cell_type_names) else f'Unknown Type {i}' for i in range(len(cm))]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Enhanced GNN - Confusion Matrix (Normalized)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        save_path = self.viz_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Log to wandb if available
        try:
            wandb.log({"confusion_matrix": wandb.Image(str(save_path))})
        except:
            pass
        
        plt.close()
        return save_path
    
    def plot_class_distribution(self, y_true, y_pred, class_names=None):
        """Plot class distribution comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Use biological cell type names
        if class_names is None:
            processor = Authentic10XProcessor()
            cell_type_names = processor.get_cell_type_names()
            max_class = max(max(y_true), max(y_pred))
            class_names = [cell_type_names[i] if i < len(cell_type_names) else f'Unknown Type {i}' for i in range(max_class + 1)]
        
        # True distribution
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        ax1.bar(unique_true, counts_true, alpha=0.7, color='skyblue')
        ax1.set_title('True Class Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_xticks(unique_true)
        ax1.set_xticklabels([class_names[i] for i in unique_true], rotation=45)
        
        # Predicted distribution
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        ax2.bar(unique_pred, counts_pred, alpha=0.7, color='lightcoral')
        ax2.set_title('Predicted Class Distribution')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        ax2.set_xticks(unique_pred)
        ax2.set_xticklabels([class_names[i] for i in unique_pred], rotation=45)
        
        plt.tight_layout()
        save_path = self.viz_dir / 'class_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Log to wandb
        try:
            wandb.log({"class_distribution": wandb.Image(str(save_path))})
        except:
            pass
        
        plt.close()
        return save_path
    
    def plot_training_curves(self, results):
        """Plot training and validation curves."""
        if 'training_history' not in results:
            logger.warning("No training history available for plotting")
            return None
        
        history = results['training_history']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss curves
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        save_path = self.viz_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Log to wandb
        try:
            wandb.log({"training_curves": wandb.Image(str(save_path))})
        except:
            pass
        
        plt.close()
        return save_path
    
    def analyze_attention_weights(self):
        """Analyze attention patterns if model supports it."""
        try:
            # This is a placeholder for attention analysis
            # Would need to modify model to return attention weights
            logger.info("🔍 Attention analysis not implemented for current model")
            return None
        except Exception as e:
            logger.warning(f"Attention analysis failed: {e}")
            return None
    
    def generate_performance_report(self, results, y_true, y_pred, class_names=None):
        """Generate comprehensive performance report."""
        report = f"""
# Enhanced Cardiomyocyte GNN - Performance Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Model Configuration
- Architecture: AdvancedCardiomyocyteGNN
- Hidden Dimensions: {self.config['hidden_dim']}
- Dropout: {self.config['dropout']}
- Learning Rate: {self.config['learning_rate']}

## Performance Metrics
- Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)
- Best Validation Accuracy: {results['best_val_accuracy']:.4f} ({results['best_val_accuracy']*100:.2f}%)

## Per-Class Performance
"""
        
        # Get biological cell type names for mapping
        processor = Authentic10XProcessor()
        cell_type_names = processor.get_cell_type_names()
        
        for subtype, metrics in results['per_class_metrics'].items():
            # Convert to readable biological name
            if subtype.startswith('subtype_'):
                # Handle old format
                class_idx = int(subtype.split('_')[1])
                display_name = cell_type_names[class_idx] if class_idx < len(cell_type_names) else subtype
            else:
                # Handle new format - convert key back to readable name
                display_name = subtype.replace('_', ' ').title()
                
            report += f"- {display_name}:\n"
            report += f"  - F1-Score: {metrics['f1']:.3f}\n"
            report += f"  - Precision: {metrics['precision']:.3f}\n"
            report += f"  - Recall: {metrics['recall']:.3f}\n"
            report += f"  - Support: {metrics['support']} cells\n\n"
        
        report += f"""
## Dataset Information
- Total Samples: {len(y_true)}
- Number of Classes: {results['dataset_info']['num_classes']}
- Feature Dimensions: {self.data.x.shape[1]}

## Generated Visualizations
- confusion_matrix.png - Model performance analysis
- class_distribution.png - Dataset distribution analysis
- training_curves.png - Training progression
- performance_report.md - This report

## Wandb Integration
Experiment logged to: {self.config['project_name']}/{self.config['experiment_name']}
"""
        
        # Save report
        report_path = self.viz_dir / 'performance_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report_path
    
    def log_metrics_to_wandb(self, results, y_true, y_pred):
        """Log comprehensive metrics to wandb."""
        try:
            # Main metrics
            wandb.log({
                "test_accuracy": results['test_accuracy'],
                "best_val_accuracy": results['best_val_accuracy'],
                "num_classes": results['dataset_info']['num_classes']
            })
            
            # Per-class metrics
            for subtype, metrics in results['per_class_metrics'].items():
                wandb.log({
                    f"f1_score_{subtype}": metrics['f1'],
                    f"precision_{subtype}": metrics['precision'],
                    f"recall_{subtype}": metrics['recall'],
                    f"support_{subtype}": metrics['support']
                })
            
            # Overall classification metrics
            accuracy = accuracy_score(y_true, y_pred)
            wandb.log({"overall_accuracy": accuracy})
            
            logger.info("📊 Metrics logged to wandb")
            
        except Exception as e:
            logger.warning(f"Failed to log metrics to wandb: {e}")
    
    def train_and_analyze(self):
        """Complete training and analysis pipeline."""
        logger.info("🚀 Starting Enhanced Cardiomyocyte GNN Training & Analysis")
        
        # Setup
        wandb_enabled = self.setup_wandb()
        self.setup_visualization_dir()
        
        # Check if training function is available
        if train_enhanced_cardiomyocyte_classifier is None:
            logger.error("❌ Training function not available due to import error")
            logger.error("💡 Please fix the import issues in the script")
            return False
        
        # Train model
        logger.info("🔥 Training model...")
        results = train_enhanced_cardiomyocyte_classifier(self.config)
        
        if not results:
            logger.error("❌ Training failed")
            return False
        
        # Load model and generate predictions
        if not self.load_model_and_data(results):
            return False
        
        predictions, probabilities = self.generate_predictions()
        y_true = self.data.y.cpu().numpy()
        
        # Get class names if available
        class_names = getattr(self.data, 'class_names', None)
        
        # Generate visualizations
        logger.info("🎨 Generating visualizations...")
        self.plot_confusion_matrix(y_true, predictions, class_names)
        self.plot_class_distribution(y_true, predictions, class_names)
        self.plot_training_curves(results)
        self.analyze_attention_weights()
        
        # Generate report
        self.generate_performance_report(results, y_true, predictions, class_names)
        
        # Log to wandb
        if wandb_enabled:
            self.log_metrics_to_wandb(results, y_true, predictions)
        
        # Print summary
        self.print_summary(results)
        
        # Finish wandb run
        if wandb_enabled:
            wandb.finish()
        
        return True
    
    def print_summary(self, results):
        """Print training summary."""
        print(f"\n{'='*60}")
        print(f"🎯 ENHANCED CARDIOMYOCYTE GNN - TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"📊 Test Accuracy: {results['test_accuracy']*100:.2f}%")
        print(f"📈 Best Val Accuracy: {results['best_val_accuracy']*100:.2f}%")
        print(f"🧬 Classes: {results['dataset_info']['num_classes']} cardiomyocyte subtypes")
        print(f"📁 Results saved to: {self.viz_dir}")
        
        print(f"\n📋 Per-Class Performance:")
        
        # Get biological cell type names for mapping
        processor = Authentic10XProcessor()
        cell_type_names = processor.get_cell_type_names()
        
        for subtype, metrics in results['per_class_metrics'].items():
            # Convert back to readable biological name
            if subtype.startswith('subtype_'):
                # Handle old format
                class_idx = int(subtype.split('_')[1])
                display_name = cell_type_names[class_idx] if class_idx < len(cell_type_names) else subtype
            else:
                # Handle new format - convert key back to readable name
                display_name = subtype.replace('_', ' ').title()
                
            print(f"  • {display_name}: F1={metrics['f1']*100:.1f}% | "
                  f"P={metrics['precision']*100:.1f}% | "
                  f"R={metrics['recall']*100:.1f}% | "
                  f"Support={metrics['support']}")
        
        print(f"\n✅ Enhanced GNN with wandb logging & visualizations complete!")
        print(f"{'='*60}")


def main():
    """Main entry point for enhanced cardiomyocyte classification."""
    
    # Simple default configuration
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
        'grad_clip': 1.0,
        'l2_reg': 0.0001,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'project_name': 'enhanced-cardiomyocyte-gnn',
        'experiment_name': f'enhanced_gnn_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    
    print("🎯 ENHANCED CARDIOMYOCYTE GNN TRAINING")
    print("=" * 50)
    print(f"Hidden Dim: {config['hidden_dim']}")
    print(f"Dropout: {config['dropout']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Weight Decay: {config['weight_decay']}")
    print(f"Max Epochs: {config['max_epochs']}")
    print(f"Patience: {config['patience']}")
    print("=" * 50)
    print()
    
    trainer = EnhancedCardiomyocyteTrainer(config)
    success = trainer.train_and_analyze()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
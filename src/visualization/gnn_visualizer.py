"""
Comprehensive visualization tools for GNN model analysis.

This module provides visualization capabilities for:
- Training metrics and performance analysis
- Confusion matrices with biological interpretations
- Loss curves and learning dynamics
- Model performance comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import pandas as pd
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")


class GNNVisualizer:
    """Comprehensive visualization tools for GNN model analysis."""
    
    def __init__(self, save_dir: str = "visualizations", 
                 class_names: Optional[List[str]] = None):
        """
        Initialize the GNN visualizer.
        
        Args:
            save_dir: Directory to save visualization plots
            class_names: List of class names for labeling
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Default cardiomyocyte subtype names
        self.class_names = class_names or [
            'Atrial CM', 'Ventricular CM', 'Conducting CM', 
            'Nodal CM', 'Epicardial CM'
        ]
        
        # Color palette for consistency
        self.colors = sns.color_palette("husl", len(self.class_names))
        
    def plot_confusion_matrix(self, 
                            y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            normalize: bool = True,
                            title: str = "Confusion Matrix",
                            save_name: str = "confusion_matrix.png") -> None:
        """
        Plot a detailed confusion matrix with biological interpretations.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize the confusion matrix
            title: Plot title
            save_name: Filename to save the plot
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            cbar_label = 'Normalized Frequency'
        else:
            fmt = 'd'
            cbar_label = 'Count'
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Main confusion matrix
        im1 = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
        ax1.figure.colorbar(im1, ax=ax1, label=cbar_label)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax1.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=10)
        
        ax1.set_title(f'{title}\n(Normalized)' if normalize else title, fontsize=14)
        ax1.set_ylabel('True Cardiomyocyte Subtype', fontsize=12)
        ax1.set_xlabel('Predicted Cardiomyocyte Subtype', fontsize=12)
        ax1.set_xticks(range(len(self.class_names)))
        ax1.set_yticks(range(len(self.class_names)))
        ax1.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax1.set_yticklabels(self.class_names)
        
        # Class-wise performance metrics
        precision = np.diag(cm) / np.sum(cm, axis=0)
        recall = np.diag(cm) / np.sum(cm, axis=1)
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        # Bar plot of metrics
        x_pos = np.arange(len(self.class_names))
        width = 0.25
        
        ax2.bar(x_pos - width, precision, width, label='Precision', alpha=0.8)
        ax2.bar(x_pos, recall, width, label='Recall', alpha=0.8)
        ax2.bar(x_pos + width, f1_score, width, label='F1-Score', alpha=0.8)
        
        ax2.set_title('Per-Class Performance Metrics', fontsize=14)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_xlabel('Cardiomyocyte Subtype', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax2.legend()
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_training_curves(self, 
                           train_losses: List[float],
                           val_losses: List[float],
                           train_accuracies: List[float],
                           val_accuracies: List[float],
                           save_name: str = "training_curves.png") -> None:
        """
        Plot comprehensive training curves with analysis.
        
        Args:
            train_losses: Training loss history
            val_losses: Validation loss history
            train_accuracies: Training accuracy history
            val_accuracies: Validation accuracy history
            save_name: Filename to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss During Training', fontsize=14)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy During Training', fontsize=14)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Loss difference (overfitting indicator)
        loss_diff = np.array(val_losses) - np.array(train_losses)
        ax3.plot(epochs, loss_diff, 'g-', linewidth=2)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_title('Overfitting Indicator (Val Loss - Train Loss)', fontsize=14)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Loss Difference', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Smoothed learning rate (if available) or gradient norm
        ax4.plot(epochs, np.gradient(train_losses), 'purple', linewidth=2, label='Loss Gradient')
        ax4.set_title('Learning Dynamics (Loss Gradient)', fontsize=14)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Loss Gradient', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_class_distribution(self, 
                              labels: np.ndarray,
                              title: str = "Cardiomyocyte Subtype Distribution",
                              save_name: str = "class_distribution.png") -> None:
        """
        Plot class distribution with biological context.
        
        Args:
            labels: Array of class labels
            title: Plot title
            save_name: Filename to save the plot
        """
        unique, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels) * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        bars = ax1.bar(self.class_names, counts, color=self.colors, alpha=0.8)
        ax1.set_title(f'{title}\n(Total: {len(labels)} cells)', fontsize=14)
        ax1.set_ylabel('Number of Cells', fontsize=12)
        ax1.set_xlabel('Cardiomyocyte Subtype', fontsize=12)
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
        
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Pie chart
        ax2.pie(counts, labels=self.class_names, colors=self.colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 10})
        ax2.set_title('Subtype Proportion', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_model_comparison(self, 
                            results_dict: Dict[str, Dict],
                            save_name: str = "model_comparison.png") -> None:
        """
        Compare multiple model results.
        
        Args:
            results_dict: Dictionary with model names as keys and results as values
            save_name: Filename to save the plot
        """
        models = list(results_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [results_dict[model].get(metric, 0) for model in models]
            ax.bar(x + i*width, values, width, label=metric.title(), alpha=0.8)
        
        ax.set_title('Model Performance Comparison', fontsize=16)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_training_report(self, 
                             training_history: Dict,
                             test_results: Dict,
                             save_name: str = "training_report.png") -> None:
        """
        Create a comprehensive training report visualization.
        
        Args:
            training_history: Dictionary containing training metrics
            test_results: Dictionary containing test results
            save_name: Filename to save the plot
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Training curves
        ax1 = plt.subplot(2, 4, 1)
        epochs = range(1, len(training_history['train_loss']) + 1)
        plt.plot(epochs, training_history['train_loss'], 'b-', label='Train')
        plt.plot(epochs, training_history['val_loss'], 'r-', label='Validation')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2 = plt.subplot(2, 4, 2)
        plt.plot(epochs, training_history['train_acc'], 'b-', label='Train')
        plt.plot(epochs, training_history['val_acc'], 'r-', label='Validation')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Final test metrics
        ax3 = plt.subplot(2, 4, 3)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [
            test_results.get('accuracy', 0),
            test_results.get('precision', 0), 
            test_results.get('recall', 0),
            test_results.get('f1_score', 0)
        ]
        bars = plt.bar(metrics, values, color=sns.color_palette("viridis", len(metrics)))
        plt.title('Final Test Performance')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Learning rate schedule (if available)
        ax4 = plt.subplot(2, 4, 4)
        if 'learning_rates' in training_history:
            plt.plot(epochs, training_history['learning_rates'])
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
        else:
            plt.text(0.5, 0.5, 'Learning Rate\nSchedule\nNot Available', 
                    ha='center', va='center', transform=ax4.transAxes)
            plt.title('Learning Rate Schedule')
        
        # Confusion matrix
        ax5 = plt.subplot(2, 4, (5, 6))
        if 'confusion_matrix' in test_results:
            cm = np.array(test_results['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names)
            plt.title('Test Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
        
        # Per-class performance
        ax6 = plt.subplot(2, 4, (7, 8))
        if 'per_class_metrics' in test_results:
            class_metrics = test_results['per_class_metrics']
            metrics_df = pd.DataFrame(class_metrics).T
            metrics_df.plot(kind='bar', ax=ax6)
            plt.title('Per-Class Performance')
            plt.xlabel('Class')
            plt.ylabel('Score')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
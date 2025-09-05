"""
Experiment tracking and model checkpointing utilities.
Integrates with Weights & Biases, TensorBoard, and custom logging.
"""

import os
import json
import yaml
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import wandb

logger = logging.getLogger(__name__)

class ExperimentTracker:
    """Comprehensive experiment tracking and management."""
    
    def __init__(
        self,
        experiment_name: str,
        project_name: str = "HybridGNNRNN-Cardiomyocyte",
        output_dir: str = "/Users/tumokgabeng/Projects/HybridGnnRnn/experiments",
        use_wandb: bool = True,
        use_tensorboard: bool = True
    ):
        self.experiment_name = experiment_name
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        
        # Create experiment directory
        self.experiment_dir = self.output_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.logs_dir = self.experiment_dir / "logs"
        self.plots_dir = self.experiment_dir / "plots"
        self.results_dir = self.experiment_dir / "results"
        
        for dir_path in [self.checkpoints_dir, self.logs_dir, self.plots_dir, self.results_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize loggers
        self.setup_logging()
        
        # Initialize W&B if requested
        if self.use_wandb:
            self.init_wandb()
        
        # Initialize TensorBoard if requested
        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=str(self.logs_dir / "tensorboard"))
        
        # Tracking data
        self.metrics_history = {
            'train': {},
            'val': {},
            'test': {}
        }
        self.best_metrics = {}
        self.artifacts = {}
        
        logger.info(f"Initialized experiment tracker: {experiment_name}")
    
    def setup_logging(self):
        """Setup file logging for the experiment."""
        log_file = self.logs_dir / f"{self.experiment_name}.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
    
    def init_wandb(self):
        """Initialize Weights & Biases tracking."""
        try:
            wandb.init(
                project=self.project_name,
                name=self.experiment_name,
                dir=str(self.logs_dir),
                save_code=True
            )
            logger.info("Initialized Weights & Biases tracking")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            self.use_wandb = False
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        # Save config to file
        config_file = self.experiment_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Log to W&B
        if self.use_wandb:
            wandb.config.update(config)
        
        logger.info(f"Logged configuration to {config_file}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        split: str = "train"
    ):
        """Log training/validation metrics."""
        # Store in history
        if split not in self.metrics_history:
            self.metrics_history[split] = {}
        
        for key, value in metrics.items():
            if key not in self.metrics_history[split]:
                self.metrics_history[split][key] = []
            self.metrics_history[split][key].append(value)
        
        # Log to W&B
        if self.use_wandb:
            wandb_metrics = {f"{split}_{key}": value for key, value in metrics.items()}
            wandb_metrics['step'] = step
            wandb.log(wandb_metrics)
        
        # Log to TensorBoard
        if self.use_tensorboard:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f"{split}/{key}", value, step)
    
    def log_model_summary(self, model, input_sample=None):
        """Log model architecture summary."""
        summary_file = self.experiment_dir / "model_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("Model Architecture Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Model summary
            if hasattr(model, 'get_model_summary'):
                summary = model.get_model_summary()
                f.write(f"Total Parameters: {summary['total_parameters']:,}\n")
                f.write(f"Trainable Parameters: {summary['trainable_parameters']:,}\n")
                if 'model_size_mb' in summary:
                    f.write(f"Model Size (MB): {summary['model_size_mb']:.2f}\n\n")
                else:
                    f.write(f"Model Size (MB): {summary['total_parameters'] * 4 / (1024*1024):.2f}\n\n")
            
            # Model structure
            f.write("Model Structure:\n")
            f.write(str(model))
        
        # Log to W&B
        if self.use_wandb and hasattr(model, 'get_model_summary'):
            summary = model.get_model_summary()
            wandb.config.update({
                'total_parameters': summary['total_parameters'],
                'trainable_parameters': summary['trainable_parameters'],
                'model_size_mb': summary.get('model_size_mb', summary['total_parameters'] * 4 / (1024*1024))
            })
        
        logger.info(f"Logged model summary to {summary_file}")
    
    def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        additional_data: Dict = None
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        if additional_data:
            checkpoint.update(additional_data)
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoints_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            
            # Update best metrics
            self.best_metrics = metrics.copy()
            
            # Log to W&B
            if self.use_wandb:
                wandb.run.summary["best_epoch"] = epoch
                for key, value in metrics.items():
                    wandb.run.summary[f"best_{key}"] = value
        
        # Save latest checkpoint
        latest_path = self.checkpoints_dir / "latest_model.pt"
        torch.save(checkpoint, latest_path)
        
        logger.info(f"Saved checkpoint for epoch {epoch}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str, model, optimizer=None, scheduler=None):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return checkpoint
    
    def save_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        split: str = "test"
    ):
        """Save model predictions and targets."""
        results_file = self.results_dir / f"{split}_predictions.npz"
        
        data_to_save = {}
        data_to_save.update({f"pred_{key}": value for key, value in predictions.items()})
        data_to_save.update({f"target_{key}": value for key, value in targets.items()})
        
        np.savez(results_file, **data_to_save)
        logger.info(f"Saved predictions to {results_file}")
    
    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        split: str = "test"
    ):
        """Generate and save classification report."""
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True
        )
        
        # Save report
        report_file = self.results_dir / f"{split}_classification_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate confusion matrix plot
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names,
            cmap='Blues'
        )
        plt.title(f'Confusion Matrix - {split.title()} Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_file = self.plots_dir / f"{split}_confusion_matrix.png"
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to W&B
        if self.use_wandb:
            wandb.log({
                f"{split}_confusion_matrix": wandb.Image(str(cm_file)),
                f"{split}_accuracy": report['accuracy'],
                f"{split}_macro_f1": report['macro avg']['f1-score']
            })
        
        logger.info(f"Generated classification report for {split} set")
        
        return report
    
    def plot_training_curves(self):
        """Generate training curves plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot loss curves
        if 'train' in self.metrics_history and 'total_loss' in self.metrics_history['train']:
            axes[0, 0].plot(self.metrics_history['train']['total_loss'], label='Train')
        if 'val' in self.metrics_history and 'total_loss' in self.metrics_history['val']:
            axes[0, 0].plot(self.metrics_history['val']['total_loss'], label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot accuracy curves
        if 'train' in self.metrics_history and 'accuracy' in self.metrics_history['train']:
            axes[0, 1].plot(self.metrics_history['train']['accuracy'], label='Train')
        if 'val' in self.metrics_history and 'accuracy' in self.metrics_history['val']:
            axes[0, 1].plot(self.metrics_history['val']['accuracy'], label='Validation')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot efficiency loss
        if 'train' in self.metrics_history and 'efficiency_loss' in self.metrics_history['train']:
            axes[1, 0].plot(self.metrics_history['train']['efficiency_loss'], label='Train')
        if 'val' in self.metrics_history and 'efficiency_loss' in self.metrics_history['val']:
            axes[1, 0].plot(self.metrics_history['val']['efficiency_loss'], label='Validation')
        axes[1, 0].set_title('Efficiency Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot maturation loss
        if 'train' in self.metrics_history and 'maturation_loss' in self.metrics_history['train']:
            axes[1, 1].plot(self.metrics_history['train']['maturation_loss'], label='Train')
        if 'val' in self.metrics_history and 'maturation_loss' in self.metrics_history['val']:
            axes[1, 1].plot(self.metrics_history['val']['maturation_loss'], label='Validation')
        axes[1, 1].set_title('Maturation Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        curves_file = self.plots_dir / "training_curves.png"
        plt.savefig(curves_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to W&B
        if self.use_wandb:
            wandb.log({"training_curves": wandb.Image(str(curves_file))})
        
        logger.info(f"Generated training curves plot: {curves_file}")
    
    def save_experiment_summary(self):
        """Save comprehensive experiment summary."""
        summary = {
            'experiment_name': self.experiment_name,
            'project_name': self.project_name,
            'timestamp': datetime.now().isoformat(),
            'best_metrics': self.best_metrics,
            'final_metrics': {
                split: {key: values[-1] if values else None for key, values in metrics.items()}
                for split, metrics in self.metrics_history.items()
            },
            'total_epochs': len(self.metrics_history.get('train', {}).get('total_loss', [])),
            'artifacts': self.artifacts
        }
        
        summary_file = self.experiment_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved experiment summary to {summary_file}")
        
        return summary
    
    def add_artifact(self, name: str, path: str, description: str = ""):
        """Add an artifact to the experiment."""
        self.artifacts[name] = {
            'path': path,
            'description': description,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log to W&B
        if self.use_wandb:
            artifact = wandb.Artifact(name, type="model")
            artifact.add_file(path)
            wandb.log_artifact(artifact)
    
    def finish(self):
        """Finalize experiment tracking."""
        # Generate final plots
        self.plot_training_curves()
        
        # Save experiment summary
        self.save_experiment_summary()
        
        # Close loggers
        if self.use_tensorboard:
            self.tb_writer.close()
        
        if self.use_wandb:
            wandb.finish()
        
        logger.info(f"Finished experiment: {self.experiment_name}")


class CheckpointManager:
    """Advanced checkpoint management with automatic cleanup."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_best: bool = True,
        save_latest: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.save_latest = save_latest
        
        self.checkpoints = []
        self.best_checkpoint = None
        self.best_metric = float('inf')
    
    def save(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        metrics: Dict[str, float],
        metric_name: str = "val_loss",
        higher_better: bool = False
    ):
        """Save checkpoint with automatic management."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Check if this is the best checkpoint
        current_metric = metrics.get(metric_name, float('inf'))
        is_best = False
        
        if higher_better:
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                is_best = True
        else:
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                is_best = True
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Track checkpoint
        self.checkpoints.append({
            'path': checkpoint_path,
            'epoch': epoch,
            'metric': current_metric
        })
        
        # Save best checkpoint
        if is_best and self.save_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.best_checkpoint = best_path
        
        # Save latest checkpoint
        if self.save_latest:
            latest_path = self.checkpoint_dir / "latest_model.pt"
            torch.save(checkpoint, latest_path)
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        return checkpoint_path, is_best
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit."""
        if len(self.checkpoints) > self.max_checkpoints:
            # Sort by epoch and keep the most recent ones
            self.checkpoints.sort(key=lambda x: x['epoch'])
            
            # Remove oldest checkpoints
            to_remove = self.checkpoints[:-self.max_checkpoints]
            for checkpoint_info in to_remove:
                if checkpoint_info['path'].exists():
                    checkpoint_info['path'].unlink()
            
            # Update checkpoint list
            self.checkpoints = self.checkpoints[-self.max_checkpoints:]
    
    def load_best(self, model, optimizer=None, scheduler=None):
        """Load the best checkpoint."""
        if self.best_checkpoint and self.best_checkpoint.exists():
            checkpoint = torch.load(self.best_checkpoint, map_location='cpu')
            
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            return checkpoint
        else:
            raise FileNotFoundError("No best checkpoint found")
    
    def load_latest(self, model, optimizer=None, scheduler=None):
        """Load the latest checkpoint."""
        latest_path = self.checkpoint_dir / "latest_model.pt"
        if latest_path.exists():
            checkpoint = torch.load(latest_path, map_location='cpu')
            
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            return checkpoint
        else:
            raise FileNotFoundError("No latest checkpoint found")


if __name__ == "__main__":
    # Example usage
    tracker = ExperimentTracker("test_experiment")
    
    # Log configuration
    config = {
        'model_type': 'hybrid_gnn_rnn',
        'learning_rate': 0.001,
        'batch_size': 32
    }
    tracker.log_config(config)
    
    # Simulate training
    for epoch in range(5):
        metrics = {
            'total_loss': 1.0 - epoch * 0.1,
            'accuracy': 0.5 + epoch * 0.1
        }
        tracker.log_metrics(metrics, epoch, 'train')
    
    tracker.finish()

"""
Large-Scale Pseudo-Spatial GNN Training Script
Train GNN on 50K cell pseudo-spatial dataset with GPU optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import logging
import time
import json
from pathlib import Path
import argparse
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Wandb for experiment tracking
import wandb

# Import our enhanced spatial GNN
from enhanced_spatial_gnn import EnhancedSpatialGNN

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class LargeScaleSpatialGNNTrainer:
    """Trainer for large-scale pseudo-spatial GNN"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # GPU optimization settings
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # Create experiment directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = f"large_pseudo_spatial_gnn_{timestamp}"
        self.exp_dir = Path('experiments_large_pseudo_spatial') / self.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Wandb
        self.setup_wandb()
        
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        logger.info(f"Experiment directory: {self.exp_dir}")
        logger.info(f"Wandb run: {wandb.run.url}")
    
    def setup_wandb(self):
        """Initialize Wandb experiment tracking"""
        wandb.init(
            project="hybrid-gnn-rnn-cardiac-spatial",
            name=self.exp_name,
            config=self.config,
            tags=["spatial-gnn", "cardiomyocyte", "large-scale", "pseudo-spatial"],
            notes="Large-scale pseudo-spatial GNN training for cardiomyocyte differentiation efficiency prediction",
            save_code=True
        )
        
        # Log system info
        wandb.config.update({
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
            "experiment_name": self.exp_name,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A"
        })
        logger.info(f"Wandb run: {wandb.run.url}")
    
    def load_data(self):
        """Load the large-scale pseudo-spatial dataset"""
        
        logger.info("📊 Loading large-scale pseudo-spatial dataset...")
        
        data_path = 'data/large_scale_pseudo_spatial_50k.pt'
        
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        # Load the PyTorch Geometric data
        data = torch.load(data_path, map_location='cpu', weights_only=False)
        
        logger.info(f"Dataset loaded: {data.num_nodes:,} nodes, {data.num_features:,} features")
        logger.info(f"Classes: {data.num_classes}")
        logger.info(f"Edges: {data.edge_index.shape[1]:,}")
        logger.info(f"Cell types: {', '.join(data.cell_types)}")
        
        # Log dataset info to Wandb
        wandb.config.update({
            "num_nodes": data.num_nodes,
            "num_features": data.num_features,
            "num_classes": data.num_classes,
            "num_edges": data.edge_index.shape[1],
            "cell_types": data.cell_types
        })
        
        return data
    
    def create_data_splits(self, data):
        """Create train/validation/test splits"""
        
        logger.info("🔀 Creating data splits...")
        
        n_nodes = data.num_nodes
        indices = np.arange(n_nodes)
        
        # Stratified split to maintain class balance
        train_idx, temp_idx = train_test_split(
            indices, 
            test_size=0.3,  # 70% train, 30% temp
            stratify=data.y.numpy(),
            random_state=42
        )
        
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.5,  # Split temp into 15% val, 15% test
            stratify=data.y[temp_idx].numpy(),
            random_state=42
        )
        
        # Create boolean masks
        data.train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        
        data.train_mask[train_idx] = True
        data.val_mask[val_idx] = True
        data.test_mask[test_idx] = True
        
        logger.info(f"Data splits - Train: {len(train_idx):,}, Val: {len(val_idx):,}, Test: {len(test_idx):,}")
        
        # Log class distribution in training set
        train_labels = data.y[data.train_mask]
        unique_labels, counts = torch.unique(train_labels, return_counts=True)
        logger.info("Training set class distribution:")
        class_distribution = {}
        for label, count in zip(unique_labels, counts):
            cell_type = data.cell_types[label.item()]
            logger.info(f"  {cell_type}: {count.item():,} cells")
            class_distribution[cell_type] = count.item()
        
        # Log to Wandb
        wandb.log({
            "train_size": len(train_idx),
            "val_size": len(val_idx), 
            "test_size": len(test_idx),
            "class_distribution": class_distribution
        })
        
        return data
    
    def build_model(self, data):
        """Build the Enhanced Spatial GNN model"""
        
        logger.info("🧠 Building Enhanced Spatial GNN model...")
        
        # Model configuration optimized for large dataset
        model_config = {
            'input_dim': data.num_features,
            'hidden_dims': [2048, 1024, 512, 256],  # Larger network for richer dataset
            'output_dim': 128,
            'num_classes': data.num_classes,
            'dropout': 0.3,  # Moderate dropout for regularization
            'num_heads': 8,  # More attention heads for complex patterns
            'conv_type': 'GAT',
            'use_residual': True,
            'use_attention': True
        }
        
        self.model = EnhancedSpatialGNN(**model_config).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Enhanced SpatialGNN initialized")
        logger.info(f"Architecture: {data.num_features} -> {model_config['hidden_dims']} -> {model_config['output_dim']} -> {data.num_classes}")
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Log model info to Wandb
        wandb.config.update({
            "model_config": model_config,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        })
        
        return model_config
    
    def setup_training(self, data):
        """Setup training components"""
        
        logger.info("⚙️ Setting up training components...")
        
        # Class weights for imbalanced dataset
        train_labels = data.y[data.train_mask]
        unique_labels, counts = torch.unique(train_labels, return_counts=True)
        class_weights = 1.0 / counts.float()
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = class_weights.to(self.device)
        
        logger.info(f"Class weights: {class_weights}")
        
        # Loss function with class weights
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer - lower learning rate for stability
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.0001,  # Conservative learning rate
            weight_decay=0.01,  # Strong regularization
            eps=1e-8
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize validation accuracy
            factor=0.8,
            patience=5,
            min_lr=1e-6
        )
        
        # Mixed precision scaler for GPU efficiency
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        if self.scaler:
            logger.info("Mixed precision training enabled")
        
        logger.info("Training components configured")
    
    def train_epoch(self, data, epoch):
        """Train for one epoch"""
        
        self.model.train()
        data = data.to(self.device)
        
        self.optimizer.zero_grad()
        
        if self.scaler is not None:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                output = self.model(data)
                out = output[0] if isinstance(output, tuple) else output  # Get classification logits
                loss = self.criterion(out[data.train_mask].float(), data.y[data.train_mask])
            
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard training
            output = self.model(data)
            out = output[0] if isinstance(output, tuple) else output  # Get classification logits
            loss = self.criterion(out[data.train_mask].float(), data.y[data.train_mask])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        # Calculate training metrics
        with torch.no_grad():
            pred = out[data.train_mask].argmax(dim=1)
            train_acc = accuracy_score(data.y[data.train_mask].cpu(), pred.cpu())
            train_f1 = f1_score(data.y[data.train_mask].cpu(), pred.cpu(), average='macro', zero_division=0)
        
        return loss.item(), train_acc, train_f1
    
    def validate(self, data):
        """Validate the model"""
        
        self.model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    out = output[0] if isinstance(output, tuple) else output  # Get classification logits
            else:
                output = self.model(data)
                out = output[0] if isinstance(output, tuple) else output  # Get classification logits
            
            # Validation loss
            val_loss = self.criterion(out[data.val_mask].float(), data.y[data.val_mask]).item()
            
            # Validation metrics
            pred = out[data.val_mask].argmax(dim=1)
            val_acc = accuracy_score(data.y[data.val_mask].cpu(), pred.cpu())
            val_f1 = f1_score(data.y[data.val_mask].cpu(), pred.cpu(), average='macro', zero_division=0)
        
        return val_loss, val_acc, val_f1
    
    def test(self, data):
        """Test the final model"""
        
        self.model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    out = output[0] if isinstance(output, tuple) else output  # Get classification logits
            else:
                output = self.model(data)
                out = output[0] if isinstance(output, tuple) else output  # Get classification logits
            
            # Test metrics
            pred = out[data.test_mask].argmax(dim=1)
            test_acc = accuracy_score(data.y[data.test_mask].cpu(), pred.cpu())
            test_f1 = f1_score(data.y[data.test_mask].cpu(), pred.cpu(), average='macro', zero_division=0)
            test_f1_weighted = f1_score(data.y[data.test_mask].cpu(), pred.cpu(), average='weighted', zero_division=0)
            
            # Detailed classification report
            target_names = [data.cell_types[i] for i in range(data.num_classes)]
            report = classification_report(
                data.y[data.test_mask].cpu(), 
                pred.cpu(), 
                target_names=target_names,
                zero_division=0,
                output_dict=True
            )
        
        return test_acc, test_f1, test_f1_weighted, report
    
    def train(self):
        """Main training loop"""
        
        logger.info("🚀 Starting Large-Scale Pseudo-Spatial GNN Training")
        
        # Load data
        data = self.load_data()
        data = self.create_data_splits(data)
        
        # Build model
        model_config = self.build_model(data)
        
        # Setup training
        self.setup_training(data)
        
        # Training configuration
        epochs = self.config.get('epochs', 50)
        early_stopping_patience = self.config.get('early_stopping_patience', 10)
        
        logger.info(f"Training for {epochs} epochs with early stopping patience {early_stopping_patience}")
        
        # Training history
        training_history = []
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        
        # Training loop
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(data, epoch)
            
            # Validate
            val_loss, val_acc, val_f1 = self.validate(data)
            
            # Scheduler step
            self.scheduler.step(val_acc)
            
            epoch_time = time.time() - start_time
            
            # Log progress
            if not (np.isnan(train_loss) or np.isnan(val_loss)):
                logger.info(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
                    f"Time: {epoch_time:.2f}s"
                )
                
                # Log to Wandb
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'train/accuracy': train_acc,
                    'train/f1': train_f1,
                    'val/loss': val_loss,
                    'val/accuracy': val_acc,
                    'val/f1': val_f1,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch_time': epoch_time
                })
            else:
                logger.warning(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Train Loss: {train_loss} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
                    f"Val Loss: {val_loss} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
                    f"Time: {epoch_time:.2f}s"
                )
            
            # Save training history
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_f1': train_f1,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'lr': self.optimizer.param_groups[0]['lr'],
                'time': epoch_time
            })
            
            # Early stopping and best model saving
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'model_config': model_config
                }, self.exp_dir / 'best_model.pth')
                
                # Log best model to Wandb
                wandb.run.summary["best_val_accuracy"] = best_val_acc
                wandb.run.summary["best_epoch"] = best_epoch
                
            else:
                patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Load best model for testing
        best_model_path = self.exp_dir / 'best_model.pth'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final testing
        test_acc, test_f1, test_f1_weighted, detailed_report = self.test(data)
        
        # Compile results
        results = {
            'best_val_accuracy': best_val_acc,
            'best_epoch': best_epoch,
            'test_accuracy': test_acc,
            'test_f1_macro': test_f1,
            'test_f1_weighted': test_f1_weighted,
            'epochs_trained': len(training_history),
            'dataset_size': data.num_nodes,
            'num_features': data.num_features,
            'num_classes': data.num_classes,
            'detailed_classification_report': detailed_report,
            'experiment_name': self.exp_name,
            'wandb_url': wandb.run.url
        }
        
        # Log final results to Wandb
        wandb.log({
            'final/test_accuracy': test_acc,
            'final/test_f1_macro': test_f1,
            'final/test_f1_weighted': test_f1_weighted,
            'final/best_val_accuracy': best_val_acc
        })
        
        # Create classification report table for Wandb
        cell_types = [data.cell_types[i] for i in range(data.num_classes)]
        report_data = []
        for cell_type in cell_types:
            if cell_type in detailed_report:
                report_data.append([
                    cell_type,
                    detailed_report[cell_type]['precision'],
                    detailed_report[cell_type]['recall'],
                    detailed_report[cell_type]['f1-score'],
                    detailed_report[cell_type]['support']
                ])
        
        if report_data:
            wandb.log({"classification_report": wandb.Table(
                columns=["Cell Type", "Precision", "Recall", "F1-Score", "Support"],
                data=report_data
            )})
        
        # Save results
        with open(self.exp_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        pd.DataFrame(training_history).to_json(self.exp_dir / "training_history.json", orient='records', indent=2)
        
        logger.info(f"🎉 Training completed!")
        logger.info(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
        logger.info(f"Test accuracy: {test_acc:.4f}")
        logger.info(f"Test F1 (macro): {test_f1:.4f}")
        logger.info(f"Test F1 (weighted): {test_f1_weighted:.4f}")
        logger.info(f"Results saved to: {self.exp_dir}")
        logger.info(f"Wandb run: {wandb.run.url}")
        
        # Close Wandb
        wandb.finish()
        
        return results

def main():
    """Main function"""
    
    # Simple configuration
    config = {
        'epochs': 50,
        'early_stopping_patience': 10
    }
    
    # Train model
    trainer = LargeScaleSpatialGNNTrainer(config)
    results = trainer.train()
    
    print(f"\n🎯 Final Results:")
    print(f"Dataset: {results['dataset_size']:,} cells")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test F1 Score (macro): {results['test_f1_macro']:.4f}")
    print(f"Test F1 Score (weighted): {results['test_f1_weighted']:.4f}")
    print(f"Wandb URL: {results['wandb_url']}")
    
    # Comparison with previous results
    print(f"\n📊 Performance Comparison:")
    print(f"Previous 8.6K spatial GNN: 14.58% accuracy")
    print(f"Current 50K pseudo-spatial: {results['test_accuracy']*100:.2f}% accuracy")
    print(f"Improvement: {(results['test_accuracy']*100 - 14.58):.2f} percentage points")

if __name__ == "__main__":
    main()

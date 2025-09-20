"""
Training Module for Cardiomyocyte Subtype Classification

This module contains the training pipeline for the AdvancedCardiomyocyteGNN model.
"""
import json
import logging
import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from pathlib import Path
from datetime import datetime
import numpy as np

from ..models.gnn_models import AdvancedCardiomyocyteGNN
from ..data_processing import Authentic10XProcessor

logger = logging.getLogger(__name__)


class CardiomyocyteTrainer:
    """Trainer class for cardiomyocyte subtype classification."""
    
    def __init__(self, config=None):
        """Initialize the trainer.
        
        Args:
            config (dict): Training configuration parameters
        """
        self.config = config or self.get_default_config()
        self.device = torch.device(self.config['device'])
        self.data_processor = Authentic10XProcessor()
        
    def get_default_config(self):
        """Get default training configuration.
        
        Returns:
            dict: Default configuration parameters
        """
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
            'l2_reg': 0.0001
        }
    
    def create_stratified_splits(self, data):
        """Create stratified train/validation/test splits.
        
        Args:
            data: PyTorch Geometric data object
            
        Returns:
            tuple: (train_mask, val_mask, test_mask)
        """
        num_nodes = data.x.shape[0]
        
        # Create stratified splits for all classes
        class_indices = [torch.where(data.y == i)[0] for i in range(data.num_classes)]
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        for class_idx in class_indices:
            n = len(class_idx)
            perm = class_idx[torch.randperm(n)]
            
            train_end = int(self.config['train_ratio'] * n)
            val_end = train_end + int(self.config['val_ratio'] * n)
            
            train_indices.extend(perm[:train_end])
            val_indices.extend(perm[train_end:val_end])
            test_indices.extend(perm[val_end:])
        
        # Convert to tensors and create masks
        train_indices = torch.tensor(train_indices)
        val_indices = torch.tensor(val_indices)
        test_indices = torch.tensor(test_indices)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        
        return train_mask, val_mask, test_mask
    
    def setup_optimizer(self, model, train_mask, data):
        """Setup optimizer with different learning rates for different components.
        
        Args:
            model: The GNN model
            train_mask: Training mask
            data: Training data
            
        Returns:
            tuple: (optimizer, criterion, scheduler)
        """
        # Calculate class weights
        train_class_counts = torch.bincount(data.y[train_mask])
        class_weights = len(data.y[train_mask]) / (data.num_classes * train_class_counts.float())
        
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        
        # Use different learning rates for different parts
        gnn_params = list(model.gat1.parameters()) + list(model.gcn1.parameters()) + \
                     list(model.gat2.parameters()) + list(model.gcn2.parameters())
        classifier_params = list(model.classifier.parameters()) + \
                           list(model.feature_fusion.parameters()) + \
                           list(model.skip_projection.parameters())
        
        optimizer = torch.optim.AdamW([
            {'params': gnn_params, 'lr': self.config['learning_rate'], 'weight_decay': self.config['weight_decay']},
            {'params': classifier_params, 'lr': self.config['classifier_lr'], 'weight_decay': self.config['classifier_wd']}
        ])
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
        
        return optimizer, criterion, scheduler
    
    def train_epoch(self, model, data, train_mask, optimizer, criterion):
        """Train for one epoch.
        
        Args:
            model: The GNN model
            data: Training data
            train_mask: Training mask
            optimizer: Optimizer
            criterion: Loss criterion
            
        Returns:
            tuple: (train_loss, train_accuracy)
        """
        model.train()
        optimizer.zero_grad()
        
        out = model(data)
        train_loss = criterion(out[train_mask], data.y[train_mask])
        
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['grad_clip'])
        optimizer.step()
        
        # Calculate accuracy
        train_pred = out[train_mask].argmax(dim=1)
        train_acc = (train_pred == data.y[train_mask]).float().mean()
        
        return train_loss.item(), train_acc.item()
    
    def validate(self, model, data, val_mask, criterion):
        """Validate the model.
        
        Args:
            model: The GNN model
            data: Validation data
            val_mask: Validation mask
            criterion: Loss criterion
            
        Returns:
            tuple: (val_loss, val_accuracy, per_class_accuracy)
        """
        model.eval()
        with torch.no_grad():
            out = model(data)
            val_loss = criterion(out[val_mask], data.y[val_mask])
            
            val_pred = out[val_mask].argmax(dim=1)
            val_acc = (val_pred == data.y[val_mask]).float().mean()
            
            # Per-class accuracy
            per_class_acc = []
            for class_id in range(data.num_classes):
                class_mask = data.y[val_mask] == class_id
                if class_mask.sum() > 0:
                    class_pred = val_pred[class_mask]
                    class_acc = (class_pred == class_id).float().mean()
                    per_class_acc.append(class_acc.item())
                else:
                    per_class_acc.append(0.0)
            
            return val_loss.item(), val_acc.item(), per_class_acc
    
    def train(self):
        """Run the complete training pipeline.
        
        Returns:
            dict: Training results and metrics
        """
        logger.info("🧬 Starting Enhanced Cardiomyocyte Subtype Classification")
        logger.info(f"Using device: {self.device}")
        
        # Load data
        try:
            data = self.data_processor.load_cached_data(device=self.device)
        except FileNotFoundError as e:
            logger.error(f"❌ {e}")
            return None
        
        # Initialize model
        model = AdvancedCardiomyocyteGNN(
            num_features=data.x.shape[1],
            num_classes=data.num_classes,
            hidden_dim=self.config['hidden_dim'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        # Create splits
        train_mask, val_mask, test_mask = self.create_stratified_splits(data)
        
        logger.info(f"Split sizes - Train: {train_mask.sum()} | Val: {val_mask.sum()} | Test: {test_mask.sum()}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Setup training
        optimizer, criterion, scheduler = self.setup_optimizer(model, train_mask, data)
        
        # Training loop
        logger.info("🚀 Starting enhanced cardiomyocyte classification...")
        
        best_val_acc = 0
        best_val_loss = float('inf')
        patience_counter = 0
        
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'per_class_acc': []
        }
        
        for epoch in range(self.config['max_epochs']):
            # Training
            train_loss, train_acc = self.train_epoch(model, data, train_mask, optimizer, criterion)
            scheduler.step()
            
            # Validation
            val_loss, val_acc, per_class_acc = self.validate(model, data, val_mask, criterion)
            
            # Store metrics
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_acc'].append(val_acc)
            training_history['per_class_acc'].append(per_class_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_cardiomyocyte_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 30 == 0 or epoch < 15:
                logger.info(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                           f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Best Val: {best_val_acc:.4f}")
                logger.info(f"  Per-class val acc: {[f'{acc:.3f}' for acc in per_class_acc]}")
            
            if patience_counter >= self.config['patience']:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model and evaluate on test set
        model.load_state_dict(torch.load('best_cardiomyocyte_model.pth'))
        test_results = self.evaluate_test_set(model, data, test_mask)
        
        # Compile results
        results = {
            'test_accuracy': test_results['accuracy'],
            'confusion_matrix': test_results['confusion_matrix'],
            'per_class_metrics': test_results['per_class_metrics'],
            'best_val_accuracy': best_val_acc,
            'best_val_loss': best_val_loss,
            'num_epochs': epoch + 1,
            'dataset_info': {
                'num_cells': int(data.x.shape[0]),
                'num_genes': int(data.x.shape[1]),
                'num_edges': int(data.edge_index.shape[1]),
                'num_classes': int(data.num_classes),
                'task': 'Enhanced Cardiomyocyte Subtype Classification',
                'class_distribution': torch.bincount(data.y).tolist(),
                'source': '10X Genomics spatial_10X0026_molecule_info.h5',
                'authentic_barcodes': True,
                'model_architecture': 'Advanced GAT+GCN with skip connections',
                'device_used': str(self.device)
            },
            'training_history': training_history,
            'model_info': model.get_model_info()
        }
        
        # Save results
        self.save_results(results)
        
        logger.info(f"🎉 Enhanced cardiomyocyte classification complete!")
        logger.info(f"Test accuracy: {test_results['accuracy']:.4f}")
        
        return results
    
    def evaluate_test_set(self, model, data, test_mask):
        """Evaluate model on test set.
        
        Args:
            model: Trained model
            data: Test data
            test_mask: Test mask
            
        Returns:
            dict: Test evaluation results
        """
        model.eval()
        with torch.no_grad():
            out = model(data)
            test_pred = out[test_mask].argmax(dim=1)
            test_acc = (test_pred == data.y[test_mask]).float().mean()
            
            # Detailed analysis
            test_true = data.y[test_mask].cpu().numpy()
            test_pred_np = test_pred.cpu().numpy()
            
            # Confusion matrix
            confusion_matrix = np.zeros((data.num_classes, data.num_classes), dtype=int)
            for true_label, pred_label in zip(test_true, test_pred_np):
                confusion_matrix[true_label][pred_label] += 1
            
            logger.info("\n🧬 ENHANCED CARDIOMYOCYTE SUBTYPE CLASSIFICATION RESULTS:")
            logger.info(f"  Overall Test Accuracy: {test_acc:.4f}")
            
            # Get biological cell type names
            cell_type_names = self.data_processor.get_cell_type_names()
            
            logger.info(f"\n📊 Confusion Matrix:")
            for i in range(data.num_classes):
                cell_type = cell_type_names[i] if i < len(cell_type_names) else f"Unknown Type {i}"
                logger.info(f"  True {cell_type}: {confusion_matrix[i].tolist()}")
            
            # Per-class metrics
            per_class_metrics = {}
            for class_id in range(data.num_classes):
                tp = confusion_matrix[class_id][class_id]
                fp = np.sum(confusion_matrix[:, class_id]) - tp
                fn = np.sum(confusion_matrix[class_id, :]) - tp
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                cell_type = cell_type_names[class_id] if class_id < len(cell_type_names) else f"Unknown Type {class_id}"
                key = cell_type.lower().replace(' ', '_')
                
                per_class_metrics[key] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'support': int(np.sum(confusion_matrix[class_id, :]))
                }
                
                logger.info(f"  {cell_type}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f} ({int(np.sum(confusion_matrix[class_id, :]))} cells)")
            
            return {
                'accuracy': float(test_acc.item()),
                'confusion_matrix': confusion_matrix.tolist(),
                'per_class_metrics': per_class_metrics
            }
    
    def save_results(self, results):
        """Save training results.
        
        Args:
            results (dict): Results to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(f'experiments_enhanced_cardiomyocyte/enhanced_cardiomyocyte_{timestamp}')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / 'enhanced_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_dir}")


def train_enhanced_cardiomyocyte_classifier(config=None):
    """Convenience function to train the enhanced cardiomyocyte classifier.
    
    Args:
        config (dict): Training configuration
        
    Returns:
        dict: Training results
    """
    trainer = CardiomyocyteTrainer(config)
    return trainer.train()
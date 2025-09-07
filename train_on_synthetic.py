"""
Clean GNN Training Script for Large Synthetic Datasets
Numerically stable implementation with comprehensive logging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
import numpy as np
import logging
import os
import json
import argparse
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StableSpatialGNN(nn.Module):
    """
    Numerically stable Spatial GNN for cardiac data analysis
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: list = [256, 128, 64],
                 num_classes: int = 5,
                 dropout: float = 0.3,
                 conv_type: str = 'GCN'):
        super(StableSpatialGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout
        self.conv_type = conv_type
        
        # Build graph convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Input layer
        if conv_type == 'GAT':
            self.convs.append(GATConv(input_dim, hidden_dims[0], heads=4, concat=False, dropout=dropout))
        else:
            self.convs.append(GCNConv(input_dim, hidden_dims[0]))
        self.norms.append(nn.LayerNorm(hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            if conv_type == 'GAT':
                self.convs.append(GATConv(hidden_dims[i], hidden_dims[i+1], heads=4, concat=False, dropout=dropout))
            else:
                self.convs.append(GCNConv(hidden_dims[i], hidden_dims[i+1]))
            self.norms.append(nn.LayerNorm(hidden_dims[i+1]))
        
        # Output layers
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
        
        # Initialize weights for stability
        self._init_weights()
        
        logger.info(f"Initialized {conv_type} model: {input_dim} -> {hidden_dims} -> {num_classes}")
    
    def _init_weights(self):
        """Initialize weights for numerical stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Smaller gain for stability
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, batch):
        """Forward pass with numerical stability checks"""
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        
        # Input validation
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("NaN/Inf detected in input features")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Graph convolution layers
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            
            # Stability check after convolution
            if torch.isnan(x).any() or torch.isinf(x).any():
                logger.warning(f"NaN/Inf detected after conv layer {i}")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            x = norm(x)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # Global pooling
        x = global_mean_pool(x, batch_idx)
        
        # Final stability check before classification
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("NaN/Inf detected after pooling")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Classification
        x = self.classifier(x)
        
        # Final output stability
        x = torch.clamp(x, min=-10, max=10)  # Prevent extreme logits
        
        return x

class SyntheticDataLoader:
    """Loader for synthetic datasets with train/val/test splits"""
    
    def __init__(self, dataset_path: str, batch_size: int = 16, val_ratio: float = 0.15, test_ratio: float = 0.15):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        
        logger.info(f"Loading synthetic dataset from {dataset_path}")
        
        # Load graphs
        graphs = torch.load(dataset_path, weights_only=False)
        logger.info(f"Loaded {len(graphs)} graphs")
        
        # Shuffle and split
        np.random.shuffle(graphs)
        
        n_val = int(len(graphs) * val_ratio)
        n_test = int(len(graphs) * test_ratio)
        n_train = len(graphs) - n_val - n_test
        
        self.train_graphs = graphs[:n_train]
        self.val_graphs = graphs[n_train:n_train+n_val] 
        self.test_graphs = graphs[n_train+n_val:]
        
        logger.info(f"Dataset splits: Train={len(self.train_graphs)}, "
                   f"Val={len(self.val_graphs)}, Test={len(self.test_graphs)}")
        
        # Create data loaders
        self.train_loader = DataLoader(self.train_graphs, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_graphs, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_graphs, batch_size=batch_size, shuffle=False)
        
        # Get dataset info
        sample_graph = graphs[0]
        self.input_dim = sample_graph.x.shape[1]
        self.num_classes = len(torch.unique(torch.cat([g.y.unsqueeze(0) for g in graphs])))
        
        logger.info(f"Dataset info: {self.input_dim} features, {self.num_classes} classes")

def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch with stability checks"""
    model.train()
    total_loss = 0.0
    predictions = []
    targets = []
    valid_batches = 0
    
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        
        # Skip batches with invalid data
        if torch.isnan(batch.x).any() or torch.isinf(batch.x).any():
            logger.warning(f"Skipping batch {batch_idx} due to invalid input")
            continue
        
        optimizer.zero_grad()
        
        try:
            # Forward pass
            out = model(batch)
            
            # Stability check
            if torch.isnan(out).any() or torch.isinf(out).any():
                logger.warning(f"Invalid output in batch {batch_idx}, skipping")
                continue
            
            # Loss calculation
            loss = criterion(out, batch.y)
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss in batch {batch_idx}, skipping")
                continue
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Record results
            total_loss += loss.item()
            predictions.extend(torch.argmax(out, dim=1).cpu().numpy())
            targets.extend(batch.y.cpu().numpy())
            valid_batches += 1
            
        except Exception as e:
            logger.warning(f"Error in batch {batch_idx}: {e}")
            continue
    
    if valid_batches == 0:
        return 0.0, 0.0, 0.0
    
    avg_loss = total_loss / valid_batches
    accuracy = accuracy_score(targets, predictions) if targets else 0.0
    f1 = f1_score(targets, predictions, average='weighted', zero_division=0) if targets else 0.0
    
    return avg_loss, accuracy, f1

def evaluate_model(model, loader, criterion, device):
    """Evaluate model with stability checks"""
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    valid_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Skip invalid batches
            if torch.isnan(batch.x).any() or torch.isinf(batch.x).any():
                continue
            
            try:
                out = model(batch)
                
                if torch.isnan(out).any() or torch.isinf(out).any():
                    continue
                
                loss = criterion(out, batch.y)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                total_loss += loss.item()
                predictions.extend(torch.argmax(out, dim=1).cpu().numpy())
                targets.extend(batch.y.cpu().numpy())
                valid_batches += 1
                
            except Exception as e:
                continue
    
    if valid_batches == 0:
        return 0.0, 0.0, 0.0
    
    avg_loss = total_loss / valid_batches
    accuracy = accuracy_score(targets, predictions) if targets else 0.0
    f1 = f1_score(targets, predictions, average='weighted', zero_division=0) if targets else 0.0
    
    return avg_loss, accuracy, f1

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train GNN on synthetic cardiac data')
    parser.add_argument('--dataset', type=str, default='medium_synthetic', 
                       choices=['small_synthetic', 'medium_synthetic', 'large_synthetic'],
                       help='Dataset size to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128, 64], 
                       help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--conv_type', type=str, default='GCN', choices=['GCN', 'GAT'],
                       help='Graph convolution type')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--save_dir', type=str, default='models/synthetic_gnn', 
                       help='Directory to save models')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset
    dataset_path = f"/Users/tumokgabeng/Projects/HybridGnnRnn/data/large_synthetic/{args.dataset}.pt"
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        logger.info("Please run create_large_synthetic_datasets.py first")
        return
    
    data_loader = SyntheticDataLoader(dataset_path, batch_size=args.batch_size)
    
    # Create model
    model = StableSpatialGNN(
        input_dim=data_loader.input_dim,
        hidden_dims=args.hidden_dims,
        num_classes=data_loader.num_classes,
        dropout=args.dropout,
        conv_type=args.conv_type
    ).to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # Training tracking
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    patience_counter = 0
    
    logger.info("🚀 Starting training...")
    
    for epoch in range(args.epochs):
        # Training
        train_loss, train_acc, train_f1 = train_epoch(
            model, data_loader.train_loader, optimizer, criterion, device
        )
        
        # Validation
        val_loss, val_acc, val_f1 = evaluate_model(
            model, data_loader.val_loader, criterion, device
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Record metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            model_path = os.path.join(args.save_dir, f'best_model_{args.dataset}_{args.conv_type}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'args': args
            }, model_path)
            logger.info(f"💾 Saved new best model (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Logging
        if epoch % 5 == 0 or epoch < 10:
            logger.info(f"Epoch {epoch+1:3d}/{args.epochs}: "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} "
                       f"(Best: {best_val_acc:.4f})")
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch+1} (patience: {args.patience})")
            break
    
    # Final evaluation
    logger.info("🧪 Final test evaluation...")
    
    # Load best model
    model_path = os.path.join(args.save_dir, f'best_model_{args.dataset}_{args.conv_type}.pth')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    # Test evaluation
    test_loss, test_acc, test_f1 = evaluate_model(
        model, data_loader.test_loader, criterion, device
    )
    
    logger.info("📊 Final Results:")
    logger.info(f"  Best Validation Accuracy: {best_val_acc:.4f}")
    logger.info(f"  Test Accuracy: {test_acc:.4f}")
    logger.info(f"  Test F1 Score: {test_f1:.4f}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'args': vars(args)
    }
    
    history_path = os.path.join(args.save_dir, f'training_history_{args.dataset}_{args.conv_type}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plot_path = os.path.join(args.save_dir, f'training_curves_{args.dataset}_{args.conv_type}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"📈 Saved training curves to {plot_path}")
    
    logger.info("✅ Training completed successfully!")

if __name__ == "__main__":
    main()

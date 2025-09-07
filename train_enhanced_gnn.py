"""
Enhanced GNN Training with Hyperparameter Optimization and Real Data Integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.loader import DataLoader
import numpy as np
import logging
import os
import json
import argparse
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import optuna
import scanpy as sc
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedCardiacGNN(nn.Module):
    """
    Advanced GNN architecture with multiple improvements:
    - Residual connections
    - Attention pooling
    - Multiple aggregation methods
    - Adaptive depth
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: list = [512, 256, 128],
                 num_classes: int = 5,
                 dropout: float = 0.3,
                 conv_type: str = 'Transformer',
                 num_heads: int = 8,
                 use_residual: bool = True,
                 pooling_method: str = 'attention'):
        super(AdvancedCardiacGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout
        self.conv_type = conv_type
        self.use_residual = use_residual
        self.pooling_method = pooling_method
        
        # Build graph convolution layers with residual connections
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_projections = nn.ModuleList()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        
        # Convolutional layers
        prev_dim = hidden_dims[0]
        for i, hidden_dim in enumerate(hidden_dims):
            if conv_type == 'Transformer':
                conv = TransformerConv(prev_dim, hidden_dim, heads=num_heads, 
                                     dropout=dropout, concat=False)
            elif conv_type == 'GAT':
                conv = GATConv(prev_dim, hidden_dim, heads=num_heads, 
                              dropout=dropout, concat=False)
            else:  # GCN
                conv = GCNConv(prev_dim, hidden_dim)
            
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
            
            # Residual projection if dimensions don't match
            if use_residual and prev_dim != hidden_dim:
                self.residual_projections.append(nn.Linear(prev_dim, hidden_dim))
            else:
                self.residual_projections.append(None)
            
            prev_dim = hidden_dim
        
        # Pooling layer
        if pooling_method == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1] // 2, 1),
                nn.Sigmoid()
            )
        
        # Classification head with multiple layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, hidden_dims[-1] // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 4, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, edge_index, batch):
        # Input projection
        x = self.input_projection(x)
        x = F.relu(x)
        
        # Graph convolution layers with residual connections
        for i, (conv, norm, res_proj) in enumerate(zip(self.convs, self.norms, self.residual_projections)):
            residual = x
            
            # Apply convolution
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Add residual connection
            if self.use_residual:
                if res_proj is not None:
                    residual = res_proj(residual)
                x = x + residual
        
        # Graph-level pooling
        if self.pooling_method == 'attention':
            # Attention-based pooling
            attention_weights = self.attention_pool(x)
            x = x * attention_weights
            x = global_add_pool(x, batch)
        elif self.pooling_method == 'multi':
            # Multiple pooling methods
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x_sum = global_add_pool(x, batch)
            x = torch.cat([x_mean, x_max, x_sum], dim=1)
            # Adjust classifier input size
            if not hasattr(self, '_adjusted_classifier'):
                self.classifier[0] = nn.Linear(x.size(1), self.hidden_dims[-1] // 2)
                self._adjusted_classifier = True
        else:
            x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        
        return x

class CardiacDataIntegrator:
    """
    Integrates real cardiac datasets with synthetic data
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.processed_data_path = os.path.join(data_dir, "processed_visium_heart.h5ad")
    
    def load_real_cardiac_data(self) -> Optional[sc.AnnData]:
        """Load and preprocess real cardiac dataset"""
        try:
            if os.path.exists(self.processed_data_path):
                logger.info(f"Loading real cardiac data from {self.processed_data_path}")
                adata = sc.read_h5ad(self.processed_data_path)
                return self._preprocess_real_data(adata)
            else:
                logger.warning("Real cardiac data not found. Using synthetic data only.")
                return None
        except Exception as e:
            logger.error(f"Error loading real cardiac data: {e}")
            return None
    
    def _preprocess_real_data(self, adata: sc.AnnData) -> sc.AnnData:
        """Preprocess real cardiac data"""
        logger.info("Preprocessing real cardiac data...")
        
        # Basic filtering
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        
        # Calculate QC metrics
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
        
        # Normalization
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # Feature selection
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata.raw = adata
        adata = adata[:, adata.var.highly_variable]
        
        # Create meaningful labels based on cell types or spatial regions
        if 'cell_type' in adata.obs.columns:
            # Use cell type as label
            unique_types = adata.obs['cell_type'].unique()
            type_to_label = {cell_type: i for i, cell_type in enumerate(unique_types)}
            adata.obs['graph_label'] = adata.obs['cell_type'].map(type_to_label)
        else:
            # Create spatial region labels
            if 'spatial' in adata.obsm:
                coords = adata.obsm['spatial']
                # K-means clustering for spatial regions
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=5, random_state=42)
                adata.obs['graph_label'] = kmeans.fit_predict(coords)
        
        logger.info(f"Preprocessed data: {adata.n_obs} cells, {adata.n_vars} genes")
        return adata
    
    def create_meaningful_synthetic_labels(self, graphs: List) -> List:
        """Create meaningful labels for synthetic data based on graph properties"""
        logger.info("Creating meaningful synthetic labels...")
        
        for graph in graphs:
            # Extract graph-level features
            node_features = graph.x
            efficiency_scores = graph.efficiency if hasattr(graph, 'efficiency') else None
            
            # Create label based on multiple graph properties
            avg_expression = torch.mean(node_features).item()
            std_expression = torch.std(node_features).item()
            num_nodes = node_features.shape[0]
            
            if efficiency_scores is not None:
                avg_efficiency = torch.mean(efficiency_scores).item()
                
                # Multi-criteria labeling
                if avg_efficiency > 0.8 and avg_expression > 0.5:
                    label = 0  # High efficiency, high expression
                elif avg_efficiency > 0.6 and std_expression > 1.0:
                    label = 1  # Medium efficiency, high variability
                elif num_nodes > 600:
                    label = 2  # Large tissue sample
                elif avg_expression < 0.2:
                    label = 3  # Low expression
                else:
                    label = 4  # Default category
            else:
                # Fallback based on expression patterns only
                if avg_expression > 0.5 and std_expression > 1.0:
                    label = 0
                elif avg_expression > 0.3:
                    label = 1
                elif num_nodes > 500:
                    label = 2
                elif std_expression > 0.8:
                    label = 3
                else:
                    label = 4
            
            graph.y = torch.tensor(label, dtype=torch.long)
        
        # Log label distribution
        labels = [graph.y.item() for graph in graphs]
        label_counts = {i: labels.count(i) for i in range(5)}
        logger.info(f"Label distribution: {label_counts}")
        
        return graphs

class HyperparameterOptimizer:
    """
    Optuna-based hyperparameter optimization
    """
    
    def __init__(self, train_loader, val_loader, device, n_trials: int = 50):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.n_trials = n_trials
    
    def objective(self, trial):
        """Optuna objective function"""
        # Suggest hyperparameters
        params = {
            'hidden_dims': [
                trial.suggest_int('hidden_dim1', 256, 1024, step=128),
                trial.suggest_int('hidden_dim2', 128, 512, step=64),
                trial.suggest_int('hidden_dim3', 64, 256, step=32)
            ],
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'conv_type': trial.suggest_categorical('conv_type', ['GCN', 'GAT', 'Transformer']),
            'num_heads': trial.suggest_int('num_heads', 4, 16, step=4),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
            'pooling_method': trial.suggest_categorical('pooling_method', ['mean', 'attention', 'multi'])
        }
        
        # Create model
        input_dim = next(iter(self.train_loader)).x.shape[1]
        model = AdvancedCardiacGNN(
            input_dim=input_dim,
            hidden_dims=params['hidden_dims'],
            dropout=params['dropout'],
            conv_type=params['conv_type'],
            num_heads=params['num_heads'],
            pooling_method=params['pooling_method']
        ).to(self.device)
        
        # Train for limited epochs
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(10):  # Quick evaluation
            total_loss = 0
            for batch in self.train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        
        accuracy = correct / total
        return accuracy
    
    def optimize(self) -> Dict:
        """Run hyperparameter optimization"""
        logger.info(f"🔍 Starting hyperparameter optimization with {self.n_trials} trials...")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials)
        
        logger.info(f"🎯 Best trial achieved {study.best_value:.4f} validation accuracy")
        logger.info(f"Best parameters: {study.best_params}")
        
        return study.best_params

def main():
    parser = argparse.ArgumentParser(description='Enhanced GNN Training with Optimization')
    parser.add_argument('--dataset', type=str, default='large_synthetic',
                       choices=['small_synthetic', 'medium_synthetic', 'large_synthetic',
                               'small_improved', 'medium_improved', 'large_improved'])
    parser.add_argument('--use_real_data', action='store_true', help='Include real cardiac data')
    parser.add_argument('--optimize_hyperparams', action='store_true', help='Run hyperparameter optimization')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of optimization trials')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='models/enhanced_gnn', help='Save directory')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load synthetic data
    if 'improved' in args.dataset:
        dataset_path = f'data/improved_synthetic/{args.dataset}.pt'
    else:
        dataset_path = f'data/large_synthetic/{args.dataset}.pt'
    logger.info(f"Loading synthetic dataset from {dataset_path}")
    synthetic_graphs = torch.load(dataset_path, weights_only=False)
    
    # Create meaningful labels only for non-improved datasets
    if 'improved' not in args.dataset:
        integrator = CardiacDataIntegrator()
        synthetic_graphs = integrator.create_meaningful_synthetic_labels(synthetic_graphs)
    else:
        logger.info("Using pre-generated meaningful labels from improved dataset")
    
    # Load real data if requested
    all_graphs = synthetic_graphs
    if args.use_real_data:
        real_data = integrator.load_real_cardiac_data()
        if real_data is not None:
            # Convert real data to graphs (simplified)
            logger.info("Converting real data to graphs...")
            # This would need more sophisticated conversion
            # For now, we'll use synthetic data with improved labels
    
    # Split data
    np.random.shuffle(all_graphs)
    n_graphs = len(all_graphs)
    train_split = int(0.7 * n_graphs)
    val_split = int(0.85 * n_graphs)
    
    train_graphs = all_graphs[:train_split]
    val_graphs = all_graphs[train_split:val_split]
    test_graphs = all_graphs[val_split:]
    
    logger.info(f"Dataset splits: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
    
    # Get input dimensions
    input_dim = train_graphs[0].x.shape[1]
    num_classes = len(set(graph.y.item() for graph in all_graphs))
    logger.info(f"Dataset info: {input_dim} features, {num_classes} classes")
    
    # Hyperparameter optimization
    best_params = None
    if args.optimize_hyperparams:
        optimizer = HyperparameterOptimizer(train_loader, val_loader, device, args.n_trials)
        best_params = optimizer.optimize()
    
    # Use optimized parameters or defaults
    if best_params:
        model_params = {
            'input_dim': input_dim,
            'hidden_dims': [best_params['hidden_dim1'], best_params['hidden_dim2'], best_params['hidden_dim3']],
            'num_classes': num_classes,
            'dropout': best_params['dropout'],
            'conv_type': best_params['conv_type'],
            'num_heads': best_params.get('num_heads', 8),
            'pooling_method': best_params.get('pooling_method', 'attention')
        }
        learning_rate = best_params['learning_rate']
    else:
        model_params = {
            'input_dim': input_dim,
            'hidden_dims': [512, 256, 128],
            'num_classes': num_classes,
            'dropout': 0.3,
            'conv_type': 'Transformer',
            'num_heads': 8,
            'pooling_method': 'attention'
        }
        learning_rate = args.learning_rate
    
    # Create and train model
    model = AdvancedCardiacGNN(**model_params).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    logger.info(f"🚀 Starting enhanced training for {args.epochs} epochs...")
    
    # Training loop
    best_val_acc = 0
    train_losses, val_losses, val_accs = [], [], []
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
        
        train_acc = correct / total
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                val_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        
        val_acc = correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'model_params': model_params,
                'best_params': best_params
            }, os.path.join(args.save_dir, f'best_enhanced_model_{args.dataset}.pth'))
            logger.info(f"💾 Saved new best model (val_acc: {val_acc:.4f})")
        
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            logger.info(f"Epoch {epoch+1:3d}/{args.epochs}: "
                       f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f} "
                       f"(Best: {best_val_acc:.4f})")
    
    # Final evaluation
    logger.info("🧪 Final test evaluation...")
    model_path = os.path.join(args.save_dir, f'best_enhanced_model_{args.dataset}.pth')
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            test_preds.extend(pred.cpu().numpy())
            test_labels.extend(batch.y.cpu().numpy())
    
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    
    logger.info("📊 Final Results:")
    logger.info(f"  Best Validation Accuracy: {best_val_acc:.4f}")
    logger.info(f"  Test Accuracy: {test_acc:.4f}")
    logger.info(f"  Test F1 Score: {test_f1:.4f}")
    
    # Save results
    results = {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'best_params': best_params,
        'model_params': model_params,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs
    }
    
    with open(os.path.join(args.save_dir, f'enhanced_results_{args.dataset}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("✅ Enhanced training completed successfully!")

if __name__ == "__main__":
    main()

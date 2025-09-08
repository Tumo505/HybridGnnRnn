#!/usr/bin/env python3
"""
Balanced Spatial GNN Training on Real Cardiac Data
Finding the sweet spot between overfitting and underfitting
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.utils import dropout_adj
from torch_geometric.data import Data
import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BalancedSpatialGNN(nn.Module):
    """Balanced spatial GNN with moderate regularization"""
    
    def __init__(self, input_dim, hidden_dim=96, num_classes=8, dropout=0.4):
        super(BalancedSpatialGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Input processing with moderate regularization
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_dropout = nn.Dropout(0.2)  # Light input dropout
        
        # Three-layer GNN architecture
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.conv3 = GCNConv(hidden_dim // 2, hidden_dim // 4)
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        
        # Progressive classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.LayerNorm(hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),  # Decreasing dropout
            nn.Linear(hidden_dim // 8, num_classes)
        )
        
        logger.info(f"Created BalancedSpatialGNN with {self.count_parameters():,} parameters")
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x, edge_index, batch=None, training=True):
        # Input processing
        x = self.input_norm(x)
        x = self.input_dropout(x)
        
        # Apply moderate edge dropout during training
        if training and self.training:
            edge_index, _ = dropout_adj(edge_index, p=0.1, training=True)
        
        # First GNN layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout * 0.6, training=self.training)
        
        # Second GNN layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout * 0.8, training=self.training)
        
        # Third GNN layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Classification
        out = self.classifier(x)
        
        return out

class BalancedSpatialProcessor:
    """Data processor with balanced preprocessing"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_and_process_spatial_data(self, data_path):
        """Load and process spatial data with balanced feature selection"""
        logger.info(f"Loading spatial data from {data_path}")
        adata = sc.read_h5ad(data_path)
        logger.info(f"Loaded spatial data: {adata.shape}")
        
        logger.info("Applying balanced preprocessing...")
        
        # Clean data
        X = adata.X.copy()
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Remove problematic values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Balanced gene selection - 500 genes (middle ground)
        gene_vars = np.var(X, axis=0)
        top_genes = np.argsort(gene_vars)[-500:]
        X = X[:, top_genes]
        
        # Moderate normalization
        X = self.scaler.fit_transform(X)
        
        # Light noise injection for regularization
        noise_factor = 0.005
        X += np.random.normal(0, noise_factor, X.shape)
        
        # Create new AnnData with balanced features
        adata_processed = sc.AnnData(X)
        adata_processed.obsm = adata.obsm.copy()
        adata_processed.obs = adata.obs.copy()
        
        logger.info(f"After balanced preprocessing: {adata_processed.shape}")
        return adata_processed
    
    def create_spatial_graph(self, adata, k_neighbors=10):
        """Create spatial graph with balanced connectivity"""
        spatial_coords = adata.obsm['spatial']
        logger.info(f"Creating balanced spatial graph with {k_neighbors} neighbors...")
        
        # Build moderately connected graph
        nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree').fit(spatial_coords)
        distances, indices = nbrs.kneighbors(spatial_coords)
        
        edge_list = []
        edge_weights = []
        
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):  # Skip self-connection
                neighbor = indices[i][j]
                distance = distances[i][j]
                
                # Balanced edge selection
                if distance < np.percentile(distances[:, 1:], 80):
                    edge_list.append([i, neighbor])
                    edge_list.append([neighbor, i])  # Make undirected
                    # Moderate edge weights
                    weight = np.exp(-distance * 1.5)
                    edge_weights.extend([weight, weight])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        
        logger.info(f"Created balanced graph with {edge_index.shape[1]} edges")
        return edge_index, edge_attr
    
    def create_spatial_labels(self, adata, n_clusters=7):
        """Create moderate number of spatial clusters"""
        spatial_coords = adata.obsm['spatial']
        
        # Use balanced number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(spatial_coords)
        
        logger.info(f"Created {n_clusters} balanced spatial clusters")
        return labels

class BalancedTrainer:
    """Trainer with balanced regularization"""
    
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        
        # Balanced training setup
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=0.001,  # Moderate learning rate
            weight_decay=0.02,  # Moderate weight decay
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=5, verbose=True
        )
        
        # Loss with moderate label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        
        self.best_val_loss = float('inf')
        self.patience = 8  # Moderate patience
        self.patience_counter = 0
        
    def train_epoch(self, data, train_mask):
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(data.x, data.edge_index, training=True)
        loss = self.criterion(out[train_mask], data.y[train_mask])
        
        loss.backward()
        
        # Moderate gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Calculate accuracy
        pred = out[train_mask].argmax(dim=1)
        acc = (pred == data.y[train_mask]).float().mean()
        
        return loss.item(), acc.item()
    
    def validate(self, data, val_mask):
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index, training=False)
            loss = self.criterion(out[val_mask], data.y[val_mask])
            pred = out[val_mask].argmax(dim=1)
            acc = (pred == data.y[val_mask]).float().mean()
        return loss.item(), acc.item()
    
    def train(self, data, train_mask, val_mask, epochs=80):
        logger.info(f"Starting balanced training for {epochs} epochs")
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(data, train_mask)
            val_loss, val_acc = self.validate(data, val_mask)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Balanced early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_balanced_gnn.pth')
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_balanced_gnn.pth'))
        logger.info("Training completed!")

def main():
    logger.info("=== Balanced Spatial GNN Training on Real Data ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize processor
    processor = BalancedSpatialProcessor()
    
    # Load and process data with balanced approach
    data_path = "data/processed_visium_heart.h5ad"
    adata = processor.load_and_process_spatial_data(data_path)
    
    # Create balanced graph
    edge_index, edge_attr = processor.create_spatial_graph(adata, k_neighbors=10)
    
    # Create balanced labels
    labels = processor.create_spatial_labels(adata, n_clusters=7)
    
    # Create PyTorch Geometric data
    graph_data = Data(
        x=torch.tensor(adata.X, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(labels, dtype=torch.long),
        pos=torch.tensor(adata.obsm['spatial'], dtype=torch.float)
    )
    
    logger.info(f"Created graph data: {graph_data}")
    
    # Balanced train/val/test split
    num_nodes = graph_data.x.shape[0]
    indices = np.arange(num_nodes)
    
    # Standard split proportions
    train_indices, temp_indices = train_test_split(indices, test_size=0.7, random_state=42, 
                                                   stratify=labels)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.43, random_state=42,
                                                 stratify=labels[temp_indices])
    
    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    logger.info(f"Data split - Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    
    # Create balanced model
    input_dim = graph_data.x.shape[1]
    num_classes = len(np.unique(labels))
    
    model = BalancedSpatialGNN(
        input_dim=input_dim,
        hidden_dim=96,  # Balanced capacity
        num_classes=num_classes,
        dropout=0.4  # Moderate dropout
    )
    
    # Train with balanced approach
    trainer = BalancedTrainer(model, device)
    trainer.train(graph_data, train_mask, val_mask, epochs=80)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index, training=False)
        
        test_pred = out[test_mask].argmax(dim=1)
        test_acc = (test_pred == graph_data.y[test_mask]).float().mean()
        
        train_pred = out[train_mask].argmax(dim=1)
        train_acc = (train_pred == graph_data.y[train_mask]).float().mean()
        
        val_pred = out[val_mask].argmax(dim=1)
        val_acc = (val_pred == graph_data.y[val_mask]).float().mean()
        
        overfitting_gap = train_acc - test_acc
        
        logger.info(f"Final Results:")
        logger.info(f"  Train Accuracy: {train_acc:.4f}")
        logger.info(f"  Validation Accuracy: {val_acc:.4f}")
        logger.info(f"  Test Accuracy: {test_acc:.4f}")
        logger.info(f"  Train-Test Gap: {overfitting_gap:.4f}")
        
        if overfitting_gap < 0.05:
            logger.info("✓ Excellent generalization!")
        elif overfitting_gap < 0.10:
            logger.info("✓ Good generalization")
        else:
            logger.info("⚠ Some overfitting detected")
            
        # Performance assessment
        if test_acc > 0.7:
            logger.info("🎯 Excellent performance achieved!")
        elif test_acc > 0.5:
            logger.info("👍 Good performance achieved!")
        else:
            logger.info("📈 Performance needs improvement")
    
    logger.info("=== Balanced Training Complete ===")

if __name__ == "__main__":
    main()

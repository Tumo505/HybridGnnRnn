#!/usr/bin/env python3
"""
Anti-Overfitting Spatial GNN Training on Real Cardiac Data
Focuses on regularization and preventing overfitting
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

class RegularizedSpatialGNN(nn.Module):
    """Heavily regularized spatial GNN to prevent overfitting"""
    
    def __init__(self, input_dim, hidden_dim=64, num_classes=8, dropout=0.7):
        super(RegularizedSpatialGNN, self).__init__()
        
        # Much smaller hidden dimensions to reduce capacity
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Input layer with heavy dropout
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_dropout = nn.Dropout(dropout)
        
        # Reduced number of GNN layers to prevent overfitting
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        
        # Batch normalization for better training stability
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        
        # Simple classifier with heavy regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        logger.info(f"Created RegularizedSpatialGNN with {self.count_parameters():,} parameters")
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x, edge_index, batch=None, training=True):
        # Input normalization and dropout
        x = self.input_norm(x)
        x = self.input_dropout(x)
        
        # Apply edge dropout during training for regularization
        if training and self.training:
            edge_index, _ = dropout_adj(edge_index, p=0.2, training=True)
        
        # First GNN layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GNN layer  
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Classification
        out = self.classifier(x)
        
        return out

class AntiOverfittingSpatialProcessor:
    """Data processor with strong regularization focus"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_and_process_spatial_data(self, data_path):
        """Load and heavily regularize spatial data"""
        logger.info(f"Loading spatial data from {data_path}")
        adata = sc.read_h5ad(data_path)
        logger.info(f"Loaded spatial data: {adata.shape}")
        
        # Much more aggressive feature selection to reduce overfitting
        logger.info("Applying aggressive preprocessing...")
        
        # Clean data thoroughly
        X = adata.X.copy()
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Remove problematic values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Very aggressive gene selection - only top 300 genes
        gene_vars = np.var(X, axis=0)
        top_genes = np.argsort(gene_vars)[-300:]  # Much fewer features
        X = X[:, top_genes]
        
        # Strong normalization
        X = self.scaler.fit_transform(X)
        
        # Add noise for regularization
        noise_factor = 0.01
        X += np.random.normal(0, noise_factor, X.shape)
        
        # Create new AnnData with reduced features
        adata_processed = sc.AnnData(X)
        adata_processed.obsm = adata.obsm.copy()
        adata_processed.obs = adata.obs.copy()
        
        logger.info(f"After aggressive preprocessing: {adata_processed.shape}")
        return adata_processed
    
    def create_spatial_graph(self, adata, k_neighbors=8):  # Fewer neighbors
        """Create spatial graph with regularization"""
        spatial_coords = adata.obsm['spatial']
        logger.info(f"Creating sparse spatial graph with {k_neighbors} neighbors...")
        
        # Build more sparse graph to prevent overfitting
        nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree').fit(spatial_coords)
        distances, indices = nbrs.kneighbors(spatial_coords)
        
        edge_list = []
        edge_weights = []
        
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):  # Skip self-connection
                neighbor = indices[i][j]
                distance = distances[i][j]
                
                # Only add edges for very close neighbors
                if distance < np.percentile(distances[:, 1:], 75):  # More selective
                    edge_list.append([i, neighbor])
                    edge_list.append([neighbor, i])  # Make undirected
                    # Weaker edge weights
                    weight = np.exp(-distance * 2.0)  # Stronger decay
                    edge_weights.extend([weight, weight])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        
        logger.info(f"Created sparse graph with {edge_index.shape[1]} edges")
        return edge_index, edge_attr
    
    def create_conservative_labels(self, adata, n_clusters=6):  # Fewer clusters
        """Create fewer, more stable clusters"""
        spatial_coords = adata.obsm['spatial']
        
        # Use fewer clusters to prevent overfitting
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(spatial_coords)
        
        logger.info(f"Created {n_clusters} conservative spatial clusters")
        return labels

class AntiOverfittingTrainer:
    """Trainer focused on preventing overfitting"""
    
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        
        # Very conservative training setup
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=0.0005,  # Much lower learning rate
            weight_decay=0.1,  # Very high weight decay
            eps=1e-8
        )
        
        # Learning rate scheduler for additional regularization
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Loss with label smoothing for regularization
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.15)  # Higher smoothing
        
        self.best_val_loss = float('inf')
        self.patience = 5  # Very early stopping
        self.patience_counter = 0
        
    def train_epoch(self, data, train_mask):
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(data.x, data.edge_index, training=True)
        loss = self.criterion(out[train_mask], data.y[train_mask])
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        
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
    
    def train(self, data, train_mask, val_mask, epochs=50):  # Fewer epochs
        logger.info(f"Starting conservative training for {epochs} epochs")
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(data, train_mask)
            val_loss, val_acc = self.validate(data, val_mask)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Conservative early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_anti_overfitting_gnn.pth')
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_anti_overfitting_gnn.pth'))
        logger.info("Training completed!")

def main():
    logger.info("=== Anti-Overfitting Spatial GNN Training ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize processor
    processor = AntiOverfittingSpatialProcessor()
    
    # Load and process data with heavy regularization
    data_path = "data/processed_visium_heart.h5ad"
    adata = processor.load_and_process_spatial_data(data_path)
    
    # Create sparse graph
    edge_index, edge_attr = processor.create_spatial_graph(adata, k_neighbors=6)
    
    # Create fewer, more conservative labels
    labels = processor.create_conservative_labels(adata, n_clusters=5)
    
    # Create PyTorch Geometric data
    graph_data = Data(
        x=torch.tensor(adata.X, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(labels, dtype=torch.long),
        pos=torch.tensor(adata.obsm['spatial'], dtype=torch.float)
    )
    
    logger.info(f"Created graph data: {graph_data}")
    
    # Conservative train/val/test split with more data for validation
    num_nodes = graph_data.x.shape[0]
    indices = np.arange(num_nodes)
    
    # Smaller training set to prevent overfitting
    train_indices, temp_indices = train_test_split(indices, test_size=0.8, random_state=42, 
                                                   stratify=labels)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42,
                                                 stratify=labels[temp_indices])
    
    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    logger.info(f"Data split - Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    
    # Create regularized model
    input_dim = graph_data.x.shape[1]
    num_classes = len(np.unique(labels))
    
    model = RegularizedSpatialGNN(
        input_dim=input_dim,
        hidden_dim=32,  # Much smaller
        num_classes=num_classes,
        dropout=0.7  # Heavy dropout
    )
    
    # Train with anti-overfitting focus
    trainer = AntiOverfittingTrainer(model, device)
    trainer.train(graph_data, train_mask, val_mask, epochs=40)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index, training=False)
        
        test_pred = out[test_mask].argmax(dim=1)
        test_acc = (test_pred == graph_data.y[test_mask]).float().mean()
        
        # Also check train accuracy to measure overfitting
        train_pred = out[train_mask].argmax(dim=1)
        train_acc = (train_pred == graph_data.y[train_mask]).float().mean()
        
        val_pred = out[val_mask].argmax(dim=1)
        val_acc = (val_pred == graph_data.y[val_mask]).float().mean()
        
        logger.info(f"Final Results:")
        logger.info(f"  Train Accuracy: {train_acc:.4f}")
        logger.info(f"  Validation Accuracy: {val_acc:.4f}")
        logger.info(f"  Test Accuracy: {test_acc:.4f}")
        logger.info(f"  Train-Test Gap: {train_acc - test_acc:.4f}")
        
        if train_acc - test_acc < 0.05:
            logger.info("✓ Good generalization - minimal overfitting!")
        else:
            logger.info("⚠ Still some overfitting detected")
    
    logger.info("=== Anti-Overfitting Training Complete ===")

if __name__ == "__main__":
    main()

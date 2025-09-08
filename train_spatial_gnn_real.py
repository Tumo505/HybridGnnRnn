#!/usr/bin/env python3
"""
Spatial GNN Training on Real Cardiac Data
Focuses on spatial relationships in cardiac tissue using Visium spatial transcriptomics
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scanpy as sc
import pandas as pd
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
import logging
from pathlib import Path
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class SpatialCardiacDataProcessor:
    """Process real spatial cardiac data for GNN training"""
    
    def __init__(self, num_neighbors: int = 10):
        self.num_neighbors = num_neighbors
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_spatial_data(self, data_path: str) -> sc.AnnData:
        """Load Visium spatial transcriptomics data"""
        logger.info(f"Loading spatial data from {data_path}")
        
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            raise FileNotFoundError(f"Cannot find data file: {data_path}")
        
        try:
            adata = sc.read_h5ad(data_path)
            logger.info(f"Loaded spatial data: {adata.shape}")
            logger.info(f"Available obsm keys: {list(adata.obsm.keys())}")
            logger.info(f"Available obs columns: {list(adata.obs.columns)}")
            return adata
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_spatial_data(self, adata: sc.AnnData) -> sc.AnnData:
        """Preprocess spatial transcriptomics data"""
        logger.info("Preprocessing spatial data...")
        
        # Copy to avoid modifying original
        adata = adata.copy()
        
        # Handle NaN/infinite values
        logger.info("Cleaning data - removing NaN and infinite values...")
        if hasattr(adata.X, 'toarray'):
            X = adata.X.toarray()
        else:
            X = adata.X
        
        # Replace NaN and infinite values with zeros
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        adata.X = X
        
        # Basic quality control
        logger.info("Applying quality control...")
        
        # Remove genes expressed in very few spots
        sc.pp.filter_genes(adata, min_cells=10)
        
        # Remove spots with very few genes
        sc.pp.filter_cells(adata, min_genes=50)  # Reduced threshold
        
        # Check if data needs normalization
        data_max = np.max(X)
        if data_max > 100:  # Raw counts
            logger.info("Normalizing expression data...")
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        else:
            logger.info("Data appears already normalized")
        
        # Select top variable genes instead of using highly_variable_genes
        logger.info("Selecting top variable genes...")
        if hasattr(adata.X, 'toarray'):
            X = adata.X.toarray()
        else:
            X = adata.X
        
        # Calculate variance for each gene
        gene_vars = np.var(X, axis=0)
        
        # Select top 1000 most variable genes
        n_top_genes = min(1000, len(gene_vars))
        top_gene_indices = np.argsort(gene_vars)[-n_top_genes:]
        
        # Keep only top variable genes
        adata = adata[:, top_gene_indices]
        
        logger.info(f"After preprocessing: {adata.shape}")
        return adata
    
    def get_spatial_coordinates(self, adata: sc.AnnData) -> np.ndarray:
        """Extract spatial coordinates from Visium data"""
        if 'spatial' in adata.obsm:
            coords = adata.obsm['spatial']
        elif 'X_spatial' in adata.obsm:
            coords = adata.obsm['X_spatial']
        else:
            logger.warning("No spatial coordinates found, using PCA coordinates")
            sc.tl.pca(adata, n_comps=2)
            coords = adata.obsm['X_pca'][:, :2]
        
        logger.info(f"Spatial coordinates shape: {coords.shape}")
        return coords
    
    def create_spatial_graph(self, adata: sc.AnnData, coordinates: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create spatial graph based on coordinate proximity"""
        logger.info("Creating spatial graph...")
        
        # Use KNN to find spatial neighbors
        nbrs = NearestNeighbors(n_neighbors=self.num_neighbors + 1, algorithm='ball_tree')
        nbrs.fit(coordinates)
        distances, indices = nbrs.kneighbors(coordinates)
        
        # Create edge index (exclude self-loops)
        edge_index = []
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors[1:]:  # Skip self (first neighbor)
                edge_index.append([i, neighbor])
                # Add reverse edge for undirected graph
                edge_index.append([neighbor, i])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create edge weights based on distance
        edge_weights = []
        for i, neighbors in enumerate(indices):
            for j, neighbor in enumerate(neighbors[1:]):
                dist = distances[i][j + 1]
                weight = np.exp(-dist / np.mean(distances))  # Gaussian weight
                edge_weights.extend([weight, weight])  # For both directions
        
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        logger.info(f"Created graph with {edge_index.shape[1]} edges")
        return edge_index, edge_weights
    
    def create_spatial_labels(self, adata: sc.AnnData) -> np.ndarray:
        """Create spatial region labels based on expression patterns"""
        logger.info("Creating spatial labels...")
        
        # Use a simpler approach - create labels based on spatial coordinates
        coordinates = self.get_spatial_coordinates(adata)
        
        # Use KMeans clustering for spatial regions
        from sklearn.cluster import KMeans
        
        # Create 8 spatial clusters
        n_clusters = 8
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        spatial_labels = kmeans.fit_predict(coordinates)
        
        logger.info(f"Created {len(np.unique(spatial_labels))} spatial clusters")
        return spatial_labels
    
    def create_torch_data(self, adata: sc.AnnData) -> List[Data]:
        """Convert spatial data to PyTorch Geometric format"""
        logger.info("Converting to PyTorch Geometric format...")
        
        # Get spatial coordinates
        coordinates = self.get_spatial_coordinates(adata)
        
        # Create spatial graph
        edge_index, edge_weights = self.create_spatial_graph(adata, coordinates)
        
        # Create labels
        labels = self.create_spatial_labels(adata)
        
        # Get expression features
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        X = self.scaler.fit_transform(X)
        
        # Create PyTorch tensors
        x = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)
        pos = torch.tensor(coordinates, dtype=torch.float)
        
        # Create single graph data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weights, y=y, pos=pos)
        
        logger.info(f"Created graph data: {data}")
        return [data]

class SpatialCardiacGNN(nn.Module):
    """Enhanced GNN for spatial cardiac tissue analysis"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 8, 
                 num_layers: int = 3, dropout: float = 0.3, use_attention: bool = True):
        super(SpatialCardiacGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            if use_attention:
                conv = GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout, concat=True)
            else:
                conv = GCNConv(hidden_dim, hidden_dim)
            
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Classification head (no global pooling needed for node classification)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        logger.info(f"Created SpatialCardiacGNN with {self.count_parameters():,} parameters")
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        
        # Input projection
        x = self.input_proj(x)
        
        # Graph convolution layers with residual connections
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = x
            x = conv(x, edge_index)
            x = norm(x)
            x = torch.relu(x)
            x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
            
            # Add residual connection (skip connection)
            if i > 0:  # Skip first layer for dimension compatibility
                x = x + residual
        
        # For node classification, we don't need global pooling
        # Apply classification head directly to node embeddings
        out = self.classifier(x)
        
        return out

class SpatialGNNTrainer:
    """Trainer for spatial GNN model - Node Classification"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, data, optimizer, criterion) -> Tuple[float, float]:
        """Train for one epoch - node classification"""
        self.model.train()
        data = data.to(self.device)
        
        optimizer.zero_grad()
        
        # Forward pass on entire graph
        out = self.model(data)
        
        # Use only training nodes for loss calculation
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Calculate training accuracy
        pred = out[data.train_mask].argmax(dim=1)
        correct = (pred == data.y[data.train_mask]).sum().item()
        total = data.train_mask.sum().item()
        accuracy = correct / total
        
        return loss.item(), accuracy
    
    def validate(self, data, criterion) -> Tuple[float, float]:
        """Validate the model - node classification"""
        self.model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            out = self.model(data)
            
            # Use validation nodes for loss calculation
            loss = criterion(out[data.val_mask], data.y[data.val_mask])
            
            # Calculate validation accuracy
            pred = out[data.val_mask].argmax(dim=1)
            correct = (pred == data.y[data.val_mask]).sum().item()
            total = data.val_mask.sum().item()
            accuracy = correct / total
        
        return loss.item(), accuracy
    
    def train(self, train_data, val_data, 
              num_epochs: int = 50, lr: float = 0.001, patience: int = 10):
        """Full training loop - node classification"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        # Initialize optimizer and criterion
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_data, optimizer, criterion)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation
            val_loss, val_acc = self.validate(val_data, criterion)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_spatial_gnn.pth')
            else:
                patience_counter += 1
            
            # Logging
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                logger.info(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_spatial_gnn.pth'))
        logger.info("Training completed!")
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('spatial_gnn_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main training function"""
    logger.info("=== Spatial GNN Training on Real Cardiac Data ===")
    
    # Configuration
    DATA_PATH = "data/processed_visium_heart.h5ad"
    BATCH_SIZE = 4  # Small batch size for spatial data
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    HIDDEN_DIM = 256
    NUM_NEIGHBORS = 10
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Data processing
        logger.info("Initializing data processor...")
        processor = SpatialCardiacDataProcessor(num_neighbors=NUM_NEIGHBORS)
        
        # Load and preprocess data
        adata = processor.load_spatial_data(DATA_PATH)
        adata = processor.preprocess_spatial_data(adata)
        
        # Create PyTorch Geometric data
        data_list = processor.create_torch_data(adata)
        
        if len(data_list) == 0:
            logger.error("No data created!")
            return
        
        # Split data (for spatial data, we typically use cross-validation or node-level splits)
        # For now, we'll use node-level splits on the same graph
        full_data = data_list[0]
        
        # Create training mask for nodes
        num_nodes = full_data.x.shape[0]
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        # 70% train, 15% val, 15% test
        indices = torch.randperm(num_nodes)
        train_end = int(0.7 * num_nodes)
        val_end = int(0.85 * num_nodes)
        
        train_mask[indices[:train_end]] = True
        val_mask[indices[train_end:val_end]] = True
        test_mask[indices[val_end:]] = True
        
        full_data.train_mask = train_mask
        full_data.val_mask = val_mask
        full_data.test_mask = test_mask
        
        # No data loaders needed for node classification
        train_data = full_data
        val_data = full_data
        
        # Model parameters
        input_dim = data_list[0].x.shape[1]
        num_classes = len(torch.unique(data_list[0].y))
        
        logger.info(f"Input dimension: {input_dim}")
        logger.info(f"Number of classes: {num_classes}")
        logger.info(f"Number of nodes: {data_list[0].x.shape[0]}")
        logger.info(f"Number of edges: {data_list[0].edge_index.shape[1]}")
        
        # Create model
        model = SpatialCardiacGNN(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            num_classes=num_classes,
            num_layers=3,
            dropout=0.3,
            use_attention=True
        )
        
        # Training
        trainer = SpatialGNNTrainer(model, device)
        trainer.train(
            train_data=train_data,
            val_data=val_data,
            num_epochs=NUM_EPOCHS,
            lr=LEARNING_RATE,
            patience=15
        )
        
        # Plot results
        trainer.plot_training_curves()
        
        # Final evaluation on test set
        model.eval()
        with torch.no_grad():
            out = model(full_data.to(device))
            test_pred = out[full_data.test_mask].argmax(dim=1)
            test_correct = (test_pred == full_data.y[full_data.test_mask]).sum().item()
            test_total = full_data.test_mask.sum().item()
            test_accuracy = test_correct / test_total
        
        logger.info(f"Final Test Accuracy: {test_accuracy:.4f}")
        
        # Print class distribution
        logger.info("Class distribution in test set:")
        unique, counts = torch.unique(full_data.y[full_data.test_mask], return_counts=True)
        for class_id, count in zip(unique, counts):
            logger.info(f"  Class {class_id}: {count} samples")
        
        logger.info("=== Training Complete ===")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

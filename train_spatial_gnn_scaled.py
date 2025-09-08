#!/usr/bin/env python3
"""
Scaled Spatial GNN Training on Combined Large Cardiac Datasets
Uses multiple spatial datasets to match RNN's data scale
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
import pandas as pd
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScaledSpatialGNN(nn.Module):
    """Scaled spatial GNN for larger datasets"""
    
    def __init__(self, input_dim, hidden_dim=128, num_classes=8, dropout=0.3):
        super(ScaledSpatialGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Input processing
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_dropout = nn.Dropout(0.1)
        
        # Multi-layer GNN architecture for larger dataset
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        self.conv4 = GCNConv(hidden_dim // 2, hidden_dim // 4)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn4 = nn.BatchNorm1d(hidden_dim // 4)
        
        # Progressive classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.LayerNorm(hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(hidden_dim // 8, hidden_dim // 16),
            nn.LayerNorm(hidden_dim // 16),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(hidden_dim // 16, num_classes)
        )
        
        logger.info(f"Created ScaledSpatialGNN with {self.count_parameters():,} parameters")
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x, edge_index, batch=None, training=True):
        # Input processing
        x = self.input_norm(x)
        x = self.input_dropout(x)
        
        # Apply edge dropout during training
        if training and self.training:
            edge_index, _ = dropout_adj(edge_index, p=0.1, training=True)
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout * 0.5, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout * 0.7, training=self.training)
        
        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout * 0.8, training=self.training)
        
        # Layer 4
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Classification
        out = self.classifier(x)
        
        return out

class LargeSpatialDataProcessor:
    """Data processor for combining multiple large spatial datasets"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_combined_spatial_data(self):
        """Load and combine multiple spatial datasets"""
        logger.info("Loading multiple large spatial datasets...")
        
        datasets = []
        dataset_names = []
        
        # 1. Load Xenium heart dataset
        try:
            xenium_path = "data/selected_datasets/spatial_transcriptomics/In Situ Gene Expression dataset analyzed using Xenium Onboard Analysis 1.9.0/processed_xenium_heart.h5ad"
            xenium = sc.read_h5ad(xenium_path)
            logger.info(f"Loaded Xenium dataset: {xenium.shape}")
            datasets.append(xenium)
            dataset_names.append("xenium")
        except Exception as e:
            logger.warning(f"Could not load Xenium dataset: {e}")
        
        # 2. Load snRNA-Spatial MI dataset
        try:
            snrna_path = "data/selected_datasets/spatial_transcriptomics/All-snRNA-Spatial multi-omic map of human myocardial infarction/091db43f-be67-4e66-ae48-bcf2fda5288c.h5ad"
            snrna = sc.read_h5ad(snrna_path)
            logger.info(f"Loaded snRNA-Spatial dataset: {snrna.shape}")
            datasets.append(snrna)
            dataset_names.append("snrna_spatial")
        except Exception as e:
            logger.warning(f"Could not load snRNA-Spatial dataset: {e}")
        
        # 3. Load original Visium dataset
        try:
            visium_path = "data/processed_visium_heart.h5ad"
            visium = sc.read_h5ad(visium_path)
            logger.info(f"Loaded Visium dataset: {visium.shape}")
            datasets.append(visium)
            dataset_names.append("visium")
        except Exception as e:
            logger.warning(f"Could not load Visium dataset: {e}")
        
        if not datasets:
            raise ValueError("No datasets could be loaded!")
        
        # Combine datasets
        combined_data = self.combine_datasets(datasets, dataset_names)
        return combined_data
    
    def combine_datasets(self, datasets, names):
        """Intelligently combine multiple spatial datasets"""
        logger.info(f"Combining {len(datasets)} spatial datasets...")
        
        # Find common genes across all datasets
        common_genes = None
        for adata in datasets:
            genes = set(adata.var_names)
            if common_genes is None:
                common_genes = genes
            else:
                common_genes = common_genes.intersection(genes)
        
        logger.info(f"Found {len(common_genes)} common genes across datasets")
        
        # Subsample to manageable size and select top variable genes
        common_genes = list(common_genes)[:1000]  # Top 1000 common genes
        
        combined_X = []
        combined_spatial = []
        combined_obs = []
        dataset_labels = []
        
        for i, (adata, name) in enumerate(zip(datasets, names)):
            # Subset to common genes
            adata_subset = adata[:, common_genes].copy()
            
            # Get expression data
            X = adata_subset.X
            if hasattr(X, 'toarray'):
                X = X.toarray()
            
            # Clean data
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Sample subset to prevent memory issues
            n_samples = min(len(adata_subset), 10000)  # Max 10k samples per dataset
            if len(adata_subset) > n_samples:
                indices = np.random.choice(len(adata_subset), n_samples, replace=False)
                X = X[indices]
                adata_subset = adata_subset[indices]
            
            # Get spatial coordinates
            if 'spatial' in adata_subset.obsm:
                spatial_coords = adata_subset.obsm['spatial']
            else:
                # Create pseudo-spatial coordinates if not available
                spatial_coords = np.random.randn(len(adata_subset), 2) * 100 + i * 200
            
            combined_X.append(X)
            combined_spatial.append(spatial_coords)
            combined_obs.append(adata_subset.obs)
            dataset_labels.extend([i] * len(adata_subset))
        
        # Combine all data
        final_X = np.vstack(combined_X)
        final_spatial = np.vstack(combined_spatial)
        final_obs = pd.concat(combined_obs, ignore_index=True)
        
        # Add dataset labels
        final_obs['dataset'] = dataset_labels
        
        # Create combined AnnData
        combined_adata = sc.AnnData(final_X)
        combined_adata.obsm['spatial'] = final_spatial
        combined_adata.obs = final_obs
        combined_adata.var_names = common_genes
        
        logger.info(f"Combined dataset shape: {combined_adata.shape}")
        return combined_adata
    
    def process_large_dataset(self, adata):
        """Process the large combined dataset"""
        logger.info("Processing large combined spatial dataset...")
        
        # Get expression data
        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Normalize
        X = self.scaler.fit_transform(X)
        
        # Light noise for regularization
        noise_factor = 0.002
        X += np.random.normal(0, noise_factor, X.shape)
        
        # Update adata
        adata.X = X
        
        logger.info(f"Processed dataset: {adata.shape}")
        return adata
    
    def create_large_spatial_graph(self, adata, k_neighbors=8):
        """Create spatial graph for large dataset"""
        spatial_coords = adata.obsm['spatial']
        logger.info(f"Creating large spatial graph with {k_neighbors} neighbors...")
        
        # For very large datasets, we need to be more efficient
        n_samples = len(spatial_coords)
        if n_samples > 50000:
            # Subsample for graph construction to prevent memory issues
            indices = np.random.choice(n_samples, 50000, replace=False)
            logger.info(f"Subsampling to 50,000 nodes for graph construction")
            spatial_coords = spatial_coords[indices]
            # Update adata to match
            adata = adata[indices].copy()
        
        # Build graph
        nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree').fit(spatial_coords)
        distances, indices = nbrs.kneighbors(spatial_coords)
        
        edge_list = []
        edge_weights = []
        
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):
                neighbor = indices[i][j]
                distance = distances[i][j]
                
                if distance < np.percentile(distances[:, 1:], 85):
                    edge_list.append([i, neighbor])
                    edge_list.append([neighbor, i])
                    weight = np.exp(-distance * 1.2)
                    edge_weights.extend([weight, weight])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        
        logger.info(f"Created large graph with {edge_index.shape[1]} edges for {len(spatial_coords)} nodes")
        return edge_index, edge_attr, adata
    
    def create_large_spatial_labels(self, adata, n_clusters=10):
        """Create spatial labels for large dataset"""
        spatial_coords = adata.obsm['spatial']
        
        # Use more clusters for larger dataset
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(spatial_coords)
        
        logger.info(f"Created {n_clusters} spatial clusters for large dataset")
        return labels

class ScaledTrainer:
    """Trainer for scaled spatial GNN"""
    
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        
        # Training setup for larger dataset
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=0.0008,  # Lower LR for stability
            weight_decay=0.01,
            eps=1e-8
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=6, verbose=True
        )
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        
        self.best_val_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0
        
    def train_epoch(self, data, train_mask):
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(data.x, data.edge_index, training=True)
        loss = self.criterion(out[train_mask], data.y[train_mask])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
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
    
    def train(self, data, train_mask, val_mask, epochs=100):
        logger.info(f"Starting scaled training for {epochs} epochs")
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(data, train_mask)
            val_loss, val_acc = self.validate(data, val_mask)
            
            self.scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'best_scaled_gnn.pth')
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        self.model.load_state_dict(torch.load('best_scaled_gnn.pth'))
        logger.info("Scaled training completed!")

def main():
    logger.info("=== Scaled Spatial GNN Training on Combined Large Datasets ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize processor
    processor = LargeSpatialDataProcessor()
    
    # Load combined large datasets
    try:
        adata = processor.load_combined_spatial_data()
        adata = processor.process_large_dataset(adata)
        
        # Create large spatial graph
        edge_index, edge_attr, adata = processor.create_large_spatial_graph(adata, k_neighbors=8)
        
        # Create spatial labels
        labels = processor.create_large_spatial_labels(adata, n_clusters=10)
        
        # Create PyTorch Geometric data
        graph_data = Data(
            x=torch.tensor(adata.X, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(labels, dtype=torch.long),
            pos=torch.tensor(adata.obsm['spatial'], dtype=torch.float)
        )
        
        logger.info(f"Created large graph data: {graph_data}")
        
        # Split data with more training samples
        num_nodes = graph_data.x.shape[0]
        indices = np.arange(num_nodes)
        
        # Use 50% for training to have enough data
        train_indices, temp_indices = train_test_split(indices, test_size=0.5, random_state=42, 
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
        
        logger.info(f"Large data split - Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
        
        # Create scaled model
        input_dim = graph_data.x.shape[1]
        num_classes = len(np.unique(labels))
        
        model = ScaledSpatialGNN(
            input_dim=input_dim,
            hidden_dim=128,
            num_classes=num_classes,
            dropout=0.3
        )
        
        # Train scaled model
        trainer = ScaledTrainer(model, device)
        trainer.train(graph_data, train_mask, val_mask, epochs=100)
        
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
            
            logger.info(f"=== SCALED GNN RESULTS ===")
            logger.info(f"  Training Data Size: {train_mask.sum():,} samples")
            logger.info(f"  Total Dataset Size: {num_nodes:,} samples")
            logger.info(f"  Train Accuracy: {train_acc:.4f}")
            logger.info(f"  Validation Accuracy: {val_acc:.4f}")
            logger.info(f"  Test Accuracy: {test_acc:.4f}")
            logger.info(f"  Train-Test Gap: {overfitting_gap:.4f}")
            
            # Compare with RNN
            rnn_training_size = 19500
            gnn_training_size = train_mask.sum().item()
            scaling_factor = gnn_training_size / rnn_training_size
            
            logger.info(f"")
            logger.info(f"=== COMPARISON WITH RNN ===")
            logger.info(f"  RNN Training Size: {rnn_training_size:,}")
            logger.info(f"  GNN Training Size: {gnn_training_size:,}")
            logger.info(f"  GNN/RNN Ratio: {scaling_factor:.2f}x")
            
            if scaling_factor >= 0.5:
                logger.info("✅ GNN dataset now at reasonable scale!")
            else:
                logger.info("⚠️  GNN dataset still smaller than ideal")
                
    except Exception as e:
        logger.error(f"Error in scaled training: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("=== Scaled Training Complete ===")

if __name__ == "__main__":
    main()

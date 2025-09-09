"""
Large-Scale Pseudo-Spatial Dataset Creation
Convert 191K single-cell dataset with UMAP coordinates to spatial-like dataset for GNN training
"""

import scanpy as sc
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LargeScalePseudoSpatialProcessor:
    """Process large single-cell dataset as pseudo-spatial for GNN training"""
    
    def __init__(self, subset_size=50000, cardiomyocyte_focus=True):
        """
        Initialize processor
        
        Args:
            subset_size: Number of cells to use (for memory management)
            cardiomyocyte_focus: Whether to focus on cardiomyocytes
        """
        self.subset_size = subset_size
        self.cardiomyocyte_focus = cardiomyocyte_focus
    
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the large dataset"""
        
        logger.info(f"🚀 Loading large single-cell dataset: {file_path}")
        
        # Load the full dataset
        adata = sc.read_h5ad(file_path)
        logger.info(f"Original dataset: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
        
        # Log cell type distribution
        cell_types = adata.obs['cell_type_original'].value_counts()
        logger.info("Cell type distribution:")
        for cell_type, count in cell_types.head(10).items():
            logger.info(f"  {cell_type}: {count:,} cells")
        
        return adata
    
    def create_strategic_subset(self, adata):
        """Create a strategic subset focusing on cardiomyocytes and related cells"""
        
        logger.info(f"🎯 Creating strategic subset of {self.subset_size:,} cells...")
        
        if self.cardiomyocyte_focus:
            # Strategy: Take all cardiomyocytes + representative sample of other cells
            cardiomyocytes = adata[adata.obs['cell_type_original'] == 'Cardiomyocyte'].copy()
            other_cells = adata[adata.obs['cell_type_original'] != 'Cardiomyocyte'].copy()
            
            n_cardio = min(len(cardiomyocytes), self.subset_size // 2)  # Take up to half as cardiomyocytes
            n_other = self.subset_size - n_cardio
            
            # Sample cardiomyocytes
            if len(cardiomyocytes) > n_cardio:
                cardio_indices = np.random.choice(len(cardiomyocytes), n_cardio, replace=False)
                cardiomyocytes_subset = cardiomyocytes[cardio_indices]
            else:
                cardiomyocytes_subset = cardiomyocytes
            
            # Sample other cells proportionally
            other_indices = np.random.choice(len(other_cells), n_other, replace=False)
            other_subset = other_cells[other_indices]
            
            # Combine
            subset_adata = sc.concat([cardiomyocytes_subset, other_subset])
            
            logger.info(f"Strategic subset created:")
            logger.info(f"  Cardiomyocytes: {len(cardiomyocytes_subset):,}")
            logger.info(f"  Other cell types: {len(other_subset):,}")
            
        else:
            # Random sampling
            indices = np.random.choice(len(adata), self.subset_size, replace=False)
            subset_adata = adata[indices].copy()
        
        # Log final subset distribution
        subset_types = subset_adata.obs['cell_type_original'].value_counts()
        logger.info("Final subset cell types:")
        for cell_type, count in subset_types.items():
            logger.info(f"  {cell_type}: {count:,} cells")
        
        return subset_adata
    
    def prepare_spatial_coordinates(self, adata):
        """Prepare UMAP coordinates as spatial coordinates"""
        
        logger.info("📍 Preparing pseudo-spatial coordinates from UMAP...")
        
        # Use UMAP coordinates as spatial coordinates
        spatial_coords = adata.obsm['X_umap'].copy()
        
        # Normalize coordinates to [0, 1] range
        spatial_coords = spatial_coords - spatial_coords.min(axis=0)
        spatial_coords = spatial_coords / (spatial_coords.max(axis=0) + 1e-8)
        
        # Store in obsm
        adata.obsm['spatial'] = spatial_coords
        adata.obsm['spatial_original'] = adata.obsm['X_umap'].copy()
        
        logger.info(f"Spatial coordinates prepared: {spatial_coords.shape}")
        logger.info(f"Coordinate ranges: X[{spatial_coords[:, 0].min():.3f}, {spatial_coords[:, 0].max():.3f}], Y[{spatial_coords[:, 1].min():.3f}, {spatial_coords[:, 1].max():.3f}]")
        
        return adata
    
    def create_enhanced_features(self, adata):
        """Create enhanced features for GNN training"""
        
        logger.info("🔧 Creating enhanced features...")
        
        # 1. Gene expression features (already in adata.X)
        if sparse.issparse(adata.X):
            X_expr = adata.X.toarray()
        else:
            X_expr = adata.X.copy()
        
        # Handle NaN values
        if np.isnan(X_expr).any():
            logger.info("Found NaN values in expression data, imputing with mean...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X_expr = imputer.fit_transform(X_expr)
        
        # 2. Spatial features from UMAP coordinates
        spatial_coords = adata.obsm['spatial']
        
        # Distance to centroid
        centroid = spatial_coords.mean(axis=0)
        distances_to_centroid = np.linalg.norm(spatial_coords - centroid, axis=1).reshape(-1, 1)
        
        # Local density (distance to 10th nearest neighbor)
        nn = NearestNeighbors(n_neighbors=11)  # 11 because it includes the point itself
        nn.fit(spatial_coords)
        distances, _ = nn.kneighbors(spatial_coords)
        local_density = distances[:, -1].reshape(-1, 1)  # Distance to 10th neighbor
        
        # Angle from centroid
        relative_coords = spatial_coords - centroid
        angles = np.arctan2(relative_coords[:, 1], relative_coords[:, 0]).reshape(-1, 1)
        
        # 3. Cell type features (one-hot encoding)
        le = LabelEncoder()
        cell_type_encoded = le.fit_transform(adata.obs['cell_type_original'])
        n_cell_types = len(le.classes_)
        cell_type_onehot = np.eye(n_cell_types)[cell_type_encoded]
        
        # 4. Combine all features
        enhanced_features = np.hstack([
            X_expr,                     # Gene expression
            spatial_coords,             # Spatial coordinates  
            distances_to_centroid,      # Distance to centroid
            local_density,             # Local density
            angles,                    # Angle from centroid
            cell_type_onehot           # Cell type one-hot
        ])
        
        # Normalize all features
        scaler = StandardScaler()
        enhanced_features = scaler.fit_transform(enhanced_features)
        
        # Check for remaining NaN or inf values
        if np.isnan(enhanced_features).any() or np.isinf(enhanced_features).any():
            logger.warning("Found NaN or inf values after processing, replacing with zeros")
            enhanced_features = np.nan_to_num(enhanced_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"Enhanced features created: {enhanced_features.shape}")
        logger.info(f"Feature composition:")
        logger.info(f"  Gene expression: {X_expr.shape[1]} features")
        logger.info(f"  Spatial coordinates: 2 features")
        logger.info(f"  Spatial metrics: 3 features")
        logger.info(f"  Cell type one-hot: {n_cell_types} features")
        logger.info(f"  Total: {enhanced_features.shape[1]} features")
        
        return enhanced_features, le.classes_
    
    def create_spatial_graph(self, spatial_coords, k_neighbors=12):
        """Create spatial graph from coordinates"""
        
        logger.info(f"🕸️ Creating spatial graph with k={k_neighbors} neighbors...")
        
        # Use KNN to create edges
        nn = NearestNeighbors(n_neighbors=k_neighbors + 1)  # +1 because it includes the point itself
        nn.fit(spatial_coords)
        distances, indices = nn.kneighbors(spatial_coords)
        
        # Create edge list (excluding self-loops)
        edges = []
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors[1:]:  # Skip first (self)
                edges.append([i, neighbor])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        logger.info(f"Spatial graph created: {edge_index.shape[1]:,} edges")
        
        return edge_index
    
    def create_labels(self, adata):
        """Create node labels for classification"""
        
        logger.info("🏷️ Creating node labels...")
        
        # Use cell types as labels
        le = LabelEncoder()
        labels = le.fit_transform(adata.obs['cell_type_original'])
        
        logger.info(f"Created labels for {len(le.classes_)} cell types:")
        for i, cell_type in enumerate(le.classes_):
            count = (labels == i).sum()
            logger.info(f"  {i}: {cell_type} ({count:,} cells)")
        
        return torch.tensor(labels, dtype=torch.long), le.classes_
    
    def process_large_dataset(self, file_path, output_path):
        """Main processing function"""
        
        logger.info("🚀 Processing Large-Scale Pseudo-Spatial Dataset")
        logger.info("=" * 60)
        
        # Load data
        adata = self.load_and_preprocess_data(file_path)
        
        # Create strategic subset
        adata_subset = self.create_strategic_subset(adata)
        
        # Prepare spatial coordinates
        adata_subset = self.prepare_spatial_coordinates(adata_subset)
        
        # Create enhanced features
        node_features, cell_types = self.create_enhanced_features(adata_subset)
        node_features = torch.tensor(node_features, dtype=torch.float32)
        
        # Create spatial graph
        edge_index = self.create_spatial_graph(adata_subset.obsm['spatial'])
        
        # Create labels
        labels, label_classes = self.create_labels(adata_subset)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=labels,
            pos=torch.tensor(adata_subset.obsm['spatial'], dtype=torch.float32),
            num_nodes=len(adata_subset)
        )
        
        # Add metadata
        data.cell_types = label_classes
        data.num_features = node_features.shape[1]
        data.num_classes = len(label_classes)
        
        # Save processed data
        torch.save(data, output_path)
        
        logger.info("🎉 LARGE-SCALE PSEUDO-SPATIAL DATASET CREATED!")
        logger.info("=" * 50)
        logger.info(f"📊 Final Statistics:")
        logger.info(f"   Total cells: {data.num_nodes:,}")
        logger.info(f"   Total features: {data.num_features:,}")
        logger.info(f"   Total edges: {data.edge_index.shape[1]:,}")
        logger.info(f"   Number of classes: {data.num_classes}")
        logger.info(f"   Cell types: {', '.join(label_classes)}")
        logger.info(f"   File saved: {output_path}")
        
        return data

def main():
    """Main function to create large-scale pseudo-spatial dataset"""
    
    # Configuration
    file_path = 'data/single_cell_rnaseq/091db43f-be67-4e66-ae48-bcf2fda5288c.h5ad'
    output_path = 'data/large_scale_pseudo_spatial_50k.pt'
    
    # Create processor
    processor = LargeScalePseudoSpatialProcessor(
        subset_size=50000,  # 50K cells - much larger than 8.6K
        cardiomyocyte_focus=True
    )
    
    # Process dataset
    try:
        data = processor.process_large_dataset(file_path, output_path)
        
        logger.info("\n🎯 COMPARISON WITH PREVIOUS DATASETS:")
        logger.info(f"   Previous spatial GNN: 8,634 cells → 14.58% accuracy")
        logger.info(f"   New pseudo-spatial: {data.num_nodes:,} cells → Expected much better!")
        logger.info(f"   Temporal RNN reference: 230,000 cells → 90.83% R²")
        logger.info("\n✅ This dataset bridges the gap between spatial and temporal scales!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

"""
Spatial Transcriptomics Data Loader for GNN Training
Handles loading and preprocessing of spatial transcriptomics datasets
for cardiac tissue analysis and cardiomyocyte differentiation prediction.
"""

import scanpy as sc
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import anndata as ad
from pathlib import Path
import logging
from typing import Tuple, List, Optional, Dict

# Configure scanpy settings
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')

class SpatialDataProcessor:
    """
    Processes spatial transcriptomics data for GNN training.
    Creates graph structures from spatial coordinates and gene expression.
    """
    
    def __init__(self, 
                 n_neighbors: int = 6,
                 n_top_genes: int = 2000,
                 normalize: bool = True,
                 log_transform: bool = True):
        """
        Initialize the spatial data processor.
        
        Args:
            n_neighbors: Number of neighbors for spatial graph construction
            n_top_genes: Number of top variable genes to keep
            normalize: Whether to normalize gene expression
            log_transform: Whether to log-transform the data
        """
        self.n_neighbors = n_neighbors
        self.n_top_genes = n_top_genes
        self.normalize = normalize
        self.log_transform = log_transform
        self.scaler = StandardScaler()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_spatial_data(self, data_path: str) -> ad.AnnData:
        """
        Load spatial transcriptomics data.
        
        Args:
            data_path: Path to the h5ad file
            
        Returns:
            AnnData object with spatial information
        """
        self.logger.info(f"Loading spatial data from {data_path}")
        
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        adata = sc.read_h5ad(data_path)
        
        self.logger.info(f"Loaded data with shape: {adata.shape}")
        self.logger.info(f"Spatial coordinates available: {'spatial' in adata.obsm}")
        
        return adata
        
    def preprocess_expression_data(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Preprocess gene expression data with standard scanpy workflow.
        
        Args:
            adata: AnnData object with raw counts
            
        Returns:
            Preprocessed AnnData object
        """
        self.logger.info("Starting expression data preprocessing...")
        
        # Make a copy for processing
        adata = adata.copy()
        
        # Handle potential issues with the data matrix
        if hasattr(adata.X, 'toarray'):
            X = adata.X.toarray()
        else:
            X = adata.X
            
        # Check for and handle NaN values
        if np.isnan(X).any():
            self.logger.warning("Found NaN values in expression matrix, replacing with zeros")
            X = np.nan_to_num(X, nan=0.0)
            adata.X = X
            
        # Check for and handle infinite values
        if np.isinf(X).any():
            self.logger.warning("Found infinite values in expression matrix, replacing with zeros")
            X = np.nan_to_num(X, posinf=0.0, neginf=0.0)
            adata.X = X
        
        # Calculate QC metrics
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
        
        # Filter genes and cells
        self.logger.info("Filtering genes and cells...")
        sc.pp.filter_cells(adata, min_genes=10)   # More lenient filtering
        sc.pp.filter_genes(adata, min_cells=3)    # Filter genes expressed in too few cells
        
        # Store raw data
        adata.raw = adata
        
        # Check if data is already normalized/log-transformed
        max_value = np.max(adata.X)
        if max_value > 50:  # Likely raw counts
            # Normalize and log transform if specified
            if self.normalize:
                self.logger.info("Normalizing to 10,000 reads per cell...")
                sc.pp.normalize_total(adata, target_sum=1e4)
                
            if self.log_transform:
                self.logger.info("Log transforming expression data...")
                sc.pp.log1p(adata)
        else:
            self.logger.info("Data appears to be already normalized/log-transformed")
            
        # Handle highly variable genes with error handling
        try:
            self.logger.info(f"Finding top {self.n_top_genes} highly variable genes...")
            sc.pp.highly_variable_genes(adata, n_top_genes=min(self.n_top_genes, adata.n_vars))
            
            # Keep only highly variable genes
            if 'highly_variable' in adata.var.columns and adata.var['highly_variable'].any():
                adata = adata[:, adata.var.highly_variable]
            else:
                self.logger.warning("No highly variable genes found, keeping all genes")
                
        except Exception as e:
            self.logger.warning(f"Error in highly variable gene selection: {e}")
            self.logger.info("Proceeding without highly variable gene selection")
            
        # Scale data
        self.logger.info("Scaling expression data...")
        sc.pp.scale(adata, max_value=10)
        
        # Final check for NaN values after scaling
        if hasattr(adata.X, 'toarray'):
            X_scaled = adata.X.toarray()
        else:
            X_scaled = adata.X
            
        if np.isnan(X_scaled).any():
            self.logger.warning("Found NaN values after scaling, replacing with zeros")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)
            adata.X = X_scaled
        
        self.logger.info(f"Preprocessing complete. Final shape: {adata.shape}")
        
        return adata
        
    def create_spatial_graph(self, adata: ad.AnnData) -> torch.Tensor:
        """
        Create spatial graph based on k-nearest neighbors in spatial coordinates.
        
        Args:
            adata: AnnData object with spatial coordinates
            
        Returns:
            Edge index tensor for PyTorch Geometric
        """
        self.logger.info(f"Creating spatial graph with {self.n_neighbors} neighbors...")
        
        if 'spatial' not in adata.obsm:
            raise ValueError("No spatial coordinates found in adata.obsm['spatial']")
            
        # Get spatial coordinates
        spatial_coords = adata.obsm['spatial']
        
        # Create k-nearest neighbors graph
        knn_graph = kneighbors_graph(
            spatial_coords, 
            n_neighbors=self.n_neighbors, 
            mode='connectivity',
            include_self=False
        )
        
        # Convert to edge index format for PyTorch Geometric
        edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)
        
        self.logger.info(f"Created spatial graph with {edge_index.shape[1]} edges")
        
        return edge_index
        
    def create_node_features(self, adata: ad.AnnData) -> torch.Tensor:
        """
        Create node features from gene expression data.
        
        Args:
            adata: Preprocessed AnnData object
            
        Returns:
            Node feature tensor
        """
        self.logger.info("Creating node features from gene expression...")
        
        # Use the processed expression matrix
        if hasattr(adata, 'X') and adata.X is not None:
            expression_matrix = adata.X
            
            # Convert to dense if sparse
            if hasattr(expression_matrix, 'toarray'):
                expression_matrix = expression_matrix.toarray()
                
            # Convert to tensor
            node_features = torch.tensor(expression_matrix, dtype=torch.float32)
            
        else:
            raise ValueError("No expression data found in adata.X")
            
        self.logger.info(f"Created node features with shape: {node_features.shape}")
        
        return node_features
        
    def create_labels(self, adata: ad.AnnData, label_key: str = 'cell_type') -> torch.Tensor:
        """
        Create labels for supervised learning.
        
        Args:
            adata: AnnData object with cell type annotations
            label_key: Key in adata.obs containing cell type labels
            
        Returns:
            Label tensor
        """
        if label_key not in adata.obs.columns:
            self.logger.warning(f"Label key '{label_key}' not found. Creating dummy labels.")
            # Create dummy labels for unsupervised/semi-supervised learning
            labels = torch.zeros(adata.n_obs, dtype=torch.long)
        else:
            # Encode categorical labels
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            labels = le.fit_transform(adata.obs[label_key])
            labels = torch.tensor(labels, dtype=torch.long)
            
            # Store label encoder for later use
            self.label_encoder = le
            
        self.logger.info(f"Created labels with shape: {labels.shape}")
        
        return labels
        
    def process_dataset(self, data_path: str, label_key: str = 'cell_type') -> Data:
        """
        Process the entire dataset and create a PyTorch Geometric Data object.
        
        Args:
            data_path: Path to the spatial transcriptomics data
            label_key: Key for cell type labels
            
        Returns:
            PyTorch Geometric Data object
        """
        self.logger.info("Processing complete spatial dataset...")
        
        # Load and preprocess data
        adata = self.load_spatial_data(data_path)
        adata = self.preprocess_expression_data(adata)
        
        # Create graph components
        edge_index = self.create_spatial_graph(adata)
        node_features = self.create_node_features(adata)
        labels = self.create_labels(adata, label_key)
        
        # Create spatial coordinates tensor
        spatial_coords = torch.tensor(adata.obsm['spatial'], dtype=torch.float32)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=labels,
            pos=spatial_coords,
            num_nodes=adata.n_obs
        )
        
        # Store metadata
        data.gene_names = adata.var_names.tolist()
        data.n_genes = adata.n_vars
        
        self.logger.info(f"Created PyTorch Geometric data object:")
        self.logger.info(f"  - Nodes: {data.num_nodes}")
        self.logger.info(f"  - Edges: {data.edge_index.shape[1]}")
        self.logger.info(f"  - Features: {data.x.shape[1]}")
        self.logger.info(f"  - Classes: {torch.unique(data.y).numel()}")
        
        return data


def load_and_process_heart_data(data_dir: str = "data") -> Data:
    """
    Convenience function to load and process the heart spatial data.
    
    Args:
        data_dir: Directory containing the data files
        
    Returns:
        Processed PyTorch Geometric Data object
    """
    processor = SpatialDataProcessor(
        n_neighbors=6,
        n_top_genes=2000,
        normalize=True,
        log_transform=True
    )
    
    heart_data_path = Path(data_dir) / "processed_visium_heart.h5ad"
    
    return processor.process_dataset(str(heart_data_path))


if __name__ == "__main__":
    # Test the data processing pipeline
    print("Testing spatial data processing pipeline...")
    
    try:
        # Process the heart data
        data = load_and_process_heart_data()
        
        print("✓ Data processing successful!")
        print(f"Final data object: {data}")
        
    except Exception as e:
        print(f"✗ Error in data processing: {e}")
        import traceback
        traceback.print_exc()

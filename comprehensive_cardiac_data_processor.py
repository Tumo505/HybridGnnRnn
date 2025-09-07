#!/usr/bin/env python3
"""
Comprehensive Real Cardiac Data Processor
Process the full GSE175634 temporal dataset and spatial data for training
"""

import os
import sys
import pandas as pd
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import mmread
import gzip
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveCardiacDataProcessor:
    """Process all available cardiac datasets for training"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.temporal_dir = self.data_dir / "selected_datasets" / "temporal_data"
        self.processed_dir = self.data_dir / "processed"
        
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_full_temporal_data(self) -> sc.AnnData:
        """Load the complete GSE175634 temporal dataset"""
        logger.info("📊 Loading full temporal cardiac data (230K+ cells)...")
        
        # Load metadata
        metadata_file = self.temporal_dir / "GSE175634_cell_metadata.tsv.gz"
        with gzip.open(metadata_file, 'rt') as f:
            metadata = pd.read_csv(f, sep='\t', index_col=0)
        
        logger.info(f"  Metadata loaded: {len(metadata)} cells")
        logger.info(f"  Columns: {list(metadata.columns)}")
        
        # Load count matrix
        counts_file = self.temporal_dir / "GSE175634_cell_counts_sctransform.mtx.gz"
        with gzip.open(counts_file, 'rb') as f:
            matrix = mmread(f).T.tocsr()  # Transpose to cells x genes
        
        logger.info(f"  Count matrix loaded: {matrix.shape}")
        
        # Load gene names
        gene_file = self.temporal_dir / "GSE175634_gene_indices_counts_sctransform.tsv.gz"
        with gzip.open(gene_file, 'rt') as f:
            gene_names = pd.read_csv(f, sep='\t', header=None)[0].values
        
        logger.info(f"  Gene names loaded: {len(gene_names)} genes")
        
        # Create AnnData object with proper gene name matching
        n_genes_matrix = matrix.shape[1]
        n_genes_names = len(gene_names)
        
        if n_genes_matrix != n_genes_names:
            logger.warning(f"  Gene count mismatch: matrix {n_genes_matrix}, names {n_genes_names}")
            # Use the minimum to avoid indexing errors
            min_genes = min(n_genes_matrix, n_genes_names)
            matrix = matrix[:, :min_genes]
            gene_names = gene_names[:min_genes]
            logger.info(f"  Adjusted to {min_genes} genes")
        
        adata = sc.AnnData(X=matrix, obs=metadata)
        adata.var_names = gene_names
        
        # Process time points
        if 'diffday' in adata.obs.columns:
            adata.obs['timepoint'] = adata.obs['diffday']
        
        logger.info(f"✅ Full temporal dataset loaded: {adata.n_obs} cells, {adata.n_vars} genes")
        logger.info(f"  Time points: {sorted(adata.obs['timepoint'].unique())}")
        
        return adata
    
    def create_temporal_sequences(self, adata: sc.AnnData, 
                                 sequence_length: int = 7,
                                 n_sequences_per_cell_type: int = 1000,
                                 n_top_genes: int = 500) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """Create temporal sequences from the full dataset"""
        logger.info(f"🔄 Creating temporal sequences (target: {n_sequences_per_cell_type} per cell type)...")
        
        # Clean data first - remove infinity and NaN values
        if hasattr(adata.X, 'data'):
            adata.X.data = np.nan_to_num(adata.X.data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Simple gene selection - use every nth gene for efficiency
        if n_top_genes < adata.n_vars:
            logger.info(f"  Selecting {n_top_genes} genes from {adata.n_vars} total genes...")
            step = max(1, adata.n_vars // n_top_genes)
            selected_genes = list(range(0, adata.n_vars, step))[:n_top_genes]
            adata_hvg = adata[:, selected_genes].copy()
        else:
            adata_hvg = adata.copy()
        
        logger.info(f"  Using {adata_hvg.n_vars} genes")
        
        sequences = []
        labels = []
        names = []
        
        # Get available time points
        time_points = sorted(adata_hvg.obs['timepoint'].unique())
        logger.info(f"  Available time points: {time_points}")
        
        # Get cell types
        if 'type' in adata_hvg.obs.columns:
            cell_types = adata_hvg.obs['type'].unique()
            logger.info(f"  Cell types: {cell_types}")
        else:
            cell_types = ['all_cells']
            adata_hvg.obs['type'] = 'all_cells'
        
        # Create sequences for each cell type
        for cell_type_idx, cell_type in enumerate(cell_types):
            logger.info(f"  Processing cell type: {cell_type}")
            
            # Filter by cell type
            if cell_type != 'all_cells':
                cell_mask = adata_hvg.obs['type'] == cell_type
                adata_ct = adata_hvg[cell_mask].copy()
            else:
                adata_ct = adata_hvg.copy()
            
            # Create sequences by sampling cells from consecutive time points
            sequences_created = 0
            
            for seq_idx in range(n_sequences_per_cell_type):
                if sequences_created >= n_sequences_per_cell_type:
                    break
                
                sequence_data = []
                valid_sequence = True
                
                # For each time point in sequence
                for tp_idx, timepoint in enumerate(time_points[:sequence_length]):
                    tp_cells = adata_ct[adata_ct.obs['timepoint'] == timepoint]
                    
                    if len(tp_cells) == 0:
                        valid_sequence = False
                        break
                    
                    # Sample a random cell from this timepoint
                    cell_idx = np.random.randint(0, len(tp_cells))
                    cell_expression = tp_cells.X[cell_idx]
                    
                    # Convert to dense array if sparse
                    if hasattr(cell_expression, 'toarray'):
                        cell_expression = cell_expression.toarray().flatten()
                    else:
                        cell_expression = np.array(cell_expression).flatten()
                    
                    sequence_data.append(cell_expression)
                
                if valid_sequence and len(sequence_data) == sequence_length:
                    sequences.append(np.array(sequence_data))
                    labels.append(cell_type_idx)
                    names.append(f"{cell_type}_seq_{sequences_created}")
                    sequences_created += 1
            
            logger.info(f"    Created {sequences_created} sequences for {cell_type}")
        
        logger.info(f"✅ Total sequences created: {len(sequences)}")
        return sequences, labels, names
    
    def load_spatial_data(self) -> sc.AnnData:
        """Load spatial transcriptomics data"""
        logger.info("🗺️ Loading spatial transcriptomics data...")
        
        spatial_file = self.data_dir / "processed_visium_heart.h5ad"
        if not spatial_file.exists():
            logger.warning("⚠️ No spatial data found")
            return None
        
        adata_spatial = sc.read_h5ad(spatial_file)
        logger.info(f"  Spatial data: {adata_spatial.n_obs} spots, {adata_spatial.n_vars} genes")
        
        return adata_spatial
    
    def create_spatial_temporal_sequences(self, adata_spatial: sc.AnnData,
                                        n_sequences: int = 500,
                                        sequence_length: int = 30) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """Convert spatial data to pseudo-temporal sequences"""
        logger.info("🔄 Creating pseudo-temporal sequences from spatial data...")
        
        sequences = []
        labels = []
        names = []
        
        # Use leiden clusters as conditions if available
        if 'leiden' in adata_spatial.obs.columns:
            clusters = adata_spatial.obs['leiden'].unique()
            logger.info(f"  Using {len(clusters)} spatial clusters as conditions")
        else:
            clusters = ['spatial']
            adata_spatial.obs['leiden'] = 'spatial'
        
        for i in range(n_sequences):
            # Sample random spots for this sequence
            spot_indices = np.random.choice(adata_spatial.n_obs, sequence_length, replace=False)
            
            # Get expression data
            expression_data = adata_spatial[spot_indices].X
            if hasattr(expression_data, 'toarray'):
                expression_data = expression_data.toarray()
            
            # Use top 200 genes for efficiency
            n_genes = min(200, expression_data.shape[1])
            gene_indices = np.random.choice(expression_data.shape[1], n_genes, replace=False)
            sequence = expression_data[:, gene_indices]
            
            # Assign label based on dominant cluster
            dominant_cluster = adata_spatial.obs.iloc[spot_indices[0]]['leiden']
            if isinstance(dominant_cluster, str):
                label = hash(dominant_cluster) % 5  # Map to 5 classes
            else:
                label = int(dominant_cluster) % 5
            
            sequences.append(sequence)
            labels.append(label)
            names.append(f"spatial_seq_{i}")
        
        logger.info(f"✅ Created {len(sequences)} spatial-temporal sequences")
        return sequences, labels, names
    
    def create_comprehensive_dataset(self, 
                                   temporal_sequences_per_type: int = 2000,
                                   spatial_sequences: int = 1000,
                                   max_sequence_length: int = 100) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """Create a comprehensive dataset from all available sources"""
        logger.info("🚀 Creating comprehensive cardiac dataset...")
        
        all_sequences = []
        all_labels = []
        all_names = []
        
        # Load and process temporal data
        try:
            adata_temporal = self.load_full_temporal_data()
            
            # Create temporal sequences
            temp_sequences, temp_labels, temp_names = self.create_temporal_sequences(
                adata_temporal, 
                n_sequences_per_cell_type=temporal_sequences_per_type
            )
            
            all_sequences.extend(temp_sequences)
            all_labels.extend(temp_labels)
            all_names.extend(temp_names)
            
            logger.info(f"  Added {len(temp_sequences)} temporal sequences")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not process temporal data: {e}")
        
        # Load and process spatial data
        try:
            adata_spatial = self.load_spatial_data()
            if adata_spatial is not None:
                spatial_seq, spatial_lab, spatial_nam = self.create_spatial_temporal_sequences(
                    adata_spatial, n_sequences=spatial_sequences
                )
                
                # Offset spatial labels to avoid conflict with temporal labels
                spatial_lab = [l + 10 for l in spatial_lab]  # Spatial conditions start at 10
                
                all_sequences.extend(spatial_seq)
                all_labels.extend(spatial_lab)
                all_names.extend(spatial_nam)
                
                logger.info(f"  Added {len(spatial_seq)} spatial-temporal sequences")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not process spatial data: {e}")
        
        logger.info(f"🎯 Total comprehensive dataset: {len(all_sequences)} sequences")
        
        # Normalize sequence lengths and feature dimensions
        if len(all_sequences) > 0:
            # Find the minimum feature dimension across all sequences
            min_features = min(seq.shape[1] for seq in all_sequences)
            logger.info(f"🔧 Standardizing to {min_features} features")
            
            normalized_sequences = []
            for seq in all_sequences:
                # Truncate sequence length
                if len(seq) > max_sequence_length:
                    seq = seq[:max_sequence_length]
                
                # Standardize feature dimension
                if seq.shape[1] > min_features:
                    seq = seq[:, :min_features]
                
                normalized_sequences.append(seq)
            
            all_sequences = normalized_sequences
        
        # Remap labels to be consecutive starting from 0
        unique_labels = sorted(set(all_labels))
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        all_labels = [label_mapping[label] for label in all_labels]
        
        logger.info(f"🔧 Remapped {len(unique_labels)} unique labels to consecutive indices")
        logger.info(f"  Label mapping: {label_mapping}")
        
        return all_sequences, all_labels, all_names

class ComprehensiveCardiacDataset(Dataset):
    """Dataset for comprehensive cardiac data"""
    
    def __init__(self, sequences, labels, names, max_length=100, normalize=True):
        self.sequences = sequences
        self.labels = labels
        self.names = names
        self.max_length = max_length
        
        if normalize:
            self._normalize_sequences()
    
    def _normalize_sequences(self):
        """Normalize sequences"""
        logger.info("🔧 Normalizing sequences...")
        for i, seq in enumerate(self.sequences):
            seq_array = np.array(seq, dtype=np.float32)
            if seq_array.size > 0:
                # Z-score normalization
                mean = np.mean(seq_array, axis=0, keepdims=True)
                std = np.std(seq_array, axis=0, keepdims=True)
                std = np.where(std == 0, 1, std)
                self.sequences[i] = (seq_array - mean) / std
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = np.array(self.sequences[idx], dtype=np.float32)
        label = self.labels[idx]  # Use actual label, don't map to 5 classes
        name = self.names[idx]
        
        # Ensure sequence has consistent shape
        seq_len, n_features = sequence.shape
        
        # Pad or truncate sequence length
        if seq_len < self.max_length:
            padding = np.zeros((self.max_length - seq_len, n_features), dtype=np.float32)
            sequence = np.vstack([sequence, padding])
            actual_length = seq_len
        else:
            sequence = sequence[:self.max_length]
            actual_length = self.max_length
        
        # Ensure all sequences have the same number of features
        if hasattr(self, '_expected_features'):
            if n_features != self._expected_features:
                if n_features < self._expected_features:
                    # Pad features
                    feature_padding = np.zeros((sequence.shape[0], self._expected_features - n_features), dtype=np.float32)
                    sequence = np.hstack([sequence, feature_padding])
                else:
                    # Truncate features
                    sequence = sequence[:, :self._expected_features]
        else:
            # Set expected features from first sequence
            self._expected_features = n_features
        
        return {
            'sequence': torch.FloatTensor(sequence),
            'label': torch.LongTensor([label]),
            'length': torch.LongTensor([actual_length]),
            'name': name
        }

def main():
    """Test comprehensive data loading"""
    processor = ComprehensiveCardiacDataProcessor("data")
    
    # Create comprehensive dataset
    sequences, labels, names = processor.create_comprehensive_dataset(
        temporal_sequences_per_type=500,  # Reduce for testing
        spatial_sequences=200
    )
    
    if len(sequences) > 0:
        dataset = ComprehensiveCardiacDataset(sequences, labels, names)
        
        print(f"\n🎯 Comprehensive Cardiac Dataset:")
        print(f"  Total sequences: {len(dataset)}")
        
        # Test data loading
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        batch = next(iter(dataloader))
        
        print(f"  Batch sequence shape: {batch['sequence'].shape}")
        print(f"  Batch labels: {batch['label'].flatten()}")
        print(f"  Batch lengths: {batch['length'].flatten()}")
        
        print("\n✅ Comprehensive cardiac dataset ready for training!")
    else:
        print("❌ No data could be loaded")

if __name__ == "__main__":
    main()

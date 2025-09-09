"""
Advanced Temporal Data Loader for Cardiomyocyte Differentiation
Optimized for large-scale training with efficient memory usage
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import gzip
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class CardiomyocyteTemporalDataset(Dataset):
    """
    Dataset class for temporal cardiomyocyte differentiation data.
    Handles GSE175634 temporal scRNA-seq data with 7 time points.
    """
    
    def __init__(
        self,
        data_path: str,
        seq_length: int = 7,
        min_cells_per_sequence: int = 50,
        target_genes: Optional[List[str]] = None,
        normalize: bool = True,
        log_transform: bool = True,
        filter_cells: bool = True,
        filter_genes: bool = True,
        highly_variable_genes: bool = True,
        n_top_genes: int = 3000,
        cache_data: bool = True
    ):
        """
        Initialize the temporal dataset.
        
        Args:
            data_path: Path to the GSE175634 data directory
            seq_length: Length of temporal sequences (default 7 for all timepoints)
            min_cells_per_sequence: Minimum cells required per timepoint
            target_genes: Specific genes to focus on (if None, use highly variable)
            normalize: Whether to normalize gene expression
            log_transform: Whether to apply log transformation
            filter_cells: Whether to filter low-quality cells
            filter_genes: Whether to filter low-expression genes
            highly_variable_genes: Whether to select highly variable genes
            n_top_genes: Number of top genes to select
            cache_data: Whether to cache processed data
        """
        self.data_path = Path(data_path)
        self.seq_length = seq_length
        self.min_cells_per_sequence = min_cells_per_sequence
        self.target_genes = target_genes
        self.normalize = normalize
        self.log_transform = log_transform
        self.filter_cells = filter_cells
        self.filter_genes = filter_genes
        self.highly_variable_genes = highly_variable_genes
        self.n_top_genes = n_top_genes
        self.cache_data = cache_data
        
        # Time point mapping
        self.timepoints = ['day0', 'day1', 'day3', 'day5', 'day7', 'day11', 'day15']
        self.timepoint_to_idx = {tp: idx for idx, tp in enumerate(self.timepoints)}
        
        # Cell type mapping
        self.cell_types = ['IPSC', 'MES', 'PROG', 'CMES', 'CF', 'CM', 'UNK']
        self.cell_type_to_idx = {ct: idx for idx, ct in enumerate(self.cell_types)}
        
        # Load and process data
        self._load_data()
        self._process_data()
        self._create_sequences()
        
    def _load_data(self):
        """Load the GSE175634 temporal data."""
        print("Loading temporal cardiomyocyte data...")
        
        # Load cell metadata
        metadata_path = self.data_path / "GSE175634_cell_metadata.tsv.gz"
        with gzip.open(metadata_path, 'rt') as f:
            self.metadata = pd.read_csv(f, sep='\t')
        
        # Load gene expression matrix
        matrix_path = self.data_path / "GSE175634_cell_counts_sctransform.mtx.gz"
        genes_path = self.data_path / "GSE175634_gene_indices_counts_sctransform.tsv.gz"
        cells_path = self.data_path / "GSE175634_cell_indices.tsv.gz"
        
        # Load gene names
        with gzip.open(genes_path, 'rt') as f:
            self.gene_names = pd.read_csv(f, sep='\t', header=0)
        
        # Load cell indices
        with gzip.open(cells_path, 'rt') as f:
            self.cell_indices = pd.read_csv(f, sep='\t', header=0)
        
        # Create AnnData object
        from scipy.io import mmread
        import tempfile
        import os
        
        # Extract matrix to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mtx') as tmp:
            with gzip.open(matrix_path, 'rb') as gz_file:
                tmp.write(gz_file.read())
            tmp_path = tmp.name
        
        try:
            # Load sparse matrix
            X = mmread(tmp_path).T.tocsr()  # Transpose to cells x genes
            
            # Create AnnData object
            self.adata = ad.AnnData(
                X=X,
                obs=self.metadata.set_index('cell'),
                var=self.gene_names.set_index('gene_name')
            )
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
        
        print(f"Loaded data: {self.adata.n_obs} cells × {self.adata.n_vars} genes")
        
    def _process_data(self):
        """Process and filter the loaded data."""
        print("Processing temporal data...")
        
        # Filter cells and genes if requested
        if self.filter_cells:
            # Filter cells with too few genes
            sc.pp.filter_cells(self.adata, min_genes=200)
            
            # Filter cells with too many genes (potential doublets)
            sc.pp.filter_cells(self.adata, max_genes=5000)
            
        if self.filter_genes:
            # Filter genes expressed in too few cells
            sc.pp.filter_genes(self.adata, min_cells=10)
        
        # Calculate QC metrics
        self.adata.var['mt'] = self.adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(self.adata, percent_top=None, log1p=False, inplace=True)
        
        # Filter cells with high mitochondrial gene expression (if the metric exists)
        if self.filter_cells and 'pct_counts_mt' in self.adata.obs.columns:
            self.adata = self.adata[self.adata.obs.pct_counts_mt < 20, :].copy()
        elif self.filter_cells:
            print("Warning: No mitochondrial gene percentage available, skipping mitochondrial filtering")
        
        # Store raw data
        self.adata.raw = self.adata
        
        # Normalize if requested
        if self.normalize:
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            
        if self.log_transform:
            sc.pp.log1p(self.adata)
        
        # Select highly variable genes if requested
        if self.highly_variable_genes:
            sc.pp.highly_variable_genes(
                self.adata, 
                n_top_genes=self.n_top_genes,
                subset=True
            )
        elif self.target_genes:
            # Filter to target genes
            target_mask = self.adata.var_names.isin(self.target_genes)
            self.adata = self.adata[:, target_mask].copy()
        
        print(f"Processed data: {self.adata.n_obs} cells × {self.adata.n_vars} genes")
        
        # Convert to dense for easier handling
        if hasattr(self.adata.X, 'toarray'):
            self.adata.X = self.adata.X.toarray()
            
    def _create_sequences(self):
        """Create temporal sequences for training."""
        print("Creating temporal sequences...")
        
        # Group by individual and create sequences
        self.sequences = []
        self.targets = []
        
        # Get unique individuals
        individuals = self.adata.obs['individual'].unique()
        
        for individual in individuals:
            # Get cells for this individual
            individual_mask = self.adata.obs['individual'] == individual
            individual_data = self.adata[individual_mask].copy()
            
            # Create sequence for this individual
            sequence_data = []
            sequence_targets = []
            
            for timepoint in self.timepoints:
                # Get cells for this timepoint
                tp_mask = individual_data.obs['diffday'] == timepoint
                tp_cells = individual_data[tp_mask]
                
                if len(tp_cells) < self.min_cells_per_sequence:
                    print(f"Skipping {individual}-{timepoint}: only {len(tp_cells)} cells")
                    continue
                
                # Average gene expression across cells for this timepoint
                mean_expression = np.mean(tp_cells.X, axis=0)
                sequence_data.append(mean_expression)
                
                # Calculate differentiation efficiency
                # Based on proportion of CM cells and CM marker expression
                cm_mask = tp_cells.obs['type'] == 'CM'
                cm_proportion = cm_mask.sum() / len(tp_cells)
                
                # Get CM marker genes (if available)
                cm_markers = ['TNNT2', 'MYH6', 'MYH7', 'ACTC1', 'NKX2-5', 'GATA4']
                available_markers = [g for g in cm_markers if g in tp_cells.var_names]
                
                if available_markers:
                    marker_expression = np.mean(tp_cells[:, available_markers].X, axis=1)
                    mean_marker_expr = np.mean(marker_expression)
                    # Combine proportion and marker expression
                    diff_efficiency = 0.7 * cm_proportion + 0.3 * mean_marker_expr
                else:
                    diff_efficiency = cm_proportion
                
                sequence_targets.append({
                    'differentiation_efficiency': diff_efficiency,
                    'cell_type_distribution': self._get_cell_type_distribution(tp_cells),
                    'timepoint': timepoint,
                    'timepoint_idx': self.timepoint_to_idx[timepoint]
                })
            
            # Only add if we have a complete or near-complete sequence
            if len(sequence_data) >= 5:  # At least 5 timepoints
                # Pad sequence if needed
                while len(sequence_data) < self.seq_length:
                    sequence_data.append(np.zeros_like(sequence_data[-1]))
                    sequence_targets.append({
                        'differentiation_efficiency': 0.0,
                        'cell_type_distribution': np.zeros(len(self.cell_types)),
                        'timepoint': 'padding',
                        'timepoint_idx': -1
                    })
                
                self.sequences.append(np.array(sequence_data))
                self.targets.append(sequence_targets)
        
        print(f"Created {len(self.sequences)} temporal sequences")
        
    def _get_cell_type_distribution(self, cells):
        """Get cell type distribution for a set of cells."""
        distribution = np.zeros(len(self.cell_types))
        for i, cell_type in enumerate(self.cell_types):
            count = (cells.obs['type'] == cell_type).sum()
            distribution[i] = count / len(cells)
        return distribution
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Get a single temporal sequence."""
        sequence = torch.FloatTensor(self.sequences[idx])
        targets = self.targets[idx]
        
        # Get final differentiation efficiency (last non-padding timepoint)
        final_diff_eff = 0.0
        final_cell_type = 0
        
        for target in reversed(targets):
            if target['timepoint'] != 'padding':
                final_diff_eff = target['differentiation_efficiency']
                # Get dominant cell type
                final_cell_type = np.argmax(target['cell_type_distribution'])
                break
        
        return {
            'sequence': sequence,
            'differentiation_efficiency': torch.FloatTensor([final_diff_eff]),
            'cell_type': torch.LongTensor([final_cell_type]),
            'sequence_length': torch.LongTensor([len([t for t in targets if t['timepoint'] != 'padding'])]),
            'targets_sequence': targets
        }


def create_temporal_dataloaders(
    data_path: str,
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 4,
    pin_memory: bool = True,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for temporal data.
    
    Args:
        data_path: Path to the temporal data
        batch_size: Batch size for training
        train_split: Proportion for training
        val_split: Proportion for validation
        test_split: Proportion for testing
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        **dataset_kwargs: Additional arguments for dataset creation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create dataset
    dataset = CardiomyocyteTemporalDataset(data_path, **dataset_kwargs)
    
    # Calculate split sizes
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_temporal_batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_temporal_batch
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_temporal_batch
    )
    
    print(f"Created dataloaders:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader


def collate_temporal_batch(batch):
    """Custom collate function for temporal sequences with variable lengths."""
    sequences = torch.stack([item['sequence'] for item in batch])
    diff_efficiencies = torch.stack([item['differentiation_efficiency'] for item in batch])
    cell_types = torch.stack([item['cell_type'] for item in batch])
    lengths = torch.stack([item['sequence_length'] for item in batch])
    
    return {
        'sequences': sequences,
        'differentiation_efficiency': diff_efficiencies,
        'cell_type': cell_types,
        'lengths': lengths.squeeze()
    }


if __name__ == "__main__":
    # Test data loading
    data_path = "path/to/GSE175634/data"
    
    try:
        train_loader, val_loader, test_loader = create_temporal_dataloaders(
            data_path=data_path,
            batch_size=16,
            n_top_genes=2000
        )
        
        # Test a batch
        for batch in train_loader:
            print(f"Batch sequences shape: {batch['sequences'].shape}")
            print(f"Batch targets shape: {batch['differentiation_efficiency'].shape}")
            break
            
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure the GSE175634 data is available in the specified path")

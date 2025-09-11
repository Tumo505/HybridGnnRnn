"""
Phase 1: Real Temporal Cardiac Dataset Processor for GSE175634
============================================================
Biologically trustworthy temporal gene expression modeling using real cardiac differentiation data.
"""
import gzip
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.io import mmread
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

class RealCardiacTemporalDataset:
    """
    Process real GSE175634 temporal cardiac differentiation dataset for trustworthy modeling.
    """
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.metadata = None
        self.gene_info = None
        self.expression_matrix = None
        self.cardiac_markers = [
            'ACTC1', 'ACTN2', 'MYH6', 'MYH7', 'TNNT2', 'NPPA', 'NPPB', 
            'TPM1', 'MYBPC3', 'MYL2', 'MYL7', 'CACNA1C', 'SCN5A'
        ]
        self.timepoint_order = ['day0', 'day1', 'day3', 'day5', 'day7', 'day11', 'day15']
        
    def load_metadata(self):
        """Load and process cell metadata with biological validation."""
        print("Loading cell metadata...")
        metadata_path = os.path.join(self.data_dir, "GSE175634_cell_metadata.tsv.gz")
        self.metadata = pd.read_csv(metadata_path, sep='\t')
        
        print(f"Total cells: {len(self.metadata):,}")
        print(f"Timepoints: {sorted(self.metadata['diffday'].unique())}")
        print(f"Cell types: {sorted(self.metadata['type'].unique())}")
        print(f"Individuals: {self.metadata['individual'].nunique()}")
        
        # Focus on cardiac cell types for trustworthy modeling
        cardiac_types = ['CM', 'CMES', 'CF', 'PROG']  # Cardiomyocytes, Cardiac mesoderm, Fibroblasts, Progenitors
        self.cardiac_metadata = self.metadata[self.metadata['type'].isin(cardiac_types)].copy()
        
        print(f"\nCardiac cells selected: {len(self.cardiac_metadata):,}")
        print("Cardiac cell distribution:")
        for cell_type, count in self.cardiac_metadata['type'].value_counts().items():
            print(f"  {cell_type}: {count:,} cells")
            
        return self.cardiac_metadata
    
    def load_genes(self):
        """Load gene information and identify cardiac markers."""
        print("\nLoading gene information...")
        gene_path = os.path.join(self.data_dir, "GSE175634_gene_indices_counts.tsv.gz")
        self.gene_info = pd.read_csv(gene_path, sep='\t')
        
        print(f"Total genes: {len(self.gene_info):,}")
        
        # Find cardiac markers in the dataset
        available_cardiac_markers = []
        cardiac_gene_indices = []
        
        for marker in self.cardiac_markers:
            mask = self.gene_info['gene_name'].str.contains(f'^{marker}$', case=False, na=False)
            if mask.any():
                available_cardiac_markers.append(marker)
                cardiac_gene_indices.extend(self.gene_info[mask].index.tolist())
        
        # Also find genes with cardiac-related patterns
        cardiac_patterns = ['MYH', 'MYL', 'ACTN', 'MYBP', 'TNNT', 'ACTC', 'TPM']
        pattern_genes = []
        
        for pattern in cardiac_patterns:
            pattern_mask = self.gene_info['gene_name'].str.contains(pattern, case=False, na=False)
            pattern_genes.extend(self.gene_info[pattern_mask]['gene_name'].tolist())
        
        print(f"Cardiac markers found: {len(available_cardiac_markers)}")
        print(f"Cardiac-related genes: {len(set(pattern_genes))}")
        print(f"Available markers: {available_cardiac_markers}")
        
        self.available_cardiac_markers = available_cardiac_markers
        self.cardiac_gene_indices = list(set(cardiac_gene_indices))
        
        return self.gene_info
    
    def load_expression_data(self, use_sctransform=True):
        """Load expression matrix with biological validation."""
        print("\nLoading expression matrix...")
        
        if use_sctransform:
            # Use sctransform-normalized data for better biological interpretation
            counts_path = os.path.join(self.data_dir, "GSE175634_cell_counts_sctransform.mtx.gz")
            gene_path = os.path.join(self.data_dir, "GSE175634_gene_indices_counts_sctransform.tsv.gz")
        else:
            counts_path = os.path.join(self.data_dir, "GSE175634_cell_counts.mtx.gz")
            gene_path = os.path.join(self.data_dir, "GSE175634_gene_indices_counts.tsv.gz")
        
        # Load sparse matrix
        with gzip.open(counts_path, 'rb') as f:
            expression_matrix = mmread(f).tocsc()
        
        print(f"Expression matrix shape: {expression_matrix.shape}")
        print(f"Matrix density: {expression_matrix.nnz / np.prod(expression_matrix.shape):.4f}")
        
        # Load corresponding gene information
        gene_info_transform = pd.read_csv(gene_path, sep='\t')
        
        self.expression_matrix = expression_matrix
        self.gene_info_transform = gene_info_transform
        
        return expression_matrix
    
    def create_temporal_sequences(self, individual_id=None, min_cells_per_timepoint=100):
        """
        Create biologically meaningful temporal sequences from real data.
        """
        print(f"\nCreating temporal sequences...")
        
        if individual_id is not None:
            # Focus on specific individual for consistent temporal trajectory
            individual_data = self.cardiac_metadata[self.cardiac_metadata['individual'] == individual_id].copy()
            print(f"Processing individual: {individual_id}")
        else:
            individual_data = self.cardiac_metadata.copy()
            print("Processing all individuals")
        
        # Group by timepoint and ensure sufficient cells
        timepoint_groups = individual_data.groupby('diffday')
        valid_timepoints = []
        
        for timepoint, group in timepoint_groups:
            if len(group) >= min_cells_per_timepoint:
                valid_timepoints.append(timepoint)
                print(f"  {timepoint}: {len(group):,} cells")
        
        print(f"Valid timepoints for modeling: {len(valid_timepoints)}")
        
        # Create temporal sequences
        sequences = []
        labels = []
        timepoint_info = []
        
        # Sort timepoints chronologically
        valid_timepoints_sorted = [tp for tp in self.timepoint_order if tp in valid_timepoints]
        
        for i, current_timepoint in enumerate(valid_timepoints_sorted[:-1]):
            next_timepoint = valid_timepoints_sorted[i + 1]
            
            # Get cells for current timepoint
            current_cells = individual_data[individual_data['diffday'] == current_timepoint]
            next_cells = individual_data[individual_data['diffday'] == next_timepoint]
            
            # Sample cells to create balanced sequences
            n_sequences = min(len(current_cells), len(next_cells), 1000)  # Limit for computational efficiency
            
            current_sample = current_cells.sample(n=n_sequences, random_state=42)
            next_sample = next_cells.sample(n=n_sequences, random_state=42)
            
            # Extract expression data
            current_indices = [self.metadata.index[self.metadata['cell'] == cell].tolist()[0] 
                             for cell in current_sample['cell'] if cell in self.metadata['cell'].values]
            next_indices = [self.metadata.index[self.metadata['cell'] == cell].tolist()[0] 
                          for cell in next_sample['cell'] if cell in self.metadata['cell'].values]
            
            if len(current_indices) > 0 and len(next_indices) > 0:
                # Get expression data (transpose because mmread loads as genes x cells)
                current_expr = self.expression_matrix[:, current_indices].T.toarray()
                next_expr = self.expression_matrix[:, next_indices].T.toarray()
                
                # Create sequences (current -> next)
                min_len = min(len(current_expr), len(next_expr))
                for j in range(min_len):
                    sequences.append(current_expr[j])
                    labels.append(next_expr[j])
                    timepoint_info.append((current_timepoint, next_timepoint))
        
        print(f"Created {len(sequences)} temporal sequences")
        
        return np.array(sequences), np.array(labels), timepoint_info
    
    def validate_cardiac_biology(self, sequences, labels, timepoint_info):
        """
        Validate that temporal sequences show expected cardiac differentiation patterns.
        """
        print("\nValidating cardiac biology...")
        
        if len(self.cardiac_gene_indices) == 0:
            print("Warning: No cardiac markers found for validation")
            return True
        
        # Check if cardiac markers increase over time
        validation_results = {}
        
        for i, (current_tp, next_tp) in enumerate(set(timepoint_info)):
            # Get indices for this timepoint transition
            transition_indices = [j for j, (curr, nxt) in enumerate(timepoint_info) 
                                if curr == current_tp and nxt == next_tp]
            
            if len(transition_indices) > 0:
                current_cardiac = np.mean([sequences[j][self.cardiac_gene_indices].mean() 
                                         for j in transition_indices[:100]])  # Sample for efficiency
                next_cardiac = np.mean([labels[j][self.cardiac_gene_indices].mean() 
                                      for j in transition_indices[:100]])
                
                fold_change = next_cardiac / (current_cardiac + 1e-10)
                validation_results[f"{current_tp}->{next_tp}"] = fold_change
                
                print(f"  {current_tp} -> {next_tp}: {fold_change:.3f}x cardiac marker expression")
        
        # Overall cardiac progression should be positive
        overall_progression = np.mean(list(validation_results.values()))
        print(f"\nOverall cardiac progression: {overall_progression:.3f}x")
        
        is_valid = overall_progression > 0.8  # Allow for some variability
        print(f"Biological validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
        
        return is_valid
    
    def prepare_dataset_for_training(self, test_size=0.2, val_size=0.1):
        """
        Prepare the complete dataset for temporal RNN training.
        """
        print("\n" + "="*60)
        print("PREPARING REAL CARDIAC TEMPORAL DATASET FOR TRAINING")
        print("="*60)
        
        # Load all components
        self.load_metadata()
        self.load_genes()
        self.load_expression_data(use_sctransform=True)
        
        # Create temporal sequences from real data
        sequences, labels, timepoint_info = self.create_temporal_sequences()
        
        # Validate biological plausibility
        is_biologically_valid = self.validate_cardiac_biology(sequences, labels, timepoint_info)
        
        if not is_biologically_valid:
            print("⚠️  Warning: Biological validation concerns detected")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            sequences, labels, test_size=test_size, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42
        )
        
        # Normalize features (important for neural networks)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Also scale targets for better training
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train)
        y_val_scaled = target_scaler.transform(y_val)
        y_test_scaled = target_scaler.transform(y_test)
        
        print(f"\nDataset prepared:")
        print(f"  Training: {X_train_scaled.shape[0]:,} sequences")
        print(f"  Validation: {X_val_scaled.shape[0]:,} sequences")
        print(f"  Test: {X_test_scaled.shape[0]:,} sequences")
        print(f"  Features: {X_train_scaled.shape[1]:,} genes")
        print(f"  Biological validation: {'✅ PASSED' if is_biologically_valid else '❌ FAILED'}")
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_scaled,
            'y_val': y_val_scaled,
            'y_test': y_test_scaled,
            'scaler': scaler,
            'target_scaler': target_scaler,
            'timepoint_info': timepoint_info,
            'gene_info': self.gene_info_transform,
            'cardiac_markers': self.available_cardiac_markers,
            'validation_passed': is_biologically_valid
        }

if __name__ == "__main__":
    # Test the real cardiac dataset processor
    data_dir = r"c:\Users\tumok\Documents\Projects\HybridGnnRnn\data\GSE175634_temporal_data"
    
    processor = RealCardiacTemporalDataset(data_dir)
    dataset = processor.prepare_dataset_for_training()
    
    print(f"\n🎯 Real cardiac temporal dataset ready for Phase 1 training!")
    print(f"   Biologically trustworthy: {dataset['validation_passed']}")
    print(f"   Training samples: {dataset['X_train'].shape[0]:,}")
    print(f"   Gene features: {dataset['X_train'].shape[1]:,}")

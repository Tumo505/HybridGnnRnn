#!/usr/bin/env python3
"""
Smart Temporal Dataset Combination for Enhanced RNN Training
Intelligently combine multiple temporal cardiac datasets for improved performance
"""

import scanpy as sc
import pandas as pd
import numpy as np
import torch
import h5py
from pathlib import Path
import logging
import warnings
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import anndata as ad
from scipy import sparse
import gc

warnings.filterwarnings('ignore')
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')

# Setup GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class TemporalDatasetCombiner:
    """Smart combination of temporal cardiac datasets"""
    
    def __init__(self):
        self.datasets = {}
        self.combined_data = None
        self.common_genes = None
        
        # Temporal mapping strategies
        self.time_mappings = {
            'GSE175634': self._map_gse175634_time,
            'GSE130731': self._map_gse130731_time,
            'GSE202398': self._map_gse202398_time
        }
        
        # Cardiac marker genes for validation
        self.cardiac_markers = [
            'TNNT2', 'MYH6', 'MYH7', 'ACTC1', 'NKX2-5', 'GATA4', 'GATA6',
            'TBX5', 'MEF2C', 'HAND1', 'HAND2', 'ISL1', 'MYL2', 'MYL7',
            'NPPA', 'NPPB', 'PLN', 'RYR2', 'SCN5A', 'CACNA1C'
        ]
        
    def load_gse175634(self):
        """Load GSE175634 temporal data"""
        print("📊 Loading GSE175634...")
        
        try:
            # Try loading from temporal data first
            temporal_path = Path("data/GSE175634_temporal_data")
            if temporal_path.exists():
                h5_files = list(temporal_path.glob("*.h5"))
                if h5_files:
                    adata = sc.read_10x_h5(h5_files[0])
                    adata.var_names_make_unique()
                    # Add basic metadata
                    adata.obs['dataset'] = 'GSE175634'
                    adata.obs['timepoint'] = 'mixed'
                    print(f"  Loaded from temporal_data: {adata.shape}")
                    return adata
            
            # Try extracted datasets
            data_path = Path("data/extracted_datasets/GSE175634")
            if data_path.exists():
                h5ad_files = list(data_path.glob("*.h5ad"))
                if h5ad_files:
                    adata = sc.read_h5ad(h5ad_files[0])
                    adata.obs['dataset'] = 'GSE175634'
                    if 'timepoint' not in adata.obs.columns:
                        adata.obs['timepoint'] = 'mixed'
                    print(f"  Loaded from extracted: {adata.shape}")
                    return adata
                    
                h5_files = list(data_path.glob("*.h5"))
                if h5_files:
                    adata = sc.read_10x_h5(h5_files[0])
                    adata.var_names_make_unique()
                    adata.obs['dataset'] = 'GSE175634'
                    adata.obs['timepoint'] = 'mixed'
                    print(f"  Loaded from h5: {adata.shape}")
                    return adata
                    
        except Exception as e:
            print(f"  Error loading GSE175634: {e}")
            
        return None
    
    def load_gse130731(self):
        """Load GSE130731 iPSC differentiation data"""
        print("📊 Loading GSE130731...")
        
        try:
            data_path = Path("data/extracted_datasets/GSE130731")
            
            # Try h5ad first
            h5ad_files = list(data_path.glob("*.h5ad"))
            if h5ad_files:
                adata = sc.read_h5ad(h5ad_files[0])
                adata.obs['dataset'] = 'GSE130731'
                if 'timepoint' not in adata.obs.columns:
                    adata.obs['timepoint'] = 'mixed'
                print(f"  Loaded: {adata.shape}")
                return adata
            
            # Try matrix files
            mtx_files = list(data_path.rglob("*matrix.mtx*"))
            if mtx_files:
                adata = sc.read_10x_mtx(mtx_files[0].parent)
                adata.var_names_make_unique()
                adata.obs['dataset'] = 'GSE130731'
                adata.obs['timepoint'] = 'mixed'
                print(f"  Loaded from mtx: {adata.shape}")
                return adata
                
        except Exception as e:
            print(f"  Error loading GSE130731: {e}")
            
        return None
    
    def load_gse202398(self):
        """Load GSE202398 cardiac data"""
        print("📊 Loading GSE202398...")
        
        try:
            data_path = Path("data/extracted_datasets/GSE202398")
            h5_files = [f for f in data_path.glob("*.h5") if "filtered" in f.name]
            
            if h5_files:
                adata = sc.read_10x_h5(h5_files[0])
                adata.var_names_make_unique()
                adata.obs['dataset'] = 'GSE202398'
                adata.obs['timepoint'] = 'static'
                print(f"  Loaded: {adata.shape}")
                return adata
                
        except Exception as e:
            print(f"  Error loading GSE202398: {e}")
            
        return None
    
    def _map_gse175634_time(self, adata):
        """Map GSE175634 temporal information"""
        # Try to infer timepoints from sample names or other metadata
        if 'sample' in adata.obs.columns:
            adata.obs['time_hours'] = 0  # Default
        else:
            adata.obs['time_hours'] = 0
        return adata
    
    def _map_gse130731_time(self, adata):
        """Map GSE130731 temporal information"""
        # iPSC differentiation - try to infer from sample names
        if 'sample' in adata.obs.columns:
            # Look for day patterns in sample names
            time_map = {'day0': 0, 'day1': 24, 'day2': 48, 'day3': 72, 'day7': 168}
            adata.obs['time_hours'] = 0  # Default
            for time_str, hours in time_map.items():
                mask = adata.obs['sample'].str.contains(time_str, case=False, na=False)
                adata.obs.loc[mask, 'time_hours'] = hours
        else:
            adata.obs['time_hours'] = 0
        return adata
    
    def _map_gse202398_time(self, adata):
        """Map GSE202398 temporal information"""
        # Static dataset
        adata.obs['time_hours'] = 0
        return adata
    
    def load_all_datasets(self):
        """Load all available datasets"""
        print("🔄 Loading all temporal datasets...")
        
        loaders = {
            'GSE175634': self.load_gse175634,
            'GSE130731': self.load_gse130731,
            'GSE202398': self.load_gse202398
        }
        
        for name, loader in loaders.items():
            try:
                adata = loader()
                if adata is not None:
                    # Apply temporal mapping
                    if name in self.time_mappings:
                        adata = self.time_mappings[name](adata)
                    
                    self.datasets[name] = adata
                    print(f"✅ {name}: {adata.shape[0]:,} cells, {adata.shape[1]:,} genes")
                else:
                    print(f"❌ Failed to load {name}")
            except Exception as e:
                print(f"❌ Error loading {name}: {e}")
        
        if not self.datasets:
            raise ValueError("No datasets were successfully loaded!")
        
        print(f"\n📈 Total datasets loaded: {len(self.datasets)}")
        return self.datasets
    
    def find_common_genes(self):
        """Find genes common across all datasets"""
        print("🧬 Finding common genes across datasets...")
        
        if not self.datasets:
            raise ValueError("No datasets loaded!")
        
        # Get gene sets for each dataset
        gene_sets = {}
        for name, adata in self.datasets.items():
            genes = set(adata.var_names)
            gene_sets[name] = genes
            print(f"  {name}: {len(genes):,} genes")
        
        # Find intersection
        self.common_genes = set.intersection(*gene_sets.values())
        print(f"  Common genes: {len(self.common_genes):,}")
        
        # Check cardiac marker coverage
        available_markers = [g for g in self.cardiac_markers if g in self.common_genes]
        print(f"  Cardiac markers in common: {len(available_markers)}/{len(self.cardiac_markers)}")
        print(f"  Available markers: {', '.join(available_markers)}")
        
        return list(self.common_genes)
    
    def preprocess_dataset(self, adata, dataset_name):
        """Standardized preprocessing for each dataset"""
        print(f"⚙️ Preprocessing {dataset_name}...")
        
        # Store raw data
        adata.raw = adata.copy()
        
        # Basic QC
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
        
        # Filter cells and genes
        print(f"  Initial shape: {adata.shape}")
        sc.pp.filter_genes(adata, min_cells=10)
        sc.pp.filter_cells(adata, min_genes=200)
        print(f"  After filtering: {adata.shape}")
        
        # Remove cells with too many genes (doublets)
        if 'n_genes_by_counts' in adata.obs.columns:
            adata = adata[adata.obs.n_genes_by_counts < 7000, :]
            print(f"  After doublet removal: {adata.shape}")
        
        # Remove high MT cells
        if 'pct_counts_mt' in adata.obs.columns:
            adata = adata[adata.obs.pct_counts_mt < 25, :]
            print(f"  After MT filtering: {adata.shape}")
        
        return adata
    
    def harmonize_datasets(self):
        """Harmonize datasets to common gene set"""
        print("🔗 Harmonizing datasets...")
        
        if not self.common_genes:
            self.find_common_genes()
        
        harmonized = {}
        total_cells = 0
        
        for name, adata in self.datasets.items():
            print(f"  Processing {name}...")
            
            # Preprocess
            adata_processed = self.preprocess_dataset(adata.copy(), name)
            
            # Subset to common genes
            common_mask = adata_processed.var_names.isin(self.common_genes)
            adata_subset = adata_processed[:, common_mask].copy()
            
            # Reorder genes to match common gene order
            gene_order = sorted(self.common_genes)
            adata_subset = adata_subset[:, gene_order].copy()
            
            print(f"    Final shape: {adata_subset.shape}")
            
            harmonized[name] = adata_subset
            total_cells += adata_subset.shape[0]
        
        print(f"✅ Harmonization complete - Total cells: {total_cells:,}")
        return harmonized
    
    def combine_datasets(self, harmonized_datasets):
        """Combine harmonized datasets"""
        print("🔀 Combining datasets...")
        
        # Collect all data
        all_X = []
        all_obs = []
        
        for name, adata in harmonized_datasets.items():
            print(f"  Adding {name}: {adata.shape[0]:,} cells")
            
            # Convert sparse to dense if needed
            if sparse.issparse(adata.X):
                X_dense = adata.X.toarray()
            else:
                X_dense = adata.X
            
            all_X.append(X_dense)
            
            # Add batch information
            obs_df = adata.obs.copy()
            obs_df['batch'] = name
            obs_df['batch_id'] = len(all_obs)  # Numeric batch ID
            all_obs.append(obs_df)
        
        # Combine
        combined_X = np.vstack(all_X)
        combined_obs = pd.concat(all_obs, ignore_index=True)
        
        # Get variable info from first dataset
        first_adata = list(harmonized_datasets.values())[0]
        var_df = first_adata.var.copy()
        
        # Create combined AnnData
        self.combined_data = ad.AnnData(
            X=combined_X,
            obs=combined_obs,
            var=var_df
        )
        
        print(f"✅ Combined dataset: {self.combined_data.shape[0]:,} cells, {self.combined_data.shape[1]:,} genes")
        
        # Add batch information summary
        batch_counts = self.combined_data.obs['batch'].value_counts()
        print(f"  Batch distribution:")
        for batch, count in batch_counts.items():
            print(f"    {batch}: {count:,} cells ({count/len(self.combined_data)*100:.1f}%)")
        
        return self.combined_data
    
    def normalize_and_scale(self):
        """Normalize and scale the combined dataset"""
        print("📊 Normalizing and scaling...")
        
        if self.combined_data is None:
            raise ValueError("No combined data available!")
        
        # Store raw
        self.combined_data.raw = self.combined_data.copy()
        
        # Normalize total counts
        sc.pp.normalize_total(self.combined_data, target_sum=1e4)
        
        # Log transform
        sc.pp.log1p(self.combined_data)
        
        # Scale
        sc.pp.scale(self.combined_data, max_value=10)
        
        print("✅ Normalization complete")
        
    def batch_correction(self):
        """Apply batch correction using Scanpy's Harmony"""
        print("🔧 Applying batch correction...")
        
        try:
            # Try using Harmony for batch correction
            sc.external.pp.harmony_integrate(
                self.combined_data, 
                key='batch',
                basis='X_pca',
                adjusted_basis='X_pca_harmony'
            )
            print("✅ Harmony batch correction applied")
        except:
            print("⚠️ Harmony not available, using simple scaling")
            # Alternative: PCA-based batch correction
            sc.tl.pca(self.combined_data, n_comps=50)
    
    def create_temporal_sequences(self, sequence_length=10):
        """Create temporal sequences for RNN training"""
        print(f"🕐 Creating temporal sequences (length={sequence_length})...")
        
        sequences = []
        targets = []
        metadata = []
        
        # Group by batch and timepoint
        for batch in self.combined_data.obs['batch'].unique():
            batch_mask = self.combined_data.obs['batch'] == batch
            batch_data = self.combined_data[batch_mask]
            
            # Sort by time if available
            if 'time_hours' in batch_data.obs.columns:
                time_order = batch_data.obs['time_hours'].argsort()
                batch_data = batch_data[time_order]
            
            # Create sequences within batch
            n_cells = batch_data.shape[0]
            
            for i in range(n_cells - sequence_length):
                # Get sequence
                seq_data = batch_data.X[i:i+sequence_length]
                target = batch_data.X[i+sequence_length]  # Next timepoint
                
                sequences.append(seq_data)
                targets.append(target)
                
                # Store metadata
                meta = {
                    'batch': batch,
                    'start_idx': i,
                    'end_idx': i + sequence_length,
                    'start_time': batch_data.obs['time_hours'].iloc[i] if 'time_hours' in batch_data.obs.columns else 0,
                    'end_time': batch_data.obs['time_hours'].iloc[i+sequence_length] if 'time_hours' in batch_data.obs.columns else 0
                }
                metadata.append(meta)
        
        print(f"✅ Created {len(sequences):,} temporal sequences")
        
        return np.array(sequences), np.array(targets), pd.DataFrame(metadata)
    
    def save_combined_dataset(self, output_path="data/combined_temporal_dataset.h5ad"):
        """Save the combined dataset"""
        print(f"💾 Saving combined dataset to {output_path}...")
        
        if self.combined_data is None:
            raise ValueError("No combined data to save!")
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        self.combined_data.write_h5ad(output_path)
        
        print(f"✅ Dataset saved: {output_path}")
        
        # Save summary statistics
        summary = {
            'total_cells': self.combined_data.shape[0],
            'total_genes': self.combined_data.shape[1],
            'datasets_combined': list(self.datasets.keys()),
            'common_genes_count': len(self.common_genes),
            'cardiac_markers_available': len([g for g in self.cardiac_markers if g in self.common_genes]),
            'batch_distribution': self.combined_data.obs['batch'].value_counts().to_dict(),
            'creation_time': datetime.now().isoformat()
        }
        
        import json
        summary_path = str(output_path).replace('.h5ad', '_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Summary saved: {summary_path}")
        
        return output_path
    
    def run_combination(self):
        """Run the complete combination pipeline"""
        print("🚀 SMART TEMPORAL DATASET COMBINATION")
        print("=" * 60)
        
        try:
            # Load all datasets
            self.load_all_datasets()
            
            # Find common genes
            self.find_common_genes()
            
            # Harmonize datasets
            harmonized = self.harmonize_datasets()
            
            # Combine datasets
            self.combine_datasets(harmonized)
            
            # Normalize and scale
            self.normalize_and_scale()
            
            # Batch correction
            self.batch_correction()
            
            # Save combined dataset
            output_path = self.save_combined_dataset()
            
            # Create temporal sequences for RNN
            sequences, targets, metadata = self.create_temporal_sequences()
            
            # Save sequences for RNN training
            sequence_data = {
                'sequences': sequences,
                'targets': targets,
                'metadata': metadata,
                'gene_names': self.combined_data.var_names.tolist(),
                'common_genes': list(self.common_genes)
            }
            
            sequence_path = "data/temporal_sequences_for_rnn.npz"
            np.savez_compressed(sequence_path, **sequence_data)
            print(f"✅ Temporal sequences saved: {sequence_path}")
            
            # Final summary
            print("\n🎉 COMBINATION COMPLETE!")
            print("=" * 60)
            print(f"Combined dataset: {self.combined_data.shape[0]:,} cells × {self.combined_data.shape[1]:,} genes")
            print(f"Temporal sequences: {len(sequences):,} sequences")
            print(f"Common genes: {len(self.common_genes):,}")
            print(f"Cardiac markers: {len([g for g in self.cardiac_markers if g in self.common_genes])}/{len(self.cardiac_markers)}")
            
            # Dataset contributions
            print(f"\nDataset contributions:")
            for batch, count in self.combined_data.obs['batch'].value_counts().items():
                print(f"  {batch}: {count:,} cells ({count/len(self.combined_data)*100:.1f}%)")
            
            return {
                'combined_data': self.combined_data,
                'sequences': sequences,
                'targets': targets,
                'metadata': metadata,
                'output_path': output_path,
                'sequence_path': sequence_path
            }
            
        except Exception as e:
            print(f"❌ Error in combination pipeline: {e}")
            raise

def main():
    """Main execution"""
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create combiner and run
    combiner = TemporalDatasetCombiner()
    results = combiner.run_combination()
    
    print(f"\n🎯 Next Steps:")
    print(f"1. Use combined dataset: {results['output_path']}")
    print(f"2. Train RNN with sequences: {results['sequence_path']}")
    print(f"3. Expected improvement: More diverse training data")
    print(f"4. Estimated training samples: {len(results['sequences']):,}")

if __name__ == "__main__":
    main()

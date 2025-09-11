#!/usr/bin/env python3
"""
Comprehensive analysis of temporal cardiac datasets to compare HVG quality
"""

import scanpy as sc
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set scanpy settings
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')

class TemporalDatasetAnalyzer:
    """Analyzer for temporal cardiac datasets"""
    
    def __init__(self):
        self.results = {}
        self.datasets = {
            'GSE175634': self.load_gse175634,
            'GSE130731': self.load_gse130731,
            'GSE202398': self.load_gse202398
        }
    
    def load_gse175634(self):
        """Load GSE175634 (current dataset)"""
        print("=== ANALYZING GSE175634 (Current Dataset) ===")
        
        try:
            data_path = Path("data/extracted_datasets/GSE175634")
            
            # Look for h5ad files first
            h5ad_files = list(data_path.glob("*.h5ad"))
            if h5ad_files:
                adata = sc.read_h5ad(h5ad_files[0])
                print(f"Loaded h5ad file: {h5ad_files[0]}")
            else:
                # Look for other formats
                h5_files = list(data_path.glob("*.h5"))
                if h5_files:
                    adata = sc.read_10x_h5(h5_files[0])
                    adata.var_names_make_unique()
                else:
                    print("No suitable files found in GSE175634")
                    return None
            
            # Add metadata
            adata.obs['dataset'] = 'GSE175634'
            if 'timepoint' not in adata.obs.columns:
                # Try to infer timepoints from sample names or add default
                adata.obs['timepoint'] = 'mixed'
            
            return self._analyze_dataset(adata, "GSE175634", has_timepoints=True)
            
        except Exception as e:
            print(f"Error loading GSE175634: {e}")
            return None
    
    def load_gse130731(self):
        """Load GSE130731 iPSC differentiation dataset"""
        print("=== ANALYZING GSE130731 (iPSC Differentiation) ===")
        
        try:
            data_path = Path("data/extracted_datasets/GSE130731")
            
            # Look for h5ad files first
            h5ad_files = list(data_path.glob("*.h5ad"))
            if h5ad_files:
                adata = sc.read_h5ad(h5ad_files[0])
            else:
                # Look for matrix files
                mtx_files = list(data_path.glob("**/matrix.mtx*"))
                if mtx_files:
                    adata = sc.read_10x_mtx(mtx_files[0].parent)
                    adata.var_names_make_unique()
                else:
                    print("No suitable files found in GSE130731")
                    return None
            
            # Add metadata
            adata.obs['dataset'] = 'GSE130731'
            if 'timepoint' not in adata.obs.columns:
                adata.obs['timepoint'] = 'mixed'
            
            return self._analyze_dataset(adata, "GSE130731", has_timepoints=True)
            
        except Exception as e:
            print(f"Error loading GSE130731: {e}")
            return None
    
    def load_gse202398(self):
        """Load GSE202398 cardiac dataset"""
        print("=== ANALYZING GSE202398 (Cardiac scRNA-seq) ===")
        
        try:
            data_path = Path("data/extracted_datasets/GSE202398")
            h5_files = [f for f in data_path.glob("*.h5") if "filtered" in f.name]
            
            if not h5_files:
                print("No filtered h5 files found in GSE202398")
                return None
            
            # Load first filtered dataset as representative
            h5_file = h5_files[0]
            adata = sc.read_10x_h5(h5_file)
            adata.var_names_make_unique()
            
            # Add metadata
            adata.obs['dataset'] = 'GSE202398'
            adata.obs['sample'] = h5_file.stem
            
            return self._analyze_dataset(adata, "GSE202398", has_timepoints=False)
            
        except Exception as e:
            print(f"Error loading GSE202398: {e}")
            return None
    
    def _analyze_dataset(self, adata, dataset_name, has_timepoints=False):
        """Comprehensive analysis of a dataset"""
        print(f"\nAnalyzing {dataset_name}...")
        print(f"Initial shape: {adata.shape}")
        
        try:
            # Store raw data
            adata.raw = adata.copy()
            
            # Basic QC metrics
            adata.var['mt'] = adata.var_names.str.startswith('MT-')
            sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
            
            # Filter cells and genes
            print("Applying quality filters...")
            
            # Filter genes
            sc.pp.filter_genes(adata, min_cells=10)
            print(f"After gene filtering: {adata.shape}")
            
            # Filter cells
            sc.pp.filter_cells(adata, min_genes=200)
            print(f"After cell filtering: {adata.shape}")
            
            # Filter cells with too many genes (potential doublets)
            if 'n_genes_by_counts' in adata.obs.columns:
                adata = adata[adata.obs.n_genes_by_counts < 7000, :]
                print(f"After doublet filtering: {adata.shape}")
            
            # Filter cells with high mitochondrial expression
            if 'pct_counts_mt' in adata.obs.columns:
                adata = adata[adata.obs.pct_counts_mt < 25, :]
                print(f"After mt filtering: {adata.shape}")
            
            # Convert to dense if needed for processing
            if hasattr(adata.X, 'toarray'):
                print("Converting sparse matrix to dense for processing...")
                # Only convert if reasonably sized
                if adata.shape[0] * adata.shape[1] < 1e8:  # Less than 100M elements
                    adata.X = adata.X.toarray()
                else:
                    print("Matrix too large, keeping sparse format")
            
            # Normalize and log transform
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            
            # Find highly variable genes
            print("Finding highly variable genes...")
            sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=False)
            
            hvg_count = adata.var.highly_variable.sum()
            print(f"Found {hvg_count} highly variable genes")
            
            # Cardiac-specific gene analysis
            cardiac_markers = [
                'TNNT2', 'MYH6', 'MYH7', 'ACTC1', 'NKX2-5', 'GATA4', 'GATA6',
                'TBX5', 'MEF2C', 'HAND1', 'HAND2', 'ISL1', 'MYL2', 'MYL7',
                'NPPA', 'NPPB', 'PLN', 'RYR2', 'SCN5A', 'CACNA1C'
            ]
            
            available_cardiac = [g for g in cardiac_markers if g in adata.var_names]
            cardiac_hvg = [g for g in available_cardiac if adata.var.loc[g, 'highly_variable']]
            
            print(f"Cardiac markers available: {len(available_cardiac)}/{len(cardiac_markers)}")
            print(f"Cardiac markers in HVG: {len(cardiac_hvg)}")
            
            # Differentiation-related genes
            diff_markers = [
                'SOX2', 'NANOG', 'POU5F1', 'KLF4', 'MYC',  # Pluripotency
                'MESP1', 'MESP2', 'BMP4', 'WNT3A',         # Early differentiation
                'TBX20', 'NKX2-5', 'GATA4'                 # Cardiac specification
            ]
            
            available_diff = [g for g in diff_markers if g in adata.var_names]
            diff_hvg = [g for g in available_diff if adata.var.loc[g, 'highly_variable']]
            
            print(f"Differentiation markers available: {len(available_diff)}/{len(diff_markers)}")
            print(f"Differentiation markers in HVG: {len(diff_hvg)}")
            
            # Calculate gene expression statistics
            try:
                if hasattr(adata.X, 'toarray'):
                    mean_expr = np.array(adata.X.mean(axis=0)).flatten()
                    var_expr = np.array(adata.X.var(axis=0)).flatten()
                else:
                    mean_expr = adata.X.mean(axis=0)
                    var_expr = adata.X.var(axis=0)
            except:
                mean_expr = None
                var_expr = None
            
            # Store results
            results = {
                'dataset_name': dataset_name,
                'n_cells': adata.shape[0],
                'n_genes_total': adata.shape[1],
                'n_genes_hvg': hvg_count,
                'cardiac_markers_available': len(available_cardiac),
                'cardiac_markers_hvg': len(cardiac_hvg),
                'diff_markers_available': len(available_diff),
                'diff_markers_hvg': len(diff_hvg),
                'has_timepoints': has_timepoints,
                'mean_genes_per_cell': adata.obs['n_genes_by_counts'].mean() if 'n_genes_by_counts' in adata.obs.columns else np.nan,
                'cardiac_hvg_genes': cardiac_hvg,
                'diff_hvg_genes': diff_hvg,
                'available_cardiac_markers': available_cardiac,
                'available_diff_markers': available_diff,
                'hvg_mean_expression': mean_expr.mean() if mean_expr is not None else np.nan,
                'hvg_var_expression': var_expr.mean() if var_expr is not None else np.nan
            }
            
            # Timepoint analysis if available
            if has_timepoints and 'timepoint' in adata.obs.columns:
                timepoints = adata.obs['timepoint'].unique()
                print(f"Timepoints: {list(timepoints)}")
                
                tp_stats = {}
                for tp in timepoints:
                    tp_mask = adata.obs['timepoint'] == tp
                    tp_stats[tp] = {
                        'n_cells': tp_mask.sum(),
                        'mean_genes': adata.obs.loc[tp_mask, 'n_genes_by_counts'].mean() if 'n_genes_by_counts' in adata.obs.columns else np.nan
                    }
                
                results['timepoint_stats'] = tp_stats
        
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            return None
        
        self.results[dataset_name] = results
        print(f"✅ Successfully analyzed {dataset_name}")
        return results
    
    def compare_datasets(self):
        """Compare all analyzed datasets"""
        print("\n" + "="*60)
        print("DATASET COMPARISON SUMMARY")
        print("="*60)
        
        comparison_data = []
        
        for name, result in self.results.items():
            comparison_data.append({
                'Dataset': name,
                'Cells': result['n_cells'],
                'Total Genes': result['n_genes_total'],
                'HVG Count': result['n_genes_hvg'],
                'Cardiac Markers': result['cardiac_markers_available'],
                'Cardiac HVG': result['cardiac_markers_hvg'],
                'Diff Markers': result['diff_markers_available'],
                'Diff HVG': result['diff_markers_hvg'],
                'Has Timepoints': result['has_timepoints'],
                'Avg Genes/Cell': f"{result['mean_genes_per_cell']:.0f}" if not np.isnan(result['mean_genes_per_cell']) else 'N/A'
            })
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        # Detailed analysis
        print("\n" + "="*60)
        print("DETAILED HVG ANALYSIS")
        print("="*60)
        
        for name, result in self.results.items():
            print(f"\n{name}:")
            print(f"  Total genes: {result['n_genes_total']:,}")
            print(f"  HVG: {result['n_genes_hvg']:,} ({result['n_genes_hvg']/result['n_genes_total']*100:.1f}%)")
            print(f"  Cardiac markers in HVG: {result['cardiac_markers_hvg']}/{result['cardiac_markers_available']}")
            print(f"  Differentiation markers in HVG: {result['diff_markers_hvg']}/{result['diff_markers_available']}")
            
            if result['cardiac_hvg_genes']:
                print(f"  Cardiac HVG genes: {', '.join(result['cardiac_hvg_genes'])}")
            
            if result['diff_hvg_genes']:
                print(f"  Diff HVG genes: {', '.join(result['diff_hvg_genes'])}")
        
        # Recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        best_dataset = max(self.results.keys(), 
                          key=lambda x: (self.results[x]['n_genes_total'], 
                                       self.results[x]['cardiac_markers_hvg'],
                                       self.results[x]['diff_markers_hvg']))
        
        print(f"🏆 Best dataset for RNN training: {best_dataset}")
        best_result = self.results[best_dataset]
        print(f"   Reasons:")
        print(f"   - Largest gene count: {best_result['n_genes_total']:,} genes")
        print(f"   - Good cardiac coverage: {best_result['cardiac_markers_hvg']}/{best_result['cardiac_markers_available']} markers in HVG")
        print(f"   - Cell count: {best_result['n_cells']:,} cells")
        
        return df
    
    def run_analysis(self):
        """Run complete analysis on all datasets"""
        print("TEMPORAL CARDIAC DATASET ANALYSIS")
        print("="*60)
        
        for dataset_name, loader_func in self.datasets.items():
            print(f"\nProcessing {dataset_name}...")
            try:
                result = loader_func()
                if result is None:
                    print(f"❌ Failed to analyze {dataset_name}")
            except Exception as e:
                print(f"❌ Error with {dataset_name}: {e}")
        
        if self.results:
            self.compare_datasets()
        else:
            print("❌ No datasets were successfully analyzed")


if __name__ == "__main__":
    analyzer = TemporalDatasetAnalyzer()
    analyzer.run_analysis()

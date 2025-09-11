"""
Comprehensive Analysis of Temporal Cardiac Datasets
Compare GSE175634, GSE130731, GSE202398, and GSE225615
Extract highly variable genes and assess dataset quality
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import os
import gzip
from scipy.io import mmread
import tempfile
warnings.filterwarnings('ignore')

# Set scanpy settings
sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=80, facecolor='white')

class TemporalDatasetAnalyzer:
    """Analyze and compare temporal cardiac datasets"""
    
    def __init__(self, output_dir="temporal_analysis_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def load_gse175634(self):
        """Load current GSE175634 dataset"""
        print("=== ANALYZING GSE175634 (Current) ===")
        
        try:
            # Load from the temporal data folder
            data_path = Path("data/GSE175634_temporal_data")
            
            # Load metadata
            metadata_path = data_path / "GSE175634_cell_metadata.tsv.gz"
            with gzip.open(metadata_path, 'rt') as f:
                metadata = pd.read_csv(f, sep='\t')
            
            # Load gene expression matrix
            matrix_path = data_path / "GSE175634_cell_counts_sctransform.mtx.gz"
            genes_path = data_path / "GSE175634_gene_indices_counts_sctransform.tsv.gz"
            cells_path = data_path / "GSE175634_cell_indices.tsv.gz"
            
            # Load gene names
            with gzip.open(genes_path, 'rt') as f:
                gene_names = pd.read_csv(f, sep='\t', header=0)
            
            # Load cell indices
            with gzip.open(cells_path, 'rt') as f:
                cell_indices = pd.read_csv(f, sep='\t', header=0)
            
            # Extract matrix to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mtx') as tmp:
                with gzip.open(matrix_path, 'rb') as gz_file:
                    tmp.write(gz_file.read())
                tmp_path = tmp.name
            
            try:
                # Load sparse matrix
                X = mmread(tmp_path).T.tocsr()  # Transpose to cells x genes
                
                # Create AnnData object
                adata = sc.AnnData(
                    X=X,
                    obs=metadata.set_index('cell'),
                    var=gene_names.set_index('gene_name')
                )
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
            
            # Process the data
            return self._analyze_dataset(adata, "GSE175634", has_timepoints=True)
            
        except Exception as e:
            print(f"Error loading GSE175634: {e}")
            return None
    
    def load_gse130731(self):
        """Load GSE130731 iPSC cardiomyocyte differentiation dataset"""
        print("=== ANALYZING GSE130731 (iPSC Differentiation) ===")
        
        try:
            data_path = Path("data/extracted_datasets/GSE130731")
            
            # Load all timepoints
            timepoints = ['0', '1', '2', '3']
            all_data = []
            
            for tp in timepoints:
                tp_path = data_path / f"iPS_{tp}.barcodes.genes.matrix"
                
                if not tp_path.exists():
                    # Extract if needed
                    tar_file = data_path / f"GSM3752596_iPS_{tp}.barcodes.genes.matrix.tar.gz"
                    if tp == '0':
                        tar_file = data_path / "GSM3752596_iPS_0.barcodes.genes.matrix.tar.gz"
                    elif tp == '1':
                        tar_file = data_path / "GSM3752598_iPS_1.barcodes.genes.matrix.tar.gz"
                    elif tp == '2':
                        tar_file = data_path / "GSM3752600_iPS_2.barcodes.genes.matrix.tar.gz"
                    elif tp == '3':
                        tar_file = data_path / "GSM3752602_iPS_3.barcodes.genes.matrix.tar.gz"
                    
                    if tar_file.exists():
                        import tarfile
                        with tarfile.open(tar_file, 'r:gz') as tar:
                            tar.extractall(data_path)
                
                # Load matrix files
                if tp_path.exists():
                    matrix_file = tp_path / "matrix.mtx"
                    barcodes_file = tp_path / "barcodes.tsv"
                    genes_file = tp_path / "genes.tsv"
                    
                    if all([f.exists() for f in [matrix_file, barcodes_file, genes_file]]):
                        # Load using scanpy
                        adata_tp = sc.read_10x_mtx(tp_path, var_names='gene_symbols')
                        adata_tp.var_names_make_unique()
                        
                        # Add timepoint information
                        adata_tp.obs['timepoint'] = f"day{tp}"
                        adata_tp.obs['timepoint_numeric'] = int(tp)
                        adata_tp.obs['dataset'] = 'GSE130731'
                        
                        all_data.append(adata_tp)
                        print(f"  Day {tp}: {adata_tp.shape[0]} cells, {adata_tp.shape[1]} genes")
            
            if all_data:
                # Concatenate all timepoints
                adata = sc.concat(all_data, join='outer')
                adata.obs_names_make_unique()
                
                return self._analyze_dataset(adata, "GSE130731", has_timepoints=True)
            else:
                print("No GSE130731 data found")
                return None
                
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
            'has_timepoints': has_timepoints,
            'cardiac_markers_available': len(available_cardiac),
            'cardiac_markers_hvg': len(cardiac_hvg),
            'diff_markers_available': len(available_diff),
            'diff_markers_hvg': len(diff_hvg),
            'mean_genes_per_cell': adata.obs.n_genes_by_counts.mean() if 'n_genes_by_counts' in adata.obs.columns else np.nan,
            'mean_counts_per_cell': adata.obs.total_counts.mean() if 'total_counts' in adata.obs.columns else np.nan,
            'hvg_genes': list(adata.var_names[adata.var.highly_variable]),
            'cardiac_hvg_genes': cardiac_hvg,
            'diff_hvg_genes': diff_hvg,
            'available_cardiac_markers': available_cardiac,
            'available_diff_markers': available_diff
        }
        
        # Timepoint analysis
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
        
        # Save comparison
        df.to_csv(self.output_dir / 'dataset_comparison.csv', index=False)
        
        # Detailed analysis
        print("\n" + "="*60)
        print("DETAILED ANALYSIS")
        print("="*60)
        
        for name, result in self.results.items():
            print(f"\n{name}:")
            print(f"  • Total genes: {result['n_genes_total']:,}")
            print(f"  • HVG: {result['n_genes_hvg']:,} ({result['n_genes_hvg']/result['n_genes_total']*100:.1f}%)")
            print(f"  • Cells: {result['n_cells']:,}")
            print(f"  • Cardiac markers: {result['cardiac_markers_hvg']}/{result['cardiac_markers_available']} in HVG")
            print(f"  • Differentiation markers: {result['diff_markers_hvg']}/{result['diff_markers_available']} in HVG")
            
            if result['has_timepoints'] and 'timepoint_stats' in result:
                print(f"  • Timepoints: {len(result['timepoint_stats'])}")
                for tp, stats in result['timepoint_stats'].items():
                    print(f"    - {tp}: {stats['n_cells']:,} cells")
        
        # Recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        best_for_genes = max(self.results.items(), key=lambda x: x[1]['n_genes_total'])
        best_for_hvg = max(self.results.items(), key=lambda x: x[1]['n_genes_hvg'])
        best_for_cardiac = max(self.results.items(), key=lambda x: x[1]['cardiac_markers_hvg'])
        best_for_cells = max(self.results.items(), key=lambda x: x[1]['n_cells'])
        
        print(f"🧬 Most genes: {best_for_genes[0]} ({best_for_genes[1]['n_genes_total']:,} genes)")
        print(f"🔍 Most HVG: {best_for_hvg[0]} ({best_for_hvg[1]['n_genes_hvg']:,} HVG)")
        print(f"❤️  Best cardiac markers: {best_for_cardiac[0]} ({best_for_cardiac[1]['cardiac_markers_hvg']} cardiac HVG)")
        print(f"📊 Most cells: {best_for_cells[0]} ({best_for_cells[1]['n_cells']:,} cells)")
        
        # Overall recommendation
        temporal_datasets = [name for name, result in self.results.items() if result['has_timepoints']]
        if temporal_datasets:
            # Score temporal datasets
            scores = {}
            for name in temporal_datasets:
                result = self.results[name]
                score = (
                    result['n_genes_hvg'] * 0.4 +  # 40% weight on HVG count
                    result['cardiac_markers_hvg'] * 100 +  # 100 points per cardiac HVG
                    result['diff_markers_hvg'] * 50 +      # 50 points per diff HVG
                    (result['n_cells'] / 1000) * 0.1       # 0.1 point per 1000 cells
                )
                scores[name] = score
            
            best_overall = max(scores.items(), key=lambda x: x[1])
            print(f"\n🏆 BEST OVERALL for temporal analysis: {best_overall[0]} (score: {best_overall[1]:.1f})")
        
        return df
    
    def analyze_hvg_overlap(self):
        """Analyze overlap of highly variable genes between datasets"""
        print("\n" + "="*60)
        print("HIGHLY VARIABLE GENES OVERLAP ANALYSIS")
        print("="*60)
        
        hvg_sets = {}
        for name, result in self.results.items():
            hvg_sets[name] = set(result['hvg_genes'])
        
        # Pairwise overlaps
        datasets = list(hvg_sets.keys())
        overlap_matrix = np.zeros((len(datasets), len(datasets)))
        
        for i, ds1 in enumerate(datasets):
            for j, ds2 in enumerate(datasets):
                if i <= j:
                    overlap = len(hvg_sets[ds1] & hvg_sets[ds2])
                    overlap_matrix[i, j] = overlap
                    overlap_matrix[j, i] = overlap
                    
                    if i != j:
                        print(f"{ds1} ∩ {ds2}: {overlap:,} genes")
        
        # Core genes (present in all datasets)
        if len(hvg_sets) > 1:
            core_genes = set.intersection(*hvg_sets.values())
            print(f"\nCore HVG (in all datasets): {len(core_genes):,} genes")
            
            # Check if core genes include important cardiac markers
            cardiac_core = [g for g in core_genes if g in ['TNNT2', 'MYH6', 'MYH7', 'ACTC1', 'NKX2-5', 'GATA4']]
            if cardiac_core:
                print(f"Core cardiac markers: {cardiac_core}")
    
    def save_results(self):
        """Save detailed results"""
        print(f"\nSaving results to {self.output_dir}/")
        
        # Save individual HVG lists
        for name, result in self.results.items():
            hvg_df = pd.DataFrame({
                'gene': result['hvg_genes'],
                'dataset': name
            })
            hvg_df.to_csv(self.output_dir / f'{name}_hvg_genes.csv', index=False)
        
        # Save combined results
        all_results = []
        for name, result in self.results.items():
            for gene in result['hvg_genes']:
                all_results.append({
                    'dataset': name,
                    'gene': gene,
                    'is_cardiac_marker': gene in ['TNNT2', 'MYH6', 'MYH7', 'ACTC1', 'NKX2-5', 'GATA4'],
                    'is_diff_marker': gene in ['SOX2', 'NANOG', 'POU5F1', 'MESP1', 'MESP2']
                })
        
        combined_df = pd.DataFrame(all_results)
        combined_df.to_csv(self.output_dir / 'all_hvg_genes.csv', index=False)
        
        print("Analysis complete!")


def main():
    """Run the complete analysis"""
    analyzer = TemporalDatasetAnalyzer()
    
    # Load and analyze all datasets
    print("Starting comprehensive temporal dataset analysis...\n")
    
    # Analyze GSE175634 (current)
    analyzer.load_gse175634()
    
    # Analyze GSE130731 (iPSC differentiation)
    analyzer.load_gse130731()
    
    # Analyze GSE202398 (cardiac scRNA-seq)
    analyzer.load_gse202398()
    
    # Compare all datasets
    comparison_df = analyzer.compare_datasets()
    
    # Analyze HVG overlaps
    analyzer.analyze_hvg_overlap()
    
    # Save results
    analyzer.save_results()
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()

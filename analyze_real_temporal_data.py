"""
Real Temporal Dataset Analysis for Biological Trustworthiness
============================================================
"""
import gzip
import pandas as pd
import numpy as np
import os

def analyze_real_temporal_data():
    """Analyze the real GSE175634 temporal cardiac dataset for biological trustworthiness."""
    
    data_dir = r"c:\Users\tumok\Documents\Projects\HybridGnnRnn\data\GSE175634_temporal_data"
    
    print("=== REAL TEMPORAL CARDIAC DATASET ANALYSIS ===")
    print(f"Data directory: {data_dir}")
    
    # List all files
    files = os.listdir(data_dir)
    print(f"\nAvailable files ({len(files)}):")
    for f in files:
        file_path = os.path.join(data_dir, f)
        size_mb = os.path.getsize(file_path) / (1024*1024)
        print(f"  {f} ({size_mb:.2f} MB)")
    
    print("\n" + "="*60)
    
    # Read experimental design
    try:
        exp_design_path = os.path.join(data_dir, "GSE175634_experimental_design.txt.gz")
        with gzip.open(exp_design_path, 'rt') as f:
            exp_design = f.read()
        print("EXPERIMENTAL DESIGN:")
        print(exp_design[:2000])  # First 2000 characters
        print("..." if len(exp_design) > 2000 else "")
    except Exception as e:
        print(f"Error reading experimental design: {e}")
    
    print("\n" + "="*60)
    
    # Read collection metadata
    try:
        metadata_path = os.path.join(data_dir, "GSE175634_collection_metadata.txt.gz")
        with gzip.open(metadata_path, 'rt') as f:
            metadata = f.read()
        print("COLLECTION METADATA:")
        print(metadata[:2000])  # First 2000 characters
        print("..." if len(metadata) > 2000 else "")
    except Exception as e:
        print(f"Error reading collection metadata: {e}")
    
    print("\n" + "="*60)
    
    # Read cell metadata (first few lines)
    try:
        cell_metadata_path = os.path.join(data_dir, "GSE175634_cell_metadata.tsv.gz")
        cell_metadata = pd.read_csv(cell_metadata_path, sep='\t', nrows=20)
        print("CELL METADATA (first 20 rows):")
        print(cell_metadata)
        print(f"\nColumn names: {list(cell_metadata.columns)}")
        
        # Full metadata for analysis
        full_metadata = pd.read_csv(cell_metadata_path, sep='\t')
        print(f"\nTotal cells: {len(full_metadata)}")
        
        # Analyze temporal structure
        if 'timepoint' in full_metadata.columns:
            timepoints = full_metadata['timepoint'].value_counts()
            print(f"\nTimepoints available:")
            print(timepoints)
        
        if 'condition' in full_metadata.columns:
            conditions = full_metadata['condition'].value_counts()
            print(f"\nConditions available:")
            print(conditions)
            
        if 'sample' in full_metadata.columns:
            samples = full_metadata['sample'].value_counts()
            print(f"\nSamples available:")
            print(samples)
            
    except Exception as e:
        print(f"Error reading cell metadata: {e}")
    
    print("\n" + "="*60)
    
    # Read gene information
    try:
        gene_path = os.path.join(data_dir, "GSE175634_gene_indices_counts.tsv.gz")
        genes = pd.read_csv(gene_path, sep='\t', nrows=20)
        print("GENE INFORMATION (first 20 genes):")
        print(genes)
        
        # Full gene list
        full_genes = pd.read_csv(gene_path, sep='\t')
        print(f"\nTotal genes: {len(full_genes)}")
        
        # Check for cardiac markers
        cardiac_markers = ['NPPA', 'NPPB', 'MYH6', 'MYH7', 'ACTN2', 'TNNT2', 'ACTC1', 'TPM1']
        available_markers = []
        if 'gene_symbol' in full_genes.columns:
            for marker in cardiac_markers:
                if marker in full_genes['gene_symbol'].values:
                    available_markers.append(marker)
        
        print(f"\nCardiac markers available: {available_markers}")
        print(f"Cardiac marker coverage: {len(available_markers)}/{len(cardiac_markers)} ({100*len(available_markers)/len(cardiac_markers):.1f}%)")
        
    except Exception as e:
        print(f"Error reading gene information: {e}")
    
    return True

if __name__ == "__main__":
    analyze_real_temporal_data()

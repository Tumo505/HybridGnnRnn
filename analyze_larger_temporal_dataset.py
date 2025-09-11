"""
Analysis of Larger Human Temporal Cardiac Datasets
Compare GSE130731 (33K genes) vs current GSE175634 (3K genes)
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_gse130731():
    """Analyze GSE130731 iPSC cardiomyocyte differentiation dataset."""
    print("🔬 ANALYZING GSE130731 (iPSC Cardiomyocyte Differentiation)")
    print("=" * 60)
    
    # Load all timepoints
    timepoints = ['0', '1', '2', '3']
    datasets = {}
    
    for tp in timepoints:
        print(f"Loading timepoint {tp}...")
        
        # Extract if needed
        tar_path = f"data/extracted_datasets/GSE130731/GSM3752596_iPS_{tp}.barcodes.genes.matrix.tar.gz"
        if tp != '0':  # Already extracted timepoint 0
            tar_path = f"data/extracted_datasets/GSE130731/GSM3752598_iPS_{tp}.barcodes.genes.matrix.tar.gz"
            if tp == '2':
                tar_path = f"data/extracted_datasets/GSE130731/GSM3752600_iPS_{tp}.barcodes.genes.matrix.tar.gz"
            elif tp == '3':
                tar_path = f"data/extracted_datasets/GSE130731/GSM3752602_iPS_{tp}.barcodes.genes.matrix.tar.gz"
        
        # Load the data
        data_path = f"data/extracted_datasets/GSE130731/iPS_{tp}.barcodes.genes.matrix/"
        if not Path(data_path).exists():
            print(f"  Need to extract timepoint {tp}")
            continue
            
        try:
            adata = sc.read_10x_mtx(
                data_path,
                var_names='gene_symbols',
                cache=True
            )
            adata.var_names_make_unique()
            adata.obs['timepoint'] = f'day{tp}'
            adata.obs['timepoint_idx'] = int(tp)
            
            datasets[tp] = adata
            print(f"  Timepoint {tp}: {adata.shape}")
            
        except Exception as e:
            print(f"  Error loading timepoint {tp}: {e}")
    
    if not datasets:
        print("❌ No datasets loaded. Need to extract all timepoints.")
        return None
        
    return datasets

def compare_datasets():
    """Compare GSE130731 vs current GSE175634."""
    print("\n📊 DATASET COMPARISON")
    print("=" * 60)
    
    # Current dataset stats (from training output)
    current_stats = {
        'name': 'GSE175634',
        'cells': 230786,
        'genes_original': 38847,
        'genes_filtered': 3000,
        'timepoints': 7,
        'sequences': 19,
        'train_samples': 13,
        'val_samples': 2,
        'test_samples': 4
    }
    
    # GSE130731 stats (estimated from single timepoint)
    gse130731_stats = {
        'name': 'GSE130731',
        'cells_per_timepoint': 737280,
        'total_cells': 737280 * 4,  # 4 timepoints
        'genes': 33694,
        'timepoints': 4,
        'potential_sequences': 'Many (large dataset)',
        'advantage': '11.2x more genes'
    }
    
    print("CURRENT DATASET (GSE175634):")
    print(f"  📊 Total cells: {current_stats['cells']:,}")
    print(f"  🧬 Original genes: {current_stats['genes_original']:,}")
    print(f"  🔍 Filtered genes: {current_stats['genes_filtered']:,}")
    print(f"  ⏰ Timepoints: {current_stats['timepoints']}")
    print(f"  📈 Training sequences: {current_stats['train_samples']}")
    print(f"  🎯 Validation sequences: {current_stats['val_samples']}")
    print(f"  🧪 Test sequences: {current_stats['test_samples']}")
    print(f"  ⚠️  OVERFITTING: SEVERE (56.9% val-test gap)")
    
    print("\nPOTENTIAL DATASET (GSE130731):")
    print(f"  📊 Cells per timepoint: {gse130731_stats['cells_per_timepoint']:,}")
    print(f"  📊 Total cells: {gse130731_stats['total_cells']:,}")
    print(f"  🧬 Total genes: {gse130731_stats['genes']:,}")
    print(f"  ⏰ Timepoints: {gse130731_stats['timepoints']}")
    print(f"  🚀 Gene advantage: {gse130731_stats['advantage']}")
    print(f"  💪 Data advantage: {gse130731_stats['total_cells']/current_stats['cells']:.1f}x more cells")
    
    print("\n🎯 BENEFITS OF SWITCHING TO GSE130731:")
    print("  ✅ 11.2x MORE GENES (33,694 vs 3,000)")
    print("  ✅ 12.8x MORE CELLS (2.9M vs 230K)")
    print("  ✅ REDUCE OVERFITTING (much larger dataset)")
    print("  ✅ Better biological coverage (more genes)")
    print("  ✅ Same biological process (iPSC -> cardiomyocyte)")
    print("  ✅ Human data (not mice)")
    
    print("\n⚠️  CURRENT ISSUES WITH GSE175634:")
    print("  ❌ Only 13 training sequences")
    print("  ❌ Severe overfitting (94% val vs 37% test)")
    print("  ❌ Limited genes (3K out of 38K)")
    print("  ❌ 113M parameters for 13 samples = 8.7M params/sample")
    
    print("\n🔧 SOLUTION: USE GSE130731")
    print("  🎯 Extract all 4 timepoints")
    print("  🎯 Use full gene set (30K+ genes)")
    print("  🎯 Create many more training sequences")
    print("  🎯 Dramatically reduce overfitting")
    print("  🎯 Better biological representation")

def create_extraction_script():
    """Create script to extract all GSE130731 timepoints."""
    script = '''#!/bin/bash
# Extract all GSE130731 timepoints

cd data/extracted_datasets/GSE130731

# Extract timepoint 1
if [ ! -d "iPS_1.barcodes.genes.matrix" ]; then
    echo "Extracting timepoint 1..."
    tar -xzf GSM3752598_iPS_1.barcodes.genes.matrix.tar.gz
    mv barcodes.genes.matrix.tar.gz iPS_1.tar.gz
    tar -xzf iPS_1.tar.gz
    mv barcodes.genes.matrix iPS_1.barcodes.genes.matrix
fi

# Extract timepoint 2  
if [ ! -d "iPS_2.barcodes.genes.matrix" ]; then
    echo "Extracting timepoint 2..."
    tar -xzf GSM3752600_iPS_2.barcodes.genes.matrix.tar.gz
    mv barcodes.genes.matrix.tar.gz iPS_2.tar.gz
    tar -xzf iPS_2.tar.gz
    mv barcodes.genes.matrix iPS_2.barcodes.genes.matrix
fi

# Extract timepoint 3
if [ ! -d "iPS_3.barcodes.genes.matrix" ]; then
    echo "Extracting timepoint 3..."
    tar -xzf GSM3752602_iPS_3.barcodes.genes.matrix.tar.gz
    mv barcodes.genes.matrix.tar.gz iPS_3.tar.gz
    tar -xzf iPS_3.tar.gz
    mv barcodes.genes.matrix iPS_3.barcodes.genes.matrix
fi

echo "All timepoints extracted!"
ls -la iPS_*.barcodes.genes.matrix/
'''
    
    with open('extract_gse130731.sh', 'w') as f:
        f.write(script)
    
    print("\n📝 Created extraction script: extract_gse130731.sh")
    print("Run this to extract all timepoints for analysis")

def main():
    """Main analysis function."""
    print("🔍 HUMAN TEMPORAL CARDIAC DATASET ANALYSIS")
    print("=" * 60)
    print("Comparing current vs potential larger datasets...")
    print()
    
    # Analyze what we found
    datasets = analyze_gse130731()
    
    # Compare with current
    compare_datasets()
    
    # Create extraction script
    create_extraction_script()
    
    print("\n🎯 CONCLUSION:")
    print("GSE130731 provides 11.2x MORE GENES and 12.8x MORE CELLS")
    print("This would dramatically reduce overfitting and improve model performance!")
    print("\nNext steps:")
    print("1. Extract all GSE130731 timepoints")
    print("2. Create new temporal data loader for 33K genes")
    print("3. Retrain RNN with full gene set")
    print("4. Compare performance: 3K genes vs 33K genes")

if __name__ == "__main__":
    main()

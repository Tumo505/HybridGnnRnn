#!/usr/bin/env python3
"""
Test the fixed data loader to ensure balanced cell type distribution.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.scgnn_loader import create_scgnn_data_loaders

def test_balanced_data_loader():
    """Test that the fixed data loader creates balanced graphs."""
    
    print("🧪 Testing Fixed Data Loader")
    print("=" * 50)
    
    # Load data with the fixed loader
    train_loader, val_loader, test_loader = create_scgnn_data_loaders(
        data_path='data/processed_visium_heart.h5ad',
        batch_size=4,
        num_neighbors=12,
        train_ratio=0.7,
        val_ratio=0.2
    )
    
    print(f"Created {len(train_loader)} training batches")
    
    # Analyze graph-level targets
    all_graph_targets = []
    
    for batch_idx, batch in enumerate(train_loader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Graphs in batch: {batch.batch.max().item() + 1}")
        
        # Calculate graph-level targets
        graph_targets = []
        for i in range(batch.batch.max().item() + 1):
            graph_mask = batch.batch == i
            graph_cell_types = batch.y_class[graph_mask]
            
            # Show cell type distribution within this graph
            unique_types, counts = np.unique(graph_cell_types.cpu().numpy(), return_counts=True)
            print(f"    Graph {i}: {len(graph_cell_types)} cells")
            print(f"      Cell types: {unique_types}")
            print(f"      Counts: {counts}")
            print(f"      Proportions: {counts / len(graph_cell_types)}")
            
            # Most common cell type becomes graph target
            most_common_idx = np.argmax(counts)
            graph_target = unique_types[most_common_idx]
            graph_targets.append(graph_target)
            print(f"      Graph target: {graph_target}")
        
        all_graph_targets.extend(graph_targets)
        print(f"  Batch graph targets: {graph_targets}")
    
    # Overall analysis
    print(f"\n📊 Overall Graph Target Analysis:")
    unique_targets, target_counts = np.unique(all_graph_targets, return_counts=True)
    print(f"  Graph targets found: {unique_targets}")
    print(f"  Target counts: {target_counts}")
    print(f"  Target proportions: {target_counts / len(all_graph_targets)}")
    
    if len(unique_targets) > 1:
        print("  ✅ SUCCESS: Multiple cell types represented as graph targets!")
    else:
        print("  ❌ STILL FAILING: Only one cell type in graph targets")
    
    return len(unique_targets) > 1

if __name__ == "__main__":
    success = test_balanced_data_loader()
    print(f"\nResult: {'✅ FIXED' if success else '❌ STILL BROKEN'}")

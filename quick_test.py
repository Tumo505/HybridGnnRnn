#!/usr/bin/env python3

import torch
import torch.nn as nn
import sys
import os

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from comprehensive_cardiac_data_processor import ComprehensiveCardiacDataProcessor
from train_scalable_rnn import ScalableRegularizedRNN

def quick_test():
    print("🔍 Quick test with real data...")
    
    # Load actual data
    processor = ComprehensiveCardiacDataProcessor()
    sequences, labels, labels_map = processor.create_comprehensive_dataset(
        temporal_sequences_per_type=100,  # Small test
        spatial_sequences=50
    )
    
    # Create data loader
    from torch.utils.data import DataLoader, TensorDataset
    import torch
    sequences = torch.FloatTensor(sequences)
    labels = torch.LongTensor(labels)
    dataset = TensorDataset(sequences, labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Get one batch
    for batch_sequences, batch_labels in dataloader:
        print(f"Batch sequences shape: {batch_sequences.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        
        # Create model
        model = ScalableRegularizedRNN(
            input_dim=200,
            hidden_dim=64,
            num_layers=2,
            num_classes=7,
            dropout=0.2,
            use_residual=True,
            use_attention=True
        )
        
        # Test forward pass
        try:
            outputs = model(batch_sequences)
            print(f"✅ Model outputs shape: {outputs.shape}")
            print(f"Expected: ({batch_sequences.shape[0]}, 7)")
            
            # Test loss calculation
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, batch_labels)
            print(f"✅ Loss calculation successful: {loss.item():.4f}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        break  # Only test first batch

if __name__ == "__main__":
    quick_test()

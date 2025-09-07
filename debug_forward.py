#!/usr/bin/env python3

import torch
import torch.nn as nn
import sys
import os

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from train_scalable_rnn import ScalableRegularizedRNN

def test_forward():
    print("🔍 Testing ScalableRegularizedRNN forward method...")
    
    # Create a simple test model
    model = ScalableRegularizedRNN(
        input_dim=200,
        hidden_dim=64,
        num_layers=2,
        num_classes=7,
        dropout=0.2,
        use_residual=True,
        use_attention=True
    )
    
    # Create test input
    batch_size = 32
    seq_len = 100  # Same as the actual training
    input_dim = 200
    
    x = torch.randn(batch_size, seq_len, input_dim)
    lengths = torch.tensor([100] * batch_size)  # All sequences full length
    
    print(f"Input shape: {x.shape}")
    print(f"Lengths: {lengths}")
    
    # Forward pass
    try:
        output = model(x, lengths)
        print(f"✅ Forward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: [{batch_size}, 7]")
        
        if output.shape == (batch_size, 7):
            print("✅ Shape is correct!")
        else:
            print(f"❌ Shape mismatch! Got {output.shape}, expected ({batch_size}, 7)")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_forward()

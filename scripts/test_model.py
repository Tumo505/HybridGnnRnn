"""
Simple Training Test Script

A simplified version for testing the core functionality.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.hybrid_model import LightweightHybridGNNRNN
from training.utils import MultiTaskLoss
from utils.memory_utils import MemoryMonitor

def test_model():
    """Test the hybrid model with synthetic data."""
    print("Testing Hybrid GNN-RNN Model...")
    
    # Initialize memory monitor
    monitor = MemoryMonitor(max_memory_gb=16.0)
    monitor.check_memory("Start")
    
    # Model parameters
    node_feature_dim = 50
    batch_size = 2
    num_nodes = 100
    sequence_length = 7
    
    print(f"Creating model with {node_feature_dim} features...")
    
    # Create lightweight model
    model = LightweightHybridGNNRNN(
        node_feature_dim=node_feature_dim,
        gnn_hidden_dim=64,
        rnn_hidden_dim=64,
        fusion_dim=128,
        prediction_tasks=['multitask'],
        use_uncertainty=True,
        memory_efficient=True
    )
    
    # Print model summary
    summary = model.get_model_summary()
    print(f"Model created successfully!")
    print(f"Total parameters: {summary['total_parameters']:,}")
    
    monitor.check_memory("After model creation")
    
    # Create synthetic data
    print("Creating synthetic data...")
    
    # Spatial data
    node_features = torch.randn(batch_size * num_nodes, node_feature_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    edge_attr = torch.randn(edge_index.size(1), 1)
    pos = torch.randn(batch_size * num_nodes, 2)
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)
    
    # Temporal data
    temporal_features = torch.randn(batch_size, sequence_length, node_feature_dim)
    temporal_mask = torch.ones(batch_size, sequence_length)
    
    print("Running forward pass...")
    monitor.check_memory("Before forward")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            batch=batch,
            temporal_features=temporal_features,
            temporal_mask=temporal_mask,
            return_embeddings=True
        )
    
    print(f"Forward pass successful!")
    print(f"Output keys: {list(outputs.keys())}")
    
    # Print output shapes
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    monitor.check_memory("After forward")
    
    # Test loss calculation
    print("\nTesting loss calculation...")
    
    criterion = MultiTaskLoss()
    criterion.add_task("multitask_differentiation_efficiency", "mse")
    criterion.add_task("multitask_maturation_logits", "classification")
    
    # Create dummy targets
    targets = {
        "multitask_differentiation_efficiency": torch.rand(batch_size),
        "multitask_maturation_logits": torch.randint(0, 3, (batch_size,))
    }
    
    # Extract predictions
    predictions = {
        "multitask_differentiation_efficiency": outputs["multitask_differentiation_efficiency"],
        "multitask_maturation_logits": outputs["multitask_maturation_logits"]
    }
    
    losses = criterion(predictions, targets)
    print(f"Loss calculation successful!")
    print(f"Losses: {list(losses.keys())}")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    monitor.check_memory("After loss")
    
    # Test prediction method
    print("\nTesting prediction method...")
    
    predictions = model.predict(
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        batch=batch,
        temporal_features=temporal_features,
        temporal_mask=temporal_mask,
        return_uncertainty=True
    )
    
    print(f"Prediction method successful!")
    print(f"Prediction keys: {list(predictions.keys())}")
    
    monitor.check_memory("After prediction")
    
    # Memory report
    print("\nMemory Report:")
    report = monitor.get_memory_report()
    print(f"Peak memory pressure: {report['peak_pressure']:.1%}")
    print(f"Current memory usage: {report['current_stats']['system_used_gb']:.1f}GB")
    
    print("\nAll tests passed successfully! ✅")
    return True

def test_training_step():
    """Test a single training step."""
    print("\nTesting training step...")
    
    # Simple model and data
    model = LightweightHybridGNNRNN(
        node_feature_dim=20,
        gnn_hidden_dim=32,
        rnn_hidden_dim=32,
        fusion_dim=64,
        prediction_tasks=['efficiency'],
        use_uncertainty=False,
        memory_efficient=True
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Synthetic batch
    batch_size = 2
    num_nodes = 50
    node_feature_dim = 20
    sequence_length = 5
    
    node_features = torch.randn(batch_size * num_nodes, node_feature_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes))
    pos = torch.randn(batch_size * num_nodes, 2)
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)
    temporal_features = torch.randn(batch_size, sequence_length, node_feature_dim)
    temporal_mask = torch.ones(batch_size, sequence_length)
    
    # Target
    target = torch.rand(batch_size)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    outputs = model(
        node_features=node_features,
        edge_index=edge_index,
        pos=pos,
        batch=batch,
        temporal_features=temporal_features,
        temporal_mask=temporal_mask
    )
    
    # Simple MSE loss
    loss = nn.MSELoss()(outputs['efficiency'], target)
    
    print(f"Loss: {loss.item():.4f}")
    
    loss.backward()
    optimizer.step()
    
    print("Training step successful! ✅")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("HYBRID GNN-RNN MODEL TEST")
    print("=" * 60)
    
    try:
        # Test model
        test_model()
        
        # Test training step
        test_training_step()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED SUCCESSFULLY! 🎉")
        print("The hybrid model is ready for training!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

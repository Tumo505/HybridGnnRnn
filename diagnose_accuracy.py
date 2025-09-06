#!/usr/bin/env python3
"""
Diagnostic Tool: Investigate Low Accuracy Issues
Analyze the model predictions and data distribution to understand the 33% accuracy problem.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models.scgnn import scGNNTransferLearning
from src.data.scgnn_loader import create_scgnn_data_loaders

def analyze_model_predictions():
    """Comprehensive analysis of model predictions and data."""
    
    print("🔍 INVESTIGATING LOW ACCURACY ISSUES")
    print("=" * 60)
    
    # Load data
    print("📊 Loading and analyzing data...")
    train_loader, val_loader, test_loader = create_scgnn_data_loaders(
        data_path='data/processed_visium_heart.h5ad',
        batch_size=4,
        num_neighbors=12,
        train_ratio=0.7,
        val_ratio=0.2
    )
    
    # Analyze data distribution
    print("\n📈 Analyzing Data Distribution:")
    
    all_labels = []
    all_reg_targets = []
    
    for batch in train_loader:
        if hasattr(batch, 'y_class'):
            all_labels.extend(batch.y_class.cpu().numpy())
        if hasattr(batch, 'y_reg'):
            all_reg_targets.extend(batch.y_reg.cpu().numpy())
    
    if all_labels:
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        total_samples = len(all_labels)
        
        print(f"  Training data:")
        print(f"    Total samples: {total_samples}")
        print(f"    Classes found: {unique_labels}")
        print(f"    Class counts: {counts}")
        print(f"    Class proportions: {counts / total_samples}")
        print(f"    Most common class: {unique_labels[np.argmax(counts)]} ({counts.max() / total_samples:.1%})")
        
        # Check if distribution is too imbalanced
        if counts.max() / total_samples > 0.8:
            print("    ⚠️  WARNING: Severe class imbalance detected!")
        elif counts.max() / total_samples > 0.6:
            print("    ⚠️  WARNING: Moderate class imbalance detected")
        else:
            print("    ✅ Class distribution looks reasonable")
    
    # Load and test model
    print("\n🤖 Loading model for detailed analysis...")
    
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch.x.size(1)
    
    # Test with fast production model
    fast_model = scGNNTransferLearning(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        num_classes=5,
        dropout=0.2,
        device='cpu'
    )
    
    fast_model.load_model('models/production/scgnn_latest.pth')
    fast_model.model.eval()
    
    print(f"  Model loaded: {sum(p.numel() for p in fast_model.model.parameters()):,} parameters")
    
    # Detailed prediction analysis
    print("\n🧪 Detailed Prediction Analysis:")
    
    all_predictions = []
    all_targets = []
    all_confidences = []
    
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            batch = batch.to('cpu')
            outputs = fast_model.model(batch)
            
            # Get raw logits and probabilities
            logits = outputs['classification']
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            confidences = torch.max(probs, dim=1)[0]
            
            print(f"\n  Batch {i+1}:")
            print(f"    Raw logits shape: {logits.shape}")
            print(f"    Raw logits:\n{logits.cpu().numpy()}")
            print(f"    Probabilities:\n{probs.cpu().numpy()}")
            print(f"    Predictions: {predictions.cpu().numpy()}")
            print(f"    Confidences: {confidences.cpu().numpy()}")
            
            # Check if model outputs are reasonable
            if torch.any(torch.isnan(logits)):
                print("    ❌ NaN values in logits!")
            if torch.any(torch.isinf(logits)):
                print("    ❌ Infinite values in logits!")
            
            # Check if all predictions are the same
            if len(torch.unique(predictions)) == 1:
                print(f"    ⚠️  All predictions are class {predictions[0].item()}")
            
            all_predictions.extend(predictions.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            
            # Get true targets for comparison
            if hasattr(batch, 'y_class'):
                # Get graph-level targets
                if hasattr(batch, 'batch'):
                    graph_labels = []
                    for j in range(batch.batch.max().item() + 1):
                        graph_mask = batch.batch == j
                        graph_cell_types = batch.y_class[graph_mask]
                        unique_types, counts = torch.unique(graph_cell_types, return_counts=True)
                        most_common_idx = torch.argmax(counts)
                        graph_labels.append(unique_types[most_common_idx])
                    graph_targets = torch.stack(graph_labels)
                else:
                    unique_types, counts = torch.unique(batch.y_class, return_counts=True)
                    most_common_idx = torch.argmax(counts)
                    graph_targets = unique_types[most_common_idx].unsqueeze(0)
                
                all_targets.extend(graph_targets.cpu().numpy())
                print(f"    True targets: {graph_targets.cpu().numpy()}")
                
                # Calculate batch accuracy
                batch_acc = (predictions == graph_targets).float().mean().item()
                print(f"    Batch accuracy: {batch_acc:.3f}")
    
    # Overall analysis
    print("\n📊 Overall Prediction Analysis:")
    
    if all_predictions and all_targets:
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_confidences = np.array(all_confidences)
        
        overall_accuracy = (all_predictions == all_targets).mean()
        print(f"  Overall accuracy: {overall_accuracy:.3f} ({overall_accuracy:.1%})")
        
        # Prediction distribution
        pred_unique, pred_counts = np.unique(all_predictions, return_counts=True)
        target_unique, target_counts = np.unique(all_targets, return_counts=True)
        
        print(f"  Predicted classes: {pred_unique}")
        print(f"  Prediction counts: {pred_counts}")
        print(f"  Prediction proportions: {pred_counts / len(all_predictions)}")
        
        print(f"  True classes: {target_unique}")
        print(f"  True counts: {target_counts}")
        print(f"  True proportions: {target_counts / len(all_targets)}")
        
        print(f"  Average confidence: {all_confidences.mean():.3f}")
        
        # Diagnosis
        print("\n🩺 Diagnosis:")
        
        if len(pred_unique) == 1:
            print(f"  ❌ Model collapsed to predicting only class {pred_unique[0]}")
            print("  💡 Possible causes:")
            print("     - Learning rate too high")
            print("     - Severe class imbalance")
            print("     - Loss function issues")
            print("     - Insufficient training")
        
        elif overall_accuracy < 0.4:
            print(f"  ⚠️  Low accuracy ({overall_accuracy:.1%}) detected")
            print("  💡 Possible causes:")
            print("     - Model underfitting")
            print("     - Learning rate too low")
            print("     - Insufficient model capacity")
            print("     - Poor feature engineering")
        
        else:
            print(f"  ✅ Accuracy seems reasonable for this task")
    
    # Check model weights
    print("\n🔍 Model Weight Analysis:")
    
    for name, param in fast_model.model.named_parameters():
        if 'classifier' in name and 'weight' in name:
            print(f"  {name}: shape {param.shape}")
            print(f"    Mean: {param.data.mean().item():.6f}")
            print(f"    Std: {param.data.std().item():.6f}")
            print(f"    Min: {param.data.min().item():.6f}")
            print(f"    Max: {param.data.max().item():.6f}")
            
            if torch.any(torch.isnan(param.data)):
                print("    ❌ NaN weights detected!")
            if param.data.std().item() < 1e-6:
                print("    ⚠️  Very small weight variance - possible dead neurons")
    
    return overall_accuracy if all_predictions and all_targets else 0

def suggest_fixes(accuracy):
    """Suggest fixes based on the accuracy analysis."""
    
    print("\n" + "=" * 60)
    print("💡 SUGGESTED FIXES")
    print("=" * 60)
    
    if accuracy < 0.3:
        print("🚨 CRITICAL: Model has severe issues")
        print("\n🔧 Immediate fixes needed:")
        print("  1. Reduce learning rate by 10x")
        print("  2. Check data preprocessing")
        print("  3. Verify loss function implementation")
        print("  4. Add class balancing")
        print("  5. Increase training epochs")
        
    elif accuracy < 0.5:
        print("⚠️  Model needs improvement")
        print("\n🔧 Recommended fixes:")
        print("  1. Adjust learning rate")
        print("  2. Increase model capacity") 
        print("  3. Better data augmentation")
        print("  4. Longer training")
        
    else:
        print("✅ Model performance is acceptable")
        print("\n🔧 Optional improvements:")
        print("  1. Fine-tune hyperparameters")
        print("  2. Add ensemble methods")
        print("  3. Advanced regularization")

if __name__ == "__main__":
    accuracy = analyze_model_predictions()
    suggest_fixes(accuracy)

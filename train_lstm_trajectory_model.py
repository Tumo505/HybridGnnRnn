#!/usr/bin/env python3
"""
LSTM Trajectory Model Training Script

Train the LSTMTrajectoryModel (from enhanced_cardiac_ensemble.py) independently
on real cardiac temporal data for gene expression trajectory classification.

Features:
- Real cardiac temporal data loading
- LSTM-based architecture with input projection
- Enhanced regularization and evaluation
- Comparison with other RNN models
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import logging
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Import the model from enhanced_cardiac_ensemble
from enhanced_cardiac_ensemble import LSTMTrajectoryModel
from real_cardiac_temporal_processor import RealCardiacTemporalDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class FeatureSelector:
    """Enhanced feature selection with cardiac gene prioritization"""
    
    def __init__(self):
        self.cardiac_genes = [
            'ACTC1', 'ACTN2', 'MYH6', 'MYH7', 'TNNT2', 'NPPA', 'NPPB', 
            'TPM1', 'MYBPC3', 'MYL2', 'MYL7', 'CACNA1C', 'SCN5A',
            'GATA4', 'GATA6', 'NKX2-5', 'TBX5', 'MEF2C', 'HAND1', 'HAND2',
            'HOPX', 'IRX4', 'ISL1', 'MSX1', 'MSX2', 'TBX3', 'TBX20'
        ]
    
    def select_features(self, X, gene_names, top_k=5000):
        """Select top features with cardiac gene prioritization"""
        print(f"   Selecting top {top_k} features from {len(gene_names)} features...")
        
        # Calculate variance for all features
        variances = np.var(X, axis=0)
        
        # Create scoring system: variance + cardiac gene bonus
        scores = variances.copy()
        
        # Add bonus for cardiac genes
        cardiac_bonus = np.max(variances) * 0.1
        for i, gene_name in enumerate(gene_names):
            if gene_name in self.cardiac_genes:
                scores[i] += cardiac_bonus
        
        # Select top features
        top_indices = np.argsort(scores)[-top_k:]
        selected_data = X[:, top_indices]
        selected_gene_names = [gene_names[i] for i in top_indices]
        
        # Count cardiac genes in selection
        cardiac_genes_selected = len([g for g in selected_gene_names if g in self.cardiac_genes])
        print(f"   Selected {len(top_indices)} features ({cardiac_genes_selected} cardiac genes)")
        
        return selected_data, selected_gene_names, top_indices

def create_sequence_labels(X_orig, y_orig, up_threshold=1.05, down_threshold=0.95):
    """Create sequence-level trajectory labels"""
    labels = []
    up_count = down_count = stable_count = 0
    
    for i in range(len(X_orig)):
        # Calculate fold changes for this sequence
        epsilon = 1e-8
        fold_changes = (y_orig[i] + epsilon) / (X_orig[i] + epsilon)
        
        # Count genes in each direction
        up_genes = np.sum(fold_changes > up_threshold)
        down_genes = np.sum(fold_changes < down_threshold)
        total_genes = len(fold_changes)
        
        # Classify sequence based on predominant direction
        up_ratio = up_genes / total_genes
        down_ratio = down_genes / total_genes
        
        if up_ratio > down_ratio and up_ratio > 0.02:  # At least 2% genes up
            label = 2  # UP trajectory
            up_count += 1
        elif down_ratio > up_ratio and down_ratio > 0.02:  # At least 2% genes down
            label = 0  # DOWN trajectory
            down_count += 1
        else:
            label = 1  # STABLE trajectory
            stable_count += 1
        
        labels.append(label)
    
    return np.array(labels), up_count, down_count, stable_count

def train_lstm_trajectory_model():
    """
    Main training function for LSTM Trajectory Model.
    """
    
    print("🚀 LSTM Trajectory Model Training")
    print("=" * 60)
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 1. Load and prepare data
    print("\\n1. Loading cardiac temporal dataset...")
    data_dir = r"data\GSE175634_temporal_data"
    processor = RealCardiacTemporalDataset(data_dir)
    dataset = processor.prepare_dataset_for_training()
    
    if not dataset['validation_passed']:
        print("❌ Dataset validation failed!")
        return None, None
    
    print("✅ Dataset validation passed")
    print(f"   Training samples: {dataset['X_train'].shape[0]:,}")
    print(f"   Genes: {dataset['X_train'].shape[1]:,}")
    print(f"   Cardiac markers: {len(dataset['cardiac_markers'])}")
    
    # 2. Convert to trajectory labels
    print("\\n2. Converting to trajectory direction labels...")
    
    # Use the processed X and y to create sequence-level trajectory labels
    X_train, X_val, X_test = dataset['X_train'], dataset['X_val'], dataset['X_test']
    y_train_reg, y_val_reg, y_test_reg = dataset['y_train'], dataset['y_val'], dataset['y_test']
    
    # Denormalize for proper fold-change calculation
    X_train_orig = dataset['scaler'].inverse_transform(X_train)
    X_val_orig = dataset['scaler'].inverse_transform(X_val)
    X_test_orig = dataset['scaler'].inverse_transform(X_test)
    
    y_train_orig = dataset['target_scaler'].inverse_transform(y_train_reg)
    y_val_orig = dataset['target_scaler'].inverse_transform(y_val_reg)
    y_test_orig = dataset['target_scaler'].inverse_transform(y_test_reg)
    
    # Create labels for all splits
    y_train, train_up, train_down, train_stable = create_sequence_labels(X_train_orig, y_train_orig)
    y_val, val_up, val_down, val_stable = create_sequence_labels(X_val_orig, y_val_orig)
    y_test, test_up, test_down, test_stable = create_sequence_labels(X_test_orig, y_test_orig)
    
    # Print distribution
    total_up = train_up + val_up + test_up
    total_down = train_down + val_down + test_down
    total_stable = train_stable + val_stable + test_stable
    total_seqs = len(y_train) + len(y_val) + len(y_test)
    
    print(f"   Sequence label distribution:")
    print(f"     DOWN: {total_down:,} ({total_down/total_seqs*100:.1f}%)")
    print(f"     STABLE: {total_stable:,} ({total_stable/total_seqs*100:.1f}%)")
    print(f"     UP: {total_up:,} ({total_up/total_seqs*100:.1f}%)")
    
    # Debug gene info structure
    gene_names = dataset['gene_info']['gene_name'].tolist()
    
    print(f"   Training: {X_train.shape[0]:,} sequences")
    print(f"   Validation: {X_val.shape[0]:,} sequences")  
    print(f"   Test: {X_test.shape[0]:,} sequences")
    print(f"   Original features: {X_train.shape[1]:,} genes")
    
    # 3. Feature selection
    print("\\n3. Applying feature selection...")
    feature_selector = FeatureSelector()
    X_train_selected, selected_genes, selected_indices = feature_selector.select_features(
        X_train, gene_names, top_k=5000)
    X_val_selected = X_val[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    
    print(f"   Reduced features: {X_train_selected.shape[1]:,} genes")
    
    # 4. Prepare data loaders
    print("\\n4. Preparing data loaders...")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_selected)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_selected)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test_selected)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 5. Initialize LSTM Trajectory Model
    print("\\n5. Initializing LSTM Trajectory Model...")
    
    model = LSTMTrajectoryModel(
        input_size=X_train_selected.shape[1],
        hidden_size=256,
        num_layers=2,
        dropout=0.5
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"   Architecture: Input Projection + LSTM + Classifier")
    print(f"   Hidden size: 256, Layers: 2, Dropout: 0.5")
    print(f"   Input projection: {X_train_selected.shape[1]} → 1024 → 512")
    
    # 6. Setup training with regularization
    print("\\n6. Setting up regularized training...")
    
    # Compute class weights for imbalanced data
    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    
    # Create full class weight tensor
    full_class_weights = np.ones(3)
    for i, cls in enumerate(unique_classes):
        full_class_weights[cls] = class_weights[i]
    
    class_weights = torch.FloatTensor(full_class_weights).to(device)
    print(f"   Class distribution in training: {np.bincount(y_train)}")
    print(f"   Class weights: DOWN={class_weights[0]:.2f}, STABLE={class_weights[1]:.2f}, UP={class_weights[2]:.2f}")
    
    # Loss functions
    focal_criterion = FocalLoss(alpha=1, gamma=2)
    weighted_criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    
    # Early stopping parameters
    best_val_loss = float('inf')
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    best_model_state = None
    
    # 7. Training loop with regularization
    print("\\n7. Training LSTM Trajectory Model...")
    
    num_epochs = 25
    results = {
        'train_losses': [],
        'val_losses': [],
        'val_accuracies': [],
        'learning_rates': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            output = model(data)
            
            # Combined loss: Focal + Weighted CrossEntropy
            loss1 = focal_criterion(output, target)
            loss2 = weighted_criterion(output, target)
            loss = 0.7 * loss1 + 0.3 * loss2
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Collect predictions
            pred = output.argmax(dim=1)
            train_preds.extend(pred.cpu().numpy())
            train_targets.extend(target.cpu().numpy())
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                loss = weighted_criterion(output, target)
                val_loss += loss.item()
                
                pred = output.argmax(dim=1)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = np.mean(np.array(val_preds) == np.array(val_targets))
        
        # Calculate recall for UP and DOWN classes
        val_targets_np = np.array(val_targets)
        val_preds_np = np.array(val_preds)
        
        up_recall = np.sum((val_targets_np == 2) & (val_preds_np == 2)) / max(np.sum(val_targets_np == 2), 1)
        down_recall = np.sum((val_targets_np == 0) & (val_preds_np == 0)) / max(np.sum(val_targets_np == 0), 1)
        
        # Store results
        results['train_losses'].append(avg_train_loss)
        results['val_losses'].append(avg_val_loss)
        results['val_accuracies'].append(val_acc)
        results['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"           UP Recall: {up_recall:.3f}, DOWN Recall: {down_recall:.3f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\\n🛑 Early stopping triggered at epoch {epoch+1}")
            print(f"   Best validation loss: {best_val_loss:.4f}")
            print(f"   Best validation accuracy: {best_val_acc:.4f}")
            break
    
    # 8. Load best model and final evaluation
    print("\\n8. Final evaluation on test set...")
    model.load_state_dict(best_model_state)
    model.eval()
    
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            pred = output.argmax(dim=1)
            
            test_preds.extend(pred.cpu().numpy())
            test_targets.extend(target.cpu().numpy())
    
    test_acc = np.mean(np.array(test_preds) == np.array(test_targets))
    
    # Generate detailed results
    print("\\n" + "="*70)
    print("🎯 LSTM TRAJECTORY MODEL RESULTS")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Model Parameters: {total_params:,}")
    print(f"Feature Reduction: {len(gene_names):,} → {X_train_selected.shape[1]:,} ({(1-X_train_selected.shape[1]/len(gene_names))*100:.1f}% reduction)")
    print(f"Architecture: LSTM Trajectory Model (from Ensemble)")
    
    # Classification report
    class_names = ['DOWN', 'STABLE', 'UP']
    report = classification_report(test_targets, test_preds, target_names=class_names, digits=2)
    print(f"\\n📊 Classification Report:")
    print(report)
    
    # Save results
    results['final_test_accuracy'] = test_acc
    results['final_val_accuracy'] = best_val_acc
    results['model_parameters'] = total_params
    results['feature_reduction'] = f"{len(gene_names)} → {X_train_selected.shape[1]}"
    results['classification_report'] = report
    results['architecture'] = "LSTM Trajectory Model (Input Projection + LSTM + Classifier)"
    
    # Save model and results
    model_path = "lstm_trajectory_model.pth"
    results_path = "lstm_trajectory_model_results.json"
    
    torch.save(model.state_dict(), model_path)
    
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_json = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in results.items()}
        json.dump(results_json, f, indent=2)
    
    print(f"\\n💾 Results saved to {results_path}")
    print(f"💾 Model saved to {model_path}")
    
    return model, results

if __name__ == "__main__":
    model, results = train_lstm_trajectory_model()
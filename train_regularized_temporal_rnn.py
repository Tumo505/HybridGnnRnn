"""
Regularized Temporal RNN Training for Cardiac Gene Expression
============================================================
Enhanced training script incorporating all regularization improvements:
- Early stopping
- Increased dropout
- Weight decay
- Feature selection
- Learning rate scheduling
- Focal loss for class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from real_cardiac_temporal_processor import RealCardiacTemporalDataset
import os
from datetime import datetime

class FocalLoss(nn.Module):
    """Focal Loss for addressing severe class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class RegularizedCardiacRNN(nn.Module):
    """
    Regularized Temporal RNN with enhanced dropout and feature selection.
    """
    
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.5):
        super(RegularizedCardiacRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Enhanced input processing with stronger regularization
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM with increased dropout
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Enhanced classifier with stronger regularization
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)  # UP, DOWN, STABLE
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project input
        x_proj = self.input_projection(x)
        
        # LSTM processing (treating each sample as a sequence of length 1)
        x_seq = x_proj.unsqueeze(1)
        lstm_out, _ = self.lstm(x_seq)
        lstm_out = lstm_out.squeeze(1)
        
        # Classification
        logits = self.classifier(lstm_out)
        
        return logits

class FeatureSelector:
    """
    Biological feature selection for cardiac gene expression.
    """
    
    def __init__(self):
        # Cardiac-specific gene sets
        self.cardiac_pathways = {
            'contractility': ['TNNT2', 'MYH6', 'MYH7', 'ACTC1', 'TPM1', 'MYBPC3'],
            'calcium_handling': ['CACNA1C', 'RYR2', 'ATP2A2', 'PLN'],
            'electrical': ['SCN5A', 'KCNH2', 'KCNQ1', 'KCNJ2'],
            'metabolism': ['PPARA', 'PPARGC1A', 'TFAM'],
            'development': ['GATA4', 'NKX2-5', 'TBX5', 'MEF2C'],
            'stress_response': ['NPPA', 'NPPB', 'CTGF'],
            'apoptosis': ['BCL2', 'BAX', 'CASP3']
        }
    
    def select_features(self, X, gene_names, top_k=5000):
        """
        Select top k features using variance and cardiac gene prioritization.
        """
        if X.shape[1] <= top_k:
            return X, gene_names, list(range(X.shape[1]))
        
        print(f"   Selecting top {top_k} features from {X.shape[1]} features...")
        
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            data_np = X.cpu().numpy()
        else:
            data_np = X
        
        # Compute feature importance scores
        feature_scores = np.zeros(data_np.shape[1])
        
        # 1. Variance-based scoring
        variances = np.var(data_np, axis=0)
        variance_scores = (variances - variances.min()) / (variances.max() - variances.min() + 1e-8)
        
        # 2. Cardiac gene prioritization
        cardiac_priority = np.zeros(len(gene_names))
        all_cardiac_genes = set()
        for pathway_genes in self.cardiac_pathways.values():
            all_cardiac_genes.update(pathway_genes)
        
        for i, gene in enumerate(gene_names):
            if gene in all_cardiac_genes:
                cardiac_priority[i] = 1.0
        
        # 3. Combined scoring (70% variance, 30% cardiac priority)
        feature_scores = 0.7 * variance_scores + 0.3 * cardiac_priority
        
        # Get top k indices
        top_indices = np.argsort(feature_scores)[-top_k:]
        
        # Select features
        if isinstance(X, torch.Tensor):
            selected_data = X[:, top_indices]
        else:
            selected_data = X[:, top_indices]
        
        selected_gene_names = [gene_names[i] for i in top_indices]
        
        cardiac_genes_selected = len([g for g in selected_gene_names if g in all_cardiac_genes])
        print(f"   Selected {len(top_indices)} features ({cardiac_genes_selected} cardiac genes)")
        
        return selected_data, selected_gene_names, top_indices

def create_sequence_trajectory_labels(train_data, val_data, test_data, up_threshold=1.2, down_threshold=0.8):
    """
    Convert temporal sequences to trajectory direction labels.
    Returns one label per sequence based on overall gene expression changes.
    
    Args:
        train_data, val_data, test_data: Lists of (X, y) sequence tuples
        up_threshold: Fold-change threshold for UP class
        down_threshold: Fold-change threshold for DOWN class
    
    Returns:
        labels: 0=DOWN, 1=STABLE, 2=UP (one per sequence)
    """
    def process_sequences(sequences):
        labels = []
        up_count = down_count = stable_count = 0
        
        for X, y in sequences:
            # Calculate fold changes for this sequence
            epsilon = 1e-8
            fold_changes = (y + epsilon) / (X + epsilon)
            
            # Count genes in each direction
            up_genes = np.sum(fold_changes > up_threshold)
            down_genes = np.sum(fold_changes < down_threshold)
            total_genes = len(fold_changes)
            
            # Classify sequence based on predominant direction
            up_ratio = up_genes / total_genes
            down_ratio = down_genes / total_genes
            
            if up_ratio > down_ratio and up_ratio > 0.05:  # At least 5% genes up
                label = 2  # UP trajectory
                up_count += 1
            elif down_ratio > up_ratio and down_ratio > 0.05:  # At least 5% genes down
                label = 0  # DOWN trajectory
                down_count += 1
            else:
                label = 1  # STABLE trajectory
                stable_count += 1
            
            labels.append(label)
        
        return np.array(labels), up_count, down_count, stable_count
    
    # Process all datasets
    train_labels, train_up, train_down, train_stable = process_sequences(train_data)
    val_labels, val_up, val_down, val_stable = process_sequences(val_data)
    test_labels, test_up, test_down, test_stable = process_sequences(test_data)
    
    # Print distribution
    total_up = train_up + val_up + test_up
    total_down = train_down + val_down + test_down
    total_stable = train_stable + val_stable + test_stable
    total_seqs = len(train_labels) + len(val_labels) + len(test_labels)
    
    print(f"   Sequence label distribution:")
    print(f"     DOWN: {total_down:,} ({total_down/total_seqs*100:.1f}%)")
    print(f"     STABLE: {total_stable:,} ({total_stable/total_seqs*100:.1f}%)")
    print(f"     UP: {total_up:,} ({total_up/total_seqs*100:.1f}%)")
    
    return train_labels, val_labels, test_labels

def create_trajectory_labels(X, y, up_threshold=1.2, down_threshold=0.8):
    """
    Convert expression values to trajectory direction labels.
    
    Args:
        X: Current timepoint expression
        y: Next timepoint expression
        up_threshold: Fold-change threshold for UP class
        down_threshold: Fold-change threshold for DOWN class
    
    Returns:
        labels: 0=DOWN, 1=STABLE, 2=UP
    """
    epsilon = 1e-8
    fold_changes = (y + epsilon) / (X + epsilon)
    
    labels = np.ones_like(fold_changes, dtype=np.long)  # Default: STABLE
    labels[fold_changes > up_threshold] = 2  # UP
    labels[fold_changes < down_threshold] = 0  # DOWN
    
    return labels

def train_regularized_temporal_rnn():
    """
    Main training function with all regularization improvements.
    """
    
    print("🚀 Regularized Temporal RNN Training")
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
    
    # Convert to trajectory labels
    print("\\n2. Converting to trajectory direction labels...")
    
    # Use the processed X and y to create sequence-level trajectory labels
    # Each X is current expression, each y is next expression
    X_train, X_val, X_test = dataset['X_train'], dataset['X_val'], dataset['X_test']
    y_train_reg, y_val_reg, y_test_reg = dataset['y_train'], dataset['y_val'], dataset['y_test']
    
    # Denormalize for proper fold-change calculation
    X_train_orig = dataset['scaler'].inverse_transform(X_train)
    X_val_orig = dataset['scaler'].inverse_transform(X_val)
    X_test_orig = dataset['scaler'].inverse_transform(X_test)
    
    y_train_orig = dataset['target_scaler'].inverse_transform(y_train_reg)
    y_val_orig = dataset['target_scaler'].inverse_transform(y_val_reg)
    y_test_orig = dataset['target_scaler'].inverse_transform(y_test_reg)
    
    # Create sequence-level labels by analyzing per-sequence gene expression changes
    def create_sequence_labels(X_orig, y_orig, up_threshold=1.05, down_threshold=0.95):
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
            
            # Classify sequence based on predominant direction (more lenient thresholds)
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
    print(f"   Gene info columns: {dataset['gene_info'].columns.tolist()}")
    gene_names = dataset['gene_info']['gene_name'].tolist()  # Use gene_name column
    
    print(f"   Training: {X_train.shape[0]:,} sequences")
    print(f"   Validation: {X_val.shape[0]:,} sequences")  
    print(f"   Test: {X_test.shape[0]:,} sequences")
    print(f"   Original features: {X_train.shape[1]:,} genes")    # 3. Feature selection
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
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # 5. Initialize model with regularization
    print("\\n5. Initializing regularized model...")
    input_size = X_train_selected.shape[1]
    model = RegularizedCardiacRNN(input_size=input_size, dropout=0.5).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # 6. Setup training with regularization
    print("\\n6. Setting up regularized training...")
    
    # Compute class weights for imbalanced data
    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    
    # Create full class weight tensor (ensure all 3 classes are represented)
    full_class_weights = np.ones(3)  # Default weight of 1 for missing classes
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
    print("\\n7. Training with regularization...")
    
    num_epochs = 25
    history = {
        'train_losses': [],
        'val_losses': [],
        'val_accuracies': [],
        'learning_rates': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(batch_x)
            
            # Combined loss (focal + weighted)
            focal_loss = focal_criterion(outputs, batch_y)
            weighted_loss = weighted_criterion(outputs, batch_y)
            loss = 0.7 * focal_loss + 0.3 * weighted_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / num_batches
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                loss = weighted_criterion(outputs, batch_y)
                val_loss += loss.item()
                
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Per-class metrics
        report = classification_report(all_labels, all_preds, 
                                     target_names=['DOWN', 'STABLE', 'UP'],
                                     output_dict=True, zero_division=0)
        
        up_recall = report['UP']['recall']
        down_recall = report['DOWN']['recall']
        
        # Save history
        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(avg_val_loss)
        history['val_accuracies'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\\n🛑 Early stopping triggered at epoch {epoch+1}")
                print(f"   Best validation loss: {best_val_loss:.4f}")
                print(f"   Best validation accuracy: {best_val_acc:.4f}")
                break
        
        # Progress reporting
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"           UP Recall: {up_recall:.3f}, DOWN Recall: {down_recall:.3f}, LR: {current_lr:.2e}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # 8. Final evaluation
    print("\\n8. Final evaluation on test set...")
    model.eval()
    test_preds = []
    test_labels_list = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            predictions = torch.argmax(outputs, dim=1)
            
            test_preds.extend(predictions.cpu().numpy())
            test_labels_list.extend(batch_y.cpu().numpy())
    
    test_accuracy = accuracy_score(test_labels_list, test_preds)
    
    print("\\n" + "="*70)
    print("🎯 REGULARIZED TEMPORAL RNN RESULTS")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Model Parameters: {total_params:,}")
    print(f"Feature Reduction: {X_train.shape[1]:,} → {X_train_selected.shape[1]:,} ({(1-X_train_selected.shape[1]/X_train.shape[1])*100:.1f}% reduction)")
    
    # Detailed classification report
    report = classification_report(test_labels_list, test_preds, 
                                 target_names=['DOWN', 'STABLE', 'UP'],
                                 output_dict=True)
    
    print(f"\\n📊 Classification Report:")
    print(classification_report(test_labels_list, test_preds, 
                              target_names=['DOWN', 'STABLE', 'UP']))
    
    # Save results
    results = {
        'test_accuracy': float(test_accuracy),
        'best_val_accuracy': float(best_val_acc),
        'total_parameters': int(total_params),
        'feature_reduction': {
            'original_features': int(X_train.shape[1]),
            'selected_features': int(X_train_selected.shape[1]),
            'reduction_percent': float((1-X_train_selected.shape[1]/X_train.shape[1])*100)
        },
        'regularization_config': {
            'dropout': 0.5,
            'weight_decay': 1e-4,
            'early_stopping_patience': patience,
            'focal_loss_gamma': 2,
            'feature_selection_top_k': 5000
        },
        'training_history': history,
        'classification_report': report,
        'training_timestamp': datetime.now().isoformat()
    }
    
    # Save model and results
    torch.save(model.state_dict(), 'regularized_cardiac_temporal_rnn.pth')
    with open('regularized_temporal_rnn_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\n💾 Results saved to regularized_temporal_rnn_results.json")
    print(f"💾 Model saved to regularized_cardiac_temporal_rnn.pth")
    
    return model, results

if __name__ == "__main__":
    model, results = train_regularized_temporal_rnn()
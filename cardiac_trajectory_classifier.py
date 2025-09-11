"""
Final Cardiac Gene Expression Trajectory Classifier
==================================================
Clean implementation of the successful direction prediction approach.
Predicts UP/DOWN/STABLE changes in gene expression over time.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from real_cardiac_temporal_processor import RealCardiacTemporalDataset
import os
from datetime import datetime

class CardiacTrajectoryClassifier(nn.Module):
    """
    Neural network to predict gene expression trajectory directions.
    Classifies each gene as UP/DOWN/STABLE between timepoints.
    """
    
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.3):
        super(CardiacTrajectoryClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input processing layers
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM for temporal pattern recognition
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Classification head - predicts 3 classes for each gene
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, input_size * 3)  # 3 classes per gene
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project input to lower dimension
        x_proj = self.input_projection(x)
        
        # Add sequence dimension for LSTM
        x_seq = x_proj.unsqueeze(1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x_seq)
        lstm_out = lstm_out.squeeze(1)
        
        # Generate predictions for each gene
        logits = self.classifier(lstm_out)
        
        # Reshape to (batch_size, input_size, 3)
        logits = logits.view(batch_size, self.input_size, 3)
        
        return logits

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

def train_model():
    """
    Train the cardiac trajectory classifier.
    """
    
    print("="*70)
    print("🧬 CARDIAC GENE EXPRESSION TRAJECTORY CLASSIFIER")
    print("="*70)
    print(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load GSE175634 dataset
    print("\n1. Loading GSE175634 temporal cardiac dataset...")
    data_dir = r"c:\Users\tumok\Documents\Projects\HybridGnnRnn\data\GSE175634_temporal_data"
    processor = RealCardiacTemporalDataset(data_dir)
    dataset = processor.prepare_dataset_for_training()
    
    if not dataset['validation_passed']:
        print("❌ Dataset failed biological validation!")
        return None, None
    
    print("✅ Dataset validation passed")
    print(f"   Training samples: {dataset['X_train'].shape[0]:,}")
    print(f"   Genes: {dataset['X_train'].shape[1]:,}")
    print(f"   Cardiac markers: {len(dataset['cardiac_markers'])}")
    
    # Convert to trajectory labels
    print("\n2. Converting to trajectory direction labels...")
    
    # Denormalize for fold-change calculation
    X_train_orig = dataset['scaler'].inverse_transform(dataset['X_train'])
    X_val_orig = dataset['scaler'].inverse_transform(dataset['X_val'])
    X_test_orig = dataset['scaler'].inverse_transform(dataset['X_test'])
    
    y_train_orig = dataset['target_scaler'].inverse_transform(dataset['y_train'])
    y_val_orig = dataset['target_scaler'].inverse_transform(dataset['y_val'])
    y_test_orig = dataset['target_scaler'].inverse_transform(dataset['y_test'])
    
    # Create labels
    train_labels = create_trajectory_labels(X_train_orig, y_train_orig)
    val_labels = create_trajectory_labels(X_val_orig, y_val_orig)
    test_labels = create_trajectory_labels(X_test_orig, y_test_orig)
    
    # Analyze distribution
    unique, counts = np.unique(train_labels, return_counts=True)
    class_names = ['DOWN', 'STABLE', 'UP']
    print("   Label distribution:")
    for label, count in zip(unique, counts):
        print(f"     {class_names[label]}: {count:,} ({count/train_labels.size*100:.1f}%)")
    
    # Create data loaders
    print("\n3. Creating data loaders...")
    batch_size = 32
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(dataset['X_train']), torch.LongTensor(train_labels)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(dataset['X_val']), torch.LongTensor(val_labels)),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(dataset['X_test']), torch.LongTensor(test_labels)),
        batch_size=batch_size, shuffle=False
    )
    
    # Initialize model
    print("\n4. Initializing model...")
    input_size = dataset['X_train'].shape[1]
    model = CardiacTrajectoryClassifier(
        input_size=input_size,
        hidden_size=256,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    print(f"   Genes: {input_size:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # Training loop
    epochs = 30
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    history = {
        'train_losses': [],
        'val_losses': [], 
        'val_accuracies': []
    }
    
    print(f"\n5. Training for {epochs} epochs...")
    print("="*70)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1:2d}"):
            batch_x, batch_labels = batch_x.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            
            # Flatten for loss computation
            logits_flat = logits.view(-1, 3)
            labels_flat = batch_labels.view(-1)
            
            loss = criterion(logits_flat, labels_flat)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_labels in val_loader:
                batch_x, batch_labels = batch_x.to(device), batch_labels.to(device)
                logits = model(batch_x)
                
                logits_flat = logits.view(-1, 3)
                labels_flat = batch_labels.view(-1)
                
                loss = criterion(logits_flat, labels_flat)
                val_loss += loss.item()
                
                preds = torch.argmax(logits_flat, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_flat.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        
        # Update history
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['val_accuracies'].append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}: Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Early stopping
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), 'cardiac_trajectory_classifier.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏰ Early stopping at epoch {epoch+1}")
                break
    
    print("\n✅ Training completed!")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('cardiac_trajectory_classifier.pth'))
    
    # Final evaluation
    print("\n6. Final evaluation...")
    model.eval()
    test_preds = []
    test_labels_list = []
    
    with torch.no_grad():
        for batch_x, batch_labels in test_loader:
            batch_x, batch_labels = batch_x.to(device), batch_labels.to(device)
            logits = model(batch_x)
            
            preds = torch.argmax(logits.view(-1, 3), dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels_list.extend(batch_labels.view(-1).cpu().numpy())
    
    test_accuracy = accuracy_score(test_labels_list, test_preds)
    
    print("\n" + "="*70)
    print("🎯 FINAL RESULTS")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Model Parameters: {total_params:,}")
    
    # Detailed classification report
    report = classification_report(test_labels_list, test_preds, 
                                 target_names=['DOWN', 'STABLE', 'UP'],
                                 output_dict=True)
    
    print(f"\n📊 Classification Report:")
    print(classification_report(test_labels_list, test_preds, 
                              target_names=['DOWN', 'STABLE', 'UP']))
    
    # Confusion matrix
    cm = confusion_matrix(test_labels_list, test_preds)
    print(f"\n📈 Confusion Matrix:")
    print("              Predicted")
    print("             DOWN  STABLE    UP")
    for i, actual in enumerate(['DOWN', 'STABLE', 'UP']):
        print(f"Actual {actual:6s} {cm[i,0]:6,} {cm[i,1]:7,} {cm[i,2]:6,}")
    
    # Save results
    results = {
        'test_accuracy': float(test_accuracy),
        'best_val_accuracy': float(best_val_acc),
        'total_parameters': int(total_params),
        'training_history': history,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'dataset_info': {
            'total_samples': int(dataset['X_train'].shape[0] + dataset['X_val'].shape[0] + dataset['X_test'].shape[0]),
            'features': int(dataset['X_train'].shape[1]),
            'cardiac_markers': len(dataset['cardiac_markers']),
            'validation_passed': bool(dataset['validation_passed'])
        },
        'training_timestamp': datetime.now().isoformat()
    }
    
    with open('cardiac_trajectory_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to cardiac_trajectory_results.json")
    print(f"💾 Model saved to cardiac_trajectory_classifier.pth")
    
    return model, results

def create_visualizations(results):
    """
    Create performance visualization plots.
    """
    
    print(f"\n7. Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training curves
    history = results['training_history']
    
    axes[0,0].plot(history['train_losses'], label='Train Loss', color='blue')
    axes[0,0].plot(history['val_losses'], label='Val Loss', color='orange')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Training Progress')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Validation accuracy
    axes[0,1].plot(history['val_accuracies'], label='Val Accuracy', color='green')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].set_title('Validation Accuracy')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Confusion matrix
    cm = np.array(results['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['DOWN', 'STABLE', 'UP'],
                yticklabels=['DOWN', 'STABLE', 'UP'],
                ax=axes[1,0])
    axes[1,0].set_title('Confusion Matrix')
    axes[1,0].set_ylabel('True')
    axes[1,0].set_xlabel('Predicted')
    
    # Performance metrics
    report = results['classification_report']
    classes = ['DOWN', 'STABLE', 'UP']
    f1_scores = [report[cls]['f1-score'] for cls in classes]
    
    bars = axes[1,1].bar(classes, f1_scores, color=['red', 'blue', 'green'], alpha=0.7)
    axes[1,1].set_ylabel('F1-Score')
    axes[1,1].set_title('Per-Class Performance')
    axes[1,1].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('cardiac_trajectory_classifier_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   📊 Visualizations saved to cardiac_trajectory_classifier_results.png")

def main():
    """
    Complete training pipeline for cardiac trajectory classifier.
    """
    
    print("🚀 Starting cardiac gene expression trajectory classification training...")
    
    # Train model
    model, results = train_model()
    
    if model is not None and results is not None:
        # Create visualizations
        create_visualizations(results)
        
        print(f"\n" + "="*70)
        print("🏁 TRAINING COMPLETE")
        print("="*70)
        print(f"✅ Successfully trained cardiac trajectory classifier")
        print(f"✅ Achieved {results['test_accuracy']*100:.2f}% test accuracy")
        print(f"✅ Model ready for biological applications")
        
        return model, results
    else:
        print("❌ Training failed!")
        return None, None

if __name__ == "__main__":
    model, results = main()

"""
Ready-to-Run Implementation: Ensemble + Biological Features
==========================================================
Complete implementation of advanced cardiac trajectory prediction.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from real_cardiac_temporal_processor import RealCardiacTemporalDataset
from enhanced_cardiac_ensemble import (
    CardiacBiologyFeatureEngineer, 
    CardiacLSTMModel, 
    CardiacAttentionModel, 
    CardiacCNNModel,
    FocalLoss,
    train_model,
    evaluate_model
)
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def main():
    """Main training pipeline with biological ensemble"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load data
    print("Loading cardiac temporal dataset...")
    data_dir = r"data\GSE175634_temporal_data"
    processor = RealCardiacTemporalDataset(data_dir)
    
    # Load all components
    processor.load_metadata()
    processor.load_genes()
    processor.load_expression_data()
    
    # Create temporal sequences
    sequences, labels, timepoint_info = processor.create_temporal_sequences()
    
    # Prepare train/val/test splits
    dataset_info = processor.prepare_dataset_for_training()
    X_train, y_train = dataset_info['X_train'], dataset_info['y_train']
    X_val, y_val = dataset_info['X_val'], dataset_info['y_val']
    X_test, y_test = dataset_info['X_test'], dataset_info['y_test']
    
    gene_names = processor.gene_info['gene_name'].values
    
    print(f"Training data: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Validation data: {X_val.shape}, Labels: {y_val.shape}")
    print(f"Test data: {X_test.shape}, Labels: {y_test.shape}")
    
    # 2. Initialize biological feature engineer
    print("Initializing biological feature engineering...")
    bio_engineer = CardiacBiologyFeatureEngineer(gene_names)
    
    # 3. Engineer biological features
    print("Engineering biological features...")
    X_train_bio = bio_engineer.engineer_features(X_train)
    X_val_bio = bio_engineer.engineer_features(X_val)
    X_test_bio = bio_engineer.engineer_features(X_test)
    
    input_size = X_train_bio.shape[1]
    print(f"Enhanced feature size: {input_size}")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_bio).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val_bio).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    X_test_tensor = torch.FloatTensor(X_test_bio).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # 4. Initialize ensemble models
    print("Creating ensemble models...")
    models = {
        'lstm': CardiacLSTMModel(input_size).to(device),
        'attention': CardiacAttentionModel(input_size).to(device),
        'cnn': CardiacCNNModel(input_size).to(device)
    }
    
    # 5. Train each model
    trained_models = {}
    model_predictions = {}
    
    for name, model in models.items():
        print(f"\nTraining {name.upper()} model...")
        
        # Train model
        trained_model = train_model(
            model, train_loader, val_loader, 
            epochs=25, device=device, model_name=name
        )
        trained_models[name] = trained_model
        
        # Get predictions
        predictions, true_labels = evaluate_model(trained_model, test_loader, device)
        model_predictions[name] = predictions
        
        # Individual model performance
        accuracy = (predictions == true_labels).mean()
        print(f"{name.upper()} Test Accuracy: {accuracy:.4f}")
    
    # 6. Ensemble prediction
    print("\nCreating ensemble predictions...")
    ensemble_pred = np.zeros((len(y_test), 3))
    
    for name, pred in model_predictions.items():
        # Convert predictions to probabilities
        pred_probs = np.eye(3)[pred]  # One-hot encoding
        ensemble_pred += pred_probs
    
    # Average ensemble
    ensemble_pred = ensemble_pred / len(model_predictions)
    final_predictions = np.argmax(ensemble_pred, axis=1)
    
    # 7. Evaluate ensemble
    ensemble_accuracy = (final_predictions == y_test).mean()
    print(f"\nEnsemble Test Accuracy: {ensemble_accuracy:.4f}")
    
    # Detailed classification report
    class_names = ['DOWN', 'STABLE', 'UP']
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, final_predictions, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, final_predictions)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Class-specific metrics
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(y_test, final_predictions)
    
    print("\nClass-Specific Results:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}")
    
    # 8. Save models
    print("\nSaving ensemble models...")
    for name, model in trained_models.items():
        torch.save(model.state_dict(), f'enhanced_cardiac_{name}_model.pth')
    
    # Save ensemble results
    results = {
        'ensemble_accuracy': ensemble_accuracy,
        'individual_accuracies': {name: (pred == y_test).mean() for name, pred in model_predictions.items()},
        'class_metrics': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        },
        'confusion_matrix': cm
    }
    
    np.save('enhanced_ensemble_results.npy', results)
    print("Enhanced ensemble training complete!")

if __name__ == "__main__":
    main()

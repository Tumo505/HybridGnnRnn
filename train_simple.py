"""
Simple training script focused on getting the models to train with real heart data.
Simplified configuration without M1 optimizations.
"""

import logging
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
import sys
import os
from datetime import datetime
import wandb

# Add src to path
sys.path.append('src')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleGNNModel(nn.Module):
    """Simplified GNN model for heart data."""
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=3):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, batch):
        # GNN layers
        x = torch.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = torch.relu(self.conv3(x, edge_index))
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        out = self.classifier(x)
        return out

def load_and_check_data():
    """Load our processed heart data and check dimensions."""
    data_path = "processed_heart_data/cardiomyocyte_datasets.pt"
    
    if not os.path.exists(data_path):
        logger.error(f"Data not found at {data_path}. Please run train_production.py first.")
        return None, None
    
    # Load data
    processed_data = torch.load(data_path, weights_only=False)
    
    # Extract spatial data
    spatial_data = []
    for key, value in processed_data.items():
        if key.startswith('spatial_') and hasattr(value, 'x'):
            spatial_data.append(value)
    
    logger.info(f"Found {len(spatial_data)} spatial samples")
    
    if len(spatial_data) > 0:
        sample = spatial_data[0]
        logger.info(f"Sample node features shape: {sample.x.shape}")
        logger.info(f"Sample edges shape: {sample.edge_index.shape}")
        input_dim = sample.x.shape[1]
        logger.info(f"Input dimension: {input_dim}")
        return spatial_data, input_dim
    
    return None, None

def create_simple_dataloaders(spatial_data, train_split=0.7, val_split=0.2):
    """Create simple train/val/test dataloaders."""
    n_samples = len(spatial_data)
    indices = np.random.permutation(n_samples)
    
    train_end = int(train_split * n_samples)
    val_end = train_end + int(val_split * n_samples)
    
    train_data = [spatial_data[i] for i in indices[:train_end]]
    val_data = [spatial_data[i] for i in indices[train_end:val_end]]
    test_data = [spatial_data[i] for i in indices[val_end:]]
    
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False)
    
    logger.info(f"Train samples: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_loader, val_loader, test_loader

def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch.x, batch.edge_index, batch.batch)
        
        # Use dummy labels for now (in real training, use actual labels)
        labels = torch.randint(0, 3, (out.shape[0],)).to(device)
        loss = criterion(out, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * out.shape[0]
        total_samples += out.shape[0]
    
    return total_loss / total_samples

def validate_epoch(model, loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    total_samples = 0
    correct = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            
            # Use dummy labels for now
            labels = torch.randint(0, 3, (out.shape[0],)).to(device)
            loss = criterion(out, labels)
            
            total_loss += loss.item() * out.shape[0]
            total_samples += out.shape[0]
            
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
    
    return total_loss / total_samples, correct / total_samples

def main():
    """Main training function."""
    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    spatial_data, input_dim = load_and_check_data()
    if spatial_data is None:
        return
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_simple_dataloaders(spatial_data)
    
    # Create model
    model = SimpleGNNModel(input_dim=input_dim, hidden_dim=128, output_dim=3)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize wandb
    wandb.init(
        project="HeartGNN-Simple",
        config={
            "model": "SimpleGNN",
            "input_dim": input_dim,
            "hidden_dim": 128,
            "learning_rate": 0.001,
            "batch_size": 8,
            "device": str(device)
        }
    )
    
    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')
    
    logger.info("🚀 Starting training...")
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Log
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_simple_model.pth")
            logger.info(f"✅ New best model saved! Val Loss: {val_loss:.4f}")
    
    # Final test
    test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
    logger.info(f"🎯 Final Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_acc
    })
    
    wandb.finish()
    logger.info("✅ Training completed!")

if __name__ == "__main__":
    main()

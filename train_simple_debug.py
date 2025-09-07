#!/usr/bin/env python3
"""
Simplified Scalable RNN Training - Debug Version
Focus on getting the basic model working correctly
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import logging
import sys
import os

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from comprehensive_cardiac_data_processor import ComprehensiveCardiacDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleCardiacRNN(nn.Module):
    """Simplified RNN for debugging"""
    
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, lengths=None):
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Global average pooling
        if lengths is not None:
            mask = torch.arange(seq_len, device=x.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1).float()
            pooled = (lstm_out * mask).sum(dim=1) / lengths.unsqueeze(-1).float()
        else:
            pooled = torch.mean(lstm_out, dim=1)
        
        # Classification
        output = self.classifier(pooled)
        return output

def test_model():
    """Test the model with synthetic data"""
    logger.info("🧪 Testing model with synthetic data...")
    
    # Create synthetic data
    batch_size = 16
    seq_len = 50
    input_dim = 200
    num_classes = 7
    
    x = torch.randn(batch_size, seq_len, input_dim)
    lengths = torch.randint(20, seq_len, (batch_size,))
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Create model
    model = SimpleCardiacRNN(input_dim, 64, 2, num_classes)
    
    # Test forward pass
    outputs = model(x, lengths)
    logger.info(f"✅ Synthetic test - Input: {x.shape}, Output: {outputs.shape}, Labels: {labels.shape}")
    
    # Test loss calculation
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    logger.info(f"✅ Synthetic test - Loss: {loss.item():.4f}")
    
    return True

def load_real_data():
    """Load real cardiac data"""
    logger.info("📊 Loading real cardiac data...")
    
    processor = ComprehensiveCardiacDataProcessor()
    sequences, labels, names = processor.create_comprehensive_dataset(
        temporal_sequences_per_type=200,  # Small dataset for testing
        spatial_sequences=100
    )
    
    logger.info(f"📊 Data loaded: {len(sequences)} sequences, {len(set(labels))} classes")
    
    # Use the custom dataset class that handles variable-length sequences
    from comprehensive_cardiac_data_processor import ComprehensiveCardiacDataset
    dataset = ComprehensiveCardiacDataset(sequences, labels, names, max_length=100, normalize=True)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    return train_loader, val_loader, len(set(labels))

def train_simple_model():
    """Train the simplified model"""
    logger.info("🚀 Training simplified model...")
    
    # Load data
    train_loader, val_loader, num_classes = load_real_data()
    
    # Get input dimension from first batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch['sequence'].shape[2]  # Use dictionary access
    logger.info(f"📏 Input dimension: {input_dim}, Classes: {num_classes}")
    
    # Create model
    model = SimpleCardiacRNN(input_dim, 64, 2, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    logger.info(f"🏗️ Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            sequences = batch['sequence']
            labels = batch['label'].squeeze()
            lengths = batch['length'].squeeze()
            
            # Forward pass
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            if batch_idx == 0:  # Debug first batch
                logger.info(f"🔍 Batch debug - Sequences: {sequences.shape}, Outputs: {outputs.shape}, Labels: {labels.shape}")
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        
        logger.info(f"📍 Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    logger.info("✅ Training completed successfully!")

def main():
    """Main function"""
    logger.info("🫀 Simplified Cardiac RNN Training")
    logger.info("=" * 50)
    
    # Test with synthetic data first
    if test_model():
        logger.info("✅ Synthetic test passed")
    else:
        logger.error("❌ Synthetic test failed")
        return
    
    # Train with real data
    try:
        train_simple_model()
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

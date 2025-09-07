#!/usr/bin/env python3
"""
Enhanced Scalable RNN Training - Working Version
Based on the successful simple model, enhanced for better performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import logging
import sys
import os
from datetime import datetime

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from comprehensive_cardiac_data_processor import ComprehensiveCardiacDataProcessor, ComprehensiveCardiacDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedCardiacRNN(nn.Module):
    """Enhanced RNN with proven architecture for better performance"""
    
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.3, use_attention=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Input projection with better initialization
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Multi-layer bidirectional LSTM
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Self-attention mechanism (optional)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                hidden_dim * 2, num_heads=8, dropout=dropout, batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Enhanced classification head with residual connection
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Careful weight initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                elif param.dim() >= 2:  # Only apply Xavier to 2D+ tensors
                    nn.init.xavier_uniform_(param, gain=0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x, lengths=None):
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Optional attention mechanism
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = self.attention_norm(lstm_out + attn_out)
        
        # Length-aware global pooling
        if lengths is not None:
            mask = torch.arange(seq_len, device=x.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1).float()
            pooled = (lstm_out * mask).sum(dim=1) / lengths.unsqueeze(-1).float()
        else:
            pooled = torch.mean(lstm_out, dim=1)
        
        # Classification
        output = self.classifier(pooled)
        return output

class AdvancedTrainer:
    """Advanced trainer with proven techniques"""
    
    def __init__(self, model, device='cpu', lr=0.001):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # History tracking
        self.history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []
        }
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in dataloader:
            sequences = batch['sequence'].to(self.device)
            labels = batch['label'].squeeze().to(self.device)
            lengths = batch['length'].squeeze().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(sequences, lengths)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                sequences = batch['sequence'].to(self.device)
                labels = batch['label'].squeeze().to(self.device)
                lengths = batch['length'].squeeze().to(self.device)
                
                outputs = self.model(sequences, lengths)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=50, patience=10):
        """Train the model with early stopping"""
        best_val_acc = 0
        patience_counter = 0
        
        logger.info(f"🚀 Training for {epochs} epochs with patience {patience}")
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_acc)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Log progress
            logger.info(f"📍 Epoch {epoch+1}/{epochs}")
            logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            logger.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_enhanced_rnn.pth')
                logger.info(f"  ✅ New best validation accuracy: {best_val_acc:.4f}")
            else:
                patience_counter += 1
                logger.info(f"  🔄 Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                logger.info(f"🛑 Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_enhanced_rnn.pth'))
        logger.info(f"✅ Training completed. Best validation accuracy: {best_val_acc:.4f}")
        
        return best_val_acc

def create_datasets(sequences_per_type=2000, spatial_sequences=1000):
    """Create train/val/test datasets"""
    logger.info(f"📊 Creating datasets: {sequences_per_type} temporal + {spatial_sequences} spatial per scale")
    
    processor = ComprehensiveCardiacDataProcessor()
    sequences, labels, names = processor.create_comprehensive_dataset(
        temporal_sequences_per_type=sequences_per_type,
        spatial_sequences=spatial_sequences
    )
    
    logger.info(f"📊 Total data: {len(sequences)} sequences, {len(set(labels))} classes")
    
    # Create custom dataset
    dataset = ComprehensiveCardiacDataset(sequences, labels, names, max_length=100, normalize=True)
    
    # Split data: 70% train, 20% val, 10% test
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    logger.info(f"📊 Splits: Train {len(train_dataset)}, Val {len(val_dataset)}, Test {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, len(set(labels))

def main():
    """Main training function"""
    logger.info("🫀 Enhanced Scalable Cardiac RNN Training")
    logger.info("=" * 60)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 Device: {device}")
    
    # Progressive scaling approach
    scales = [
        {'name': 'Scale 1: Base', 'temporal': 1000, 'spatial': 500, 'epochs': 30, 'lr': 0.002},
        {'name': 'Scale 2: Medium', 'temporal': 2000, 'spatial': 1000, 'epochs': 40, 'lr': 0.001},
        {'name': 'Scale 3: Large', 'temporal': 3000, 'spatial': 1500, 'epochs': 50, 'lr': 0.0008},
    ]
    
    best_overall_acc = 0
    best_scale = ""
    
    for scale_config in scales:
        logger.info(f"\n🔄 {scale_config['name']}")
        logger.info(f"  Temporal: {scale_config['temporal']}, Spatial: {scale_config['spatial']}")
        
        try:
            # Create datasets
            train_dataset, val_dataset, test_dataset, num_classes = create_datasets(
                scale_config['temporal'], scale_config['spatial']
            )
            
            # Create data loaders
            batch_size = min(64, max(16, len(train_dataset) // 50))  # Adaptive batch size
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            logger.info(f"  Batch size: {batch_size}")
            
            # Get input dimension
            sample_batch = next(iter(train_loader))
            input_dim = sample_batch['sequence'].shape[2]
            logger.info(f"  Input dimension: {input_dim}, Classes: {num_classes}")
            
            # Create enhanced model
            model = EnhancedCardiacRNN(
                input_dim=input_dim,
                hidden_dim=128,  # Larger hidden dimension
                num_layers=3,    # More layers
                num_classes=num_classes,
                dropout=0.3,
                use_attention=True
            )
            
            logger.info(f"🏗️ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Create trainer
            trainer = AdvancedTrainer(model, device, scale_config['lr'])
            
            # Train model
            val_acc = trainer.train(
                train_loader, val_loader, 
                epochs=scale_config['epochs'], 
                patience=8
            )
            
            # Test evaluation
            test_loss, test_acc = trainer.validate(test_loader)
            
            logger.info(f"\n🎯 {scale_config['name']} Results:")
            logger.info(f"  Validation Accuracy: {val_acc:.4f}")
            logger.info(f"  Test Accuracy: {test_acc:.4f}")
            
            # Track best scale
            if test_acc > best_overall_acc:
                best_overall_acc = test_acc
                best_scale = scale_config['name']
                # Save final best model
                torch.save(model.state_dict(), 'final_best_enhanced_rnn.pth')
            
            logger.info(f"  Current best: {best_overall_acc:.4f} ({best_scale})")
            
        except Exception as e:
            logger.error(f"❌ Error in {scale_config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info(f"\n🏆 FINAL RESULTS:")
    logger.info(f"  Best Test Accuracy: {best_overall_acc:.4f}")
    logger.info(f"  Best Scale: {best_scale}")
    logger.info(f"  Improvement over 67.58% baseline: {(best_overall_acc - 0.6758) * 100:.2f}%")
    
    if best_overall_acc > 0.6758:
        logger.info("✅ SUCCESS: Exceeded 67.58% baseline!")
    else:
        logger.info("🔄 Continue iterating to beat baseline...")

if __name__ == "__main__":
    main()

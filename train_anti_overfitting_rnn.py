#!/usr/bin/env python3
"""
Anti-Overfitting Comprehensive RNN Training
Enhanced regularization and overfitting prevention
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import logging
from pathlib import Path
import time
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from comprehensive_cardiac_data_processor import ComprehensiveCardiacDataProcessor, ComprehensiveCardiacDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('anti_overfitting_rnn_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RegularizedTemporalRNN(nn.Module):
    """Regularized RNN with strong overfitting prevention"""
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=5, dropout=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Reduced complexity - smaller hidden dimension
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Simpler LSTM with heavy dropout
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # Reduced complexity
        )
        
        # Simple attention with regularization
        self.attention_weights = nn.Linear(hidden_dim, 1)
        
        # Regularized classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Batch normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Conservative weight initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param, gain=0.1)  # Small gain
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x, lengths=None):
        batch_size, seq_len, _ = x.shape
        
        # Project input with regularization
        x = self.input_projection(x)  # [batch, seq, hidden]
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # [batch, seq, hidden]
        
        # Apply layer norm
        lstm_out = self.layer_norm(lstm_out)
        
        # Simple attention mechanism
        attention_scores = self.attention_weights(lstm_out)  # [batch, seq, 1]
        
        if lengths is not None:
            # Mask padding
            mask = torch.arange(seq_len, device=x.device)[None, :] < lengths[:, None]
            attention_scores = attention_scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        
        attention_weights = torch.softmax(attention_scores, dim=1)  # [batch, seq, 1]
        
        # Weighted average
        context = torch.sum(lstm_out * attention_weights, dim=1)  # [batch, hidden]
        
        # Classification with heavy regularization
        output = self.classifier(context)
        return output

class AntiOverfittingTrainer:
    """Trainer with aggressive overfitting prevention"""
    
    def __init__(self, model, device='auto', learning_rate=0.0001):  # Much lower LR
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        
        # Conservative optimizer settings
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.1,  # Strong L2 regularization
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # Label smoothing for regularization
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Aggressive learning rate scheduling
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.3, patience=3
        )
        
        # Early stopping with patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = 8  # Aggressive early stopping
        
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        logger.info(f"🔧 Using device: {self.device}")
        logger.info(f"🏗️ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"📊 Aggressive overfitting prevention enabled")
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(dataloader):
            sequences = batch['sequence'].to(self.device)
            labels = batch['label'].squeeze().to(self.device)
            lengths = batch['length'].squeeze().to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(sequences, lengths)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            if batch_idx % 100 == 0:
                logger.info(f"  Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")
        
        return total_loss / len(dataloader), total_correct / total_samples
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                sequences = batch['sequence'].to(self.device)
                labels = batch['label'].squeeze().to(self.device)
                lengths = batch['length'].squeeze().to(self.device)
                
                outputs = self.model(sequences, lengths)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy, all_predictions, all_labels
    
    def train(self, train_loader, val_loader, epochs=30):  # Fewer epochs
        logger.info(f"🚀 Anti-overfitting training for {epochs} epochs...")
        
        best_val_acc = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            logger.info(f"\n📍 Epoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, _, _ = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Calculate train-val gap for overfitting detection
            acc_gap = train_acc - val_acc
            loss_gap = val_loss - train_loss
            
            logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            logger.info(f"  📊 Acc Gap: {acc_gap:.4f}, Loss Gap: {loss_gap:.4f}")
            
            # Overfitting warning
            if acc_gap > 0.15 or loss_gap > 0.5:
                logger.warning(f"⚠️ Overfitting detected! Gap too large")
            
            # Early stopping based on validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                checkpoint_dir = Path("checkpoints") / f"regularized_rnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'history': self.history
                }, checkpoint_dir / "best_model.pt")
                
                logger.info(f"  🎯 New best validation loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                logger.info(f"  Patience: {self.patience_counter}/{self.early_stopping_patience}")
                
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"🛑 Early stopping: validation loss not improving")
                    break
        
        training_time = time.time() - start_time
        logger.info(f"\n✅ Training completed in {training_time:.2f} seconds")
        logger.info(f"🏆 Best validation loss: {self.best_val_loss:.4f}")
        
        return self.best_val_loss

def main():
    """Main training function with overfitting prevention"""
    logger.info("🫀 Anti-Overfitting Cardiac RNN Training")
    logger.info("=" * 60)
    
    # Load dataset with smaller size for overfitting prevention
    logger.info("📂 Loading cardiac dataset (reduced size)...")
    processor = ComprehensiveCardiacDataProcessor("data")
    
    sequences, labels, names = processor.create_comprehensive_dataset(
        temporal_sequences_per_type=500,   # Reduced from 2000
        spatial_sequences=300,             # Reduced from 1000
        max_sequence_length=50             # Reduced from 100
    )
    
    if len(sequences) == 0:
        logger.error("❌ No data available for training")
        return
    
    logger.info(f"📊 Total sequences: {len(sequences)} (reduced for overfitting prevention)")
    
    # Create dataset
    dataset = ComprehensiveCardiacDataset(sequences, labels, names)
    
    # Split with larger validation set for better overfitting detection
    train_size = int(0.6 * len(dataset))  # Reduced training set
    val_size = int(0.3 * len(dataset))    # Larger validation set
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Smaller batch size to reduce overfitting
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    logger.info(f"📈 Data splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Get input dimension
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch['sequence'].shape[2]
    logger.info(f"🔢 Input dimension: {input_dim}")
    
    # Create regularized model
    model = RegularizedTemporalRNN(
        input_dim=input_dim,
        hidden_dim=64,      # Much smaller
        num_layers=2,       # Fewer layers
        num_classes=5,
        dropout=0.5         # Heavy dropout
    )
    
    # Create trainer with anti-overfitting measures
    trainer = AntiOverfittingTrainer(model, learning_rate=0.0001)  # Low LR
    
    # Train with overfitting monitoring
    best_val_loss = trainer.train(train_loader, val_loader, epochs=30)
    
    # Final test evaluation
    logger.info("\n🧪 Final test evaluation...")
    test_loss, test_acc, test_predictions, test_labels = trainer.validate(test_loader)
    
    logger.info(f"🎯 Final Results:")
    logger.info(f"  Test Loss: {test_loss:.4f}")
    logger.info(f"  Test Accuracy: {test_acc:.4f}")
    
    # Plot training curves to visualize overfitting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs_range = range(1, len(trainer.history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs_range, trainer.history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs_range, trainer.history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs_range, trainer.history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs_range, trainer.history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training vs Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('anti_overfitting_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Overfitting analysis
    final_train_acc = trainer.history['train_acc'][-1]
    final_val_acc = trainer.history['val_acc'][-1]
    acc_gap = final_train_acc - final_val_acc
    
    logger.info(f"\n📊 Overfitting Analysis:")
    logger.info(f"  Final Training Accuracy: {final_train_acc:.4f}")
    logger.info(f"  Final Validation Accuracy: {final_val_acc:.4f}")
    logger.info(f"  Accuracy Gap: {acc_gap:.4f}")
    
    if acc_gap < 0.05:
        logger.info("✅ No significant overfitting detected")
    elif acc_gap < 0.10:
        logger.info("⚠️ Mild overfitting detected")
    else:
        logger.info("❌ Significant overfitting detected")
    
    logger.info(f"\n🎉 Anti-overfitting training completed!")

if __name__ == "__main__":
    main()

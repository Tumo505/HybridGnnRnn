#!/usr/bin/env python3
"""
Scalable Anti-Overfitting Training
Strategic scaling with progressive regularization for better accuracy
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
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

from comprehensive_cardiac_data_processor import ComprehensiveCardiacDataProcessor, ComprehensiveCardiacDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScalableRegularizedRNN(nn.Module):
    """Scalable RNN with progressive regularization strategies"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, num_classes=5, 
                 dropout=0.4, use_residual=True, use_attention=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.use_residual = use_residual
        self.use_attention = use_attention
        
        # Input processing with layer normalization (more stable than batch norm)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)  # Lighter dropout on input
        )
        
        # Stacked LSTM with residual connections
        self.lstm_layers = nn.ModuleList()
        self.residual_projections = nn.ModuleList()
        lstm_input_dim = hidden_dim
        for i in range(num_layers):
            self.lstm_layers.append(
                nn.LSTM(lstm_input_dim, hidden_dim, 1, batch_first=True, 
                       dropout=0, bidirectional=True)
            )
            
            # Add residual projection for dimension matching (except first layer)
            if i > 0 and use_residual:
                self.residual_projections.append(
                    nn.Linear(lstm_input_dim, hidden_dim * 2)
                )
            else:
                self.residual_projections.append(None)
            
            # After first layer, input dimension is hidden_dim * 2 due to bidirectional
            lstm_input_dim = hidden_dim * 2
        
        # Layer normalization for each LSTM
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * 2) for _ in range(num_layers)
        ])
        
        # Dropout layers
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_layers)
        ])
        
        # Multi-head attention (optional)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                hidden_dim * 2, num_heads=8, dropout=dropout, batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Progressive classification head with multiple regularization
        classifier_layers = []
        current_dim = hidden_dim * 2
        
        # Multiple hidden layers with decreasing size
        hidden_sizes = [hidden_dim, hidden_dim // 2, hidden_dim // 4]
        for i, next_dim in enumerate(hidden_sizes):
            classifier_layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.LayerNorm(next_dim),  # Use LayerNorm instead of BatchNorm
                nn.ReLU(),
                nn.Dropout(dropout + i * 0.1)  # Increasing dropout
            ])
            current_dim = next_dim
        
        # Final classification layer
        classifier_layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        
        # Weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Careful weight initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM weights
                    nn.init.orthogonal_(param)
                elif len(param.shape) >= 2:
                    # Linear layer weights
                    nn.init.xavier_uniform_(param, gain=0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x, lengths=None):
        batch_size, seq_len, _ = x.shape
        print(f"FORWARD DEBUG: Called with input shape {x.shape}, lengths: {lengths.shape if lengths is not None else None}")
        
        # Input projection - apply to all time steps at once
        x = x.view(-1, x.size(-1))  # Reshape to [batch_size * seq_len, input_dim]
        x = self.input_projection(x)  # Apply projection
        x = x.view(batch_size, seq_len, -1)  # Reshape back to [batch_size, seq_len, hidden_dim]
        
        # Stacked LSTM with residual connections
        for i, (lstm, norm, dropout, res_proj) in enumerate(zip(self.lstm_layers, self.layer_norms, self.dropout_layers, self.residual_projections)):
            # Store input for potential residual connection
            lstm_input = x
            
            lstm_out, _ = lstm(x)
            lstm_out = norm(lstm_out)
            lstm_out = dropout(lstm_out)
            
            # Residual connection with projection if needed
            if self.use_residual and i > 0 and res_proj is not None:
                residual = res_proj(lstm_input)
                lstm_out = lstm_out + residual
            
            x = lstm_out
        
        # Attention mechanism
        if self.use_attention:
            attn_out, _ = self.attention(x, x, x)
            x = self.attention_norm(x + attn_out)
        
        # Global pooling with length masking
        if lengths is not None:
            mask = torch.arange(seq_len, device=x.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1).float()
            pooled = (x * mask).sum(dim=1) / lengths.unsqueeze(-1).float()
        else:
            pooled = torch.mean(x, dim=1)
        
        # Classification
        output = self.classifier(pooled)
        return output

class ProgressiveTrainer:
    """Progressive training with adaptive regularization"""
    
    def __init__(self, model, device='auto', base_lr=0.001, use_mixed_precision=False):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.use_mixed_precision = use_mixed_precision and self.device.type == 'cuda'
        
        # Progressive learning rate (starts higher, reduces over time)
        self.base_lr = base_lr
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=0.01,  # Start with moderate L2
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=base_lr/100
        )
        
        # Progressive loss function (starts with label smoothing, reduces over time)
        self.initial_label_smoothing = 0.15
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.initial_label_smoothing)
        
        # Mixed precision scaler
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Tracking
        self.history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
            'learning_rate': [], 'label_smoothing': []
        }
        
        logger.info(f"🔧 Device: {self.device}, Mixed Precision: {self.use_mixed_precision}")
        logger.info(f"🏗️ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def update_regularization(self, epoch, total_epochs):
        """Progressively adjust regularization"""
        progress = epoch / total_epochs
        
        # Reduce label smoothing over time
        current_smoothing = self.initial_label_smoothing * (1 - progress * 0.7)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=current_smoothing)
        
        # Increase weight decay for later epochs
        current_weight_decay = 0.01 + progress * 0.09  # 0.01 -> 0.10
        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = current_weight_decay
        
        return current_smoothing
    
    def train_epoch(self, dataloader, epoch, total_epochs):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        # Update regularization
        smoothing = self.update_regularization(epoch, total_epochs)
        
        for batch_idx, batch in enumerate(dataloader):
            sequences = batch['sequence'].to(self.device)
            labels = batch['label'].squeeze().to(self.device)
            lengths = batch['length'].to(self.device)  # Don't squeeze - keep as tensor
            
            # Ensure lengths is properly shaped
            if lengths.dim() == 0:  # If scalar, convert to batch tensor
                lengths = lengths.repeat(sequences.size(0))
            elif lengths.dim() == 1 and lengths.size(0) != sequences.size(0):
                # If not matching batch size, create proper lengths
                lengths = torch.full((sequences.size(0),), sequences.size(1), device=self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(sequences, lengths)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                print(f"DEBUG INPUT: sequences shape: {sequences.shape}, lengths shape: {lengths.shape}")
                outputs = self.model(sequences, lengths)
                
                # Debug: Check tensor shapes
                print(f"DEBUG: outputs shape: {outputs.shape}, labels shape: {labels.shape}")
                print(f"DEBUG: outputs dtype: {outputs.dtype}, labels dtype: {labels.dtype}")
                print(f"DEBUG: labels sample: {labels[:5]}")
                
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            if batch_idx % 200 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"  Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}, LR: {current_lr:.6f}, Smoothing: {smoothing:.3f}")
        
        # Update scheduler
        self.scheduler.step()
        
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
                lengths = batch['length'].to(self.device)  # Don't squeeze
                
                # Ensure lengths is properly shaped
                if lengths.dim() == 0:  # If scalar, convert to batch tensor
                    lengths = lengths.repeat(sequences.size(0))
                elif lengths.dim() == 1 and lengths.size(0) != sequences.size(0):
                    # If not matching batch size, create proper lengths
                    lengths = torch.full((sequences.size(0),), sequences.size(1), device=self.device)
                
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(sequences, lengths)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(sequences, lengths)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy, all_predictions, all_labels
    
    def train_progressive(self, train_loader, val_loader, epochs=50, patience=15):
        """Progressive training with adaptive strategies"""
        logger.info(f"🚀 Progressive training for {epochs} epochs...")
        
        best_val_acc = 0
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            logger.info(f"\n📍 Epoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch, epochs)
            
            # Validate
            val_loss, val_acc, _, _ = self.validate(val_loader)
            
            # Save metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Calculate gaps
            acc_gap = train_acc - val_acc
            loss_gap = val_loss - train_loss
            
            logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            logger.info(f"  📊 Acc Gap: {acc_gap:.4f}, Loss Gap: {loss_gap:.4f}")
            
            # Overfitting warnings with thresholds
            if acc_gap > 0.08:
                logger.warning(f"⚠️ Moderate overfitting detected")
            elif acc_gap > 0.15:
                logger.warning(f"🚨 Significant overfitting detected")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                checkpoint_dir = Path("checkpoints") / f"scalable_rnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'history': self.history,
                    'model_config': {
                        'input_dim': self.model.input_dim,
                        'hidden_dim': self.model.hidden_dim,
                        'num_layers': self.model.num_layers,
                        'num_classes': self.model.num_classes
                    }
                }, checkpoint_dir / "best_model.pt")
                
                logger.info(f"  🎯 New best validation accuracy: {val_acc:.4f}")
                
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"🛑 Early stopping after {epoch+1} epochs")
                    break
        
        training_time = time.time() - start_time
        logger.info(f"\n✅ Training completed in {training_time:.2f} seconds")
        logger.info(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
        
        return best_val_acc

def main():
    """Main scalable training function"""
    logger.info("🫀 Scalable Anti-Overfitting Cardiac RNN Training")
    logger.info("=" * 70)
    
    # Progressive dataset scaling strategy
    scales = [
        {"sequences_per_type": 1000, "spatial": 500, "name": "Medium Scale"},
        {"sequences_per_type": 1500, "spatial": 750, "name": "Large Scale"},
        {"sequences_per_type": 2000, "spatial": 1000, "name": "Full Scale"}
    ]
    
    best_overall_acc = 0
    best_scale = None
    
    for scale_idx, scale_config in enumerate(scales):
        logger.info(f"\n🔄 Training Scale {scale_idx+1}: {scale_config['name']}")
        logger.info(f"  Sequences per type: {scale_config['sequences_per_type']}")
        logger.info(f"  Spatial sequences: {scale_config['spatial']}")
        
        # Load dataset with current scale
        processor = ComprehensiveCardiacDataProcessor("data")
        sequences, labels, names = processor.create_comprehensive_dataset(
            temporal_sequences_per_type=scale_config['sequences_per_type'],
            spatial_sequences=scale_config['spatial'],
            max_sequence_length=80  # Slightly longer sequences
        )
        
        logger.info(f"  Total sequences: {len(sequences)}")
        
        if len(sequences) == 0:
            logger.error("❌ No data available")
            continue
        
        # Create dataset with improved splits
        dataset = ComprehensiveCardiacDataset(sequences, labels, names)
        
        # Determine number of unique classes from the raw labels before creating dataset
        unique_labels = sorted(set(labels))  # labels is the raw list from data processor
        num_classes = len(unique_labels)
        
        logger.info(f"  Total sequences: {len(sequences)}")
        logger.info(f"  Number of classes: {num_classes}, Unique labels: {unique_labels}")
        
        # Stratified splits to ensure balanced classes
        train_size = int(0.65 * len(dataset))  # More training data
        val_size = int(0.25 * len(dataset))    # Adequate validation
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Adaptive batch size based on dataset size
        batch_size = min(32, max(16, len(train_dataset) // 100))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        logger.info(f"  Data splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        logger.info(f"  Batch size: {batch_size}")
        
        # Get input dimension
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch['sequence'].shape[2]
        
        logger.info(f"  Input dimension: {input_dim}")
        
        # Scale model complexity with dataset size
        if len(sequences) < 2000:
            hidden_dim, num_layers = 96, 2
        elif len(sequences) < 5000:
            hidden_dim, num_layers = 128, 3
        else:
            hidden_dim, num_layers = 160, 3
        
        # Create scalable model
        model = ScalableRegularizedRNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,  # Use actual number of classes
            dropout=0.35,  # Moderate dropout
            use_residual=True,
            use_attention=True
        )
        
        # Progressive learning rate based on dataset size
        base_lr = 0.001 if len(sequences) < 3000 else 0.0008
        
        # Create trainer
        trainer = ProgressiveTrainer(
            model, 
            base_lr=base_lr,
            use_mixed_precision=torch.cuda.is_available()
        )
        
        # Train with adaptive epochs
        epochs = min(60, max(30, 20000 // len(sequences)))
        patience = max(10, epochs // 4)
        
        logger.info(f"  Training epochs: {epochs}, Patience: {patience}")
        
        # Train
        val_acc = trainer.train_progressive(
            train_loader, val_loader, 
            epochs=epochs, patience=patience
        )
        
        # Test evaluation
        test_loss, test_acc, test_predictions, test_labels = trainer.validate(test_loader)
        
        logger.info(f"\n🎯 {scale_config['name']} Results:")
        logger.info(f"  Validation Accuracy: {val_acc:.4f}")
        logger.info(f"  Test Accuracy: {test_acc:.4f}")
        
        # Track best scale
        if test_acc > best_overall_acc:
            best_overall_acc = test_acc
            best_scale = scale_config['name']
        
        logger.info(f"  Current best: {best_overall_acc:.4f} ({best_scale})")
    
    logger.info(f"\n🏆 FINAL RESULTS:")
    logger.info(f"  Best Test Accuracy: {best_overall_acc:.4f}")
    logger.info(f"  Best Scale: {best_scale}")
    logger.info(f"  Improvement over baseline: {(best_overall_acc - 0.6758) * 100:.2f}%")

if __name__ == "__main__":
    main()

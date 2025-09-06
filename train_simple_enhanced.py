#!/usr/bin/env python3
"""
Simple Enhanced Training Script for Spatial GNN with Synthetic Data
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models.spatial_gnn import SpatialGNN
from src.data.enhanced_cardiac_loader import create_enhanced_cardiac_loaders

def setup_logging():
    """Setup logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"simple_gnn_training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    return log_file

def main():
    """Main training function."""
    
    # Setup logging
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting Simple Enhanced Spatial GNN Training")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Create data loaders
        logger.info("📊 Creating enhanced data loaders...")
        train_loader, val_loader, test_loader = create_enhanced_cardiac_loaders(
            include_synthetic=True,
            batch_size=8,
            num_neighbors=10
        )
        
        logger.info(f"Data loaders created:")
        logger.info(f"  Train: {len(train_loader)} batches")
        logger.info(f"  Val: {len(val_loader) if val_loader else 0} batches")
        logger.info(f"  Test: {len(test_loader) if test_loader else 0} batches")
        
        # Get input dimensions from a sample batch
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch.x.shape[1]
        num_classes = len(torch.unique(sample_batch.y_class))
        
        logger.info(f"Input dimension: {input_dim}")
        logger.info(f"Number of classes: {num_classes}")
        
        # Create model (simplified)
        logger.info("🔧 Creating spatial GNN model...")
        model = SpatialGNN(
            input_dim=input_dim,
            hidden_dims=[256, 128],
            output_dim=64,
            num_classes=num_classes,
            conv_type='GCN',
            use_attention=False,
            dropout=0.3,
            use_batch_norm=False
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model created with {total_params:,} parameters")
        
        # Simple classification only
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # Training loop
        logger.info("🔥 Starting training...")
        
        epochs = 20
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass - only classification
                class_logits, _ = model(batch)
                
                # Create graph-level targets for classification
                num_graphs = batch.batch.max().item() + 1
                graph_targets = []
                
                for graph_idx in range(num_graphs):
                    graph_mask = batch.batch == graph_idx
                    graph_classes = batch.y_class[graph_mask]
                    # Use most frequent class as target
                    graph_target = torch.mode(graph_classes).values
                    graph_targets.append(graph_target)
                
                graph_targets = torch.stack(graph_targets)
                
                # Loss and backward
                loss = criterion(class_logits, graph_targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(class_logits.data, 1)
                total += graph_targets.size(0)
                correct += (predicted == graph_targets).sum().item()
            
            # Calculate metrics
            avg_loss = total_loss / len(train_loader)
            accuracy = 100 * correct / total
            
            logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            # Validation
            if val_loader and epoch % 5 == 0:
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        class_logits, _ = model(batch)
                        
                        # Create graph-level targets
                        num_graphs = batch.batch.max().item() + 1
                        graph_targets = []
                        
                        for graph_idx in range(num_graphs):
                            graph_mask = batch.batch == graph_idx
                            graph_classes = batch.y_class[graph_mask]
                            graph_target = torch.mode(graph_classes).values
                            graph_targets.append(graph_target)
                        
                        graph_targets = torch.stack(graph_targets)
                        
                        loss = criterion(class_logits, graph_targets)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(class_logits.data, 1)
                        val_total += graph_targets.size(0)
                        val_correct += (predicted == graph_targets).sum().item()
                
                val_accuracy = 100 * val_correct / val_total
                logger.info(f"  Validation - Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%")
        
        # Save model
        save_dir = Path('models/simple_enhanced_gnn')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = save_dir / 'final_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_info': {
                'input_dim': input_dim,
                'num_classes': num_classes,
                'total_params': total_params,
                'final_accuracy': accuracy
            }
        }, model_path)
        
        logger.info(f"💾 Saved model to {model_path}")
        logger.info("🎊 Training completed successfully!")
        logger.info("✅ Successfully trained with synthetic data augmentation")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

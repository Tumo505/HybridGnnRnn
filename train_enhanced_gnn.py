#!/usr/bin/env python3
"""
Enhanced Training Script for Spatial GNN with Synthetic Data
Trains our original spatial GNN model with both original and synthetic cardiac data.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import argparse
import wandb
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, r2_score

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models.spatial_gnn import SpatialGNN
from src.data.enhanced_cardiac_loader import create_enhanced_cardiac_loaders

def setup_logging():
    """Setup logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"enhanced_gnn_training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    return log_file

def evaluate_model(model, data_loader, device, criterion_class, criterion_reg):
    """
    Evaluate model performance on a data loader.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader to evaluate on
        device: Device to run evaluation on
        criterion_class: Classification loss function
        criterion_reg: Regression loss function
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0
    total_class_loss = 0
    total_reg_loss = 0
    all_class_preds = []
    all_class_targets = []
    all_reg_preds = []
    all_reg_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            # Forward pass
            class_logits, reg_output = model(batch)
            
            # For graph-level predictions, create graph-level targets
            num_graphs = batch.batch.max().item() + 1
            graph_class_targets = []
            graph_reg_targets = []
            
            for graph_idx in range(num_graphs):
                # Get nodes for this graph
                graph_mask = batch.batch == graph_idx
                
                # Classification: take the mode (most common class) for this graph
                graph_classes = batch.y_class[graph_mask]
                graph_class = torch.mode(graph_classes).values
                graph_class_targets.append(graph_class)
                
                # Regression: take the mean efficiency for this graph
                graph_efficiencies = batch.y_efficiency[graph_mask]
                valid_eff = graph_efficiencies[~torch.isnan(graph_efficiencies)]
                if len(valid_eff) > 0:
                    graph_eff = torch.mean(valid_eff)
                else:
                    graph_eff = torch.tensor(0.5, device=batch.y_efficiency.device)
                graph_reg_targets.append(graph_eff)
            
            graph_class_targets = torch.stack(graph_class_targets)
            graph_reg_targets = torch.stack(graph_reg_targets)
            
            # Classification loss and metrics
            class_loss = criterion_class(class_logits, graph_class_targets)
            class_preds = torch.argmax(class_logits, dim=1)
            
            # Regression loss and metrics
            valid_mask = ~torch.isnan(graph_reg_targets)
            if valid_mask.sum() > 0:
                reg_loss = criterion_reg(reg_output[valid_mask].squeeze(), graph_reg_targets[valid_mask])
                reg_preds = reg_output[valid_mask].squeeze()
                reg_targets = graph_reg_targets[valid_mask]
            else:
                reg_loss = torch.tensor(0.0, device=device)
                reg_preds = torch.tensor([], device=device)
                reg_targets = torch.tensor([], device=device)
            
            # Total loss
            total_loss += (class_loss + reg_loss).item()
            total_class_loss += class_loss.item()
            total_reg_loss += reg_loss.item()
            
            # Collect predictions and targets
            all_class_preds.extend(class_preds.cpu().numpy())
            all_class_targets.extend(graph_class_targets.cpu().numpy())
            if len(reg_preds) > 0:
                all_reg_preds.extend(reg_preds.cpu().numpy())
                all_reg_targets.extend(reg_targets.cpu().numpy())    # Calculate metrics
    n_batches = len(data_loader)
    avg_loss = total_loss / n_batches
    avg_class_loss = total_class_loss / n_batches
    avg_reg_loss = total_reg_loss / n_batches
    
    # Classification metrics
    class_accuracy = accuracy_score(all_class_targets, all_class_preds)
    class_f1 = f1_score(all_class_targets, all_class_preds, average='weighted')
    
    # Regression metrics
    if len(all_reg_preds) > 0:
        reg_r2 = r2_score(all_reg_targets, all_reg_preds)
        reg_mse = np.mean((np.array(all_reg_targets) - np.array(all_reg_preds)) ** 2)
    else:
        reg_r2 = 0.0
        reg_mse = 0.0
    
    return {
        'total_loss': avg_loss,
        'class_loss': avg_class_loss,
        'reg_loss': avg_reg_loss,
        'class_accuracy': class_accuracy,
        'class_f1': class_f1,
        'reg_r2': reg_r2,
        'reg_mse': reg_mse,
        'n_samples': len(all_class_targets)
    }

def train_epoch(model, train_loader, optimizer, criterion_class, criterion_reg, device, log_freq=10):
    """
    Train model for one epoch.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion_class: Classification loss function
        criterion_reg: Regression loss function
        device: Device to train on
        log_freq: How often to log progress
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    
    total_loss = 0
    total_class_loss = 0
    total_reg_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        class_logits, reg_output = model(batch)
        
        # Classification loss
        class_loss = criterion_class(class_logits, batch.y_class)
        
        # Regression loss (only on valid efficiency scores)
        valid_mask = ~torch.isnan(batch.y_efficiency)
        if valid_mask.sum() > 0:
            reg_loss = criterion_reg(reg_output[valid_mask].squeeze(), batch.y_efficiency[valid_mask])
        else:
            reg_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Total loss
        total_loss_batch = class_loss + reg_loss
        
        # Backward pass
        total_loss_batch.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        
        # Accumulate losses
        total_loss += total_loss_batch.item()
        total_class_loss += class_loss.item()
        total_reg_loss += reg_loss.item()
        
        # Log progress
        if (batch_idx + 1) % log_freq == 0:
            logging.info(f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                        f"Loss={total_loss_batch.item():.4f}, "
                        f"Class={class_loss.item():.4f}, "
                        f"Reg={reg_loss.item():.4f}")
    
    n_batches = len(train_loader)
    return {
        'total_loss': total_loss / n_batches,
        'class_loss': total_class_loss / n_batches,
        'reg_loss': total_reg_loss / n_batches
    }

def main():
    """Main training function."""
    
    parser = argparse.ArgumentParser(description='Enhanced Spatial GNN Training')
    parser.add_argument('--experiment_name', default='enhanced_spatial_gnn', help='Experiment name')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[512, 256, 128], help='Hidden dimensions')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_neighbors', type=int, default=10, help='Number of graph neighbors')
    parser.add_argument('--use_synthetic', action='store_true', default=True, help='Use synthetic data')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--save_dir', default='models/enhanced_gnn', help='Model save directory')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting Enhanced Spatial GNN Training")
    logger.info(f"Arguments: {args}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize WandB
    if args.use_wandb:
        wandb.init(
            project='enhanced-cardiac-gnn',
            name=args.experiment_name,
            config=vars(args)
        )
    
    try:
        # Create data loaders
        logger.info("📊 Creating enhanced data loaders...")
        train_loader, val_loader, test_loader = create_enhanced_cardiac_loaders(
            include_synthetic=args.use_synthetic,
            batch_size=args.batch_size,
            num_neighbors=args.num_neighbors
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
        
        # Create model
        logger.info("🔧 Creating enhanced spatial GNN model...")
        model = SpatialGNN(
            input_dim=input_dim,
            hidden_dims=args.hidden_dims,
            output_dim=64,
            num_classes=num_classes,
            conv_type='GCN',
            use_attention=True,
            dropout=args.dropout,
            use_batch_norm=True
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model created with {total_params:,} parameters")
        
        # Define loss functions
        criterion_class = nn.CrossEntropyLoss()
        criterion_reg = nn.MSELoss()
        
        # Define optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training loop
        logger.info("🔥 Starting training...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(args.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
            
            # Train
            train_metrics = train_epoch(
                model, train_loader, optimizer, 
                criterion_class, criterion_reg, device
            )
            
            logger.info(f"Train - Loss: {train_metrics['total_loss']:.4f}, "
                       f"Class: {train_metrics['class_loss']:.4f}, "
                       f"Reg: {train_metrics['reg_loss']:.4f}")
            
            # Validate
            if val_loader:
                val_metrics = evaluate_model(
                    model, val_loader, device, 
                    criterion_class, criterion_reg
                )
                
                logger.info(f"Val - Loss: {val_metrics['total_loss']:.4f}, "
                           f"Acc: {val_metrics['class_accuracy']:.4f}, "
                           f"F1: {val_metrics['class_f1']:.4f}, "
                           f"R²: {val_metrics['reg_r2']:.4f}")
                
                # Learning rate scheduling
                scheduler.step(val_metrics['total_loss'])
                
                # Early stopping
                if val_metrics['total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['total_loss']
                    patience_counter = 0
                    
                    # Save best model
                    save_dir = Path(args.save_dir)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    best_model_path = save_dir / 'best_model.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics,
                        'args': args
                    }, best_model_path)
                    
                    logger.info(f"💾 Saved best model to {best_model_path}")
                    
                else:
                    patience_counter += 1
                    
                if patience_counter >= args.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Log to WandB
            if args.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'train_loss': train_metrics['total_loss'],
                    'train_class_loss': train_metrics['class_loss'],
                    'train_reg_loss': train_metrics['reg_loss'],
                    'learning_rate': optimizer.param_groups[0]['lr']
                }
                
                if val_loader:
                    log_dict.update({
                        'val_loss': val_metrics['total_loss'],
                        'val_accuracy': val_metrics['class_accuracy'],
                        'val_f1': val_metrics['class_f1'],
                        'val_r2': val_metrics['reg_r2'],
                        'val_mse': val_metrics['reg_mse']
                    })
                
                wandb.log(log_dict)
        
        # Final evaluation on test set
        if test_loader:
            logger.info("\n📊 Final evaluation on test set...")
            
            # Load best model
            if val_loader:
                checkpoint = torch.load(save_dir / 'best_model.pth')
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Loaded best model for final evaluation")
            
            test_metrics = evaluate_model(
                model, test_loader, device,
                criterion_class, criterion_reg
            )
            
            logger.info("🎉 Final Test Results:")
            logger.info(f"  Loss: {test_metrics['total_loss']:.4f}")
            logger.info(f"  Accuracy: {test_metrics['class_accuracy']:.4f}")
            logger.info(f"  F1-Score: {test_metrics['class_f1']:.4f}")
            logger.info(f"  R² Score: {test_metrics['reg_r2']:.4f}")
            logger.info(f"  MSE: {test_metrics['reg_mse']:.4f}")
            logger.info(f"  Samples: {test_metrics['n_samples']}")
            
            # Save final model
            final_model_path = save_dir / 'final_model.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'test_metrics': test_metrics,
                'args': args,
                'model_info': {
                    'input_dim': input_dim,
                    'hidden_dims': args.hidden_dims,
                    'num_classes': num_classes,
                    'total_params': total_params
                }
            }, final_model_path)
            
            logger.info(f"💾 Saved final model to {final_model_path}")
            
            if args.use_wandb:
                wandb.log({
                    'final_test_loss': test_metrics['total_loss'],
                    'final_test_accuracy': test_metrics['class_accuracy'],
                    'final_test_f1': test_metrics['class_f1'],
                    'final_test_r2': test_metrics['reg_r2'],
                    'final_test_mse': test_metrics['reg_mse']
                })
        
        logger.info("\n🎊 Enhanced Spatial GNN Training Completed Successfully!")
        logger.info(f"📝 Log file: {log_file}")
        
        if args.use_synthetic:
            logger.info("✅ Successfully trained with synthetic data augmentation")
            logger.info("✅ Model should have improved overfitting resistance")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        if args.use_wandb:
            wandb.log({"error": str(e), "status": "failed"})
        
        raise
    
    finally:
        if args.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main()

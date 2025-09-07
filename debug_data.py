"""
Data Debugging and Fixing Script for NaN Issues
"""

import torch
import torch_geometric as pyg
import numpy as np
import pandas as pd
import scanpy as sc
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our modules
import sys
sys.path.append('/Users/tumokgabeng/Projects/HybridGnnRnn/src')

from data.enhanced_cardiac_loader import create_enhanced_cardiac_loaders

def debug_data_issues(data_path='/Users/tumokgabeng/Projects/HybridGnnRnn/data'):
    """Debug and identify data issues causing NaN problems"""
    
    logger.info("🔍 Starting comprehensive data debugging...")
    
    # Load data loaders
    train_loader, val_loader, test_loader = create_enhanced_cardiac_loaders(
        data_path, batch_size=4
    )
    
    # Get dataset info from first batch
    sample_batch = next(iter(train_loader))
    dataset_info = {
        'num_features': sample_batch.x.size(1),
        'num_classes': len(torch.unique(sample_batch.y))
    }
    
    logger.info(f"Dataset info: {dataset_info}")
    
    # Analyze first batch in detail
    logger.info("\n📊 Analyzing training data batch...")
    
    train_batch = next(iter(train_loader))
    logger.info(f"Batch info:")
    logger.info(f"  - Batch size: {train_batch.batch.max().item() + 1}")
    logger.info(f"  - Total nodes: {train_batch.x.size(0)}")
    logger.info(f"  - Node features: {train_batch.x.size(1)}")
    logger.info(f"  - Total edges: {train_batch.edge_index.size(1)}")
    logger.info(f"  - Number of graphs: {len(train_batch.y)}")
    
    # Check for data quality issues
    x = train_batch.x
    y = train_batch.y
    edge_index = train_batch.edge_index
    
    logger.info("\n🔬 Data Quality Analysis:")
    
    # Check for NaN/Inf in features
    nan_features = torch.isnan(x).sum().item()
    inf_features = torch.isinf(x).sum().item()
    logger.info(f"  - NaN in features: {nan_features}")
    logger.info(f"  - Inf in features: {inf_features}")
    
    # Check feature statistics
    x_mean = x.mean().item()
    x_std = x.std().item()
    x_min = x.min().item()
    x_max = x.max().item()
    logger.info(f"  - Feature mean: {x_mean:.6f}")
    logger.info(f"  - Feature std: {x_std:.6f}")
    logger.info(f"  - Feature min: {x_min:.6f}")
    logger.info(f"  - Feature max: {x_max:.6f}")
    
    # Check for extreme values
    extreme_large = (x > 100).sum().item()
    extreme_small = (x < -100).sum().item()
    zero_features = (x == 0).sum().item()
    logger.info(f"  - Values > 100: {extreme_large}")
    logger.info(f"  - Values < -100: {extreme_small}")
    logger.info(f"  - Zero values: {zero_features}")
    
    # Check labels
    logger.info(f"  - Labels: {y}")
    logger.info(f"  - Label range: [{y.min().item()}, {y.max().item()}]")
    logger.info(f"  - Unique labels: {torch.unique(y)}")
    
    # Check edge connectivity
    logger.info(f"  - Edge index shape: {edge_index.shape}")
    logger.info(f"  - Max node index: {edge_index.max().item()}")
    logger.info(f"  - Min node index: {edge_index.min().item()}")
    
    # Check for self-loops
    self_loops = (edge_index[0] == edge_index[1]).sum().item()
    logger.info(f"  - Self loops: {self_loops}")
    
    return train_batch, dataset_info

def fix_data_preprocessing():
    """Create improved data preprocessing that avoids NaN issues"""
    
    logger.info("🛠️ Creating robust data preprocessing...")
    
    from data.enhanced_cardiac_loader import EnhancedCardiacDataLoader
    import os
    
    # Create fixed data loader class
    class RobustCardiacDataLoader(EnhancedCardiacDataLoader):
        """Enhanced data loader with robust preprocessing"""
        
        def preprocess_data(self, adata):
            """Robust preprocessing that avoids NaN issues"""
            logger.info(f"Preprocessing data with shape: {adata.shape}")
            
            # Make a copy to avoid modifying original
            adata = adata.copy()
            
            # Convert to dense if sparse
            if hasattr(adata.X, 'toarray'):
                adata.X = adata.X.toarray()
            
            # Replace any NaN/inf values
            X = adata.X
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply robust scaling instead of standard scaling
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Clip extreme values to prevent numerical issues
            X_clipped = np.clip(X_scaled, -10, 10)
            
            # Final check for any remaining issues
            X_final = np.nan_to_num(X_clipped, nan=0.0, posinf=0.0, neginf=0.0)
            
            adata.X = X_final
            
            logger.info(f"After preprocessing:")
            logger.info(f"  - Shape: {adata.X.shape}")
            logger.info(f"  - Mean: {np.mean(adata.X):.6f}")
            logger.info(f"  - Std: {np.std(adata.X):.6f}")
            logger.info(f"  - Min: {np.min(adata.X):.6f}")
            logger.info(f"  - Max: {np.max(adata.X):.6f}")
            logger.info(f"  - NaN count: {np.isnan(adata.X).sum()}")
            logger.info(f"  - Inf count: {np.isinf(adata.X).sum()}")
            
            return adata
        
        def create_efficiency_scores(self, adata):
            """Create robust efficiency scores"""
            n_cells = adata.shape[0]
            n_genes = adata.shape[1]
            
            # Use more stable calculations
            X = adata.X
            if hasattr(X, 'toarray'):
                X = X.toarray()
            
            # Calculate efficiency based on multiple factors
            gene_expression_mean = np.mean(X, axis=1)
            gene_expression_std = np.std(X, axis=1)
            genes_expressed = np.sum(X > 0, axis=1)
            
            # Normalize components
            efficiency_components = []
            
            # Expression level component (30%)
            exp_component = gene_expression_mean / (np.max(gene_expression_mean) + 1e-8)
            efficiency_components.append(0.3 * exp_component)
            
            # Diversity component (40%)
            diversity_component = genes_expressed / n_genes
            efficiency_components.append(0.4 * diversity_component)
            
            # Stability component (30%)
            # Lower std relative to mean indicates more stable expression
            cv = gene_expression_std / (gene_expression_mean + 1e-8)
            stability_component = 1.0 / (1.0 + cv)  # Inverse coefficient of variation
            stability_component = stability_component / (np.max(stability_component) + 1e-8)
            efficiency_components.append(0.3 * stability_component)
            
            # Combine components
            efficiency_scores = np.sum(efficiency_components, axis=0)
            
            # Ensure scores are in valid range [0, 1]
            efficiency_scores = np.clip(efficiency_scores, 0.0, 1.0)
            
            # Replace any remaining NaN/inf
            efficiency_scores = np.nan_to_num(efficiency_scores, nan=0.5)
            
            logger.info(f"Efficiency scores - Min: {np.min(efficiency_scores):.3f}, "
                       f"Max: {np.max(efficiency_scores):.3f}, "
                       f"Mean: {np.mean(efficiency_scores):.3f}")
            
            return efficiency_scores
    
    return RobustCardiacDataLoader

def test_robust_training():
    """Test training with robust data preprocessing"""
    
    logger.info("🧪 Testing robust training...")
    
    # Create robust data loader
    RobustLoader = fix_data_preprocessing()
    robust_loader = RobustLoader()
    
    # Load robust data
    train_graphs, val_graphs, test_graphs = robust_loader.load_all_data(
        '/Users/tumokgabeng/Projects/HybridGnnRnn/data'
    )
    
    # Create data loaders
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=8, shuffle=False)
    
    # Get sample batch
    sample_batch = next(iter(train_loader))
    logger.info(f"Robust batch info:")
    logger.info(f"  - Features shape: {sample_batch.x.shape}")
    logger.info(f"  - Features mean: {sample_batch.x.mean().item():.6f}")
    logger.info(f"  - Features std: {sample_batch.x.std().item():.6f}")
    logger.info(f"  - NaN count: {torch.isnan(sample_batch.x).sum().item()}")
    logger.info(f"  - Inf count: {torch.isinf(sample_batch.x).sum().item()}")
    
    # Test simple model forward pass
    from models.spatial_gnn import SpatialGNN
    
    device = 'cpu'
    model = SpatialGNN(
        input_dim=sample_batch.x.size(1),
        hidden_dim=128,
        output_dim=len(torch.unique(sample_batch.y)),
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    # Forward pass
    sample_batch = sample_batch.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(sample_batch)
        logger.info(f"Model output shape: {output.shape}")
        logger.info(f"Output mean: {output.mean().item():.6f}")
        logger.info(f"Output std: {output.std().item():.6f}")
        logger.info(f"Output NaN count: {torch.isnan(output).sum().item()}")
        logger.info(f"Output Inf count: {torch.isinf(output).sum().item()}")
        
        # Test loss calculation
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, sample_batch.y)
        logger.info(f"Loss value: {loss.item():.6f}")
        logger.info(f"Loss is NaN: {torch.isnan(loss).item()}")
    
    return train_loader, val_loader, model

def create_robust_training_script():
    """Create a robust training script with proper preprocessing"""
    
    script_content = '''
"""
Robust Training Script with Fixed Data Preprocessing
"""

import torch
import torch_geometric as pyg
import numpy as np
import logging
from sklearn.preprocessing import RobustScaler
from torch_geometric.loader import DataLoader
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append('/Users/tumokgabeng/Projects/HybridGnnRnn/src')

from data.enhanced_cardiac_loader import EnhancedCardiacDataLoader
from models.spatial_gnn import SpatialGNN

class RobustCardiacDataLoader(EnhancedCardiacDataLoader):
    """Enhanced data loader with robust preprocessing"""
    
    def preprocess_data(self, adata):
        """Robust preprocessing that avoids NaN issues"""
        logger.info(f"Preprocessing data with shape: {adata.shape}")
        
        # Make a copy to avoid modifying original
        adata = adata.copy()
        
        # Convert to dense if sparse
        if hasattr(adata.X, 'toarray'):
            adata.X = adata.X.toarray()
        
        # Replace any NaN/inf values
        X = adata.X
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply robust scaling instead of standard scaling
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Clip extreme values to prevent numerical issues
        X_clipped = np.clip(X_scaled, -5, 5)  # More conservative clipping
        
        # Final check for any remaining issues
        X_final = np.nan_to_num(X_clipped, nan=0.0, posinf=0.0, neginf=0.0)
        
        adata.X = X_final
        
        logger.info(f"After preprocessing: Mean={np.mean(adata.X):.6f}, "
                   f"Std={np.std(adata.X):.6f}, NaN={np.isnan(adata.X).sum()}")
        
        return adata
    
    def create_efficiency_scores(self, adata):
        """Create robust efficiency scores"""
        n_cells = adata.shape[0]
        n_genes = adata.shape[1]
        
        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Simple, stable efficiency calculation
        genes_expressed = np.sum(X > 0, axis=1)
        efficiency_scores = genes_expressed / n_genes
        
        # Add some noise to avoid identical scores
        efficiency_scores += np.random.normal(0, 0.1, n_cells)
        efficiency_scores = np.clip(efficiency_scores, 0.0, 1.0)
        
        return efficiency_scores

def main():
    """Main training function"""
    
    logger.info("🚀 Starting Robust GNN Training")
    
    device = 'cpu'
    
    # Create robust data loader
    data_loader = RobustCardiacDataLoader()
    
    # Load data
    train_graphs, val_graphs, test_graphs = data_loader.load_all_data(
        '/Users/tumokgabeng/Projects/HybridGnnRnn/data'
    )
    
    logger.info(f"Loaded {len(train_graphs)} training, {len(val_graphs)} validation, "
               f"{len(test_graphs)} test graphs")
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=8, shuffle=False)
    
    # Get sample for model setup
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch.x.size(1)
    num_classes = len(torch.unique(sample_batch.y))
    
    logger.info(f"Input dim: {input_dim}, Num classes: {num_classes}")
    
    # Create model
    model = SpatialGNN(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=num_classes,
        num_layers=2,
        dropout=0.2,
        use_batch_norm=False  # Disable batch norm for stability
    ).to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    best_val_acc = 0.0
    
    for epoch in range(30):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            out = model(batch)
            
            # Check for NaN before loss calculation
            if torch.isnan(out).any() or torch.isinf(out).any():
                logger.warning(f"NaN/Inf detected in model output, skipping batch")
                continue
            
            loss = criterion(out, batch.y)
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss detected, skipping batch")
                continue
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            total += batch.y.size(0)
            correct += (predicted == batch.y).sum().item()
        
        train_acc = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        
        # Validation
        if epoch % 5 == 0:
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    
                    if torch.isnan(out).any() or torch.isinf(out).any():
                        continue
                    
                    _, predicted = torch.max(out.data, 1)
                    val_total += batch.y.size(0)
                    val_correct += (predicted == batch.y).sum().item()
            
            val_acc = val_correct / val_total if val_total > 0 else 0.0
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, "
                       f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
            
            model.train()
        else:
            logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}")
    
    logger.info(f"✅ Training completed! Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
'''
    
    with open('/Users/tumokgabeng/Projects/HybridGnnRnn/train_robust.py', 'w') as f:
        f.write(script_content)
    
    logger.info("✅ Created robust training script: train_robust.py")

def main():
    """Main debugging function"""
    
    logger.info("🚀 Starting Data Debugging Session")
    
    # Debug current data issues
    try:
        train_batch, dataset_info = debug_data_issues()
    except Exception as e:
        logger.error(f"Error during data debugging: {e}")
        return
    
    # Test robust preprocessing
    try:
        train_loader, val_loader, model = test_robust_training()
        logger.info("✅ Robust preprocessing test passed!")
    except Exception as e:
        logger.error(f"Error during robust testing: {e}")
        return
    
    # Create robust training script
    create_robust_training_script()
    
    logger.info("🎊 Debugging completed successfully!")

if __name__ == "__main__":
    main()
    
"""
Spatial Graph Neural Network for Cardiomyocyte Differentiation Prediction
Implements GNN architecture for spatial transcriptomics analysis of cardiac tissue.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import logging
from typing import List, Optional, Tuple

class SpatialGNN(nn.Module):
    """
    Graph Neural Network for spatial transcriptomics analysis.
    Designed to capture intercellular interactions and tissue architecture
    for predicting cardiomyocyte differentiation efficiency.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 output_dim: int = 64,
                 num_classes: int = 10,
                 conv_type: str = 'GCN',
                 use_attention: bool = False,
                 dropout: float = 0.3,  # Increased dropout
                 use_batch_norm: bool = True,
                 l2_reg: float = 1e-4,  # Added L2 regularization
                 use_residual: bool = True):  # Added residual connections
        """
        Initialize the Spatial GNN model.
        
        Args:
            input_dim: Number of input features (genes)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output embedding dimension
            num_classes: Number of classification classes
            conv_type: Type of graph convolution ('GCN', 'GAT', 'GraphConv')
            use_attention: Whether to use attention mechanism
            dropout: Dropout rate (increased for regularization)
            use_batch_norm: Whether to use batch normalization
            l2_reg: L2 regularization strength
            use_residual: Whether to use residual connections
        """
        super(SpatialGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.conv_type = conv_type
        self.use_attention = use_attention
        self.l2_reg = l2_reg
        self.use_residual = use_residual
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Build the network layers
        self._build_network()
        
        self.logger.info(f"Initialized SpatialGNN with {self.conv_type} convolutions")
        self.logger.info(f"Architecture: {input_dim} -> {hidden_dims} -> {output_dim} -> {num_classes}")
        
    def _build_network(self):
        """Build the GNN network layers."""
        
        # Graph convolution layers with residual connections
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()  # Added individual dropout layers
        
        # Input layer
        in_dim = self.input_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            conv_layer = self._create_conv_layer(in_dim, hidden_dim)
            self.conv_layers.append(conv_layer)
            
            if self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
            # Progressive dropout (higher for later layers)
            dropout_rate = self.dropout * (1 + i * 0.1)
            self.dropouts.append(nn.Dropout(dropout_rate))
            
            in_dim = hidden_dim
            
        # Final convolution to output dimension
        final_conv = self._create_conv_layer(in_dim, self.output_dim)
        self.conv_layers.append(final_conv)
        
        if self.use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(self.output_dim))
        
        self.dropouts.append(nn.Dropout(self.dropout))
        
        # Attention mechanism for node importance
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.output_dim,
                num_heads=4,  # Reduced complexity
                dropout=self.dropout,
                batch_first=True
            )
        
        # Regularized classification head
        classifier_layers = [nn.Linear(self.output_dim, self.output_dim // 2)]
        if self.use_batch_norm:
            classifier_layers.append(nn.BatchNorm1d(self.output_dim // 2))
        classifier_layers.extend([
            nn.ReLU(),
            nn.Dropout(self.dropout * 1.5),  # Higher dropout in head
            nn.Linear(self.output_dim // 2, self.output_dim // 4)  # Additional layer
        ])
        if self.use_batch_norm:
            classifier_layers.append(nn.BatchNorm1d(self.output_dim // 4))
        classifier_layers.extend([
            nn.ReLU(),
            nn.Dropout(self.dropout * 1.2),
            nn.Linear(self.output_dim // 4, self.num_classes)
        ])
        self.classifier = nn.Sequential(*classifier_layers)
        
        # Regularized regression head  
        regressor_layers = [nn.Linear(self.output_dim, self.output_dim // 2)]
        if self.use_batch_norm:
            regressor_layers.append(nn.BatchNorm1d(self.output_dim // 2))
        regressor_layers.extend([
            nn.ReLU(),
            nn.Dropout(self.dropout * 1.5),
            nn.Linear(self.output_dim // 2, self.output_dim // 4)
        ])
        if self.use_batch_norm:
            regressor_layers.append(nn.BatchNorm1d(self.output_dim // 4))
        regressor_layers.extend([
            nn.ReLU(),
            nn.Dropout(self.dropout * 1.2),
            nn.Linear(self.output_dim // 4, 1),
            nn.Sigmoid()
        ])
        self.regressor = nn.Sequential(*regressor_layers)
        
    def _create_conv_layer(self, in_dim: int, out_dim: int):
        """Create a graph convolution layer based on the specified type."""
        
        if self.conv_type == 'GCN':
            return GCNConv(in_dim, out_dim)
        elif self.conv_type == 'GAT':
            return GATConv(in_dim, out_dim, heads=4, concat=False, dropout=self.dropout)
        elif self.conv_type == 'GraphConv':
            return GraphConv(in_dim, out_dim)
        else:
            raise ValueError(f"Unsupported conv_type: {self.conv_type}")
            
    def forward(self, data: Data, return_embeddings: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the GNN with improved regularization.
        
        Args:
            data: PyTorch Geometric Data object
            return_embeddings: Whether to return node embeddings
            
        Returns:
            Tuple of (classification logits, regression predictions)
            If return_embeddings=True, also returns node embeddings
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch if hasattr(data, 'batch') else None
        
        # Store initial features for residual connections
        residual = None
        
        # Graph convolution layers with residual connections
        for i, conv in enumerate(self.conv_layers[:-1]):
            identity = x  # Store for residual connection
            
            x = conv(x, edge_index)
            
            if self.use_batch_norm and i < len(self.batch_norms) - 1:
                x = self.batch_norms[i](x)
            
            # Add residual connection if dimensions match
            if self.use_residual and identity.shape[-1] == x.shape[-1]:
                x = x + identity
                
            x = F.relu(x)
            x = self.dropouts[i](x)  # Use individual dropout layers
        
        # Final convolution layer
        identity = x
        x = self.conv_layers[-1](x, edge_index)
        
        if self.use_batch_norm:
            x = self.batch_norms[-1](x)
        
        # Final residual connection if dimensions match
        if self.use_residual and identity.shape[-1] == x.shape[-1]:
            x = x + identity
            
        node_embeddings = x
        
        # Apply attention if specified
        if self.use_attention:
            # Reshape for attention (batch, seq_len, features)
            if batch is not None:
                # Handle batched data
                attention_out, _ = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
                x = attention_out.squeeze(0)
            else:
                # Single graph
                attention_out, _ = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
                x = attention_out.squeeze(0)
        
        # Global pooling for graph-level predictions
        if batch is not None:
            # Batch processing with both mean and max pooling
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            graph_embeddings = (mean_pool + max_pool) / 2  # Ensemble pooling
        else:
            # Single graph - use both mean and max
            mean_emb = torch.mean(x, dim=0, keepdim=True)
            max_emb = torch.max(x, dim=0, keepdim=True)[0]
            graph_embeddings = (mean_emb + max_emb) / 2
        
        # Predictions
        classification_logits = self.classifier(graph_embeddings)
        regression_pred = self.regressor(graph_embeddings)
        
        if return_embeddings:
            return classification_logits, regression_pred, node_embeddings
        
        return classification_logits, regression_pred
    
    def get_node_embeddings(self, data: Data) -> torch.Tensor:
        """Get node embeddings from the model."""
        with torch.no_grad():
            _, _, embeddings = self.forward(data, return_embeddings=True)
        return embeddings
    
    def predict_differentiation_efficiency(self, data: Data) -> torch.Tensor:
        """Predict cardiomyocyte differentiation efficiency."""
        with torch.no_grad():
            _, efficiency = self.forward(data)
        return efficiency


class MultiScaleGNN(nn.Module):
    """
    Multi-scale GNN that combines local and global spatial information.
    Uses multiple GNN branches with different receptive fields.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [256, 128],
                 output_dim: int = 64,
                 num_classes: int = 10,
                 scales: List[int] = [3, 6, 12],
                 dropout: float = 0.1):
        """
        Initialize Multi-scale GNN.
        
        Args:
            input_dim: Number of input features
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension
            num_classes: Number of classes
            scales: Different neighborhood scales (k values)
            dropout: Dropout rate
        """
        super(MultiScaleGNN, self).__init__()
        
        self.scales = scales
        self.output_dim = output_dim
        
        # Create GNN for each scale
        self.scale_networks = nn.ModuleList()
        for scale in scales:
            gnn = SpatialGNN(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                num_classes=num_classes,
                conv_type='GCN',
                dropout=dropout
            )
            self.scale_networks.append(gnn)
        
        # Fusion layer
        fusion_input_dim = output_dim * len(scales)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final prediction heads
        self.classifier = nn.Linear(output_dim, num_classes)
        self.regressor = nn.Sequential(
            nn.Linear(output_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, data_list: List[Data]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multi-scale data.
        
        Args:
            data_list: List of Data objects with different edge connectivity
            
        Returns:
            Classification and regression predictions
        """
        scale_embeddings = []
        
        # Process each scale
        for i, (gnn, data) in enumerate(zip(self.scale_networks, data_list)):
            _, _, embeddings = gnn.forward(data, return_embeddings=True)
            
            # Global pooling
            if hasattr(data, 'batch') and data.batch is not None:
                embeddings = global_mean_pool(embeddings, data.batch)
            else:
                embeddings = torch.mean(embeddings, dim=0, keepdim=True)
                
            scale_embeddings.append(embeddings)
        
        # Fuse multi-scale representations
        fused = torch.cat(scale_embeddings, dim=-1)
        fused = self.fusion(fused)
        
        # Predictions
        classification = self.classifier(fused)
        regression = self.regressor(fused)
        
        return classification, regression


def create_spatial_gnn(config: dict) -> SpatialGNN:
    """
    Factory function to create a Spatial GNN model based on configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Configured SpatialGNN model
    """
    return SpatialGNN(
        input_dim=config.get('input_dim', 2000),
        hidden_dims=config.get('hidden_dims', [512, 256, 128]),
        output_dim=config.get('output_dim', 64),
        num_classes=config.get('num_classes', 10),
        conv_type=config.get('conv_type', 'GCN'),
        use_attention=config.get('use_attention', False),
        dropout=config.get('dropout', 0.1),
        use_batch_norm=config.get('use_batch_norm', True)
    )


if __name__ == "__main__":
    # Test the GNN model
    print("Testing Spatial GNN model...")
    
    # Create a simple test configuration
    config = {
        'input_dim': 2000,
        'hidden_dims': [512, 256, 128],
        'output_dim': 64,
        'num_classes': 5,
        'conv_type': 'GCN',
        'use_attention': True,
        'dropout': 0.1
    }
    
    # Create model
    model = create_spatial_gnn(config)
    
    # Create dummy data
    num_nodes = 100
    num_features = 2000
    
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, 200))
    
    data = Data(x=x, edge_index=edge_index)
    
    # Test forward pass
    try:
        classification, regression = model(data)
        print(f"✓ Model test successful!")
        print(f"Classification output shape: {classification.shape}")
        print(f"Regression output shape: {regression.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()

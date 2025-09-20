"""
Advanced GNN Model for Cardiomyocyte Subtype Classification

This module contains the AdvancedCardiomyocyteGNN model that combines GAT and GCN
layers with skip connections for predicting cardiomyocyte differentiation states.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import add_self_loops


class AdvancedCardiomyocyteGNN(torch.nn.Module):
    """Advanced GNN for cardiomyocyte subtype classification.
    
    This model uses a combination of Graph Attention Networks (GAT) and Graph
    Convolutional Networks (GCN) with skip connections and feature fusion to
    classify cardiomyocyte subtypes based on spatial gene expression data.
    
    Args:
        num_features (int): Number of input features (genes)
        num_classes (int): Number of cardiomyocyte subtypes to classify
        hidden_dim (int): Hidden dimension size
        dropout (float): Dropout probability
    """
    
    def __init__(self, num_features, num_classes=5, hidden_dim=128, dropout=0.4):
        super().__init__()
        self.dropout = dropout
        self.num_classes = num_classes
        
        # Multi-scale feature extraction
        self.gat1 = GATConv(num_features, hidden_dim, heads=6, dropout=dropout, concat=False)
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=4, dropout=dropout, concat=False)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim//2)
        
        # Skip connections and feature fusion
        self.skip_projection = torch.nn.Linear(num_features, hidden_dim//2)
        self.feature_fusion = torch.nn.Linear(hidden_dim + hidden_dim//2, hidden_dim//2)
        
        # Advanced classifier with residual connections
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim//2, hidden_dim//4),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim//4, hidden_dim//8),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout//2),
            torch.nn.Linear(hidden_dim//8, num_classes)
        )
        
        # Layer normalization
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)
        self.norm3 = torch.nn.LayerNorm(hidden_dim//2)
        
    def forward(self, data):
        """Forward pass through the network.
        
        Args:
            data: PyTorch Geometric data object with x (node features) and edge_index
            
        Returns:
            torch.Tensor: Logits for cardiomyocyte subtype classification
        """
        x, edge_index = data.x, data.edge_index
        original_x = x
        
        # Add self loops for better connectivity
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # First layer: GAT + GCN combination
        x1_gat = F.relu(self.norm1(self.gat1(x, edge_index)))
        x1_gat = F.dropout(x1_gat, p=self.dropout, training=self.training)
        
        x1_gcn = F.relu(self.gcn1(x1_gat, edge_index))
        x1_gcn = F.dropout(x1_gcn, p=self.dropout, training=self.training)
        
        # Second layer: Another GAT + GCN combination  
        x2_gat = F.relu(self.norm2(self.gat2(x1_gcn, edge_index)))
        x2_gat = F.dropout(x2_gat, p=self.dropout, training=self.training)
        
        x2_gcn = F.relu(self.gcn2(x2_gat, edge_index))
        x2_gcn = F.dropout(x2_gcn, p=self.dropout//2, training=self.training)
        
        # Skip connection from input
        skip_features = F.relu(self.skip_projection(original_x))
        
        # Feature fusion
        combined_features = torch.cat([x1_gcn, x2_gcn], dim=1)
        fused_features = F.relu(self.feature_fusion(combined_features))
        
        # Add skip connection
        final_features = self.norm3(fused_features + skip_features)
        
        # Classification
        out = self.classifier(final_features)
        return out
    
    def get_model_info(self):
        """Get information about the model architecture.
        
        Returns:
            dict: Model information including parameter count and architecture details
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'AdvancedCardiomyocyteGNN',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.num_classes,
            'architecture': 'GAT+GCN with skip connections',
            'components': {
                'gat_layers': 2,
                'gcn_layers': 2,
                'skip_connections': True,
                'feature_fusion': True,
                'layer_normalization': True
            }
        }
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
        
        # Input feature normalization
        self.input_norm = torch.nn.BatchNorm1d(num_features)
        
        # Multi-scale feature extraction with reduced complexity
        self.gat1 = GATConv(num_features, hidden_dim, heads=4, dropout=dropout, concat=False)  # Reduced heads from 6 to 4
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        
        self.gat2 = GATConv(hidden_dim, hidden_dim//2, heads=2, dropout=dropout, concat=False)  # Reduced heads from 4 to 2
        self.gcn2 = GCNConv(hidden_dim//2, hidden_dim//2)  # Reduced output dim
        
        # Skip connections and feature fusion
        self.skip_projection = torch.nn.Linear(num_features, hidden_dim//2)
        self.feature_fusion = torch.nn.Linear(hidden_dim + hidden_dim//2, hidden_dim//2)
        
        # Simplified classifier with batch norm
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim//2, hidden_dim//4),
            torch.nn.BatchNorm1d(hidden_dim//4),  # Added batch norm
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim//4, num_classes)  # Removed extra layer
        )
        
        # Layer normalization
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim//2)
        self.norm3 = torch.nn.LayerNorm(hidden_dim//2)
        
        # Learnable dropout rates
        self.adaptive_dropout1 = torch.nn.Dropout(dropout)
        self.adaptive_dropout2 = torch.nn.Dropout(dropout * 0.8)  # Reduced dropout in later layers
        
    def forward(self, data, return_attention=False):
        """Forward pass through the network.
        
        Args:
            data: PyTorch Geometric data object with x (node features) and edge_index
            return_attention: Whether to return attention weights for visualization
            
        Returns:
            torch.Tensor or tuple: Logits for cardiomyocyte subtype classification
                                  If return_attention=True, returns (logits, attention_dict)
        """
        x, edge_index = data.x, data.edge_index
        original_x = x
        attention_weights = {}
        
        # Input normalization
        x = self.input_norm(x)
        
        # Add self loops for better connectivity
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # First layer: GAT + GCN combination with improved regularization
        if return_attention:
            x1_gat, (edge_index_gat1, alpha1) = self.gat1(x, edge_index, return_attention_weights=True)
            attention_weights['gat1'] = self._edge_attention_to_matrix(alpha1, edge_index_gat1, x.size(0))
        else:
            x1_gat = self.gat1(x, edge_index)
        
        x1_gat = F.relu(self.norm1(x1_gat))
        x1_gat = self.adaptive_dropout1(x1_gat)  # Use adaptive dropout
        
        x1_gcn = F.relu(self.gcn1(x1_gat, edge_index))
        x1_gcn = self.adaptive_dropout1(x1_gcn)
        
        # Second layer: Another GAT + GCN combination  
        if return_attention:
            x2_gat, (edge_index_gat2, alpha2) = self.gat2(x1_gcn, edge_index, return_attention_weights=True)
            attention_weights['gat2'] = self._edge_attention_to_matrix(alpha2, edge_index_gat2, x1_gcn.size(0))
        else:
            x2_gat = self.gat2(x1_gcn, edge_index)
        
        x2_gat = F.relu(self.norm2(x2_gat))
        x2_gat = self.adaptive_dropout2(x2_gat)  # Reduced dropout
        
        x2_gcn = F.relu(self.gcn2(x2_gat, edge_index))
        x2_gcn = self.adaptive_dropout2(x2_gcn)
        
        # Skip connection from normalized input
        skip_features = F.relu(self.skip_projection(x))  # Use normalized input
        
        # Feature fusion
        combined_features = torch.cat([x1_gcn, x2_gcn], dim=1)
        fused_features = F.relu(self.feature_fusion(combined_features))
        
        # Add skip connection
        final_features = self.norm3(fused_features + skip_features)
        
        # Classification
        out = self.classifier(final_features)
        
        if return_attention:
            return out, attention_weights
        return out
    
    def _edge_attention_to_matrix(self, edge_attention, edge_index, num_nodes):
        """Convert edge-wise attention weights to adjacency matrix format.
        
        Args:
            edge_attention: Attention weights for each edge [num_edges]
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes
            
        Returns:
            torch.Tensor: Attention matrix [num_nodes, num_nodes]
        """
        attention_matrix = torch.zeros(num_nodes, num_nodes, device=edge_attention.device)
        
        # Handle multi-head attention by taking mean if needed
        if edge_attention.dim() > 1:
            edge_attention = edge_attention.mean(dim=1)
        
        attention_matrix[edge_index[0], edge_index[1]] = edge_attention
        return attention_matrix
    
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
"""
Enhanced Spatial GNN with Advanced Features for Better Accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data
import logging

logger = logging.getLogger(__name__)

class MultiHeadGraphAttention(nn.Module):
    """Multi-head graph attention mechanism"""
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        output = self.output(attended)
        return output, attention_weights.mean(dim=1)  # Return average attention weights

class ResidualGCNBlock(nn.Module):
    """Residual GCN block with skip connections"""
    def __init__(self, input_dim, hidden_dim, conv_type='GCN', dropout=0.3):
        super().__init__()
        
        if conv_type == 'GCN':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif conv_type == 'GAT':
            self.conv1 = GATConv(input_dim, hidden_dim // 4, heads=4, concat=True, dropout=dropout)
            self.conv2 = GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True, dropout=dropout)
        else:
            self.conv1 = GraphConv(input_dim, hidden_dim)
            self.conv2 = GraphConv(hidden_dim, hidden_dim)
            
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection projection if dimensions don't match
        if input_dim != hidden_dim:
            self.skip_projection = nn.Linear(input_dim, hidden_dim)
        else:
            self.skip_projection = None
            
    def forward(self, x, edge_index):
        # First convolution
        out = self.conv1(x, edge_index)
        out = self.norm1(out)
        out = F.gelu(out)  # Use GELU activation
        out = self.dropout(out)
        
        # Second convolution
        out = self.conv2(out, edge_index)
        out = self.norm2(out)
        
        # Skip connection
        if self.skip_projection is not None:
            x = self.skip_projection(x)
        
        out = out + x  # Residual connection
        out = F.gelu(out)
        out = self.dropout(out)
        
        return out

class EnhancedSpatialGNN(nn.Module):
    """Enhanced Spatial GNN with advanced features for better accuracy"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: list = [512, 256, 128],
                 output_dim: int = 128,
                 num_classes: int = 5,
                 conv_type: str = 'GAT',
                 num_heads: int = 4,
                 dropout: float = 0.4,
                 use_residual: bool = True,
                 use_attention: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.conv_type = conv_type
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_attention = use_attention
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Residual GCN blocks
        self.conv_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            input_size = hidden_dims[0] if i == 0 else hidden_dims[i-1]
            output_size = hidden_dims[i]
            
            if use_residual:
                block = ResidualGCNBlock(input_size, output_size, conv_type, dropout)
            else:
                if conv_type == 'GCN':
                    block = GCNConv(input_size, output_size)
                elif conv_type == 'GAT':
                    block = GATConv(input_size, output_size // num_heads, heads=num_heads, concat=True, dropout=dropout)
                else:
                    block = GraphConv(input_size, output_size)
                    
            self.conv_blocks.append(block)
        
        # Final projection to embedding space
        self.embedding_projection = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Multi-head attention for graph-level representation
        if use_attention:
            self.graph_attention = MultiHeadGraphAttention(output_dim, output_dim, num_heads)
        
        # Advanced classification head with multiple paths
        self.classifier = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),  # 3x because of mean+max+add pooling
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim // 2),
            nn.LayerNorm(output_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(output_dim // 2, num_classes)
        )
        
        # Regression head for efficiency prediction
        self.regressor = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim // 2),
            nn.LayerNorm(output_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(output_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"Enhanced SpatialGNN initialized with {conv_type} convolutions")
        logger.info(f"Architecture: {input_dim} -> {hidden_dims} -> {output_dim} -> {num_classes}")
        
    def _init_weights(self, module):
        """Initialize weights with Xavier initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, data: Data, return_embeddings: bool = False):
        """Enhanced forward pass with better feature extraction"""
        x, edge_index, batch = data.x, data.edge_index, data.batch if hasattr(data, 'batch') else None
        
        # Input projection
        x = self.input_projection(x)
        
        # Apply graph convolution blocks
        for i, block in enumerate(self.conv_blocks):
            if self.use_residual and hasattr(block, 'forward'):
                x = block(x, edge_index)
            else:
                x = block(x, edge_index)
                x = F.gelu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Project to embedding space
        node_embeddings = self.embedding_projection(x)
        
        # Multi-scale graph pooling
        if batch is not None:
            # Multiple pooling strategies
            mean_pool = global_mean_pool(node_embeddings, batch)
            max_pool = global_max_pool(node_embeddings, batch)
            add_pool = global_add_pool(node_embeddings, batch)
            
            # Combine pooling strategies
            graph_embeddings = torch.cat([mean_pool, max_pool, add_pool], dim=1)
        else:
            # Single graph
            mean_emb = torch.mean(node_embeddings, dim=0, keepdim=True)
            max_emb = torch.max(node_embeddings, dim=0, keepdim=True)[0]
            add_emb = torch.sum(node_embeddings, dim=0, keepdim=True)
            graph_embeddings = torch.cat([mean_emb, max_emb, add_emb], dim=1)
        
        # Apply attention mechanism if enabled
        if self.use_attention and batch is not None:
            # Reshape for attention (treat each graph as a sequence of 1)
            attention_input = graph_embeddings.view(1, graph_embeddings.size(0), -1)
            attended_features, attention_weights = self.graph_attention(attention_input)
            graph_embeddings = attended_features.squeeze(0)
        
        # Predictions
        classification_logits = self.classifier(graph_embeddings)
        regression_pred = self.regressor(graph_embeddings)
        
        if return_embeddings:
            return classification_logits, regression_pred, node_embeddings
        
        return classification_logits, regression_pred
    
    def get_attention_weights(self, data: Data):
        """Get attention weights for visualization"""
        self.eval()
        with torch.no_grad():
            x, edge_index, batch = data.x, data.edge_index, data.batch if hasattr(data, 'batch') else None
            
            # Forward pass through convolutions
            x = self.input_projection(x)
            for block in self.conv_blocks:
                if self.use_residual:
                    x = block(x, edge_index)
                else:
                    x = block(x, edge_index)
                    x = F.gelu(x)
            
            node_embeddings = self.embedding_projection(x)
            
            if batch is not None and self.use_attention:
                mean_pool = global_mean_pool(node_embeddings, batch)
                max_pool = global_max_pool(node_embeddings, batch)
                add_pool = global_add_pool(node_embeddings, batch)
                graph_embeddings = torch.cat([mean_pool, max_pool, add_pool], dim=1)
                
                attention_input = graph_embeddings.view(1, graph_embeddings.size(0), -1)
                _, attention_weights = self.graph_attention(attention_input)
                return attention_weights.squeeze(0)
            
        return None

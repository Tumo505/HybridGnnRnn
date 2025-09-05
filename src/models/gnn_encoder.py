"""
Graph Neural Network Encoder for Spatial Data

This module implements the GNN component of the hybrid framework for modeling
spatial interactions in cardiac tissue data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np


class SpatialGNNEncoder(nn.Module):
    """
    Graph Neural Network encoder for spatial transcriptomics data.
    
    Processes spatial coordinates and gene expression to learn tissue architecture
    and cell-cell interactions for cardiomyocyte differentiation prediction.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 3,
        gnn_type: str = "GraphSAGE",
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        attention_heads: int = 4
    ):
        super(SpatialGNNEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout
        self.use_positional_encoding = use_positional_encoding
        
        # Positional encoding for spatial coordinates
        if self.use_positional_encoding:
            self.pos_encoder = PositionalEncoder(coord_dim=2, embed_dim=32)
            self.input_projection = nn.Linear(input_dim + 32, hidden_dim)
        else:
            self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        
        if gnn_type == "GCN":
            for i in range(num_layers):
                in_dim = hidden_dim if i > 0 else hidden_dim
                self.gnn_layers.append(GCNConv(in_dim, hidden_dim))
                
        elif gnn_type == "GAT":
            for i in range(num_layers):
                in_dim = hidden_dim if i > 0 else hidden_dim
                out_dim = hidden_dim if i < num_layers - 1 else hidden_dim
                self.gnn_layers.append(
                    GATConv(in_dim, out_dim // attention_heads, heads=attention_heads, dropout=dropout)
                )
                
        elif gnn_type == "GraphSAGE":
            for i in range(num_layers):
                in_dim = hidden_dim if i > 0 else hidden_dim
                self.gnn_layers.append(GraphSAGE(in_dim, hidden_dim, num_layers=1))
        
        # Batch normalization and dropout
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # Graph-level pooling for batch processing
        self.global_pool = global_mean_pool
        
    def forward(self, x, edge_index, pos=None, batch=None):
        """
        Forward pass of the GNN encoder.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, input_dim]
            edge_index (torch.Tensor): Edge indices [2, num_edges]
            pos (torch.Tensor, optional): Spatial coordinates [num_nodes, 2]
            batch (torch.Tensor, optional): Batch assignment [num_nodes]
            
        Returns:
            torch.Tensor: Encoded spatial representations [batch_size, output_dim]
        """
        # Add positional encoding if enabled
        if self.use_positional_encoding and pos is not None:
            pos_encoding = self.pos_encoder(pos)
            x = torch.cat([x, pos_encoding], dim=-1)
        
        # Input projection
        x = self.input_projection(x)
        x = F.relu(x)
        
        # GNN layers
        for i, (gnn_layer, batch_norm, dropout) in enumerate(
            zip(self.gnn_layers, self.batch_norms, self.dropout_layers)
        ):
            residual = x
            
            # Apply GNN layer
            if self.gnn_type == "GraphSAGE":
                x = gnn_layer(x, edge_index)
            else:
                x = gnn_layer(x, edge_index)
            
            # Batch normalization and activation
            x = batch_norm(x)
            x = F.relu(x)
            
            # Residual connection (if dimensions match)
            if residual.size(-1) == x.size(-1):
                x = x + residual
            
            # Dropout
            x = dropout(x)
        
        # Output projection
        x = self.output_projection(x)
        
        # Global pooling for graph-level representation
        if batch is not None:
            x = self.global_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        return x


class PositionalEncoder(nn.Module):
    """
    Positional encoding for spatial coordinates using sinusoidal functions.
    """
    
    def __init__(self, coord_dim: int = 2, embed_dim: int = 32, max_freq: float = 10.0):
        super(PositionalEncoder, self).__init__()
        
        self.coord_dim = coord_dim
        self.embed_dim = embed_dim
        self.max_freq = max_freq
        
        # Create frequency bands
        freq_bands = torch.linspace(1.0, max_freq, embed_dim // (2 * coord_dim))
        self.register_buffer('freq_bands', freq_bands)
        
    def forward(self, pos):
        """
        Apply positional encoding to spatial coordinates.
        
        Args:
            pos (torch.Tensor): Spatial coordinates [num_nodes, coord_dim]
            
        Returns:
            torch.Tensor: Positional encodings [num_nodes, embed_dim]
        """
        encodings = []
        
        for i in range(self.coord_dim):
            coord = pos[:, i:i+1]  # [num_nodes, 1]
            
            # Apply sinusoidal encoding
            for freq in self.freq_bands:
                encodings.append(torch.sin(coord * freq))
                encodings.append(torch.cos(coord * freq))
        
        return torch.cat(encodings, dim=-1)


class SpatialGraphConstructor:
    """
    Utility class for constructing spatial graphs from coordinate data.
    """
    
    @staticmethod
    def create_spatial_graph(coordinates, k_neighbors=6, distance_threshold=None):
        """
        Create spatial graph from coordinates.
        
        Args:
            coordinates (np.ndarray): Spatial coordinates [num_spots, 2]
            k_neighbors (int): Number of nearest neighbors
            distance_threshold (float, optional): Maximum distance for edges
            
        Returns:
            torch.Tensor: Edge indices [2, num_edges]
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(coordinates)
        distances, indices = nbrs.kneighbors(coordinates)
        
        # Create edge list
        edges = []
        for i in range(len(coordinates)):
            for j in range(1, len(indices[i])):  # Skip self (index 0)
                neighbor_idx = indices[i][j]
                distance = distances[i][j]
                
                # Add edge if within distance threshold
                if distance_threshold is None or distance <= distance_threshold:
                    edges.append([i, neighbor_idx])
        
        # Convert to tensor and make undirected
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        # Make graph undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_index = torch.unique(edge_index, dim=1)
        
        return edge_index
    
    @staticmethod
    def create_batch_from_spatial_data(spatial_data_list):
        """
        Create PyTorch Geometric batch from list of spatial data.
        
        Args:
            spatial_data_list (list): List of dictionaries with spatial data
            
        Returns:
            torch_geometric.data.Batch: Batched graph data
        """
        data_list = []
        
        for data_dict in spatial_data_list:
            # Extract components
            x = torch.tensor(data_dict['features'], dtype=torch.float)
            pos = torch.tensor(data_dict['coordinates'], dtype=torch.float)
            edge_index = data_dict['edge_index']
            
            # Create PyG Data object
            data = Data(x=x, pos=pos, edge_index=edge_index)
            
            # Add labels if available
            if 'labels' in data_dict:
                data.y = torch.tensor(data_dict['labels'], dtype=torch.float)
            
            data_list.append(data)
        
        return Batch.from_data_list(data_list)


if __name__ == "__main__":
    # Example usage and testing
    torch.manual_seed(42)
    
    # Create sample spatial data
    num_spots = 100
    input_dim = 500  # Number of genes
    coordinates = np.random.uniform(0, 10, (num_spots, 2))
    features = torch.randn(num_spots, input_dim)
    
    # Create spatial graph
    graph_constructor = SpatialGraphConstructor()
    edge_index = graph_constructor.create_spatial_graph(coordinates, k_neighbors=6)
    pos = torch.tensor(coordinates, dtype=torch.float)
    
    print(f"Created spatial graph:")
    print(f"  Spots: {num_spots}")
    print(f"  Features: {input_dim}")
    print(f"  Edges: {edge_index.size(1)}")
    
    # Create GNN encoder
    gnn_encoder = SpatialGNNEncoder(
        input_dim=input_dim,
        hidden_dim=256,
        output_dim=128,
        num_layers=3,
        gnn_type="GraphSAGE",
        use_positional_encoding=True
    )
    
    print(f"\nGNN Encoder:")
    print(f"  Parameters: {sum(p.numel() for p in gnn_encoder.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        output = gnn_encoder(features, edge_index, pos)
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

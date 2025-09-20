"""
Adaptive graph construction for dynamic Graph Neural Networks.

This module implements dynamic edge pruning and addition during training
based on attention weights, node similarity metrics, and biological
relevance for improved model performance and interpretability.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.utils import to_undirected, coalesce
import warnings
warnings.filterwarnings('ignore')


class AdaptiveGraphConstructor:
    """Dynamic graph construction for GNN training."""
    
    def __init__(self, 
                 attention_threshold: float = 0.1,
                 similarity_threshold: float = 0.8,
                 max_edges_per_node: int = 50,
                 update_frequency: int = 10):
        """
        Initialize adaptive graph constructor.
        
        Args:
            attention_threshold: Minimum attention weight to maintain edge
            similarity_threshold: Minimum similarity to add new edge
            max_edges_per_node: Maximum number of edges per node
            update_frequency: How often to update graph (in epochs)
        """
        self.attention_threshold = attention_threshold
        self.similarity_threshold = similarity_threshold
        self.max_edges_per_node = max_edges_per_node
        self.update_frequency = update_frequency
        
        # Track graph evolution
        self.graph_history = []
        self.edge_statistics = {
            'added_edges': 0,
            'removed_edges': 0,
            'total_updates': 0
        }
        
    def compute_node_similarity(self, 
                              node_features: torch.Tensor,
                              similarity_metric: str = 'cosine') -> torch.Tensor:
        """
        Compute pairwise node similarity matrix.
        
        Args:
            node_features: Node feature matrix [num_nodes, num_features]
            similarity_metric: Similarity metric ('cosine', 'euclidean', 'pearson')
            
        Returns:
            Similarity matrix [num_nodes, num_nodes]
        """
        num_nodes = node_features.size(0)
        
        if similarity_metric == 'cosine':
            # Normalize features
            normalized_features = F.normalize(node_features, p=2, dim=1)
            similarity_matrix = torch.mm(normalized_features, normalized_features.t())
            
        elif similarity_metric == 'euclidean':
            # Compute pairwise Euclidean distances
            distances = torch.cdist(node_features, node_features, p=2)
            # Convert to similarity (higher = more similar)
            max_dist = distances.max()
            similarity_matrix = 1 - (distances / max_dist)
            
        elif similarity_metric == 'pearson':
            # Compute Pearson correlation coefficient
            centered_features = node_features - node_features.mean(dim=1, keepdim=True)
            correlation_matrix = torch.mm(centered_features, centered_features.t())
            # Normalize by standard deviations
            std_features = torch.std(node_features, dim=1, keepdim=True)
            std_outer = torch.mm(std_features, std_features.t())
            similarity_matrix = correlation_matrix / (std_outer + 1e-8)
            
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")
        
        return similarity_matrix
    
    def prune_edges_by_attention(self, 
                               edge_index: torch.Tensor,
                               attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Prune edges with low attention weights.
        
        Args:
            edge_index: Current edge indices [2, num_edges]
            attention_weights: Attention weights for each edge
            
        Returns:
            Pruned edge indices
        """
        # Handle multi-dimensional attention weights
        if attention_weights.dim() > 1:
            attention_weights = attention_weights.mean(dim=1)
        
        # Find edges with sufficient attention
        keep_mask = attention_weights > self.attention_threshold
        pruned_edge_index = edge_index[:, keep_mask]
        
        # Update statistics
        removed_count = (~keep_mask).sum().item()
        self.edge_statistics['removed_edges'] += removed_count
        
        return pruned_edge_index
    
    def add_edges_by_similarity(self,
                              edge_index: torch.Tensor,
                              similarity_matrix: torch.Tensor,
                              num_nodes: int) -> torch.Tensor:
        """
        Add new edges based on node similarity.
        
        Args:
            edge_index: Current edge indices
            similarity_matrix: Node similarity matrix
            num_nodes: Number of nodes in graph
            
        Returns:
            Updated edge indices with new edges
        """
        # Create adjacency matrix from current edges
        current_adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        current_adj[edge_index[0], edge_index[1]] = 1
        
        # Find potential new edges based on similarity
        potential_edges = similarity_matrix > self.similarity_threshold
        
        # Remove existing edges and self-loops
        potential_edges = potential_edges & (~current_adj.bool())
        potential_edges.fill_diagonal_(False)
        
        # Get candidate edge indices
        candidate_i, candidate_j = torch.where(potential_edges)
        candidate_similarities = similarity_matrix[candidate_i, candidate_j]
        
        # Sort by similarity (highest first)
        sorted_indices = torch.argsort(candidate_similarities, descending=True)
        
        # Limit edges per node
        new_edges = []
        node_edge_count = torch.zeros(num_nodes, device=edge_index.device)
        
        # Count existing edges per node
        unique_nodes, counts = torch.unique(edge_index[0], return_counts=True)
        node_edge_count[unique_nodes] = counts.float()
        
        for idx in sorted_indices:
            i, j = candidate_i[idx], candidate_j[idx]
            
            # Check if we can add more edges to these nodes
            if (node_edge_count[i] < self.max_edges_per_node and 
                node_edge_count[j] < self.max_edges_per_node):
                
                new_edges.extend([[i, j], [j, i]])  # Add both directions
                node_edge_count[i] += 1
                node_edge_count[j] += 1
        
        if new_edges:
            new_edge_tensor = torch.tensor(new_edges, device=edge_index.device).t()
            updated_edge_index = torch.cat([edge_index, new_edge_tensor], dim=1)
            
            # Update statistics
            self.edge_statistics['added_edges'] += len(new_edges) // 2  # Count undirected edges
        else:
            updated_edge_index = edge_index
        
        return updated_edge_index
    
    def adaptive_graph_update(self,
                            data: Data,
                            attention_weights: Optional[torch.Tensor] = None,
                            epoch: int = 0) -> Data:
        """
        Perform adaptive graph update based on attention and similarity.
        
        Args:
            data: PyTorch Geometric data object
            attention_weights: Optional attention weights from model
            epoch: Current training epoch
            
        Returns:
            Updated data object with modified graph structure
        """
        # Only update graph at specified intervals
        if epoch % self.update_frequency != 0:
            return data
        
        num_nodes = data.x.size(0)
        edge_index = data.edge_index
        
        # Remove self-loops temporarily
        edge_index, _ = remove_self_loops(edge_index)
        
        # Prune edges by attention if available
        if attention_weights is not None:
            # Convert attention matrix to edge weights if needed
            if attention_weights.dim() == 2:  # Adjacency matrix format
                edge_weights = attention_weights[edge_index[0], edge_index[1]]
            else:  # Already edge weights
                edge_weights = attention_weights
            
            edge_index = self.prune_edges_by_attention(edge_index, edge_weights)
        
        # Compute node similarity
        similarity_matrix = self.compute_node_similarity(data.x, 'cosine')
        
        # Add new edges based on similarity
        edge_index = self.add_edges_by_similarity(edge_index, similarity_matrix, num_nodes)
        
        # Ensure graph is undirected and remove duplicates
        edge_index = to_undirected(edge_index)
        edge_index = coalesce(edge_index, num_nodes=num_nodes)[0]
        
        # Add self-loops back
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        # Create updated data object
        updated_data = Data(x=data.x, edge_index=edge_index)
        if hasattr(data, 'y'):
            updated_data.y = data.y
        if hasattr(data, 'batch'):
            updated_data.batch = data.batch
        
        # Record graph statistics
        self.graph_history.append({
            'epoch': epoch,
            'num_edges': edge_index.size(1),
            'num_nodes': num_nodes,
            'avg_degree': edge_index.size(1) / num_nodes
        })
        
        self.edge_statistics['total_updates'] += 1
        
        return updated_data
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive graph evolution statistics.
        
        Returns:
            Dictionary containing graph statistics
        """
        stats = {
            'edge_changes': self.edge_statistics.copy(),
            'graph_evolution': self.graph_history.copy()
        }
        
        if self.graph_history:
            # Calculate evolution metrics
            initial_edges = self.graph_history[0]['num_edges']
            final_edges = self.graph_history[-1]['num_edges']
            
            stats['evolution_metrics'] = {
                'initial_edges': initial_edges,
                'final_edges': final_edges,
                'edge_change_ratio': final_edges / initial_edges if initial_edges > 0 else 0,
                'total_epochs_tracked': len(self.graph_history)
            }
        
        return stats
    
    def plot_graph_evolution(self, save_path: str = None):
        """
        Plot graph evolution over training.
        
        Args:
            save_path: Optional path to save plot
        """
        if not self.graph_history:
            print("No graph history available for plotting.")
            return
        
        import matplotlib.pyplot as plt
        
        epochs = [entry['epoch'] for entry in self.graph_history]
        num_edges = [entry['num_edges'] for entry in self.graph_history]
        avg_degrees = [entry['avg_degree'] for entry in self.graph_history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot number of edges
        ax1.plot(epochs, num_edges, 'b-o', linewidth=2, markersize=4)
        ax1.set_title('Graph Edge Evolution')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Number of Edges')
        ax1.grid(True, alpha=0.3)
        
        # Plot average degree
        ax2.plot(epochs, avg_degrees, 'r-o', linewidth=2, markersize=4)
        ax2.set_title('Average Node Degree Evolution')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Average Degree')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def biological_edge_filtering(self,
                                edge_index: torch.Tensor,
                                node_features: torch.Tensor,
                                feature_names: List[str],
                                biological_constraints: Dict[str, List[str]] = None) -> torch.Tensor:
        """
        Filter edges based on biological constraints and gene expression patterns.
        
        Args:
            edge_index: Current edge indices
            node_features: Node feature matrix
            feature_names: List of feature (gene) names
            biological_constraints: Dictionary of biological pathway constraints
            
        Returns:
            Biologically filtered edge indices
        """
        if biological_constraints is None:
            # Default cardiac-specific constraints
            biological_constraints = {
                'calcium_signaling': ['CACNA1C', 'RYR2', 'ATP2A2', 'PLN'],
                'electrical_conduction': ['SCN5A', 'KCNH2', 'KCNQ1', 'HCN4'],
                'contractile': ['MYH6', 'MYH7', 'TNNT2', 'TNNI3']
            }
        
        # Create feature mapping
        feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        
        # Calculate pathway-based similarity
        pathway_similarities = torch.zeros(node_features.size(0), node_features.size(0))
        
        for pathway_genes in biological_constraints.values():
            # Find indices of pathway genes
            pathway_indices = [feature_to_idx[gene] for gene in pathway_genes 
                             if gene in feature_to_idx]
            
            if len(pathway_indices) > 1:
                # Calculate similarity within pathway
                pathway_features = node_features[:, pathway_indices]
                pathway_sim = F.cosine_similarity(pathway_features.unsqueeze(1), 
                                                pathway_features.unsqueeze(0), dim=2)
                pathway_similarities += pathway_sim
        
        # Normalize by number of pathways
        pathway_similarities /= len(biological_constraints)
        
        # Filter edges based on biological similarity
        edge_similarities = pathway_similarities[edge_index[0], edge_index[1]]
        bio_threshold = 0.3  # Biological similarity threshold
        
        keep_mask = edge_similarities > bio_threshold
        filtered_edge_index = edge_index[:, keep_mask]
        
        return filtered_edge_index
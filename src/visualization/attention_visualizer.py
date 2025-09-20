"""
Attention mechanism visualization for Graph Neural Networks.

This module provides tools to visualize and interpret attention weights
from GAT (Graph Attention Network) layers, enabling understanding of
which nodes and connections the model focuses on during prediction.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import warnings
warnings.filterwarnings('ignore')


class AttentionVisualizer:
    """Visualization tools for GAT attention mechanisms."""
    
    def __init__(self, save_dir: str = "attention_visualizations"):
        """
        Initialize the attention visualizer.
        
        Args:
            save_dir: Directory to save attention visualization plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def plot_attention_heatmap(self,
                             attention_weights: torch.Tensor,
                             node_ids: Optional[List[int]] = None,
                             title: str = "Attention Heatmap",
                             save_name: str = "attention_heatmap.png") -> None:
        """
        Plot attention weights as a heatmap.
        
        Args:
            attention_weights: Attention weight matrix [num_nodes, num_nodes]
            node_ids: Optional list of node identifiers
            title: Plot title
            save_name: Filename to save the plot
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        mask = attention_weights == 0  # Mask zero attention weights
        sns.heatmap(attention_weights, 
                   mask=mask,
                   cmap='YlOrRd', 
                   square=True,
                   cbar_kws={'label': 'Attention Weight'},
                   xticklabels=node_ids[:50] if node_ids else False,  # Limit labels for readability
                   yticklabels=node_ids[:50] if node_ids else False)
        
        plt.title(f'{title}\n(Showing top 50x50 nodes if applicable)', fontsize=14)
        plt.xlabel('Target Nodes', fontsize=12)
        plt.ylabel('Source Nodes', fontsize=12)
        
        if node_ids and len(node_ids) <= 50:
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_attention_graph(self,
                           data: Data,
                           attention_weights: torch.Tensor,
                           node_labels: Optional[torch.Tensor] = None,
                           threshold: float = 0.1,
                           max_nodes: int = 100,
                           title: str = "Attention Graph",
                           save_name: str = "attention_graph.png") -> None:
        """
        Plot graph with attention-weighted edges.
        
        Args:
            data: PyTorch Geometric data object
            attention_weights: Attention weights for edges
            node_labels: Optional node class labels for coloring
            threshold: Minimum attention weight to display edge
            max_nodes: Maximum number of nodes to display
            title: Plot title
            save_name: Filename to save the plot
        """
        # Convert to NetworkX graph
        G = to_networkx(data, to_undirected=True)
        
        # Limit graph size for visualization
        if len(G.nodes()) > max_nodes:
            # Select most connected nodes
            node_degrees = dict(G.degree())
            top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            top_node_ids = [node for node, degree in top_nodes]
            G = G.subgraph(top_node_ids)
        
        # Prepare attention weights
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Create figure
        plt.figure(figsize=(15, 12))
        
        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        if node_labels is not None:
            if isinstance(node_labels, torch.Tensor):
                node_labels = node_labels.detach().cpu().numpy()
            node_colors = node_labels[:len(G.nodes())]
            cmap = plt.cm.Set3
        else:
            node_colors = 'lightblue'
            cmap = None
        
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors,
                              node_size=100,
                              cmap=cmap,
                              alpha=0.8)
        
        # Draw edges with attention weights
        edge_list = list(G.edges())
        edge_weights = []
        
        for edge in edge_list:
            i, j = edge
            if i < attention_weights.shape[0] and j < attention_weights.shape[1]:
                weight = attention_weights[i, j]
                if weight > threshold:
                    edge_weights.append(weight)
                else:
                    edge_weights.append(0)
            else:
                edge_weights.append(0)
        
        # Filter edges by attention threshold
        filtered_edges = [(edge, weight) for edge, weight in zip(edge_list, edge_weights) if weight > threshold]
        
        if filtered_edges:
            edges, weights = zip(*filtered_edges)
            nx.draw_networkx_edges(G, pos,
                                  edgelist=edges,
                                  width=[w * 5 for w in weights],  # Scale for visibility
                                  alpha=0.6,
                                  edge_color=weights,
                                  edge_cmap=plt.cm.Reds)
        
        plt.title(f'{title}\n(Attention threshold: {threshold}, {len(G.nodes())} nodes)', fontsize=14)
        plt.axis('off')
        
        # Add colorbar for attention weights
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, 
                                  norm=plt.Normalize(vmin=threshold, vmax=max(edge_weights) if edge_weights else 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, shrink=0.8)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_node_attention_distribution(self,
                                       attention_weights: torch.Tensor,
                                       node_ids: Optional[List[int]] = None,
                                       top_k: int = 20,
                                       title: str = "Node Attention Distribution",
                                       save_name: str = "node_attention_dist.png") -> None:
        """
        Plot distribution of attention weights per node.
        
        Args:
            attention_weights: Attention weight matrix
            node_ids: Optional node identifiers
            top_k: Number of top nodes to show
            title: Plot title
            save_name: Filename to save the plot
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Calculate attention statistics per node
        node_attention_sum = np.sum(attention_weights, axis=1)
        node_attention_max = np.max(attention_weights, axis=1)
        node_attention_mean = np.mean(attention_weights, axis=1)
        
        # Get top nodes by total attention
        top_indices = np.argsort(node_attention_sum)[-top_k:][::-1]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Bar plot of top nodes by total attention
        labels = [f'Node {i}' if node_ids is None else f'Node {node_ids[i]}' 
                 for i in top_indices]
        
        ax1.bar(range(len(top_indices)), node_attention_sum[top_indices], 
               color='skyblue', alpha=0.8)
        ax1.set_title(f'Top {top_k} Nodes by Total Attention')
        ax1.set_xlabel('Node Rank')
        ax1.set_ylabel('Total Attention Weight')
        ax1.set_xticks(range(len(top_indices)))
        ax1.set_xticklabels([f'{i+1}' for i in range(len(top_indices))])
        
        # Distribution of all attention weights
        ax2.hist(attention_weights.flatten(), bins=50, alpha=0.7, color='lightcoral')
        ax2.set_title('Distribution of All Attention Weights')
        ax2.set_xlabel('Attention Weight')
        ax2.set_ylabel('Frequency')
        ax2.set_yscale('log')
        
        # Scatter plot: max vs mean attention
        ax3.scatter(node_attention_mean, node_attention_max, alpha=0.6, color='forestgreen')
        ax3.set_title('Node Attention: Max vs Mean')
        ax3.set_xlabel('Mean Attention Weight')
        ax3.set_ylabel('Max Attention Weight')
        
        # Highlight top nodes
        for idx in top_indices[:5]:  # Top 5 nodes
            ax3.scatter(node_attention_mean[idx], node_attention_max[idx], 
                       color='red', s=100, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_attention_layer_comparison(self,
                                      attention_layers: Dict[str, torch.Tensor],
                                      title: str = "Attention Layer Comparison",
                                      save_name: str = "attention_layers.png") -> None:
        """
        Compare attention patterns across different GAT layers.
        
        Args:
            attention_layers: Dict mapping layer names to attention weights
            title: Plot title
            save_name: Filename to save the plot
        """
        num_layers = len(attention_layers)
        fig, axes = plt.subplots(2, (num_layers + 1) // 2, figsize=(6 * ((num_layers + 1) // 2), 12))
        
        if num_layers == 1:
            axes = [axes]
        elif num_layers <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for idx, (layer_name, attention_weights) in enumerate(attention_layers.items()):
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = attention_weights.detach().cpu().numpy()
            
            # Show subset for visualization
            display_size = min(50, attention_weights.shape[0])
            subset = attention_weights[:display_size, :display_size]
            
            im = axes[idx].imshow(subset, cmap='YlOrRd', aspect='auto')
            axes[idx].set_title(f'{layer_name}\nAttention Weights')
            axes[idx].set_xlabel('Target Nodes')
            axes[idx].set_ylabel('Source Nodes')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for idx in range(num_layers, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_attention_patterns(self,
                                 attention_weights: torch.Tensor,
                                 node_features: torch.Tensor,
                                 class_labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Analyze attention patterns and return statistical insights.
        
        Args:
            attention_weights: Attention weight matrix
            node_features: Node feature matrix
            class_labels: Optional node class labels
            
        Returns:
            Dictionary containing attention analysis results
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        if isinstance(node_features, torch.Tensor):
            node_features = node_features.detach().cpu().numpy()
        if class_labels is not None and isinstance(class_labels, torch.Tensor):
            class_labels = class_labels.detach().cpu().numpy()
        
        analysis = {}
        
        # Basic statistics
        analysis['total_edges'] = np.sum(attention_weights > 0)
        analysis['mean_attention'] = np.mean(attention_weights[attention_weights > 0])
        analysis['std_attention'] = np.std(attention_weights[attention_weights > 0])
        analysis['max_attention'] = np.max(attention_weights)
        analysis['sparsity'] = 1 - (np.sum(attention_weights > 0) / attention_weights.size)
        
        # Node-level statistics
        node_in_attention = np.sum(attention_weights, axis=0)  # Attention received
        node_out_attention = np.sum(attention_weights, axis=1)  # Attention given
        
        analysis['top_attended_nodes'] = np.argsort(node_in_attention)[-10:].tolist()
        analysis['top_attending_nodes'] = np.argsort(node_out_attention)[-10:].tolist()
        
        # Class-based analysis if labels provided
        if class_labels is not None:
            class_attention = {}
            unique_classes = np.unique(class_labels)
            
            for cls in unique_classes:
                class_mask = class_labels == cls
                class_attention[f'class_{cls}'] = {
                    'mean_received': np.mean(node_in_attention[class_mask]),
                    'mean_given': np.mean(node_out_attention[class_mask]),
                    'count': np.sum(class_mask)
                }
            
            analysis['class_attention'] = class_attention
        
        return analysis
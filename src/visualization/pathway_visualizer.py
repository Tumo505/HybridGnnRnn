"""
Pathway visualization for intercellular signaling analysis.

This module provides tools to visualize and analyze intercellular signaling
pathways, cell-cell communication patterns, and biological pathway activation
through graph connectivity and message passing visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class PathwayVisualizer:
    """Visualization tools for biological pathways and intercellular signaling."""
    
    def __init__(self, save_dir: str = "pathway_visualizations"):
        """
        Initialize the pathway visualizer.
        
        Args:
            save_dir: Directory to save pathway visualization plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Cardiomyocyte-related pathway information
        self.cardiac_pathways = {
            'calcium_signaling': {
                'genes': ['CACNA1C', 'RYR2', 'ATP2A2', 'PLN', 'CASQ2', 'CAMK2D'],
                'description': 'Calcium handling and excitation-contraction coupling'
            },
            'electrical_conduction': {
                'genes': ['SCN5A', 'KCNH2', 'KCNQ1', 'KCNJ2', 'HCN4', 'CONNEXIN43'],
                'description': 'Electrical conduction and action potential'
            },
            'metabolism': {
                'genes': ['PPARA', 'PGC1A', 'CPT1B', 'ACADM', 'HADHA', 'PDK4'],
                'description': 'Metabolic pathways and energy production'
            },
            'sarcomere': {
                'genes': ['MYH6', 'MYH7', 'TNNT2', 'TNNI3', 'TPM1', 'ACTC1'],
                'description': 'Sarcomere structure and contraction'
            },
            'signaling': {
                'genes': ['ADRA1A', 'ADRB1', 'NPPA', 'NPPB', 'ACE', 'AGTR1'],
                'description': 'Adrenergic and neurohumoral signaling'
            }
        }
        
        # Cell type markers
        self.cell_markers = {
            'Atrial CM': ['NPPA', 'MYL7', 'IRX4'],
            'Ventricular CM': ['MYL2', 'IRX4', 'HEY2'],
            'Conducting CM': ['HCN4', 'SHOX2', 'TBX3'],
            'Nodal CM': ['SHOX2', 'TBX3', 'ISL1'],
            'Epicardial CM': ['TBX18', 'WT1', 'UPK3B']
        }
        
    def analyze_pathway_activity(self,
                               node_features: torch.Tensor,
                               feature_names: List[str],
                               cell_labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Analyze pathway activity across different cell types.
        
        Args:
            node_features: Node feature matrix [num_nodes, num_features]
            feature_names: List of feature (gene) names
            cell_labels: Optional cell type labels
            
        Returns:
            Dictionary containing pathway activity analysis
        """
        if isinstance(node_features, torch.Tensor):
            node_features = node_features.detach().cpu().numpy()
        if cell_labels is not None and isinstance(cell_labels, torch.Tensor):
            cell_labels = cell_labels.detach().cpu().numpy()
        
        # Create feature name to index mapping
        feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        
        pathway_activities = {}
        
        # Calculate pathway activity for each pathway
        for pathway_name, pathway_info in self.cardiac_pathways.items():
            pathway_genes = pathway_info['genes']
            
            # Find indices of pathway genes that exist in our features
            gene_indices = []
            found_genes = []
            for gene in pathway_genes:
                if gene in feature_to_idx:
                    gene_indices.append(feature_to_idx[gene])
                    found_genes.append(gene)
            
            if gene_indices:
                # Calculate pathway activity as mean expression of pathway genes
                pathway_expression = node_features[:, gene_indices]
                pathway_activity = np.mean(pathway_expression, axis=1)
                
                pathway_activities[pathway_name] = {
                    'activity_scores': pathway_activity,
                    'found_genes': found_genes,
                    'gene_indices': gene_indices,
                    'mean_activity': np.mean(pathway_activity),
                    'std_activity': np.std(pathway_activity)
                }
        
        # Cell type specific analysis if labels provided
        if cell_labels is not None:
            cell_type_activities = {}
            unique_labels = np.unique(cell_labels)
            
            for label in unique_labels:
                cell_mask = cell_labels == label
                cell_activities = {}
                
                for pathway_name, pathway_data in pathway_activities.items():
                    cell_pathway_activity = pathway_data['activity_scores'][cell_mask]
                    cell_activities[pathway_name] = {
                        'mean': np.mean(cell_pathway_activity),
                        'std': np.std(cell_pathway_activity),
                        'count': np.sum(cell_mask)
                    }
                
                cell_type_activities[f'cell_type_{label}'] = cell_activities
            
            pathway_activities['cell_type_analysis'] = cell_type_activities
        
        return pathway_activities
        
    def plot_pathway_heatmap(self,
                           pathway_activities: Dict[str, Any],
                           title: str = "Pathway Activity Heatmap",
                           save_name: str = "pathway_heatmap.png") -> None:
        """Plot pathway activity as a heatmap."""
        if 'cell_type_analysis' not in pathway_activities:
            print("Cell type analysis not available. Run analyze_pathway_activity with cell labels.")
            return
        
        cell_type_data = pathway_activities['cell_type_analysis']
        
        # Prepare data for heatmap
        pathway_names = list(self.cardiac_pathways.keys())
        cell_types = list(cell_type_data.keys())
        
        heatmap_data = []
        for cell_type in cell_types:
            row = []
            for pathway in pathway_names:
                if pathway in cell_type_data[cell_type]:
                    activity = cell_type_data[cell_type][pathway]['mean']
                    row.append(activity)
                else:
                    row.append(0)
            heatmap_data.append(row)
        
        heatmap_data = np.array(heatmap_data)
        
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(heatmap_data, 
                   xticklabels=[name.replace('_', ' ').title() for name in pathway_names],
                   yticklabels=[ct.replace('cell_type_', 'Type ') for ct in cell_types],
                   annot=True, 
                   fmt='.3f',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Pathway Activity Score'})
        
        plt.title(title, fontsize=14)
        plt.xlabel('Cardiac Pathways', fontsize=12)
        plt.ylabel('Cell Types', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_intercellular_communication(self,
                                       data: Data,
                                       attention_weights: Optional[torch.Tensor] = None,
                                       cell_labels: Optional[torch.Tensor] = None,
                                       threshold: float = 0.1,
                                       title: str = "Intercellular Communication Network",
                                       save_name: str = "communication_network.png") -> None:
        """
        Visualize intercellular communication patterns.
        
        Args:
            data: PyTorch Geometric data object
            attention_weights: Optional attention weights for edge importance
            cell_labels: Optional cell type labels
            threshold: Minimum communication strength to display
            title: Plot title
            save_name: Filename to save the plot
        """
        # Convert to NetworkX graph
        G = to_networkx(data, to_undirected=True)
        
        # Limit graph size for visualization
        if len(G.nodes()) > 200:
            # Sample nodes for visualization
            sampled_nodes = np.random.choice(list(G.nodes()), 200, replace=False)
            G = G.subgraph(sampled_nodes)
        
        plt.figure(figsize=(15, 12))
        
        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Node colors based on cell types
        if cell_labels is not None:
            if isinstance(cell_labels, torch.Tensor):
                cell_labels = cell_labels.detach().cpu().numpy()
            
            node_colors = []
            color_map = plt.cm.Set3
            for node in G.nodes():
                if node < len(cell_labels):
                    node_colors.append(color_map(cell_labels[node]))
                else:
                    node_colors.append('lightgray')
        else:
            node_colors = 'lightblue'
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors,
                              node_size=80,
                              alpha=0.8)
        
        # Draw edges with communication strength
        edge_list = list(G.edges())
        edge_weights = []
        
        if attention_weights is not None:
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = attention_weights.detach().cpu().numpy()
            
            for edge in edge_list:
                i, j = edge
                if (i < attention_weights.shape[0] and j < attention_weights.shape[1]):
                    weight = max(attention_weights[i, j], attention_weights[j, i])  # Symmetric
                    edge_weights.append(weight)
                else:
                    edge_weights.append(0.1)  # Default weight
        else:
            edge_weights = [0.1] * len(edge_list)  # Default weights
        
        # Filter edges by threshold
        filtered_edges = []
        filtered_weights = []
        for edge, weight in zip(edge_list, edge_weights):
            if weight > threshold:
                filtered_edges.append(edge)
                filtered_weights.append(weight)
        
        if filtered_edges:
            nx.draw_networkx_edges(G, pos,
                                  edgelist=filtered_edges,
                                  width=[w * 3 for w in filtered_weights],
                                  alpha=0.6,
                                  edge_color=filtered_weights,
                                  edge_cmap=plt.cm.Reds)
        
        plt.title(f'{title}\n(Threshold: {threshold}, {len(G.nodes())} cells)', fontsize=14)
        plt.axis('off')
        
        # Add legend for cell types if available
        if cell_labels is not None:
            from matplotlib.patches import Patch
            unique_labels = np.unique(cell_labels)
            legend_elements = []
            class_names = ['Atrial CM', 'Ventricular CM', 'Conducting CM', 'Nodal CM', 'Epicardial CM']
            
            for i, label in enumerate(unique_labels[:len(class_names)]):
                legend_elements.append(Patch(facecolor=color_map(label), 
                                           label=class_names[label] if label < len(class_names) else f'Type {label}'))
            
            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        # Add colorbar for communication strength
        if filtered_weights:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, 
                                      norm=plt.Normalize(vmin=threshold, vmax=max(filtered_weights)))
            sm.set_array([])
            cbar = plt.colorbar(sm, shrink=0.8)
            cbar.set_label('Communication Strength', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_signaling_gradients(self,
                                  node_features: torch.Tensor,
                                  positions: torch.Tensor,
                                  feature_names: List[str],
                                  gradient_genes: List[str] = None) -> Dict[str, Any]:
        """
        Analyze signaling gradients across spatial positions.
        
        Args:
            node_features: Node feature matrix
            positions: Spatial positions of nodes [num_nodes, 2]
            feature_names: List of feature names
            gradient_genes: Specific genes to analyze for gradients
            
        Returns:
            Dictionary containing gradient analysis results
        """
        if isinstance(node_features, torch.Tensor):
            node_features = node_features.detach().cpu().numpy()
        if isinstance(positions, torch.Tensor):
            positions = positions.detach().cpu().numpy()
        
        if gradient_genes is None:
            # Default to important signaling molecules
            gradient_genes = ['NPPA', 'NPPB', 'NKX2-5', 'TBX5', 'HAND1', 'HAND2']
        
        # Create feature mapping
        feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        
        gradient_analysis = {}
        
        for gene in gradient_genes:
            if gene in feature_to_idx:
                gene_idx = feature_to_idx[gene]
                gene_expression = node_features[:, gene_idx]
                
                # Calculate spatial gradients
                x_positions = positions[:, 0]
                y_positions = positions[:, 1]
                
                # Simple gradient calculation using finite differences
                dx_gradient = np.gradient(gene_expression, x_positions)
                dy_gradient = np.gradient(gene_expression, y_positions)
                gradient_magnitude = np.sqrt(dx_gradient**2 + dy_gradient**2)
                
                gradient_analysis[gene] = {
                    'expression': gene_expression,
                    'dx_gradient': dx_gradient,
                    'dy_gradient': dy_gradient,
                    'magnitude': gradient_magnitude,
                    'max_gradient': np.max(gradient_magnitude),
                    'mean_gradient': np.mean(gradient_magnitude)
                }
        
        return gradient_analysis
        
    def plot_signaling_gradients(self,
                               gradient_analysis: Dict[str, Any],
                               positions: torch.Tensor,
                               title: str = "Signaling Gradients",
                               save_name: str = "signaling_gradients.png") -> None:
        """Plot signaling gradients in spatial context."""
        if isinstance(positions, torch.Tensor):
            positions = positions.detach().cpu().numpy()
        
        genes = list(gradient_analysis.keys())
        num_genes = len(genes)
        
        if num_genes == 0:
            print("No gradient data available.")
            return
        
        # Create subplots
        cols = min(3, num_genes)
        rows = (num_genes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if num_genes == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, gene in enumerate(genes):
            ax = axes[i] if i < len(axes) else axes[0]
            
            gene_data = gradient_analysis[gene]
            
            # Create scatter plot with gradient magnitude as color
            scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                               c=gene_data['magnitude'], 
                               cmap='viridis', 
                               alpha=0.7,
                               s=30)
            
            ax.set_title(f'{gene} Gradient\n(Max: {gene_data["max_gradient"]:.3f})')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Gradient Magnitude')
            
            # Add gradient vectors (simplified)
            # Sample some points to avoid overcrowding
            step = max(1, len(positions) // 50)
            sample_indices = range(0, len(positions), step)
            
            ax.quiver(positions[sample_indices, 0], 
                     positions[sample_indices, 1],
                     gene_data['dx_gradient'][sample_indices], 
                     gene_data['dy_gradient'][sample_indices],
                     alpha=0.6, width=0.003, scale=20)
        
        # Hide unused subplots
        for i in range(num_genes, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_pathway_summary(self,
                             node_features: torch.Tensor,
                             feature_names: List[str],
                             cell_labels: Optional[torch.Tensor] = None,
                             attention_weights: Optional[torch.Tensor] = None,
                             save_name: str = "pathway_summary.png") -> Dict[str, Any]:
        """Create comprehensive pathway analysis summary."""
        # Analyze pathway activities
        pathway_activities = self.analyze_pathway_activity(node_features, feature_names, cell_labels)
        
        # Create summary plot
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Pathway activity overview
        ax1 = plt.subplot(2, 3, 1)
        pathway_names = list(self.cardiac_pathways.keys())
        pathway_means = [pathway_activities[name]['mean_activity'] for name in pathway_names]
        
        bars = plt.bar(range(len(pathway_names)), pathway_means, 
                      color=plt.cm.Set3(range(len(pathway_names))))
        plt.title('Overall Pathway Activity')
        plt.ylabel('Mean Activity Score')
        plt.xticks(range(len(pathway_names)), 
                  [name.replace('_', ' ').title() for name in pathway_names], 
                  rotation=45)
        
        # 2. Cell type specific pathway activity (if available)
        if 'cell_type_analysis' in pathway_activities:
            ax2 = plt.subplot(2, 3, 2)
            self.plot_pathway_heatmap(pathway_activities, title="", save_name="temp.png")
            plt.close()  # Close the separate figure
            
            # Recreate heatmap in subplot
            cell_type_data = pathway_activities['cell_type_analysis']
            pathway_names = list(self.cardiac_pathways.keys())
            cell_types = list(cell_type_data.keys())
            
            heatmap_data = []
            for cell_type in cell_types:
                row = []
                for pathway in pathway_names:
                    if pathway in cell_type_data[cell_type]:
                        activity = cell_type_data[cell_type][pathway]['mean']
                        row.append(activity)
                    else:
                        row.append(0)
                heatmap_data.append(row)
            
            heatmap_data = np.array(heatmap_data)
            im = plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
            plt.title('Cell Type Pathway Activity')
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xticks(range(len(pathway_names)), 
                      [name[:8] for name in pathway_names], rotation=45)
            plt.yticks(range(len(cell_types)), 
                      [ct.replace('cell_type_', 'T') for ct in cell_types])
        
        # 3. Gene expression distribution for key pathways
        ax3 = plt.subplot(2, 3, 3)
        calcium_genes = self.cardiac_pathways['calcium_signaling']['genes']
        feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        
        found_calcium_genes = [gene for gene in calcium_genes if gene in feature_to_idx]
        if found_calcium_genes:
            calcium_indices = [feature_to_idx[gene] for gene in found_calcium_genes]
            if isinstance(node_features, torch.Tensor):
                calcium_expression = node_features[:, calcium_indices].detach().cpu().numpy()
            else:
                calcium_expression = node_features[:, calcium_indices]
            
            plt.boxplot(calcium_expression, labels=[gene[:6] for gene in found_calcium_genes])
            plt.title('Calcium Signaling Gene Expression')
            plt.ylabel('Expression Level')
            plt.xticks(rotation=45)
        
        # 4. Pathway gene coverage
        ax4 = plt.subplot(2, 3, 4)
        coverage_data = []
        for pathway_name, pathway_info in self.cardiac_pathways.items():
            total_genes = len(pathway_info['genes'])
            found_genes = len(pathway_activities[pathway_name]['found_genes']) if pathway_name in pathway_activities else 0
            coverage = found_genes / total_genes
            coverage_data.append(coverage)
        
        bars = plt.bar(range(len(pathway_names)), coverage_data, 
                      color=plt.cm.viridis(coverage_data))
        plt.title('Pathway Gene Coverage')
        plt.ylabel('Fraction of Genes Found')
        plt.xticks(range(len(pathway_names)), 
                  [name.replace('_', ' ').title() for name in pathway_names], 
                  rotation=45)
        plt.ylim(0, 1)
        
        # Add percentage labels
        for bar, cov in zip(bars, coverage_data):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{cov:.1%}', ha='center', va='bottom')
        
        # 5. Pathway correlation matrix
        ax5 = plt.subplot(2, 3, 5)
        if len(pathway_activities) > 1:
            # Calculate correlations between pathway activities
            pathway_scores = []
            pathway_labels = []
            for name in pathway_names:
                if name in pathway_activities:
                    pathway_scores.append(pathway_activities[name]['activity_scores'])
                    pathway_labels.append(name)
            
            if len(pathway_scores) > 1:
                correlation_matrix = np.corrcoef(pathway_scores)
                im = plt.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                plt.title('Pathway Correlation Matrix')
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.xticks(range(len(pathway_labels)), 
                          [name[:8] for name in pathway_labels], rotation=45)
                plt.yticks(range(len(pathway_labels)), 
                          [name[:8] for name in pathway_labels])
        
        # 6. Summary statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.text(0.1, 0.9, 'Pathway Analysis Summary', fontsize=14, fontweight='bold',
                transform=ax6.transAxes)
        
        summary_text = []
        summary_text.append(f"Total Pathways Analyzed: {len(pathway_activities)}")
        summary_text.append(f"Total Features: {len(feature_names)}")
        
        if cell_labels is not None:
            unique_labels = np.unique(cell_labels.detach().cpu().numpy() if isinstance(cell_labels, torch.Tensor) else cell_labels)
            summary_text.append(f"Cell Types: {len(unique_labels)}")
        
        # Most active pathway
        if pathway_activities:
            max_pathway = max(pathway_activities.keys(), 
                            key=lambda x: pathway_activities[x]['mean_activity'] if 'mean_activity' in pathway_activities[x] else 0)
            summary_text.append(f"Most Active: {max_pathway.replace('_', ' ').title()}")
        
        for i, text in enumerate(summary_text):
            ax6.text(0.1, 0.7 - i*0.1, text, fontsize=10, transform=ax6.transAxes)
        
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        
        plt.suptitle('Cardiac Pathway Analysis Summary', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
        return pathway_activities
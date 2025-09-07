"""
Large Synthetic Dataset Generator for GNN Training
Creates realistic cardiac single-cell datasets with proper spatial structure
"""

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch_geometric as pyg
from torch_geometric.data import Data
import os
import logging
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntheticCardiacDataGenerator:
    """
    Generate large-scale synthetic cardiac single-cell datasets
    with realistic expression patterns and spatial structure
    """
    
    def __init__(self, 
                 n_genes: int = 2000,
                 seed: int = 42):
        self.n_genes = n_genes
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Define cardiac cell types with realistic proportions
        self.cell_types = {
            'Cardiomyocytes': {'proportion': 0.35, 'markers': 200, 'efficiency_range': (0.7, 1.0)},
            'Fibroblasts': {'proportion': 0.25, 'markers': 150, 'efficiency_range': (0.4, 0.7)},
            'Endothelial': {'proportion': 0.20, 'markers': 120, 'efficiency_range': (0.3, 0.6)},
            'Immune': {'proportion': 0.15, 'markers': 100, 'efficiency_range': (0.2, 0.5)},
            'Smooth_muscle': {'proportion': 0.05, 'markers': 80, 'efficiency_range': (0.1, 0.4)}
        }
        
        logger.info(f"Initialized synthetic data generator with {len(self.cell_types)} cell types")
    
    def generate_gene_expression_matrix(self, n_cells: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate realistic gene expression matrix with cell-type-specific patterns
        """
        logger.info(f"Generating expression matrix for {n_cells} cells and {self.n_genes} genes")
        
        # Assign cell types based on proportions
        cell_type_assignments = []
        cell_type_names = list(self.cell_types.keys())
        
        for cell_type, info in self.cell_types.items():
            n_type_cells = int(n_cells * info['proportion'])
            cell_type_assignments.extend([cell_type] * n_type_cells)
        
        # Fill remaining cells if any
        while len(cell_type_assignments) < n_cells:
            cell_type_assignments.append(np.random.choice(cell_type_names))
        
        # Shuffle to avoid clustering by type
        np.random.shuffle(cell_type_assignments)
        cell_type_assignments = cell_type_assignments[:n_cells]
        
        # Initialize expression matrix
        X = np.zeros((n_cells, self.n_genes))
        
        # Generate base expression levels
        base_expression = np.random.gamma(2, 0.5, self.n_genes)  # Shape parameter for realistic distribution
        
        # Generate cell-type-specific expression patterns
        for i, cell_type in enumerate(cell_type_assignments):
            info = self.cell_types[cell_type]
            
            # Start with base expression
            cell_expression = base_expression.copy()
            
            # Add cell-type-specific markers
            n_markers = info['markers']
            marker_genes = np.random.choice(self.n_genes, n_markers, replace=False)
            
            # Increase expression of marker genes
            marker_boost = np.random.uniform(3, 8, n_markers)
            cell_expression[marker_genes] *= marker_boost
            
            # Add biological noise
            noise_factor = np.random.gamma(1, 0.3, self.n_genes)
            cell_expression *= noise_factor
            
            # Add technical dropout (more realistic for single-cell)
            dropout_prob = np.random.beta(2, 8, self.n_genes)  # Most genes have low dropout
            dropout_mask = np.random.random(self.n_genes) > dropout_prob
            cell_expression *= dropout_mask
            
            # Ensure non-negative and add small baseline
            cell_expression = np.maximum(cell_expression, 0) + np.random.exponential(0.1, self.n_genes)
            
            X[i] = cell_expression
        
        # Create efficiency scores based on cell type
        efficiency_scores = np.zeros(n_cells)
        for i, cell_type in enumerate(cell_type_assignments):
            eff_range = self.cell_types[cell_type]['efficiency_range']
            efficiency_scores[i] = np.random.uniform(eff_range[0], eff_range[1])
        
        # Add some noise to efficiency scores
        efficiency_scores += np.random.normal(0, 0.05, n_cells)
        efficiency_scores = np.clip(efficiency_scores, 0, 1)
        
        logger.info(f"Generated expression matrix: {X.shape}")
        logger.info(f"Cell type distribution: {pd.Series(cell_type_assignments).value_counts().to_dict()}")
        logger.info(f"Efficiency range: [{efficiency_scores.min():.3f}, {efficiency_scores.max():.3f}]")
        
        return X, efficiency_scores, cell_type_assignments
    
    def generate_spatial_coordinates(self, n_cells: int, cell_types: List[str]) -> np.ndarray:
        """
        Generate realistic spatial coordinates with tissue-like structure
        """
        logger.info(f"Generating spatial coordinates for {n_cells} cells")
        
        # Create tissue-like spatial structure
        coords = np.zeros((n_cells, 2))
        
        # Define spatial regions for different cell types
        regions = {
            'Cardiomyocytes': {'center': (0, 0), 'spread': 15},
            'Fibroblasts': {'center': (5, 5), 'spread': 20},
            'Endothelial': {'center': (-5, 5), 'spread': 12},
            'Immune': {'center': (0, -8), 'spread': 10},
            'Smooth_muscle': {'center': (8, -3), 'spread': 8}
        }
        
        for i, cell_type in enumerate(cell_types):
            if cell_type in regions:
                region = regions[cell_type]
                center = region['center']
                spread = region['spread']
                
                # Add some randomness to avoid perfect clustering
                angle = np.random.uniform(0, 2*np.pi)
                distance = np.random.exponential(spread/3)
                
                x = center[0] + distance * np.cos(angle) + np.random.normal(0, 2)
                y = center[1] + distance * np.sin(angle) + np.random.normal(0, 2)
                
                coords[i] = [x, y]
            else:
                # Random position for unknown cell types
                coords[i] = np.random.normal(0, 20, 2)
        
        logger.info(f"Spatial coordinates range: X[{coords[:, 0].min():.1f}, {coords[:, 0].max():.1f}], "
                   f"Y[{coords[:, 1].min():.1f}, {coords[:, 1].max():.1f}]")
        
        return coords
    
    def create_graph_from_data(self, X: np.ndarray, coords: np.ndarray, 
                              efficiency_scores: np.ndarray, k: int = 10) -> Data:
        """
        Create PyTorch Geometric graph from expression and spatial data
        """
        logger.info(f"Creating graph with k={k} nearest neighbors")
        
        # Normalize expression data
        scaler = RobustScaler()
        X_normalized = scaler.fit_transform(X)
        
        # Clip extreme values for numerical stability
        X_normalized = np.clip(X_normalized, -5, 5)
        
        # Replace any remaining NaN/inf
        X_normalized = np.nan_to_num(X_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create k-NN graph based on spatial coordinates
        nbrs = NearestNeighbors(n_neighbors=min(k+1, X.shape[0])).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        # Build edge list
        edge_list = []
        for i in range(len(indices)):
            for j in indices[i][1:]:  # Skip self-connection
                edge_list.append([i, j])
                edge_list.append([j, i])  # Add reverse edge for undirected graph
        
        # Remove duplicates and convert to tensor
        edge_list = list(set(tuple(edge) for edge in edge_list))
        if len(edge_list) == 0:
            # Fallback: create minimal connectivity
            edge_list = [[0, 1], [1, 0]] if X.shape[0] > 1 else [[0, 0]]
        
        edge_index = torch.tensor(edge_list).t().contiguous()
        
        # Create efficiency-based labels (5 bins)
        labels = np.digitize(efficiency_scores, bins=np.linspace(0, 1, 6)) - 1
        labels = np.clip(labels, 0, 4)  # Ensure valid class indices
        
        # Convert to tensors
        x = torch.tensor(X_normalized, dtype=torch.float32)
        y = torch.tensor(np.random.choice(5), dtype=torch.long)  # Graph-level label
        pos = torch.tensor(coords, dtype=torch.float32)
        
        # Create graph data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            pos=pos,
            efficiency=torch.tensor(efficiency_scores, dtype=torch.float32)
        )
        
        logger.info(f"Created graph: {x.shape[0]} nodes, {edge_index.shape[1]} edges")
        logger.info(f"Graph label: {y.item()}, Node features: {x.shape[1]}")
        
        return data
    
    def generate_dataset(self, 
                        n_graphs: int = 100,
                        cells_per_graph_range: Tuple[int, int] = (300, 800),
                        k_neighbors: int = 10,
                        save_path: str = None) -> List[Data]:
        """
        Generate a complete synthetic dataset
        """
        logger.info(f"🚀 Generating {n_graphs} synthetic graphs...")
        logger.info(f"Cells per graph: {cells_per_graph_range[0]}-{cells_per_graph_range[1]}")
        
        graphs = []
        
        for i in range(n_graphs):
            if i % 10 == 0:
                logger.info(f"Generating graph {i+1}/{n_graphs}")
            
            # Random number of cells for this graph
            n_cells = np.random.randint(cells_per_graph_range[0], cells_per_graph_range[1])
            
            # Generate expression data
            X, efficiency_scores, cell_types = self.generate_gene_expression_matrix(n_cells)
            
            # Generate spatial coordinates
            coords = self.generate_spatial_coordinates(n_cells, cell_types)
            
            # Create graph
            graph = self.create_graph_from_data(X, coords, efficiency_scores, k_neighbors)
            graphs.append(graph)
        
        if save_path:
            logger.info(f"💾 Saving dataset to {save_path}")
            torch.save(graphs, save_path)
        
        logger.info(f"✅ Generated {len(graphs)} graphs successfully!")
        self._print_dataset_statistics(graphs)
        
        return graphs
    
    def _print_dataset_statistics(self, graphs: List[Data]):
        """Print comprehensive dataset statistics"""
        
        n_nodes = [g.x.shape[0] for g in graphs]
        n_edges = [g.edge_index.shape[1] for g in graphs]
        n_features = graphs[0].x.shape[1] if graphs else 0
        
        logger.info("📊 Dataset Statistics:")
        logger.info(f"  Total graphs: {len(graphs)}")
        logger.info(f"  Nodes per graph: {np.mean(n_nodes):.1f} ± {np.std(n_nodes):.1f} "
                   f"[{min(n_nodes)}, {max(n_nodes)}]")
        logger.info(f"  Edges per graph: {np.mean(n_edges):.1f} ± {np.std(n_edges):.1f} "
                   f"[{min(n_edges)}, {max(n_edges)}]")
        logger.info(f"  Features per node: {n_features}")
        logger.info(f"  Total nodes: {sum(n_nodes):,}")
        logger.info(f"  Total edges: {sum(n_edges):,}")

def create_multiple_datasets():
    """Create multiple synthetic datasets of different sizes"""
    
    logger.info("🏗️ Creating multiple synthetic datasets...")
    
    # Create output directory
    output_dir = "/Users/tumokgabeng/Projects/HybridGnnRnn/data/large_synthetic"
    os.makedirs(output_dir, exist_ok=True)
    
    generator = SyntheticCardiacDataGenerator(n_genes=2000, seed=42)
    
    # Dataset configurations
    datasets = [
        {
            'name': 'small_synthetic',
            'n_graphs': 50,
            'cells_range': (200, 400),
            'description': 'Small dataset for quick testing'
        },
        {
            'name': 'medium_synthetic', 
            'n_graphs': 150,
            'cells_range': (300, 600),
            'description': 'Medium dataset for training'
        },
        {
            'name': 'large_synthetic',
            'n_graphs': 300,
            'cells_range': (400, 800),
            'description': 'Large dataset for robust training'
        }
    ]
    
    created_datasets = {}
    
    for dataset_config in datasets:
        logger.info(f"\n🔨 Creating {dataset_config['name']}: {dataset_config['description']}")
        
        # Generate dataset
        graphs = generator.generate_dataset(
            n_graphs=dataset_config['n_graphs'],
            cells_per_graph_range=dataset_config['cells_range'],
            k_neighbors=10,
            save_path=os.path.join(output_dir, f"{dataset_config['name']}.pt")
        )
        
        created_datasets[dataset_config['name']] = {
            'graphs': graphs,
            'path': os.path.join(output_dir, f"{dataset_config['name']}.pt"),
            'config': dataset_config
        }
    
    # Create a comprehensive summary
    logger.info("\n📋 Summary of Created Datasets:")
    total_graphs = 0
    total_nodes = 0
    
    for name, info in created_datasets.items():
        graphs = info['graphs']
        n_graphs = len(graphs)
        n_nodes = sum(g.x.shape[0] for g in graphs)
        
        total_graphs += n_graphs
        total_nodes += n_nodes
        
        logger.info(f"  {name}:")
        logger.info(f"    - Graphs: {n_graphs}")
        logger.info(f"    - Total nodes: {n_nodes:,}")
        logger.info(f"    - Avg nodes/graph: {n_nodes/n_graphs:.1f}")
        logger.info(f"    - File: {info['path']}")
    
    logger.info(f"\n🎯 Total across all datasets:")
    logger.info(f"  - Total graphs: {total_graphs}")
    logger.info(f"  - Total nodes: {total_nodes:,}")
    
    return created_datasets

def visualize_synthetic_data(dataset_path: str, n_samples: int = 3):
    """Visualize sample graphs from synthetic dataset"""
    
    logger.info(f"📊 Visualizing synthetic data from {dataset_path}")
    
    # Load dataset
    graphs = torch.load(dataset_path, weights_only=False)
    
    # Create visualization
    fig, axes = plt.subplots(2, n_samples, figsize=(5*n_samples, 10))
    if n_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(min(n_samples, len(graphs))):
        graph = graphs[i]
        
        # Plot spatial structure
        ax1 = axes[0, i]
        pos = graph.pos.numpy()
        efficiency = graph.efficiency.numpy()
        
        scatter = ax1.scatter(pos[:, 0], pos[:, 1], c=efficiency, 
                            cmap='viridis', s=20, alpha=0.7)
        ax1.set_title(f'Graph {i+1}: Spatial Structure\n{graph.x.shape[0]} cells')
        ax1.set_xlabel('X coordinate')
        ax1.set_ylabel('Y coordinate')
        plt.colorbar(scatter, ax=ax1, label='Efficiency Score')
        
        # Plot expression distribution
        ax2 = axes[1, i]
        expression = graph.x.numpy()
        ax2.hist(expression.flatten(), bins=50, alpha=0.7, density=True)
        ax2.set_title(f'Expression Distribution\n{graph.x.shape[1]} genes')
        ax2.set_xlabel('Expression Level')
        ax2.set_ylabel('Density')
        ax2.set_yscale('log')
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = dataset_path.replace('.pt', '_visualization.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    logger.info(f"💾 Saved visualization to {viz_path}")
    
    plt.show()

def main():
    """Main function to create synthetic datasets"""
    
    logger.info("🚀 Starting Large Synthetic Dataset Creation")
    
    # Create datasets
    datasets = create_multiple_datasets()
    
    # Visualize one dataset
    medium_dataset_path = datasets['medium_synthetic']['path']
    visualize_synthetic_data(medium_dataset_path, n_samples=3)
    
    logger.info("✅ All synthetic datasets created successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Use train_on_synthetic.py to train models")
    logger.info("2. Compare performance across different dataset sizes")
    logger.info("3. Tune hyperparameters on the large dataset")

if __name__ == "__main__":
    main()

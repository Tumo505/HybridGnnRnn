"""
Fixed Synthetic Dataset Generator with Meaningful Labels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
import numpy as np
import logging
import os
import json
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple
import networkx as nx

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedSyntheticCardiacDataGenerator:
    """
    Enhanced synthetic data generator with meaningful graph-level labels
    """
    
    def __init__(self, n_genes: int = 2000, random_seed: int = 42):
        self.n_genes = n_genes
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Define cardiac cell types with realistic proportions
        self.cell_types = {
            'Cardiomyocytes': {'proportion': 0.35, 'markers': 150, 'spatial_center': (0, 0)},
            'Fibroblasts': {'proportion': 0.25, 'markers': 120, 'spatial_center': (15, 10)},
            'Endothelial': {'proportion': 0.20, 'markers': 100, 'spatial_center': (-10, 15)},
            'Immune': {'proportion': 0.15, 'markers': 80, 'spatial_center': (10, -15)},
            'Smooth_muscle': {'proportion': 0.05, 'markers': 60, 'spatial_center': (-15, -10)}
        }
        
        # Define conditions that create meaningful graph-level labels
        self.graph_conditions = {
            0: {'name': 'Healthy', 'inflammation': 0.1, 'fibrosis': 0.1, 'cardiomyocyte_stress': 0.1},
            1: {'name': 'Mild_Inflammation', 'inflammation': 0.4, 'fibrosis': 0.2, 'cardiomyocyte_stress': 0.2},
            2: {'name': 'Fibrotic', 'inflammation': 0.2, 'fibrosis': 0.6, 'cardiomyocyte_stress': 0.3},
            3: {'name': 'Ischemic', 'inflammation': 0.3, 'fibrosis': 0.3, 'cardiomyocyte_stress': 0.7},
            4: {'name': 'Severe_Disease', 'inflammation': 0.7, 'fibrosis': 0.5, 'cardiomyocyte_stress': 0.8}
        }
    
    def generate_condition_specific_expression(self, n_cells: int, condition_id: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate gene expression matrix based on cardiac condition
        """
        condition = self.graph_conditions[condition_id]
        logger.info(f"Generating {condition['name']} condition with {n_cells} cells")
        
        # Assign cell types based on condition (disease affects proportions)
        cell_type_assignments = []
        cell_type_names = list(self.cell_types.keys())
        
        # Adjust proportions based on condition
        adjusted_proportions = self._adjust_cell_proportions(condition_id)
        
        for cell_type, proportion in adjusted_proportions.items():
            n_type_cells = int(n_cells * proportion)
            cell_type_assignments.extend([cell_type] * n_type_cells)
        
        # Fill remaining cells
        while len(cell_type_assignments) < n_cells:
            cell_type_assignments.append(np.random.choice(cell_type_names))
        
        # Shuffle to avoid clustering by type
        np.random.shuffle(cell_type_assignments)
        cell_type_assignments = cell_type_assignments[:n_cells]
        
        # Initialize expression matrix
        X = np.zeros((n_cells, self.n_genes))
        efficiency_scores = np.zeros(n_cells)
        
        # Generate condition-specific gene signatures
        inflammatory_genes = np.random.choice(self.n_genes, 200, replace=False)
        fibrosis_genes = np.random.choice(self.n_genes, 150, replace=False)
        stress_genes = np.random.choice(self.n_genes, 180, replace=False)
        
        # Generate base expression levels
        base_expression = np.random.gamma(2, 0.3, self.n_genes)
        
        for i, cell_type in enumerate(cell_type_assignments):
            info = self.cell_types[cell_type]
            
            # Start with base expression
            cell_expression = base_expression.copy()
            
            # Add cell-type-specific markers
            n_markers = info['markers']
            marker_genes = np.random.choice(self.n_genes, n_markers, replace=False)
            marker_boost = np.random.uniform(2, 5, n_markers)
            cell_expression[marker_genes] *= marker_boost
            
            # Apply condition-specific modifications
            if cell_type == 'Immune':
                # Immune cells upregulate inflammatory genes
                inflammatory_boost = 1 + condition['inflammation'] * np.random.uniform(1, 3, len(inflammatory_genes))
                cell_expression[inflammatory_genes] *= inflammatory_boost
            
            elif cell_type == 'Fibroblasts':
                # Fibroblasts upregulate fibrosis genes
                fibrosis_boost = 1 + condition['fibrosis'] * np.random.uniform(1, 4, len(fibrosis_genes))
                cell_expression[fibrosis_genes] *= fibrosis_boost
            
            elif cell_type == 'Cardiomyocytes':
                # Cardiomyocytes upregulate stress genes
                stress_boost = 1 + condition['cardiomyocyte_stress'] * np.random.uniform(1, 3, len(stress_genes))
                cell_expression[stress_genes] *= stress_boost
                
                # In disease, cardiomyocytes may have reduced function
                if condition_id > 1:
                    function_genes = np.random.choice(marker_genes, len(marker_genes)//2, replace=False)
                    function_reduction = 1 - condition['cardiomyocyte_stress'] * 0.5
                    cell_expression[function_genes] *= function_reduction
            
            # Calculate efficiency score based on condition and cell type
            base_efficiency = 0.8 if cell_type == 'Cardiomyocytes' else 0.6
            condition_penalty = (condition['inflammation'] + condition['fibrosis'] + condition['cardiomyocyte_stress']) / 3
            efficiency_scores[i] = max(0.1, base_efficiency - condition_penalty * 0.7 + np.random.normal(0, 0.1))
            
            # Add biological noise
            noise_factor = np.random.gamma(1, 0.2, self.n_genes)
            cell_expression *= noise_factor
            
            # Add technical dropout (realistic for single-cell)
            dropout_prob = np.random.beta(2, 10, self.n_genes)
            dropout_mask = np.random.random(self.n_genes) > dropout_prob
            cell_expression *= dropout_mask
            
            # Ensure non-negative values
            cell_expression = np.maximum(cell_expression, 0)
            
            X[i] = cell_expression
        
        logger.info(f"Generated expression matrix: {X.shape}")
        logger.info(f"Cell type distribution: {dict(zip(*np.unique(cell_type_assignments, return_counts=True)))}")
        logger.info(f"Efficiency range: [{efficiency_scores.min():.3f}, {efficiency_scores.max():.3f}]")
        
        return X, efficiency_scores, cell_type_assignments
    
    def _adjust_cell_proportions(self, condition_id: int) -> Dict[str, float]:
        """Adjust cell type proportions based on cardiac condition"""
        base_proportions = {k: v['proportion'] for k, v in self.cell_types.items()}
        
        if condition_id == 0:  # Healthy
            return base_proportions
        elif condition_id == 1:  # Mild Inflammation
            # Slight increase in immune cells
            base_proportions['Immune'] *= 1.5
            base_proportions['Cardiomyocytes'] *= 0.95
        elif condition_id == 2:  # Fibrotic
            # Increase fibroblasts, decrease cardiomyocytes
            base_proportions['Fibroblasts'] *= 1.8
            base_proportions['Cardiomyocytes'] *= 0.8
        elif condition_id == 3:  # Ischemic
            # Balanced pathology
            base_proportions['Immune'] *= 1.3
            base_proportions['Fibroblasts'] *= 1.2
            base_proportions['Cardiomyocytes'] *= 0.85
        elif condition_id == 4:  # Severe Disease
            # Major shifts in all cell types
            base_proportions['Immune'] *= 2.0
            base_proportions['Fibroblasts'] *= 1.6
            base_proportions['Cardiomyocytes'] *= 0.7
        
        # Normalize proportions
        total = sum(base_proportions.values())
        return {k: v/total for k, v in base_proportions.items()}
    
    def generate_spatial_coordinates(self, n_cells: int, cell_types: List[str], condition_id: int) -> np.ndarray:
        """Generate realistic spatial coordinates"""
        logger.info(f"Generating spatial coordinates for {n_cells} cells")
        
        coords = np.zeros((n_cells, 2))
        condition = self.graph_conditions[condition_id]
        
        # Disease affects spatial organization
        spatial_noise = 5 + condition['inflammation'] * 10  # More disorganized in disease
        
        for i, cell_type in enumerate(cell_types):
            if cell_type in self.cell_types:
                center = self.cell_types[cell_type]['spatial_center']
                
                # Add condition-specific spatial perturbations
                if condition_id > 1:  # Disease states
                    # Add spatial clustering due to tissue damage
                    cluster_factor = np.random.choice([0.5, 1.5], p=[0.7, 0.3])
                    radius = np.random.exponential(8) * cluster_factor
                else:
                    radius = np.random.exponential(10)
                
                angle = np.random.uniform(0, 2*np.pi)
                x = center[0] + radius * np.cos(angle) + np.random.normal(0, spatial_noise)
                y = center[1] + radius * np.sin(angle) + np.random.normal(0, spatial_noise)
                
                coords[i] = [x, y]
            else:
                coords[i] = np.random.normal(0, 20, 2)
        
        logger.info(f"Spatial coordinates range: X[{coords[:, 0].min():.1f}, {coords[:, 0].max():.1f}], "
                   f"Y[{coords[:, 1].min():.1f}, {coords[:, 1].max():.1f}]")
        
        return coords
    
    def create_graph_from_data(self, X: np.ndarray, coords: np.ndarray, 
                              efficiency_scores: np.ndarray, condition_id: int, k: int = 10) -> Data:
        """Create PyTorch Geometric graph from expression and spatial data"""
        logger.info(f"Creating graph with k={k} nearest neighbors")
        
        # Normalize expression data
        scaler = RobustScaler()
        X_normalized = scaler.fit_transform(X)
        X_normalized = np.clip(X_normalized, -5, 5)
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
            edge_list = [[0, 1], [1, 0]] if X.shape[0] > 1 else [[0, 0]]
        
        edge_index = torch.tensor(edge_list).t().contiguous()
        
        # Convert to tensors
        x = torch.tensor(X_normalized, dtype=torch.float32)
        y = torch.tensor(condition_id, dtype=torch.long)  # Meaningful label!
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
        logger.info(f"Graph label: {y.item()} ({self.graph_conditions[condition_id]['name']})")
        
        return data
    
    def generate_balanced_dataset(self, 
                                 n_graphs: int = 300,
                                 cells_per_graph_range: Tuple[int, int] = (400, 800),
                                 k_neighbors: int = 10) -> List[Data]:
        """Generate a balanced dataset with equal samples per condition"""
        logger.info(f"🚀 Generating {n_graphs} balanced synthetic graphs...")
        
        graphs = []
        graphs_per_condition = n_graphs // len(self.graph_conditions)
        
        for condition_id in range(len(self.graph_conditions)):
            condition_name = self.graph_conditions[condition_id]['name']
            logger.info(f"📊 Generating {graphs_per_condition} graphs for condition: {condition_name}")
            
            for i in range(graphs_per_condition):
                # Random number of cells for this graph
                n_cells = np.random.randint(cells_per_graph_range[0], cells_per_graph_range[1])
                
                # Generate condition-specific expression data
                X, efficiency_scores, cell_types = self.generate_condition_specific_expression(n_cells, condition_id)
                
                # Generate spatial coordinates
                coords = self.generate_spatial_coordinates(n_cells, cell_types, condition_id)
                
                # Create graph
                graph = self.create_graph_from_data(X, coords, efficiency_scores, condition_id, k_neighbors)
                graphs.append(graph)
        
        # Add remaining graphs if needed
        remaining = n_graphs - len(graphs)
        for i in range(remaining):
            condition_id = np.random.randint(0, len(self.graph_conditions))
            n_cells = np.random.randint(cells_per_graph_range[0], cells_per_graph_range[1])
            X, efficiency_scores, cell_types = self.generate_condition_specific_expression(n_cells, condition_id)
            coords = self.generate_spatial_coordinates(n_cells, cell_types, condition_id)
            graph = self.create_graph_from_data(X, coords, efficiency_scores, condition_id, k_neighbors)
            graphs.append(graph)
        
        # Shuffle the dataset
        np.random.shuffle(graphs)
        
        # Log label distribution
        labels = [graph.y.item() for graph in graphs]
        label_counts = {i: labels.count(i) for i in range(5)}
        condition_names = {i: self.graph_conditions[i]['name'] for i in range(5)}
        
        logger.info("📋 Final dataset label distribution:")
        for label_id, count in label_counts.items():
            logger.info(f"  {label_id} ({condition_names[label_id]}): {count} graphs")
        
        return graphs

def create_improved_datasets():
    """Create improved synthetic datasets with meaningful labels"""
    logger.info("🚀 Creating improved synthetic datasets with meaningful labels...")
    
    # Create output directory
    output_dir = "data/improved_synthetic"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize generator
    generator = ImprovedSyntheticCardiacDataGenerator(n_genes=2000, random_seed=42)
    
    # Create datasets of different sizes
    datasets = {
        'small_improved': {'n_graphs': 50, 'cells_range': (200, 400)},
        'medium_improved': {'n_graphs': 150, 'cells_range': (300, 600)},
        'large_improved': {'n_graphs': 300, 'cells_range': (400, 800)}
    }
    
    created_datasets = {}
    total_graphs = 0
    total_nodes = 0
    
    for dataset_name, config in datasets.items():
        logger.info(f"📊 Creating {dataset_name} dataset...")
        
        graphs = generator.generate_balanced_dataset(
            n_graphs=config['n_graphs'],
            cells_per_graph_range=config['cells_range'],
            k_neighbors=10
        )
        
        # Save dataset
        dataset_path = os.path.join(output_dir, f"{dataset_name}.pt")
        torch.save(graphs, dataset_path)
        
        # Calculate statistics
        n_nodes = sum(graph.x.shape[0] for graph in graphs)
        avg_nodes = n_nodes / len(graphs)
        
        created_datasets[dataset_name] = {
            'path': dataset_path,
            'n_graphs': len(graphs),
            'n_nodes': n_nodes,
            'avg_nodes': avg_nodes
        }
        
        total_graphs += len(graphs)
        total_nodes += n_nodes
        
        logger.info(f"✅ {dataset_name}:")
        logger.info(f"  - Graphs: {len(graphs)}")
        logger.info(f"  - Total nodes: {n_nodes:,}")
        logger.info(f"  - Avg nodes/graph: {avg_nodes:.1f}")
        logger.info(f"  - File: {dataset_path}")
    
    logger.info("🎯 Summary of Improved Datasets:")
    for name, info in created_datasets.items():
        logger.info(f"  {name}:")
        logger.info(f"    - Graphs: {info['n_graphs']}")
        logger.info(f"    - Total nodes: {info['n_nodes']:,}")
        logger.info(f"    - Avg nodes/graph: {info['avg_nodes']:.1f}")
        logger.info(f"    - File: {info['path']}")
    
    logger.info(f"🎯 Total across all improved datasets:")
    logger.info(f"  - Total graphs: {total_graphs}")
    logger.info(f"  - Total nodes: {total_nodes:,}")
    
    return created_datasets

if __name__ == "__main__":
    create_improved_datasets()

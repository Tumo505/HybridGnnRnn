"""
Comprehensive visualization tools for analyzing model predictions, embeddings, and results.
Includes spatial plots, temporal trajectories, attention maps, and interactive dashboards.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import umap.umap_ as umap

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelVisualization:
    """Comprehensive visualization toolkit for the Hybrid GNN-RNN model."""
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Color schemes
        self.cell_type_colors = {
            'pluripotent': '#1f77b4',
            'mesoderm': '#ff7f0e', 
            'cardiac_progenitor': '#2ca02c',
            'immature_cardiomyocyte': '#d62728',
            'mature_cardiomyocyte': '#9467bd'
        }
        
        self.stage_colors = {
            0: '#1f77b4',  # pluripotent
            1: '#ff7f0e',  # mesoderm
            2: '#2ca02c',  # cardiac_progenitor
            3: '#d62728',  # immature_cardiomyocyte
            4: '#9467bd'   # mature_cardiomyocyte
        }
    
    def plot_spatial_embeddings(
        self,
        embeddings: np.ndarray,
        coordinates: np.ndarray,
        cell_types: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None,
        title: str = "Spatial Embeddings",
        save_path: Optional[str] = None
    ):
        """Visualize spatial embeddings with tissue coordinates."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Original spatial coordinates colored by cell type
        if cell_types is not None:
            scatter = axes[0, 0].scatter(
                coordinates[:, 0], coordinates[:, 1],
                c=[self.stage_colors.get(ct, '#gray') for ct in cell_types],
                s=20, alpha=0.7
            )
            axes[0, 0].set_title('Spatial Distribution by Cell Type')
        else:
            axes[0, 0].scatter(coordinates[:, 0], coordinates[:, 1], s=20, alpha=0.7)
            axes[0, 0].set_title('Spatial Distribution')
        
        axes[0, 0].set_xlabel('X Coordinate')
        axes[0, 0].set_ylabel('Y Coordinate')
        
        # 2. t-SNE of embeddings
        if embeddings.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embeddings_2d = tsne.fit_transform(embeddings)
        else:
            embeddings_2d = embeddings
        
        if cell_types is not None:
            scatter = axes[0, 1].scatter(
                embeddings_2d[:, 0], embeddings_2d[:, 1],
                c=[self.stage_colors.get(ct, '#gray') for ct in cell_types],
                s=20, alpha=0.7
            )
            axes[0, 1].set_title('t-SNE of Spatial Embeddings (by Cell Type)')
        else:
            axes[0, 1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=20, alpha=0.7)
            axes[0, 1].set_title('t-SNE of Spatial Embeddings')
        
        axes[0, 1].set_xlabel('t-SNE 1')
        axes[0, 1].set_ylabel('t-SNE 2')
        
        # 3. Original coordinates colored by predictions
        if predictions is not None:
            scatter = axes[1, 0].scatter(
                coordinates[:, 0], coordinates[:, 1],
                c=predictions, cmap='viridis', s=20, alpha=0.7
            )
            plt.colorbar(scatter, ax=axes[1, 0])
            axes[1, 0].set_title('Spatial Distribution by Predictions')
        else:
            axes[1, 0].scatter(coordinates[:, 0], coordinates[:, 1], s=20, alpha=0.7)
            axes[1, 0].set_title('Spatial Distribution')
        
        axes[1, 0].set_xlabel('X Coordinate')
        axes[1, 0].set_ylabel('Y Coordinate')
        
        # 4. t-SNE colored by predictions
        if predictions is not None:
            scatter = axes[1, 1].scatter(
                embeddings_2d[:, 0], embeddings_2d[:, 1],
                c=predictions, cmap='viridis', s=20, alpha=0.7
            )
            plt.colorbar(scatter, ax=axes[1, 1])
            axes[1, 1].set_title('t-SNE of Embeddings (by Predictions)')
        else:
            axes[1, 1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=20, alpha=0.7)
            axes[1, 1].set_title('t-SNE of Embeddings')
        
        axes[1, 1].set_xlabel('t-SNE 1')
        axes[1, 1].set_ylabel('t-SNE 2')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_temporal_trajectories(
        self,
        temporal_embeddings: np.ndarray,
        time_points: np.ndarray,
        cell_ids: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None,
        title: str = "Temporal Trajectories",
        save_path: Optional[str] = None
    ):
        """Visualize temporal differentiation trajectories."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Reduce dimensionality for visualization
        if temporal_embeddings.shape[1] > 2:
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(temporal_embeddings)
        else:
            embeddings_2d = temporal_embeddings
        
        # 1. Trajectory plot colored by time
        scatter = axes[0, 0].scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=time_points, cmap='plasma', s=30, alpha=0.7
        )
        plt.colorbar(scatter, ax=axes[0, 0], label='Time Point')
        axes[0, 0].set_title('Temporal Trajectories (by Time)')
        axes[0, 0].set_xlabel('PC1')
        axes[0, 0].set_ylabel('PC2')
        
        # 2. Individual cell trajectories
        if cell_ids is not None:
            unique_cells = np.unique(cell_ids)
            for i, cell_id in enumerate(unique_cells[:10]):  # Show first 10 cells
                mask = cell_ids == cell_id
                if np.sum(mask) > 1:  # Only plot if cell has multiple time points
                    cell_traj = embeddings_2d[mask]
                    time_traj = time_points[mask]
                    sorted_idx = np.argsort(time_traj)
                    axes[0, 1].plot(
                        cell_traj[sorted_idx, 0], 
                        cell_traj[sorted_idx, 1], 
                        'o-', alpha=0.7, linewidth=2
                    )
        axes[0, 1].set_title('Individual Cell Trajectories')
        axes[0, 1].set_xlabel('PC1')
        axes[0, 1].set_ylabel('PC2')
        
        # 3. Time point progression
        time_means = []
        time_stds = []
        unique_times = np.unique(time_points)
        
        for t in unique_times:
            mask = time_points == t
            time_means.append(np.mean(embeddings_2d[mask], axis=0))
            time_stds.append(np.std(embeddings_2d[mask], axis=0))
        
        time_means = np.array(time_means)
        time_stds = np.array(time_stds)
        
        axes[1, 0].errorbar(
            unique_times, time_means[:, 0], yerr=time_stds[:, 0],
            label='PC1', marker='o', capsize=5
        )
        axes[1, 0].errorbar(
            unique_times, time_means[:, 1], yerr=time_stds[:, 1],
            label='PC2', marker='s', capsize=5
        )
        axes[1, 0].set_title('Trajectory Progression Over Time')
        axes[1, 0].set_xlabel('Time Point')
        axes[1, 0].set_ylabel('Principal Component Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. Predictions vs Time
        if predictions is not None:
            for t in unique_times:
                mask = time_points == t
                if np.sum(mask) > 0:
                    axes[1, 1].scatter(
                        [t] * np.sum(mask), predictions[mask],
                        alpha=0.6, s=20
                    )
            
            # Add trend line
            time_pred_means = [np.mean(predictions[time_points == t]) for t in unique_times]
            axes[1, 1].plot(unique_times, time_pred_means, 'r-', linewidth=3, label='Mean Prediction')
            
            axes[1, 1].set_title('Predictions Over Time')
            axes[1, 1].set_xlabel('Time Point')
            axes[1, 1].set_ylabel('Prediction Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_attention_maps(
        self,
        attention_weights: np.ndarray,
        coordinates: Optional[np.ndarray] = None,
        gene_names: Optional[List[str]] = None,
        top_k: int = 20,
        title: str = "Attention Analysis",
        save_path: Optional[str] = None
    ):
        """Visualize attention weights and patterns."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Attention heatmap
        if attention_weights.ndim == 3:  # Multi-head attention
            attention_mean = np.mean(attention_weights, axis=0)
        else:
            attention_mean = attention_weights
        
        sns.heatmap(
            attention_mean[:50, :50],  # Show subset for visibility
            cmap='YlOrRd', ax=axes[0, 0],
            cbar_kws={'label': 'Attention Weight'}
        )
        axes[0, 0].set_title('Attention Weight Matrix')
        axes[0, 0].set_xlabel('Key Position')
        axes[0, 0].set_ylabel('Query Position')
        
        # 2. Attention distribution
        attention_sums = np.sum(attention_mean, axis=1)
        axes[0, 1].hist(attention_sums, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Attention Weight Distribution')
        axes[0, 1].set_xlabel('Total Attention Weight')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Top attended positions
        top_indices = np.argsort(attention_sums)[-top_k:]
        top_weights = attention_sums[top_indices]
        
        bars = axes[1, 0].barh(range(len(top_indices)), top_weights)
        axes[1, 0].set_yticks(range(len(top_indices)))
        if gene_names:
            labels = [gene_names[i] if i < len(gene_names) else f'Position {i}' for i in top_indices]
            axes[1, 0].set_yticklabels(labels)
        else:
            axes[1, 0].set_yticklabels([f'Position {i}' for i in top_indices])
        axes[1, 0].set_title(f'Top {top_k} Attended Positions')
        axes[1, 0].set_xlabel('Attention Weight')
        
        # 4. Spatial attention pattern
        if coordinates is not None:
            scatter = axes[1, 1].scatter(
                coordinates[:, 0], coordinates[:, 1],
                c=attention_sums[:len(coordinates)], 
                cmap='YlOrRd', s=30, alpha=0.7
            )
            plt.colorbar(scatter, ax=axes[1, 1], label='Attention Weight')
            axes[1, 1].set_title('Spatial Attention Pattern')
            axes[1, 1].set_xlabel('X Coordinate')
            axes[1, 1].set_ylabel('Y Coordinate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        task_type: str = "classification",
        title: str = "Model Performance",
        save_path: Optional[str] = None
    ):
        """Comprehensive model performance visualization."""
        
        if task_type == "classification":
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 1. Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(
                cm, annot=True, fmt='d', ax=axes[0, 0],
                xticklabels=class_names, yticklabels=class_names,
                cmap='Blues'
            )
            axes[0, 0].set_title('Confusion Matrix')
            axes[0, 0].set_xlabel('Predicted')
            axes[0, 0].set_ylabel('Actual')
            
            # 2. Class-wise Performance
            from sklearn.metrics import classification_report
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            
            classes = class_names if class_names else [f'Class {i}' for i in range(len(np.unique(y_true)))]
            metrics = ['precision', 'recall', 'f1-score']
            metric_values = {metric: [report[cls][metric] for cls in classes] for metric in metrics}
            
            x = np.arange(len(classes))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                axes[0, 1].bar(x + i*width, metric_values[metric], width, label=metric, alpha=0.8)
            
            axes[0, 1].set_xlabel('Classes')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].set_title('Class-wise Performance')
            axes[0, 1].set_xticks(x + width)
            axes[0, 1].set_xticklabels(classes, rotation=45)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Prediction Distribution
            axes[0, 2].hist(y_pred, bins=len(np.unique(y_true)), alpha=0.7, edgecolor='black')
            axes[0, 2].set_title('Prediction Distribution')
            axes[0, 2].set_xlabel('Predicted Class')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].grid(True, alpha=0.3)
            
            if y_prob is not None:
                # 4. ROC Curves (for binary/multiclass)
                if len(np.unique(y_true)) == 2:
                    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                    auc_score = np.trapz(tpr, fpr)
                    axes[1, 0].plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.3f})')
                    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
                    axes[1, 0].set_xlabel('False Positive Rate')
                    axes[1, 0].set_ylabel('True Positive Rate')
                    axes[1, 0].set_title('ROC Curve')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)
                
                # 5. Precision-Recall Curve
                if len(np.unique(y_true)) == 2:
                    precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
                    axes[1, 1].plot(recall, precision)
                    axes[1, 1].set_xlabel('Recall')
                    axes[1, 1].set_ylabel('Precision')
                    axes[1, 1].set_title('Precision-Recall Curve')
                    axes[1, 1].grid(True, alpha=0.3)
                
                # 6. Prediction Confidence
                max_probs = np.max(y_prob, axis=1)
                correct = (y_pred == y_true)
                
                axes[1, 2].hist(max_probs[correct], bins=20, alpha=0.7, label='Correct', density=True)
                axes[1, 2].hist(max_probs[~correct], bins=20, alpha=0.7, label='Incorrect', density=True)
                axes[1, 2].set_xlabel('Prediction Confidence')
                axes[1, 2].set_ylabel('Density')
                axes[1, 2].set_title('Prediction Confidence Distribution')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
        
        else:  # Regression
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Predictions vs Actual
            axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
            min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
            axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            axes[0, 0].set_xlabel('Actual Values')
            axes[0, 0].set_ylabel('Predicted Values')
            axes[0, 0].set_title('Predictions vs Actual')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Residuals
            residuals = y_pred - y_true
            axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
            axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
            axes[0, 1].set_xlabel('Predicted Values')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residual Plot')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Residual Distribution
            axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Residuals')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Residual Distribution')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Error Metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            axes[1, 1].text(0.1, 0.8, f'MSE: {mse:.4f}', transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].text(0.1, 0.6, f'MAE: {mae:.4f}', transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].text(0.1, 0.4, f'R²: {r2:.4f}', transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].set_title('Performance Metrics')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_dashboard(
        self,
        spatial_embeddings: np.ndarray,
        temporal_embeddings: np.ndarray,
        coordinates: np.ndarray,
        predictions: np.ndarray,
        cell_types: Optional[np.ndarray] = None,
        time_points: Optional[np.ndarray] = None,
        save_path: str = "interactive_dashboard.html"
    ):
        """Create interactive dashboard with plotly."""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Spatial Distribution', 'Spatial Embeddings (t-SNE)',
                'Temporal Trajectories', 'Predictions vs Time',
                'Embedding Comparison', 'Model Confidence'
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        # Prepare data
        if spatial_embeddings.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42)
            spatial_2d = tsne.fit_transform(spatial_embeddings)
        else:
            spatial_2d = spatial_embeddings
        
        if temporal_embeddings.shape[1] > 2:
            pca = PCA(n_components=2)
            temporal_2d = pca.fit_transform(temporal_embeddings)
        else:
            temporal_2d = temporal_embeddings
        
        # Color mapping
        if cell_types is not None:
            colors = [self.stage_colors.get(ct, '#gray') for ct in cell_types]
        else:
            colors = ['blue'] * len(coordinates)
        
        # 1. Spatial Distribution
        fig.add_trace(
            go.Scatter(
                x=coordinates[:, 0], y=coordinates[:, 1],
                mode='markers',
                marker=dict(color=colors, size=8, opacity=0.7),
                name='Cells',
                text=[f'Cell {i}: Type {ct}' for i, ct in enumerate(cell_types)] if cell_types is not None else None
            ),
            row=1, col=1
        )
        
        # 2. Spatial Embeddings (t-SNE)
        fig.add_trace(
            go.Scatter(
                x=spatial_2d[:, 0], y=spatial_2d[:, 1],
                mode='markers',
                marker=dict(color=colors, size=8, opacity=0.7),
                name='Spatial Embeddings'
            ),
            row=1, col=2
        )
        
        # 3. Temporal Trajectories
        if time_points is not None:
            fig.add_trace(
                go.Scatter(
                    x=temporal_2d[:, 0], y=temporal_2d[:, 1],
                    mode='markers',
                    marker=dict(color=time_points, colorscale='Viridis', size=8, opacity=0.7),
                    name='Temporal Embeddings'
                ),
                row=1, col=3
            )
        
        # 4. Predictions vs Time
        if time_points is not None:
            fig.add_trace(
                go.Scatter(
                    x=time_points, y=predictions,
                    mode='markers',
                    marker=dict(color=predictions, colorscale='Plasma', size=8, opacity=0.7),
                    name='Predictions'
                ),
                row=2, col=1
            )
        
        # 5. Embedding Comparison
        fig.add_trace(
            go.Scatter(
                x=spatial_2d[:, 0], y=temporal_2d[:, 0],
                mode='markers',
                marker=dict(color=predictions, colorscale='RdYlBu', size=8, opacity=0.7),
                name='Embedding Correlation'
            ),
            row=2, col=2
        )
        
        # 6. Prediction Distribution
        fig.add_trace(
            go.Histogram(
                x=predictions,
                nbinsx=20,
                name='Prediction Distribution'
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title="Hybrid GNN-RNN Model Analysis Dashboard",
            showlegend=False,
            height=800,
            width=1200
        )
        
        # Save interactive plot
        output_path = os.path.join(self.output_dir, save_path)
        pyo.plot(fig, filename=output_path, auto_open=False)
        
        print(f"Interactive dashboard saved to: {output_path}")
        
        return fig
    
    def plot_gene_expression_patterns(
        self,
        expression_data: np.ndarray,
        gene_names: List[str],
        coordinates: np.ndarray,
        cell_types: Optional[np.ndarray] = None,
        top_genes: int = 12,
        save_path: Optional[str] = None
    ):
        """Visualize spatial gene expression patterns."""
        
        # Select top variable genes
        gene_vars = np.var(expression_data, axis=0)
        top_indices = np.argsort(gene_vars)[-top_genes:]
        
        # Create subplot grid
        rows = int(np.ceil(top_genes / 4))
        fig, axes = plt.subplots(rows, 4, figsize=(20, 5*rows))
        axes = axes.flatten() if top_genes > 4 else [axes]
        
        for i, gene_idx in enumerate(top_indices):
            if i >= len(axes):
                break
                
            expression = expression_data[:, gene_idx]
            gene_name = gene_names[gene_idx] if gene_idx < len(gene_names) else f'Gene {gene_idx}'
            
            scatter = axes[i].scatter(
                coordinates[:, 0], coordinates[:, 1],
                c=expression, cmap='YlOrRd', s=20, alpha=0.8
            )
            
            axes[i].set_title(f'{gene_name}\n(Var: {gene_vars[gene_idx]:.3f})')
            axes[i].set_xlabel('X Coordinate')
            axes[i].set_ylabel('Y Coordinate')
            
            # Add colorbar
            plt.colorbar(scatter, ax=axes[i], shrink=0.8)
        
        # Hide unused subplots
        for i in range(len(top_indices), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_uncertainty_analysis(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        y_true: np.ndarray,
        coordinates: Optional[np.ndarray] = None,
        title: str = "Uncertainty Analysis",
        save_path: Optional[str] = None
    ):
        """Analyze model uncertainty and prediction confidence."""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Uncertainty vs Accuracy
        errors = np.abs(predictions - y_true)
        axes[0, 0].scatter(uncertainties, errors, alpha=0.6)
        axes[0, 0].set_xlabel('Prediction Uncertainty')
        axes[0, 0].set_ylabel('Prediction Error')
        axes[0, 0].set_title('Uncertainty vs Error')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add correlation
        correlation = np.corrcoef(uncertainties, errors)[0, 1]
        axes[0, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=axes[0, 0].transAxes, fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 2. Uncertainty Distribution
        axes[0, 1].hist(uncertainties, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(np.mean(uncertainties), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(uncertainties):.3f}')
        axes[0, 1].set_xlabel('Uncertainty')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Uncertainty Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. High vs Low Uncertainty Performance
        threshold = np.median(uncertainties)
        high_unc_mask = uncertainties > threshold
        low_unc_mask = uncertainties <= threshold
        
        high_unc_error = np.mean(errors[high_unc_mask])
        low_unc_error = np.mean(errors[low_unc_mask])
        
        categories = ['High Uncertainty', 'Low Uncertainty']
        mean_errors = [high_unc_error, low_unc_error]
        
        bars = axes[1, 0].bar(categories, mean_errors, color=['red', 'green'], alpha=0.7)
        axes[1, 0].set_ylabel('Mean Absolute Error')
        axes[1, 0].set_title('Error by Uncertainty Level')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, mean_errors):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Spatial Uncertainty Pattern
        if coordinates is not None:
            scatter = axes[1, 1].scatter(
                coordinates[:, 0], coordinates[:, 1],
                c=uncertainties, cmap='Reds', s=20, alpha=0.7
            )
            plt.colorbar(scatter, ax=axes[1, 1], label='Uncertainty')
            axes[1, 1].set_title('Spatial Uncertainty Pattern')
            axes[1, 1].set_xlabel('X Coordinate')
            axes[1, 1].set_ylabel('Y Coordinate')
        else:
            # Uncertainty vs Prediction scatter
            axes[1, 1].scatter(predictions, uncertainties, alpha=0.6)
            axes[1, 1].set_xlabel('Predictions')
            axes[1, 1].set_ylabel('Uncertainty')
            axes[1, 1].set_title('Predictions vs Uncertainty')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        
        plt.show()


def create_comprehensive_analysis(
    model_outputs: Dict,
    true_labels: Dict,
    coordinates: np.ndarray,
    output_dir: str = "comprehensive_analysis"
):
    """Create a comprehensive analysis report with all visualizations."""
    
    viz = ModelVisualization(output_dir)
    
    # Extract data
    spatial_emb = model_outputs.get('spatial_embeddings', np.array([]))
    temporal_emb = model_outputs.get('temporal_embeddings', np.array([]))
    predictions = model_outputs.get('predictions', np.array([]))
    uncertainties = model_outputs.get('uncertainties', np.array([]))
    
    # 1. Spatial Analysis
    if len(spatial_emb) > 0:
        viz.plot_spatial_embeddings(
            spatial_emb, coordinates,
            cell_types=true_labels.get('cell_types'),
            predictions=predictions,
            save_path="spatial_analysis.png"
        )
    
    # 2. Temporal Analysis
    if len(temporal_emb) > 0:
        viz.plot_temporal_trajectories(
            temporal_emb,
            time_points=true_labels.get('time_points', np.arange(len(temporal_emb))),
            predictions=predictions,
            save_path="temporal_analysis.png"
        )
    
    # 3. Performance Analysis
    if len(predictions) > 0:
        task_type = "regression" if np.issubdtype(predictions.dtype, np.floating) else "classification"
        viz.plot_model_performance(
            true_labels.get('targets', np.zeros_like(predictions)),
            predictions,
            task_type=task_type,
            save_path="performance_analysis.png"
        )
    
    # 4. Uncertainty Analysis
    if len(uncertainties) > 0:
        viz.plot_uncertainty_analysis(
            predictions, uncertainties,
            true_labels.get('targets', np.zeros_like(predictions)),
            coordinates=coordinates,
            save_path="uncertainty_analysis.png"
        )
    
    # 5. Interactive Dashboard
    if len(spatial_emb) > 0 and len(temporal_emb) > 0:
        viz.create_interactive_dashboard(
            spatial_emb, temporal_emb, coordinates, predictions,
            cell_types=true_labels.get('cell_types'),
            time_points=true_labels.get('time_points'),
            save_path="interactive_dashboard.html"
        )
    
    print(f"Comprehensive analysis completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    # Example usage
    viz = ModelVisualization("test_visualizations")
    
    # Generate sample data
    n_samples = 1000
    coordinates = np.random.randn(n_samples, 2)
    spatial_embeddings = np.random.randn(n_samples, 64)
    temporal_embeddings = np.random.randn(n_samples, 64)
    predictions = np.random.rand(n_samples)
    cell_types = np.random.randint(0, 5, n_samples)
    
    # Test visualizations
    viz.plot_spatial_embeddings(
        spatial_embeddings, coordinates, cell_types, predictions,
        save_path="test_spatial.png"
    )
    
    print("Test visualizations completed!")

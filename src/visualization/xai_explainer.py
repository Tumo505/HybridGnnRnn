"""
Explainable AI (XAI) tools for Graph Neural Networks.

This module provides interpretability methods for understanding GNN predictions
including gradient-based explanations, feature importance analysis, and 
node influence mapping for biological insights.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class GNNExplainer:
    """Explainable AI tools for Graph Neural Networks."""
    
    def __init__(self, model: torch.nn.Module, save_dir: str = "xai_explanations"):
        """
        Initialize the GNN explainer.
        
        Args:
            model: Trained GNN model
            save_dir: Directory to save explanation visualizations
        """
        self.model = model
        self.model.eval()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Feature names for biological interpretation
        self.feature_names = None
        self.class_names = [
            'Atrial CM', 'Ventricular CM', 'Conducting CM', 
            'Nodal CM', 'Epicardial CM'
        ]
        
    def set_feature_names(self, feature_names: List[str]):
        """Set feature names for biological interpretation."""
        self.feature_names = feature_names
        
    def gradient_based_explanation(self,
                                 data: Data,
                                 target_class: Optional[int] = None,
                                 method: str = 'vanilla_gradients') -> Dict[str, torch.Tensor]:
        """
        Generate gradient-based explanations for model predictions.
        
        Args:
            data: PyTorch Geometric data object
            target_class: Target class for explanation (if None, uses predicted class)
            method: Explanation method ('vanilla_gradients', 'integrated_gradients', 'guided_backprop')
            
        Returns:
            Dictionary containing attribution scores and metadata
        """
        self.model.eval()
        
        # Get model prediction
        with torch.no_grad():
            output = self.model(data)
            predicted_class = output.argmax(dim=1)
        
        if target_class is None:
            target_class = predicted_class[0].item()
        
        # Enable gradient computation for input features
        data.x.requires_grad_(True)
        
        if method == 'vanilla_gradients':
            # Standard gradient computation
            output = self.model(data)
            loss = output[0, target_class]
            loss.backward()
            attributions = data.x.grad.clone()
            
        elif method == 'integrated_gradients':
            # Integrated gradients implementation
            attributions = self._integrated_gradients(data, target_class, steps=50)
            
        elif method == 'guided_backprop':
            # Guided backpropagation
            attributions = self._guided_backprop(data, target_class)
            
        else:
            raise ValueError(f"Unknown attribution method: {method}")
        
        # Clean up gradients
        data.x.grad = None
        
        return {
            'attributions': attributions,
            'predicted_class': predicted_class[0].item(),
            'target_class': target_class,
            'prediction_confidence': F.softmax(output, dim=1).max().item(),
            'method': method
        }
    
    def _integrated_gradients(self, data: Data, target_class: int, steps: int = 50) -> torch.Tensor:
        """Compute integrated gradients."""
        # Create baseline (zeros)
        baseline = torch.zeros_like(data.x)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps)
        gradients = []
        
        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated_input = baseline + alpha * (data.x - baseline)
            interpolated_input.requires_grad_(True)
            
            # Create new data object
            interpolated_data = Data(x=interpolated_input, edge_index=data.edge_index,
                                   batch=data.batch if hasattr(data, 'batch') else None)
            
            # Forward pass
            output = self.model(interpolated_data)
            loss = output[0, target_class]
            
            # Backward pass
            loss.backward()
            gradients.append(interpolated_input.grad.clone())
            
            # Clean up
            interpolated_input.grad = None
        
        # Average gradients and multiply by input difference
        avg_gradients = torch.stack(gradients).mean(dim=0)
        integrated_gradients = (data.x - baseline) * avg_gradients
        
        return integrated_gradients
    
    def _guided_backprop(self, data: Data, target_class: int) -> torch.Tensor:
        """Compute guided backpropagation (simplified version)."""
        # For simplicity, this is similar to vanilla gradients
        # A full implementation would modify ReLU backwards pass
        output = self.model(data)
        loss = output[0, target_class]
        loss.backward()
        return data.x.grad.clone()
        
    def node_importance_analysis(self,
                               data: Data,
                               target_class: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze importance of individual nodes in the graph.
        
        Args:
            data: PyTorch Geometric data object
            target_class: Target class for analysis
            
        Returns:
            Dictionary containing node importance scores and analysis
        """
        self.model.eval()
        num_nodes = data.x.size(0)
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(data)
            baseline_pred = baseline_output.argmax(dim=1)
            if target_class is None:
                target_class = baseline_pred[0].item()
        
        node_importance = []
        
        # Test importance by masking each node
        for node_idx in range(num_nodes):
            # Create masked data (set node features to zero)
            masked_data = data.clone()
            masked_data.x[node_idx] = 0
            
            with torch.no_grad():
                masked_output = self.model(masked_data)
                
            # Calculate importance as change in prediction confidence
            baseline_conf = F.softmax(baseline_output, dim=1)[0, target_class].item()
            masked_conf = F.softmax(masked_output, dim=1)[0, target_class].item()
            importance = baseline_conf - masked_conf
            
            node_importance.append({
                'node_idx': node_idx,
                'importance': importance,
                'baseline_conf': baseline_conf,
                'masked_conf': masked_conf
            })
        
        # Sort by importance
        node_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            'node_importance': node_importance,
            'target_class': target_class,
            'baseline_prediction': baseline_pred[0].item(),
            'top_important_nodes': [x['node_idx'] for x in node_importance[:10]],
            'least_important_nodes': [x['node_idx'] for x in node_importance[-10:]]
        }
        
    def feature_importance_analysis(self,
                                  data: Data,
                                  target_class: Optional[int] = None,
                                  top_k: int = 20) -> Dict[str, Any]:
        """
        Analyze importance of individual features (genes).
        
        Args:
            data: PyTorch Geometric data object
            target_class: Target class for analysis
            top_k: Number of top features to return
            
        Returns:
            Dictionary containing feature importance scores
        """
        # Get gradient-based attributions
        explanations = self.gradient_based_explanation(data, target_class, 'integrated_gradients')
        attributions = explanations['attributions']
        
        # Calculate feature importance across all nodes
        feature_importance = torch.abs(attributions).mean(dim=0)
        
        # Get top features
        top_indices = torch.argsort(feature_importance, descending=True)[:top_k]
        top_features = feature_importance[top_indices]
        
        feature_analysis = {
            'feature_importance': feature_importance,
            'top_feature_indices': top_indices.tolist(),
            'top_feature_scores': top_features.tolist(),
            'target_class': explanations['target_class']
        }
        
        # Add feature names if available
        if self.feature_names:
            feature_analysis['top_feature_names'] = [
                self.feature_names[idx] for idx in top_indices.tolist()
            ]
        
        return feature_analysis
        
    def plot_feature_importance(self,
                              feature_analysis: Dict[str, Any],
                              title: str = "Top Important Features",
                              save_name: str = "feature_importance.png") -> None:
        """Plot feature importance analysis."""
        top_scores = feature_analysis['top_feature_scores']
        top_indices = feature_analysis['top_feature_indices']
        
        # Use feature names if available, otherwise use indices
        if 'top_feature_names' in feature_analysis:
            labels = feature_analysis['top_feature_names']
        else:
            labels = [f'Feature {idx}' for idx in top_indices]
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(top_scores)))
        
        bars = plt.barh(range(len(labels)), top_scores, color=colors)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel('Importance Score (Integrated Gradients)')
        plt.title(f'{title}\n(Class: {self.class_names[feature_analysis["target_class"]]})')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, top_scores)):
            plt.text(score + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_node_importance(self,
                           node_analysis: Dict[str, Any],
                           data: Data,
                           top_k: int = 20,
                           title: str = "Node Importance Analysis",
                           save_name: str = "node_importance.png") -> None:
        """Plot node importance analysis."""
        node_importance = node_analysis['node_importance'][:top_k]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot of node importance
        node_indices = [x['node_idx'] for x in node_importance]
        importance_scores = [x['importance'] for x in node_importance]
        
        colors = plt.cm.RdYlGn(np.array(importance_scores) / max(importance_scores))
        bars = ax1.bar(range(len(node_indices)), importance_scores, color=colors)
        ax1.set_title(f'Top {top_k} Most Important Nodes')
        ax1.set_xlabel('Node Rank')
        ax1.set_ylabel('Importance Score')
        ax1.set_xticks(range(len(node_indices)))
        ax1.set_xticklabels([f'N{idx}' for idx in node_indices], rotation=45)
        
        # Add value labels
        for bar, score in zip(bars, importance_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Distribution of importance scores
        all_scores = [x['importance'] for x in node_analysis['node_importance']]
        ax2.hist(all_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(all_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_scores):.3f}')
        ax2.set_title('Distribution of Node Importance Scores')
        ax2.set_xlabel('Importance Score')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_explanation_summary(self,
                                   data: Data,
                                   target_class: Optional[int] = None,
                                   save_name: str = "explanation_summary.png") -> Dict[str, Any]:
        """
        Generate comprehensive explanation summary.
        
        Args:
            data: PyTorch Geometric data object
            target_class: Target class for explanation
            save_name: Filename to save summary plot
            
        Returns:
            Dictionary containing all explanation results
        """
        # Get all explanations
        gradient_exp = self.gradient_based_explanation(data, target_class)
        node_analysis = self.node_importance_analysis(data, target_class)
        feature_analysis = self.feature_importance_analysis(data, target_class)
        
        # Create comprehensive summary plot
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Prediction confidence
        ax1 = plt.subplot(2, 4, 1)
        with torch.no_grad():
            output = self.model(data)
            probs = F.softmax(output, dim=1)[0]
        
        bars = plt.bar(range(len(self.class_names)), probs.tolist(), 
                      color=plt.cm.Set3(range(len(self.class_names))))
        plt.title('Prediction Confidence')
        plt.xlabel('Cardiomyocyte Subtype')
        plt.ylabel('Probability')
        plt.xticks(range(len(self.class_names)), self.class_names, rotation=45)
        
        # Highlight predicted class
        predicted_idx = probs.argmax().item()
        bars[predicted_idx].set_color('red')
        bars[predicted_idx].set_alpha(0.8)
        
        # 2. Feature attribution heatmap
        ax2 = plt.subplot(2, 4, 2)
        attr_matrix = gradient_exp['attributions'].abs()
        # Show subset for visualization
        display_nodes = min(50, attr_matrix.size(0))
        display_features = min(50, attr_matrix.size(1))
        subset = attr_matrix[:display_nodes, :display_features]
        
        im = plt.imshow(subset.detach().cpu().numpy(), cmap='YlOrRd', aspect='auto')
        plt.title('Feature Attributions\n(Nodes × Features)')
        plt.xlabel('Features')
        plt.ylabel('Nodes')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        
        # 3. Top important features
        ax3 = plt.subplot(2, 4, 3)
        top_features = feature_analysis['top_feature_scores'][:10]
        feature_labels = (feature_analysis.get('top_feature_names', 
                         [f'F{i}' for i in feature_analysis['top_feature_indices']])[:10])
        
        plt.barh(range(len(top_features)), top_features, color='lightcoral')
        plt.yticks(range(len(feature_labels)), feature_labels)
        plt.xlabel('Importance Score')
        plt.title('Top 10 Important Features')
        plt.gca().invert_yaxis()
        
        # 4. Node importance distribution
        ax4 = plt.subplot(2, 4, 4)
        node_scores = [x['importance'] for x in node_analysis['node_importance']]
        plt.hist(node_scores, bins=20, alpha=0.7, color='lightgreen')
        plt.axvline(np.mean(node_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(node_scores):.3f}')
        plt.title('Node Importance Distribution')
        plt.xlabel('Importance Score')
        plt.ylabel('Count')
        plt.legend()
        
        # 5-8. Per-class feature importance (if multiple classes)
        for i in range(4):
            ax = plt.subplot(2, 4, 5 + i)
            if i < len(self.class_names):
                # Get feature importance for this class
                class_exp = self.gradient_based_explanation(data, target_class=i)
                class_features = torch.abs(class_exp['attributions']).mean(dim=0)
                top_indices = torch.argsort(class_features, descending=True)[:10]
                top_scores = class_features[top_indices]
                
                plt.bar(range(len(top_scores)), top_scores.detach().cpu().numpy(), 
                       color=plt.cm.Set3(i), alpha=0.7)
                plt.title(f'{self.class_names[i][:8]}...\nTop Features')
                plt.xlabel('Feature Rank')
                plt.ylabel('Importance')
                plt.xticks(range(len(top_scores)), [f'{idx}' for idx in top_indices.tolist()], 
                          rotation=45)
            else:
                plt.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                        transform=ax.transAxes)
                plt.title('Empty')
        
        plt.suptitle(f'GNN Explanation Summary\nPredicted: {self.class_names[predicted_idx]} '
                    f'(Confidence: {probs.max():.3f})', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'gradient_explanation': gradient_exp,
            'node_importance': node_analysis,
            'feature_importance': feature_analysis,
            'prediction_probabilities': probs.tolist(),
            'predicted_class': predicted_idx
        }
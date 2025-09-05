"""
Prediction Heads for Hybrid GNN-RNN Framework

This module implements various prediction heads for the hybrid framework,
including regression, classification, and multi-task heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head for differentiation efficiency and maturation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_maturation_classes: int = 3,
        use_uncertainty: bool = True
    ):
        super(MultiTaskHead, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_maturation_classes = num_maturation_classes
        self.use_uncertainty = use_uncertainty
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Differentiation efficiency head (regression)
        self.efficiency_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        # Maturation classification head
        self.maturation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_maturation_classes)
        )
        
        # Uncertainty estimation heads (if enabled)
        if self.use_uncertainty:
            self.efficiency_uncertainty = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus()  # Ensures positive uncertainty
            )
            
            self.maturation_uncertainty = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus()
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of multi-task head.
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of predictions
        """
        # Shared representation
        shared_repr = self.shared_layers(x)
        
        # Task-specific predictions
        efficiency_pred = self.efficiency_head(shared_repr)
        maturation_logits = self.maturation_head(shared_repr)
        maturation_probs = F.softmax(maturation_logits, dim=-1)
        
        outputs = {
            'differentiation_efficiency': efficiency_pred.squeeze(-1),
            'maturation_logits': maturation_logits,
            'maturation_probs': maturation_probs
        }
        
        # Add uncertainty estimates if enabled
        if self.use_uncertainty:
            efficiency_uncertainty = self.efficiency_uncertainty(shared_repr)
            maturation_uncertainty = self.maturation_uncertainty(shared_repr)
            
            outputs.update({
                'efficiency_uncertainty': efficiency_uncertainty.squeeze(-1),
                'maturation_uncertainty': maturation_uncertainty.squeeze(-1)
            })
        
        return outputs


class DifferentiationEfficiencyHead(nn.Module):
    """
    Specialized head for predicting cardiomyocyte differentiation efficiency.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        output_activation: str = "sigmoid"  # "sigmoid", "tanh", "none"
    ):
        super(DifferentiationEfficiencyHead, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_activation = output_activation
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Output activation
        if output_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif output_activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for efficiency prediction.
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Efficiency predictions [batch_size]
        """
        output = self.layers(x)
        output = self.activation(output)
        return output.squeeze(-1)


class MaturationClassificationHead(nn.Module):
    """
    Specialized head for predicting cardiomyocyte maturation stage.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,  # immature, intermediate, mature
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_ordinal: bool = True
    ):
        super(MaturationClassificationHead, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.use_ordinal = use_ordinal
        
        self.feature_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        if use_ordinal:
            # Ordinal classification (assumes ordered classes)
            self.ordinal_layer = nn.Linear(hidden_dim // 2, num_classes - 1)
        else:
            # Standard multi-class classification
            self.classification_layer = nn.Linear(hidden_dim // 2, num_classes)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for maturation classification.
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
            
        Returns:
            Dict[str, torch.Tensor]: Classification outputs
        """
        features = self.feature_layers(x)
        
        if self.use_ordinal:
            # Ordinal classification
            ordinal_logits = self.ordinal_layer(features)
            ordinal_probs = torch.sigmoid(ordinal_logits)
            
            # Convert ordinal probabilities to class probabilities
            class_probs = self._ordinal_to_class_probs(ordinal_probs)
            
            return {
                'ordinal_logits': ordinal_logits,
                'ordinal_probs': ordinal_probs,
                'class_probs': class_probs,
                'predictions': torch.argmax(class_probs, dim=-1)
            }
        else:
            # Standard classification
            logits = self.classification_layer(features)
            probs = F.softmax(logits, dim=-1)
            
            return {
                'logits': logits,
                'probs': probs,
                'predictions': torch.argmax(probs, dim=-1)
            }
    
    def _ordinal_to_class_probs(self, ordinal_probs: torch.Tensor) -> torch.Tensor:
        """Convert ordinal probabilities to class probabilities."""
        batch_size = ordinal_probs.size(0)
        num_ordinal = ordinal_probs.size(1)
        num_classes = num_ordinal + 1
        
        class_probs = torch.zeros(batch_size, num_classes, device=ordinal_probs.device)
        
        # First class: (1 - p1)
        class_probs[:, 0] = 1 - ordinal_probs[:, 0]
        
        # Middle classes: (p_{k-1} - p_k)
        for k in range(1, num_ordinal):
            class_probs[:, k] = ordinal_probs[:, k-1] - ordinal_probs[:, k]
        
        # Last class: p_{K-1}
        class_probs[:, -1] = ordinal_probs[:, -1]
        
        return class_probs


class FunctionalMaturationHead(nn.Module):
    """
    Head for predicting functional maturation markers.
    """
    
    def __init__(
        self,
        input_dim: int,
        functional_markers: Dict[str, int],  # e.g., {'contractility': 1, 'calcium_handling': 1}
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super(FunctionalMaturationHead, self).__init__()
        
        self.input_dim = input_dim
        self.functional_markers = functional_markers
        self.hidden_dim = hidden_dim
        
        # Shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads for each functional marker
        self.marker_heads = nn.ModuleDict()
        for marker_name, output_dim in functional_markers.items():
            if output_dim == 1:
                # Regression head
                head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid()
                )
            else:
                # Classification head
                head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, output_dim)
                )
            
            self.marker_heads[marker_name] = head
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for functional maturation prediction.
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
            
        Returns:
            Dict[str, torch.Tensor]: Functional marker predictions
        """
        shared_features = self.shared_layers(x)
        
        outputs = {}
        for marker_name, head in self.marker_heads.items():
            prediction = head(shared_features)
            
            # Process output based on type
            if self.functional_markers[marker_name] == 1:
                # Regression output
                outputs[marker_name] = prediction.squeeze(-1)
            else:
                # Classification output
                outputs[f"{marker_name}_logits"] = prediction
                outputs[f"{marker_name}_probs"] = F.softmax(prediction, dim=-1)
        
        return outputs


class UncertaintyHead(nn.Module):
    """
    Head for uncertainty estimation using evidential learning.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        uncertainty_type: str = "aleatoric"  # "aleatoric", "epistemic", "both"
    ):
        super(UncertaintyHead, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.uncertainty_type = uncertainty_type
        
        # Mean prediction
        self.mean_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Aleatoric uncertainty (data uncertainty)
        if uncertainty_type in ["aleatoric", "both"]:
            self.aleatoric_head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.Softplus()  # Ensure positive
            )
        
        # Epistemic uncertainty (model uncertainty) via Monte Carlo Dropout
        if uncertainty_type in ["epistemic", "both"]:
            self.epistemic_dropout = nn.Dropout(0.5)
    
    def forward(
        self, 
        x: torch.Tensor, 
        num_samples: int = 10,
        training: bool = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
            num_samples (int): Number of MC samples for epistemic uncertainty
            training (bool): Whether in training mode
            
        Returns:
            Dict[str, torch.Tensor]: Predictions with uncertainty estimates
        """
        # Mean prediction
        mean_pred = self.mean_head(x)
        
        outputs = {'mean': mean_pred}
        
        # Aleatoric uncertainty
        if self.uncertainty_type in ["aleatoric", "both"]:
            aleatoric_var = self.aleatoric_head(x)
            outputs['aleatoric_uncertainty'] = aleatoric_var
        
        # Epistemic uncertainty via MC Dropout
        if self.uncertainty_type in ["epistemic", "both"] and not (training is False):
            mc_predictions = []
            
            # Enable dropout for MC sampling
            self.train()
            
            for _ in range(num_samples):
                x_dropped = self.epistemic_dropout(x)
                mc_pred = self.mean_head(x_dropped)
                mc_predictions.append(mc_pred)
            
            mc_predictions = torch.stack(mc_predictions, dim=0)  # [num_samples, batch_size, output_dim]
            
            # Calculate epistemic uncertainty
            epistemic_mean = mc_predictions.mean(dim=0)
            epistemic_var = mc_predictions.var(dim=0)
            
            outputs.update({
                'epistemic_mean': epistemic_mean,
                'epistemic_uncertainty': epistemic_var,
                'mc_predictions': mc_predictions
            })
        
        return outputs


class DomainAdaptationHead(nn.Module):
    """
    Domain adaptation head for reducing batch/site effects.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_domains: int,
        hidden_dim: int = 128,
        gradient_reversal_lambda: float = 1.0
    ):
        super(DomainAdaptationHead, self).__init__()
        
        self.input_dim = input_dim
        self.num_domains = num_domains
        self.gradient_reversal_lambda = gradient_reversal_lambda
        
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(gradient_reversal_lambda),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_domains)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for domain classification.
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Domain logits [batch_size, num_domains]
        """
        return self.domain_classifier(x)


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient reversal layer for domain adaptation.
    """
    
    @staticmethod
    def forward(ctx, x, lambda_val=1.0):
        ctx.lambda_val = lambda_val
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


def GradientReversalLayer(lambda_val=1.0):
    """Factory function for gradient reversal layer."""
    def grl_layer(x):
        return GradientReversalLayer.apply(x, lambda_val)
    return grl_layer


if __name__ == "__main__":
    # Example usage and testing
    torch.manual_seed(42)
    
    # Sample features
    batch_size = 16
    input_dim = 256
    
    features = torch.randn(batch_size, input_dim)
    
    print(f"Testing prediction heads:")
    print(f"  Batch size: {batch_size}")
    print(f"  Input dim: {input_dim}")
    
    # Test MultiTaskHead
    multi_task_head = MultiTaskHead(
        input_dim=input_dim,
        hidden_dim=128,
        num_maturation_classes=3,
        use_uncertainty=True
    )
    
    print(f"\nMultiTaskHead:")
    print(f"  Parameters: {sum(p.numel() for p in multi_task_head.parameters()):,}")
    
    with torch.no_grad():
        outputs = multi_task_head(features)
        print(f"  Outputs: {list(outputs.keys())}")
        for key, value in outputs.items():
            print(f"    {key}: {value.shape}")
    
    # Test FunctionalMaturationHead
    functional_head = FunctionalMaturationHead(
        input_dim=input_dim,
        functional_markers={'contractility': 1, 'calcium_handling': 3},
        hidden_dim=128
    )
    
    print(f"\nFunctionalMaturationHead:")
    print(f"  Parameters: {sum(p.numel() for p in functional_head.parameters()):,}")
    
    with torch.no_grad():
        func_outputs = functional_head(features)
        print(f"  Outputs: {list(func_outputs.keys())}")
        for key, value in func_outputs.items():
            print(f"    {key}: {value.shape}")

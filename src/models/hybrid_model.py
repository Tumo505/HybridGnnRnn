"""
Hybrid GNN-RNN Model for Cardiomyocyte Differentiation Prediction

This module implements the main hybrid model that combines spatial (GNN) and
temporal (RNN) encoders with fusion mechanisms and prediction heads.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union, List
import warnings

from .gnn_encoder import SpatialGNNEncoder
from .rnn_encoder import TemporalRNNEncoder
from .fusion import SpatioTemporalFusion
from .heads import (
    MultiTaskHead, 
    DifferentiationEfficiencyHead,
    MaturationClassificationHead,
    FunctionalMaturationHead,
    UncertaintyHead,
    DomainAdaptationHead
)


class HybridGNNRNN(nn.Module):
    """
    Hybrid GNN-RNN model for predicting cardiomyocyte differentiation efficiency.
    
    Combines spatial information (cell-cell interactions, tissue architecture) 
    via Graph Neural Networks with temporal dynamics (differentiation trajectory)
    via Recurrent Neural Networks.
    """
    
    def __init__(
        self,
        # Data dimensions
        node_feature_dim: int,
        edge_feature_dim: int = None,
        spatial_dim: int = 2,
        sequence_length: int = 7,
        
        # Model architecture
        gnn_hidden_dim: int = 256,
        rnn_hidden_dim: int = 256,
        fusion_dim: int = 512,
        
        # Model configurations
        gnn_type: str = "GraphSAGE",  # "GraphSAGE", "GAT", "GCN"
        rnn_type: str = "LSTM",  # "LSTM", "GRU"
        fusion_type: str = "attention",  # "attention", "gated", "adaptive"
        
        # Training configurations
        dropout: float = 0.1,
        num_gnn_layers: int = 3,
        num_rnn_layers: int = 2,
        bidirectional: bool = True,
        
        # Prediction tasks
        prediction_tasks: List[str] = ["efficiency", "maturation"],
        num_maturation_classes: int = 3,
        functional_markers: Dict[str, int] = None,
        
        # Advanced features
        use_positional_encoding: bool = True,
        use_temporal_attention: bool = True,
        use_uncertainty: bool = True,
        use_domain_adaptation: bool = False,
        num_domains: int = 1,
        
        # Memory optimization
        use_checkpoint: bool = True,
        memory_efficient: bool = True
    ):
        super(HybridGNNRNN, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.spatial_dim = spatial_dim
        self.sequence_length = sequence_length
        self.prediction_tasks = prediction_tasks
        self.use_uncertainty = use_uncertainty
        self.use_domain_adaptation = use_domain_adaptation
        self.memory_efficient = memory_efficient
        self.use_checkpoint = use_checkpoint
        
        # Initialize spatial encoder (GNN)
        self.spatial_encoder = SpatialGNNEncoder(
            input_dim=node_feature_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_hidden_dim,
            num_layers=num_gnn_layers,
            gnn_type=gnn_type,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding
        )
        
        # Initialize temporal encoder (RNN)
        self.temporal_encoder = TemporalRNNEncoder(
            input_dim=node_feature_dim,
            hidden_dim=rnn_hidden_dim,
            output_dim=rnn_hidden_dim,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            dropout=dropout,
            use_attention=use_temporal_attention,
            use_cell_type_embedding=False  # Disable for now
        )
        
        # Initialize fusion module
        spatial_out_dim = gnn_hidden_dim
        # The temporal encoder outputs to output_dim (same as rnn_hidden_dim), not the internal RNN dimension
        temporal_out_dim = rnn_hidden_dim
        
        self.fusion_module = SpatioTemporalFusion(
            spatial_dim=spatial_out_dim,
            temporal_dim=temporal_out_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
            fusion_strategy=fusion_type if fusion_type in ["concat", "add", "multiply", "gated"] else "concat"
        )
        
        # Initialize prediction heads
        self.prediction_heads = nn.ModuleDict()
        
        if "efficiency" in prediction_tasks:
            self.prediction_heads["efficiency"] = DifferentiationEfficiencyHead(
                input_dim=fusion_dim,
                hidden_dim=fusion_dim // 2,
                dropout=dropout
            )
        
        if "maturation" in prediction_tasks:
            self.prediction_heads["maturation"] = MaturationClassificationHead(
                input_dim=fusion_dim,
                num_classes=num_maturation_classes,
                hidden_dim=fusion_dim // 2,
                dropout=dropout
            )
        
        if "multitask" in prediction_tasks:
            self.prediction_heads["multitask"] = MultiTaskHead(
                input_dim=fusion_dim,
                hidden_dim=fusion_dim // 2,
                dropout=dropout,
                num_maturation_classes=num_maturation_classes,
                use_uncertainty=use_uncertainty
            )
        
        if functional_markers and "functional" in prediction_tasks:
            self.prediction_heads["functional"] = FunctionalMaturationHead(
                input_dim=fusion_dim,
                functional_markers=functional_markers,
                hidden_dim=fusion_dim // 2,
                dropout=dropout
            )
        
        # Uncertainty head (if enabled)
        if use_uncertainty and "uncertainty" in prediction_tasks:
            self.prediction_heads["uncertainty"] = UncertaintyHead(
                input_dim=fusion_dim,
                output_dim=1,
                hidden_dim=fusion_dim // 2,
                dropout=dropout,
                uncertainty_type="both"
            )
        
        # Domain adaptation head (if enabled)
        if use_domain_adaptation:
            self.domain_adapter = DomainAdaptationHead(
                input_dim=fusion_dim,
                num_domains=num_domains,
                hidden_dim=fusion_dim // 2
            )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def forward(
        self,
        # Spatial data
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        
        # Temporal data
        temporal_features: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
        cell_types: Optional[torch.Tensor] = None,
        
        # Additional inputs
        domain_labels: Optional[torch.Tensor] = None,
        
        # Control flags
        return_embeddings: bool = False,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the hybrid model.
        
        Args:
            # Spatial inputs
            node_features (torch.Tensor): Node features [N, F]
            edge_index (torch.Tensor): Edge indices [2, E]
            edge_attr (torch.Tensor, optional): Edge attributes [E, edge_dim]
            pos (torch.Tensor, optional): Spatial coordinates [N, spatial_dim]
            batch (torch.Tensor, optional): Batch assignment [N]
            
            # Temporal inputs
            temporal_features (torch.Tensor, optional): Temporal features [B, T, F]
            temporal_mask (torch.Tensor, optional): Temporal mask [B, T]
            cell_types (torch.Tensor, optional): Cell type labels [B, T]
            
            # Additional inputs
            domain_labels (torch.Tensor, optional): Domain labels [B]
            
            # Control flags
            return_embeddings (bool): Whether to return intermediate embeddings
            return_attention (bool): Whether to return attention weights
            
        Returns:
            Dict[str, torch.Tensor]: Model outputs including predictions
        """
        outputs = {}
        
        # Spatial encoding
        if self.use_checkpoint and self.training:
            spatial_embeddings = torch.utils.checkpoint.checkpoint(
                self._forward_spatial,
                node_features,
                edge_index,
                edge_attr,
                pos,
                batch
            )
        else:
            spatial_embeddings = self._forward_spatial(
                node_features, edge_index, edge_attr, pos, batch
            )
        
        # Temporal encoding (if temporal data provided)
        if temporal_features is not None:
            if self.use_checkpoint and self.training:
                temporal_embeddings = torch.utils.checkpoint.checkpoint(
                    self._forward_temporal,
                    temporal_features,
                    temporal_mask,
                    cell_types
                )
            else:
                temporal_embeddings = self._forward_temporal(
                    temporal_features, temporal_mask, cell_types
                )
        else:
            # Use spatial embeddings as temporal if no temporal data
            temporal_embeddings = spatial_embeddings
        
        # Fusion
        fused_embeddings = self.fusion_module(
            spatial_embeddings, 
            temporal_embeddings
        )
        
        # Store intermediate embeddings if requested
        if return_embeddings:
            outputs.update({
                "spatial_embeddings": spatial_embeddings,
                "temporal_embeddings": temporal_embeddings,
                "fused_embeddings": fused_embeddings
            })
        
        # Predictions
        for task_name, head in self.prediction_heads.items():
            if task_name == "uncertainty":
                task_output = head(
                    fused_embeddings, 
                    num_samples=10 if self.training else 5,
                    training=self.training
                )
            else:
                task_output = head(fused_embeddings)
            
            # Add task prefix to output keys
            if isinstance(task_output, dict):
                for key, value in task_output.items():
                    outputs[f"{task_name}_{key}"] = value
            else:
                outputs[task_name] = task_output
        
        # Domain adaptation (if enabled)
        if self.use_domain_adaptation:
            domain_logits = self.domain_adapter(fused_embeddings)
            outputs["domain_logits"] = domain_logits
        
        return outputs
    
    def _forward_spatial(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through spatial encoder."""
        return self.spatial_encoder(
            x=node_features,
            edge_index=edge_index,
            pos=pos,
            batch=batch
        )
    
    def _forward_temporal(
        self,
        temporal_features: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
        cell_types: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through temporal encoder."""
        lengths = temporal_mask.sum(-1) if temporal_mask is not None else None
        return self.temporal_encoder(
            x=temporal_features,
            lengths=lengths,
            cell_types=cell_types
        )
    
    def predict(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        temporal_features: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Prediction method for inference.
        
        Args:
            Same as forward() but with return_uncertainty flag
            
        Returns:
            Dict[str, torch.Tensor]: Predictions with optional uncertainty
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(
                node_features=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=pos,
                batch=batch,
                temporal_features=temporal_features,
                temporal_mask=temporal_mask,
                return_embeddings=False,
                return_attention=False
            )
        
        # Extract main predictions
        predictions = {}
        
        if "efficiency" in outputs:
            predictions["differentiation_efficiency"] = outputs["efficiency"]
        
        if "multitask_differentiation_efficiency" in outputs:
            predictions["differentiation_efficiency"] = outputs["multitask_differentiation_efficiency"]
            predictions["maturation_probs"] = outputs["multitask_maturation_probs"]
        
        if "maturation_probs" in outputs:
            predictions["maturation_probs"] = outputs["maturation_probs"]
            predictions["maturation_prediction"] = torch.argmax(outputs["maturation_probs"], dim=-1)
        
        # Add uncertainty if requested and available
        if return_uncertainty and self.use_uncertainty:
            for key in outputs:
                if "uncertainty" in key:
                    predictions[key] = outputs[key]
        
        return predictions
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model summary including parameter counts and architecture info.
        
        Returns:
            Dict[str, Any]: Model summary
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Component parameter counts
        spatial_params = sum(p.numel() for p in self.spatial_encoder.parameters())
        temporal_params = sum(p.numel() for p in self.temporal_encoder.parameters())
        fusion_params = sum(p.numel() for p in self.fusion_module.parameters())
        head_params = sum(p.numel() for head in self.prediction_heads.values() 
                         for p in head.parameters())
        
        summary = {
            "model_type": "HybridGNNRNN",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "component_parameters": {
                "spatial_encoder": spatial_params,
                "temporal_encoder": temporal_params,
                "fusion_module": fusion_params,
                "prediction_heads": head_params
            },
            "architecture": {
                "spatial_encoder": self.spatial_encoder.__class__.__name__,
                "temporal_encoder": self.temporal_encoder.__class__.__name__,
                "fusion_module": self.fusion_module.__class__.__name__,
                "prediction_tasks": self.prediction_tasks
            },
            "memory_optimization": {
                "gradient_checkpointing": self.use_checkpoint,
                "memory_efficient": self.memory_efficient
            }
        }
        
        return summary


class LightweightHybridGNNRNN(HybridGNNRNN):
    """
    Lightweight version of the hybrid model for memory-constrained environments.
    """
    
    def __init__(self, **kwargs):
        # Override default parameters for memory efficiency
        lightweight_defaults = {
            "gnn_hidden_dim": 128,
            "rnn_hidden_dim": 128,
            "fusion_dim": 256,
            "num_gnn_layers": 2,
            "num_rnn_layers": 1,
            "dropout": 0.2,
            "use_checkpoint": True,
            "memory_efficient": True,
            "use_positional_encoding": False,
            "use_temporal_attention": False
        }
        
        # Update with provided kwargs
        for key, value in lightweight_defaults.items():
            kwargs.setdefault(key, value)
        
        super().__init__(**kwargs)


if __name__ == "__main__":
    # Example usage and testing
    torch.manual_seed(42)
    
    # Sample data dimensions
    batch_size = 4
    num_nodes = 1000
    node_feature_dim = 100
    edge_feature_dim = 10
    sequence_length = 7
    spatial_dim = 2
    
    print(f"Testing Hybrid GNN-RNN Model:")
    print(f"  Batch size: {batch_size}")
    print(f"  Nodes per graph: {num_nodes}")
    print(f"  Node features: {node_feature_dim}")
    
    # Create model
    model = HybridGNNRNN(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        spatial_dim=spatial_dim,
        sequence_length=sequence_length,
        gnn_hidden_dim=128,
        rnn_hidden_dim=128,
        fusion_dim=256,
        prediction_tasks=["multitask"],
        use_uncertainty=True,
        memory_efficient=True
    )
    
    print(f"\nModel Summary:")
    summary = model.get_model_summary()
    print(f"  Total parameters: {summary['total_parameters']:,}")
    print(f"  Trainable parameters: {summary['trainable_parameters']:,}")
    
    # Sample spatial data
    node_features = torch.randn(batch_size * num_nodes, node_feature_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 4))
    edge_attr = torch.randn(edge_index.size(1), edge_feature_dim)
    pos = torch.randn(batch_size * num_nodes, spatial_dim)
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)
    
    # Sample temporal data
    temporal_features = torch.randn(batch_size, sequence_length, node_feature_dim)
    temporal_mask = torch.ones(batch_size, sequence_length)
    
    print(f"\nTesting forward pass...")
    model.eval()
    
    with torch.no_grad():
        outputs = model(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            batch=batch,
            temporal_features=temporal_features,
            temporal_mask=temporal_mask,
            return_embeddings=True
        )
    
    print(f"  Outputs: {list(outputs.keys())}")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"    {key}: {value.shape}")
    
    # Test prediction method
    print(f"\nTesting prediction method...")
    predictions = model.predict(
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        batch=batch,
        temporal_features=temporal_features,
        temporal_mask=temporal_mask,
        return_uncertainty=True
    )
    
    print(f"  Predictions: {list(predictions.keys())}")
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            print(f"    {key}: {value.shape}")

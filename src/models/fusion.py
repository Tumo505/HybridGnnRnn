"""
Fusion Module for Hybrid GNN-RNN Framework

This module implements the attention-based fusion mechanism that combines
spatial (GNN) and temporal (RNN) representations for integrated prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any


class SpatioTemporalFusion(nn.Module):
    """
    Attention-based fusion module for combining spatial and temporal representations.
    
    Uses cross-attention mechanisms to integrate spatial tissue architecture 
    information with temporal differentiation dynamics.
    """
    
    def __init__(
        self,
        spatial_dim: int,
        temporal_dim: int,
        fusion_dim: int = 256,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
        use_self_attention: bool = True,
        fusion_strategy: str = "concat"  # "concat", "add", "multiply", "gated"
    ):
        super(SpatioTemporalFusion, self).__init__()
        
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.fusion_dim = fusion_dim
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.use_cross_attention = use_cross_attention
        self.use_self_attention = use_self_attention
        self.fusion_strategy = fusion_strategy
        
        # Project inputs to common dimension
        self.spatial_projection = nn.Linear(spatial_dim, fusion_dim)
        self.temporal_projection = nn.Linear(temporal_dim, fusion_dim)
        
        # Cross-attention between spatial and temporal
        if self.use_cross_attention:
            self.spatial_to_temporal_attention = MultiHeadAttention(
                fusion_dim, num_attention_heads, dropout
            )
            self.temporal_to_spatial_attention = MultiHeadAttention(
                fusion_dim, num_attention_heads, dropout
            )
        
        # Self-attention for integrated representation
        if self.use_self_attention:
            self.self_attention = MultiHeadAttention(
                fusion_dim, num_attention_heads, dropout
            )
        
        # Fusion layers based on strategy
        if fusion_strategy == "concat":
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, fusion_dim)
            )
        elif fusion_strategy == "gated":
            self.gate_layer = GatedFusion(fusion_dim, dropout)
        elif fusion_strategy in ["add", "multiply"]:
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Output normalization
        self.layer_norm = nn.LayerNorm(fusion_dim)
        
    def forward(
        self, 
        spatial_features: torch.Tensor, 
        temporal_features: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the fusion module.
        
        Args:
            spatial_features (torch.Tensor): Spatial representations [batch_size, spatial_dim]
            temporal_features (torch.Tensor): Temporal representations [batch_size, temporal_dim]
            spatial_mask (torch.Tensor, optional): Mask for spatial features
            temporal_mask (torch.Tensor, optional): Mask for temporal features
            
        Returns:
            torch.Tensor: Fused spatiotemporal representation [batch_size, fusion_dim]
        """
        batch_size = spatial_features.size(0)
        
        # Project to common dimension
        spatial_proj = self.spatial_projection(spatial_features)  # [batch_size, fusion_dim]
        temporal_proj = self.temporal_projection(temporal_features)  # [batch_size, fusion_dim]
        
        # Add batch dimension for attention if needed
        spatial_proj = spatial_proj.unsqueeze(1)  # [batch_size, 1, fusion_dim]
        temporal_proj = temporal_proj.unsqueeze(1)  # [batch_size, 1, fusion_dim]
        
        # Cross-attention between modalities
        if self.use_cross_attention:
            # Spatial attends to temporal
            spatial_attended, _ = self.spatial_to_temporal_attention(
                spatial_proj, temporal_proj, temporal_proj,
                key_padding_mask=temporal_mask
            )
            
            # Temporal attends to spatial
            temporal_attended, _ = self.temporal_to_spatial_attention(
                temporal_proj, spatial_proj, spatial_proj,
                key_padding_mask=spatial_mask
            )
        else:
            spatial_attended = spatial_proj
            temporal_attended = temporal_proj
        
        # Remove sequence dimension
        spatial_attended = spatial_attended.squeeze(1)  # [batch_size, fusion_dim]
        temporal_attended = temporal_attended.squeeze(1)  # [batch_size, fusion_dim]
        
        # Fusion strategy
        if self.fusion_strategy == "concat":
            # Concatenate and project
            concatenated = torch.cat([spatial_attended, temporal_attended], dim=-1)
            fused = self.fusion_layer(concatenated)
            
        elif self.fusion_strategy == "add":
            # Element-wise addition
            fused = spatial_attended + temporal_attended
            fused = self.fusion_layer(fused)
            
        elif self.fusion_strategy == "multiply":
            # Element-wise multiplication
            fused = spatial_attended * temporal_attended
            fused = self.fusion_layer(fused)
            
        elif self.fusion_strategy == "gated":
            # Gated fusion
            fused = self.gate_layer(spatial_attended, temporal_attended)
        
        # Self-attention on fused representation
        if self.use_self_attention:
            fused_expanded = fused.unsqueeze(1)  # [batch_size, 1, fusion_dim]
            fused_attended, _ = self.self_attention(
                fused_expanded, fused_expanded, fused_expanded
            )
            fused = fused_attended.squeeze(1)  # [batch_size, fusion_dim]
        
        # Layer normalization
        fused = self.layer_norm(fused)
        
        return fused


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query (torch.Tensor): Query tensor [batch_size, seq_len, embed_dim]
            key (torch.Tensor): Key tensor [batch_size, seq_len, embed_dim]
            value (torch.Tensor): Value tensor [batch_size, seq_len, embed_dim]
            key_padding_mask (torch.Tensor, optional): Padding mask
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (attended_output, attention_weights)
        """
        batch_size, seq_len, embed_dim = query.size()
        
        # Linear projections
        Q = self.q_linear(query)  # [batch_size, seq_len, embed_dim]
        K = self.k_linear(key)    # [batch_size, seq_len, embed_dim]
        V = self.v_linear(value)  # [batch_size, seq_len, embed_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attended_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, key_padding_mask
        )
        
        # Reshape and project output
        attended_output = attended_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        output = self.out_linear(attended_output)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot-product attention computation."""
        d_k = Q.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)
        
        # Apply attention to values
        attended_output = torch.matmul(attention_weights, V)
        
        return attended_output, attention_weights


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism for combining two feature vectors.
    """
    
    def __init__(self, feature_dim: int, dropout: float = 0.1):
        super(GatedFusion, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Gate computation
        self.gate_linear = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        
        # Feature transformation
        self.transform_spatial = nn.Linear(feature_dim, feature_dim)
        self.transform_temporal = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        spatial_features: torch.Tensor, 
        temporal_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of gated fusion.
        
        Args:
            spatial_features (torch.Tensor): Spatial features [batch_size, feature_dim]
            temporal_features (torch.Tensor): Temporal features [batch_size, feature_dim]
            
        Returns:
            torch.Tensor: Fused features [batch_size, feature_dim]
        """
        # Compute gate
        concatenated = torch.cat([spatial_features, temporal_features], dim=-1)
        gate = self.gate_linear(concatenated)  # [batch_size, feature_dim]
        
        # Transform features
        spatial_transformed = self.transform_spatial(spatial_features)
        temporal_transformed = self.transform_temporal(temporal_features)
        
        # Gated combination
        fused = gate * spatial_transformed + (1 - gate) * temporal_transformed
        fused = self.dropout(fused)
        
        return fused


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion that learns optimal combination weights.
    """
    
    def __init__(
        self, 
        spatial_dim: int, 
        temporal_dim: int, 
        fusion_dim: int,
        num_experts: int = 4
    ):
        super(AdaptiveFusion, self).__init__()
        
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.fusion_dim = fusion_dim
        self.num_experts = num_experts
        
        # Expert networks
        self.spatial_experts = nn.ModuleList([
            nn.Linear(spatial_dim, fusion_dim) for _ in range(num_experts)
        ])
        self.temporal_experts = nn.ModuleList([
            nn.Linear(temporal_dim, fusion_dim) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gating_network = nn.Sequential(
            nn.Linear(spatial_dim + temporal_dim, num_experts * 2),
            nn.ReLU(),
            nn.Linear(num_experts * 2, num_experts * 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(
        self, 
        spatial_features: torch.Tensor, 
        temporal_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of adaptive fusion.
        
        Args:
            spatial_features (torch.Tensor): Spatial features [batch_size, spatial_dim]
            temporal_features (torch.Tensor): Temporal features [batch_size, temporal_dim]
            
        Returns:
            torch.Tensor: Adaptively fused features [batch_size, fusion_dim]
        """
        batch_size = spatial_features.size(0)
        
        # Compute gating weights
        concatenated = torch.cat([spatial_features, temporal_features], dim=-1)
        gates = self.gating_network(concatenated)  # [batch_size, num_experts * 2]
        
        spatial_gates = gates[:, :self.num_experts]  # [batch_size, num_experts]
        temporal_gates = gates[:, self.num_experts:]  # [batch_size, num_experts]
        
        # Apply experts
        spatial_outputs = []
        temporal_outputs = []
        
        for i in range(self.num_experts):
            spatial_out = self.spatial_experts[i](spatial_features)
            temporal_out = self.temporal_experts[i](temporal_features)
            
            spatial_outputs.append(spatial_out)
            temporal_outputs.append(temporal_out)
        
        # Weighted combination
        spatial_combined = torch.zeros(batch_size, self.fusion_dim, device=spatial_features.device)
        temporal_combined = torch.zeros(batch_size, self.fusion_dim, device=temporal_features.device)
        
        for i in range(self.num_experts):
            spatial_combined += spatial_gates[:, i:i+1] * spatial_outputs[i]
            temporal_combined += temporal_gates[:, i:i+1] * temporal_outputs[i]
        
        # Final combination
        fused = spatial_combined + temporal_combined
        
        return fused


if __name__ == "__main__":
    # Example usage and testing
    torch.manual_seed(42)
    
    # Sample features
    batch_size = 16
    spatial_dim = 128
    temporal_dim = 128
    fusion_dim = 256
    
    spatial_features = torch.randn(batch_size, spatial_dim)
    temporal_features = torch.randn(batch_size, temporal_dim)
    
    print(f"Testing fusion modules:")
    print(f"  Batch size: {batch_size}")
    print(f"  Spatial dim: {spatial_dim}")
    print(f"  Temporal dim: {temporal_dim}")
    print(f"  Fusion dim: {fusion_dim}")
    
    # Test SpatioTemporalFusion
    fusion_module = SpatioTemporalFusion(
        spatial_dim=spatial_dim,
        temporal_dim=temporal_dim,
        fusion_dim=fusion_dim,
        num_attention_heads=8,
        fusion_strategy="concat"
    )
    
    print(f"\nSpatioTemporalFusion:")
    print(f"  Parameters: {sum(p.numel() for p in fusion_module.parameters()):,}")
    
    with torch.no_grad():
        fused_output = fusion_module(spatial_features, temporal_features)
        print(f"  Output shape: {fused_output.shape}")
        print(f"  Output range: [{fused_output.min().item():.3f}, {fused_output.max().item():.3f}]")
    
    # Test AdaptiveFusion
    adaptive_fusion = AdaptiveFusion(
        spatial_dim=spatial_dim,
        temporal_dim=temporal_dim,
        fusion_dim=fusion_dim,
        num_experts=4
    )
    
    print(f"\nAdaptiveFusion:")
    print(f"  Parameters: {sum(p.numel() for p in adaptive_fusion.parameters()):,}")
    
    with torch.no_grad():
        adaptive_output = adaptive_fusion(spatial_features, temporal_features)
        print(f"  Output shape: {adaptive_output.shape}")
        print(f"  Output range: [{adaptive_output.min().item():.3f}, {adaptive_output.max().item():.3f}]")

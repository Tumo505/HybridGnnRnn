"""
Advanced Temporal RNN Model for Cardiomyocyte Differentiation Prediction
Optimized for NVIDIA RTX 5070 - Full Scale Implementation

This model implements a sophisticated LSTM-based architecture for predicting
cardiomyocyte differentiation efficiency from temporal gene expression data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
import math
from typing import Dict, List, Tuple, Optional, Union
import numpy as np


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for temporal features."""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = query.size(0), query.size(1)
        residual = query
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.w_o(attn_output)
        return self.layer_norm(output + residual)


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences."""
    
    def __init__(self, d_model: int, max_seq_length: int = 10):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class GeneExpressionEmbedding(nn.Module):
    """Advanced gene expression embedding with attention mechanism."""
    
    def __init__(self, input_dim: int, embed_dim: int, num_attention_heads: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Multi-layer embedding
        self.embedding_layers = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
        # Gene attention mechanism
        self.gene_attention = nn.MultiheadAttention(
            embed_dim, num_attention_heads, dropout=0.1, batch_first=True
        )
        
        # Pathway enrichment layers
        self.pathway_projection = nn.Linear(embed_dim, embed_dim // 2)
        self.pathway_attention = nn.Linear(embed_dim // 2, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial embedding
        embedded = self.embedding_layers(x)  # (batch, seq, embed_dim)
        
        # Self-attention across genes
        attn_output, _ = self.gene_attention(embedded, embedded, embedded)
        
        # Pathway-level attention
        pathway_features = F.relu(self.pathway_projection(attn_output))
        pathway_weights = F.softmax(self.pathway_attention(pathway_features), dim=-1)
        
        # Weighted combination
        enhanced_embedding = attn_output * pathway_weights
        
        return enhanced_embedding


class AdvancedLSTMCell(nn.Module):
    """Advanced LSTM cell with highway connections and layer normalization."""
    
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # LSTM gates
        self.weight_ih = nn.Linear(input_size, 4 * hidden_size)
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size)
        
        # Layer normalization
        self.ln_ih = nn.LayerNorm(4 * hidden_size)
        self.ln_hh = nn.LayerNorm(4 * hidden_size)
        self.ln_cell = nn.LayerNorm(hidden_size)
        
        # Highway connection
        self.highway_gate = nn.Linear(hidden_size, hidden_size)
        self.highway_transform = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_tensor: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = hidden
        
        # LSTM computations with layer norm
        gi = self.ln_ih(self.weight_ih(input_tensor))
        gh = self.ln_hh(self.weight_hh(h_prev))
        i_i, i_f, i_g, i_o = gi.chunk(4, 1)
        h_i, h_f, h_g, h_o = gh.chunk(4, 1)
        
        input_gate = torch.sigmoid(i_i + h_i)
        forget_gate = torch.sigmoid(i_f + h_f)
        cell_gate = torch.tanh(i_g + h_g)
        output_gate = torch.sigmoid(i_o + h_o)
        
        c_new = forget_gate * c_prev + input_gate * cell_gate
        c_new = self.ln_cell(c_new)
        
        h_new = output_gate * torch.tanh(c_new)
        
        # Highway connection
        highway_gate = torch.sigmoid(self.highway_gate(h_new))
        highway_transform = torch.relu(self.highway_transform(h_new))
        h_new = highway_gate * highway_transform + (1 - highway_gate) * h_new
        
        h_new = self.dropout(h_new)
        
        return h_new, c_new


class AdvancedTemporalRNN(nn.Module):
    """
    Advanced Temporal RNN for Cardiomyocyte Differentiation Prediction
    
    Features:
    - Multi-layer LSTM with attention
    - Gene expression embedding
    - Positional encoding
    - Multi-head self-attention
    - Residual connections
    - Advanced regularization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_attention_heads: int = 16,
        embedding_dim: int = 512,
        output_dim: int = 1,
        dropout: float = 0.15,
        bidirectional: bool = True,
        use_attention: bool = True,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Calculate actual hidden dimension for bidirectional
        self.actual_hidden_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Gene expression embedding
        self.gene_embedding = GeneExpressionEmbedding(
            input_dim, embedding_dim, num_attention_heads // 2
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_length=10)
        
        # Multi-layer LSTM
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                embedding_dim if i == 0 else self.actual_hidden_dim,
                hidden_dim,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if i < num_layers - 1 else 0
            )
            for i in range(num_layers)
        ])
        
        # Layer normalization for each LSTM layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.actual_hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Multi-head attention mechanism
        if use_attention:
            self.temporal_attention = MultiHeadAttention(
                self.actual_hidden_dim, num_attention_heads, dropout
            )
            
        # Residual connections
        if use_residual and embedding_dim == self.actual_hidden_dim:
            self.residual_projection = None
        elif use_residual:
            self.residual_projection = nn.Linear(embedding_dim, self.actual_hidden_dim)
        else:
            self.residual_projection = None
            
        # Final prediction layers
        self.prediction_head = nn.Sequential(
            nn.Linear(self.actual_hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 4, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            
            nn.Linear(64, output_dim)
        )
        
        # Cell type classification head (auxiliary task)
        self.cell_type_head = nn.Sequential(
            nn.Linear(self.actual_hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 7)  # 7 cell types
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            lengths: Sequence lengths for packed sequences
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing predictions and optional attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Gene expression embedding
        embedded = self.gene_embedding(x)  # (batch, seq, embedding_dim)
        
        # Add positional encoding
        embedded = self.pos_encoding(embedded)
        
        # Store original embedding for residual connection
        residual_input = embedded
        
        # Multi-layer LSTM processing
        lstm_output = embedded
        hidden_states = []
        
        for i, (lstm_layer, layer_norm) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            # LSTM forward pass
            if lengths is not None:
                # Ensure lengths is the correct format for pack_padded_sequence
                lengths_cpu = lengths.cpu()
                if lengths_cpu.dim() == 0:
                    lengths_cpu = lengths_cpu.unsqueeze(0)
                lengths_cpu = lengths_cpu.long()  # Ensure int64 type
                
                # Pack sequences for variable length
                packed_input = nn.utils.rnn.pack_padded_sequence(
                    lstm_output, lengths_cpu, batch_first=True, enforce_sorted=False
                )
                packed_output, _ = lstm_layer(packed_input)
                lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_output, batch_first=True
                )
            else:
                lstm_output, _ = lstm_layer(lstm_output)
            
            # Layer normalization
            lstm_output = layer_norm(lstm_output)
            
            # Store hidden states
            hidden_states.append(lstm_output)
        
        # Apply attention mechanism
        attention_weights = None
        if self.use_attention:
            attended_output = self.temporal_attention(lstm_output, lstm_output, lstm_output)
            if return_attention:
                # For attention visualization (simplified)
                attention_weights = torch.ones(batch_size, seq_len, seq_len)
            lstm_output = attended_output
        
        # Residual connection
        if self.use_residual and self.residual_projection is not None:
            residual_input = self.residual_projection(residual_input)
        
        if self.use_residual and residual_input.shape == lstm_output.shape:
            lstm_output = lstm_output + residual_input
        
        # Global average pooling across time dimension
        if lengths is not None:
            # Ensure lengths is properly formatted
            if lengths.dim() == 0:
                lengths = lengths.unsqueeze(0)
            lengths = lengths.to(x.device)
            
            # Masked average pooling
            mask = torch.arange(seq_len, device=x.device).expand(
                batch_size, seq_len
            ) < lengths.unsqueeze(1)
            masked_output = lstm_output * mask.unsqueeze(-1).float()
            pooled_output = masked_output.sum(dim=1) / lengths.unsqueeze(-1).float()
        else:
            pooled_output = lstm_output.mean(dim=1)
        
        # Final predictions
        differentiation_pred = self.prediction_head(pooled_output)
        cell_type_pred = self.cell_type_head(pooled_output)
        
        output = {
            'differentiation_efficiency': differentiation_pred,
            'cell_type_prediction': cell_type_pred,
            'hidden_states': hidden_states,
            'final_representation': pooled_output
        }
        
        if return_attention and attention_weights is not None:
            output['attention_weights'] = attention_weights
            
        return output
    
    def get_temporal_representations(self, x: torch.Tensor) -> torch.Tensor:
        """Extract temporal representations for downstream analysis."""
        with torch.no_grad():
            output = self.forward(x)
            return output['final_representation']


class TemporalRNNLoss(nn.Module):
    """
    Combined loss function for temporal RNN training.
    Includes differentiation efficiency loss and auxiliary cell type classification loss.
    """
    
    def __init__(
        self, 
        differentiation_weight: float = 1.0,
        cell_type_weight: float = 0.3,
        smoothness_weight: float = 0.1
    ):
        super().__init__()
        self.differentiation_weight = differentiation_weight
        self.cell_type_weight = cell_type_weight
        self.smoothness_weight = smoothness_weight
        
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        
        # Differentiation efficiency loss (primary task)
        if 'differentiation_efficiency' in targets:
            diff_loss = self.mse_loss(
                predictions['differentiation_efficiency'],
                targets['differentiation_efficiency']
            )
            losses['differentiation_loss'] = diff_loss
        else:
            losses['differentiation_loss'] = torch.tensor(0.0, device=predictions['differentiation_efficiency'].device)
        
        # Cell type classification loss (auxiliary task)
        if 'cell_type' in targets:
            cell_type_loss = self.ce_loss(
                predictions['cell_type_prediction'],
                targets['cell_type']
            )
            losses['cell_type_loss'] = cell_type_loss
        else:
            losses['cell_type_loss'] = torch.tensor(0.0, device=predictions['cell_type_prediction'].device)
        
        # Temporal smoothness regularization
        if len(predictions['hidden_states']) > 1:
            smoothness_loss = 0
            for i in range(1, len(predictions['hidden_states'])):
                smoothness_loss += self.l1_loss(
                    predictions['hidden_states'][i],
                    predictions['hidden_states'][i-1]
                )
            smoothness_loss /= (len(predictions['hidden_states']) - 1)
            losses['smoothness_loss'] = smoothness_loss
        else:
            losses['smoothness_loss'] = torch.tensor(0.0, device=predictions['differentiation_efficiency'].device)
        
        # Total loss
        total_loss = (
            self.differentiation_weight * losses['differentiation_loss'] +
            self.cell_type_weight * losses['cell_type_loss'] +
            self.smoothness_weight * losses['smoothness_loss']
        )
        
        losses['total_loss'] = total_loss
        
        return losses


def create_advanced_rnn_model(
    input_dim: int,
    config: Optional[Dict] = None
) -> AdvancedTemporalRNN:
    """
    Factory function to create an advanced RNN model with optimal configuration.
    
    Args:
        input_dim: Input feature dimension
        config: Optional configuration dictionary
        
    Returns:
        Configured AdvancedTemporalRNN model
    """
    default_config = {
        'hidden_dim': 1024,
        'num_layers': 4,
        'num_attention_heads': 16,
        'embedding_dim': 512,
        'output_dim': 1,
        'dropout': 0.15,
        'bidirectional': True,
        'use_attention': True,
        'use_residual': True
    }
    
    if config:
        default_config.update(config)
    
    return AdvancedTemporalRNN(input_dim=input_dim, **default_config)


if __name__ == "__main__":
    # Test model creation
    model = create_advanced_rnn_model(input_dim=2000)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size, seq_len, input_dim = 32, 7, 2000
    x = torch.randn(batch_size, seq_len, input_dim)
    
    with torch.no_grad():
        output = model(x)
        print(f"Differentiation prediction shape: {output['differentiation_efficiency'].shape}")
        print(f"Cell type prediction shape: {output['cell_type_prediction'].shape}")

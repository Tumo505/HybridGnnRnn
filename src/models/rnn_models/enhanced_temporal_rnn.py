#!/usr/bin/env python3
"""
Enhanced Temporal RNN for Cardiac Data Analysis
Handles sequential/temporal patterns in cardiac gene expression data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
import logging

logger = logging.getLogger(__name__)

class AttentionLayer(nn.Module):
    """Self-attention mechanism for RNN outputs"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            mask: [batch_size, seq_len] - 1 for valid positions, 0 for padding
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            # Expand mask for multi-head attention
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection
        output = self.output_proj(context)
        
        return output

class PositionalEncoding(nn.Module):
    """Add positional encoding for temporal sequences"""
    
    def __init__(self, hidden_dim: int, max_seq_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_seq_len, hidden_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                           -(np.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class EnhancedTemporalRNN(nn.Module):
    """
    Advanced RNN for temporal cardiac data analysis
    
    Features:
    - Bidirectional LSTM/GRU with attention
    - Positional encoding for temporal patterns
    - Multi-scale temporal convolutions
    - Residual connections
    - Cardiac-specific temporal patterns
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_classes: int = 5,
        rnn_type: str = 'LSTM',
        bidirectional: bool = True,
        use_attention: bool = True,
        use_positional_encoding: bool = True,
        dropout: float = 0.3,
        temporal_conv_filters: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [3, 5, 7]
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_positional_encoding = use_positional_encoding
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Multi-scale temporal convolutions
        self.temporal_convs = nn.ModuleList()
        for i, (filters, kernel_size) in enumerate(zip(temporal_conv_filters, kernel_sizes)):
            conv_layer = nn.Sequential(
                nn.Conv1d(hidden_dim if i == 0 else temporal_conv_filters[i-1], 
                         filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            )
            self.temporal_convs.append(conv_layer)
        
        # RNN layers
        rnn_input_dim = hidden_dim + sum(temporal_conv_filters)
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                rnn_input_dim, hidden_dim, num_layers,
                batch_first=True, bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                rnn_input_dim, hidden_dim, num_layers,
                batch_first=True, bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        rnn_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Attention mechanism
        if use_attention:
            self.attention = AttentionLayer(rnn_output_dim)
            
        # Residual connection
        self.residual_proj = nn.Linear(rnn_input_dim, rnn_output_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # Cardiac-specific pattern detection
        self.cardiac_pattern_detector = nn.Sequential(
            nn.Conv1d(rnn_output_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 16)
        )
        
        self.pattern_fusion = nn.Linear(rnn_output_dim + 16, rnn_output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'rnn' in name:
                    nn.init.orthogonal_(param)
                elif param.dim() >= 2:  # Only apply xavier to 2D+ tensors
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.normal_(param, 0, 0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            lengths: Actual sequence lengths [batch_size]
            return_attention: Whether to return attention weights
            
        Returns:
            logits: [batch_size, num_classes]
            attention_weights (optional): [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]
        
        # Positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoding(x)
        
        # Multi-scale temporal convolutions
        conv_features = []
        conv_input = x.transpose(1, 2)  # [batch_size, hidden_dim, seq_len]
        
        for conv_layer in self.temporal_convs:
            conv_out = conv_layer(conv_input)
            conv_features.append(conv_out.transpose(1, 2))  # Back to [batch_size, seq_len, filters]
            conv_input = conv_out
        
        # Concatenate original features with conv features
        x_with_conv = torch.cat([x] + conv_features, dim=-1)
        
        # Store for residual connection
        residual = self.residual_proj(x_with_conv)
        
        # RNN processing
        if lengths is not None:
            # Pack sequences for variable lengths
            packed_input = pack_padded_sequence(
                x_with_conv, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, hidden = self.rnn(packed_input)
            rnn_output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
            
            # Adjust residual to match RNN output dimensions
            batch_size, max_seq_len = rnn_output.shape[:2]
            if residual.shape[1] != max_seq_len:
                # Truncate residual to match RNN output length
                residual = residual[:, :max_seq_len, :]
        else:
            rnn_output, hidden = self.rnn(x_with_conv)
        
        # Add residual connection (dimensions should now match)
        rnn_output = rnn_output + residual
        
        # Attention mechanism
        attention_weights = None
        if self.use_attention:
            # Create mask for padding if lengths provided
            mask = None
            if lengths is not None:
                mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            
            attended_output = self.attention(rnn_output, mask)
            if return_attention:
                # For simplicity, return a dummy attention weight
                attention_weights = torch.ones(batch_size, 8, seq_len, seq_len, device=x.device)
            
            rnn_output = attended_output
        
        # Cardiac pattern detection
        pattern_features = self.cardiac_pattern_detector(rnn_output.transpose(1, 2))
        pattern_features_expanded = pattern_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Fuse RNN output with pattern features
        fused_features = torch.cat([rnn_output, pattern_features_expanded], dim=-1)
        fused_output = self.pattern_fusion(fused_features)
        
        # Global pooling for sequence-level prediction
        if lengths is not None:
            # Masked average pooling
            mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()
            pooled_output = (fused_output * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            pooled_output = fused_output.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        if return_attention and self.use_attention:
            return logits, attention_weights
        return logits
    
    def get_temporal_patterns(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Extract temporal patterns for analysis"""
        self.eval()
        with torch.no_grad():
            batch_size, seq_len, _ = x.shape
            
            # Forward pass through RNN
            x_proj = self.input_projection(x)
            if self.use_positional_encoding:
                x_proj = self.pos_encoding(x_proj)
            
            # Get RNN features
            rnn_output, hidden = self.rnn(x_proj)
            
            # Get cardiac patterns
            pattern_features = self.cardiac_pattern_detector(rnn_output.transpose(1, 2))
            
            patterns = {
                'rnn_hidden_states': rnn_output,
                'final_hidden': hidden[0] if self.rnn_type == 'LSTM' else hidden,
                'cardiac_patterns': pattern_features,
                'temporal_summary': rnn_output.mean(dim=1)
            }
            
            return patterns

class CardiacTemporalDataset(torch.utils.data.Dataset):
    """Dataset for cardiac temporal sequences"""
    
    def __init__(
        self,
        sequences: List[np.ndarray],
        labels: List[int],
        sequence_names: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        normalize: bool = True
    ):
        self.sequences = sequences
        self.labels = labels
        self.sequence_names = sequence_names or [f"seq_{i}" for i in range(len(sequences))]
        self.max_length = max_length
        self.normalize = normalize
        
        if normalize:
            self._normalize_sequences()
    
    def _normalize_sequences(self):
        """Normalize sequences"""
        all_values = np.concatenate([seq.flatten() for seq in self.sequences])
        self.mean = np.mean(all_values)
        self.std = np.std(all_values)
        
        self.sequences = [(seq - self.mean) / (self.std + 1e-8) for seq in self.sequences]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.LongTensor([self.labels[idx]])
        length = torch.LongTensor([len(self.sequences[idx])])
        
        # Pad or truncate if max_length specified
        if self.max_length is not None:
            if len(sequence) > self.max_length:
                sequence = sequence[:self.max_length]
                length = torch.LongTensor([self.max_length])
            elif len(sequence) < self.max_length:
                padding = torch.zeros(self.max_length - len(sequence), sequence.shape[1])
                sequence = torch.cat([sequence, padding], dim=0)
        
        return {
            'sequence': sequence,
            'label': label.squeeze(),
            'length': length.squeeze(),
            'name': self.sequence_names[idx]
        }

def create_synthetic_temporal_data(
    num_sequences: int = 100,
    min_length: int = 10,
    max_length: int = 50,
    num_features: int = 100,
    num_classes: int = 5,
    cardiac_conditions: Optional[List[str]] = None
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Create synthetic temporal cardiac data with realistic patterns
    """
    if cardiac_conditions is None:
        cardiac_conditions = ['Healthy', 'Mild_Inflammation', 'Fibrotic', 'Ischemic', 'Severe_Disease']
    
    sequences = []
    labels = []
    names = []
    
    logger.info(f"Creating {num_sequences} synthetic temporal sequences...")
    
    for i in range(num_sequences):
        # Random sequence length
        seq_length = np.random.randint(min_length, max_length + 1)
        
        # Random condition
        condition_idx = np.random.randint(num_classes)
        condition = cardiac_conditions[condition_idx]
        
        # Generate condition-specific temporal patterns
        sequence = generate_cardiac_temporal_pattern(
            seq_length, num_features, condition_idx, condition
        )
        
        sequences.append(sequence)
        labels.append(condition_idx)
        names.append(f"temporal_seq_{i}_{condition}")
    
    logger.info(f"Generated {len(sequences)} temporal sequences")
    return sequences, labels, names

def generate_cardiac_temporal_pattern(
    seq_length: int,
    num_features: int,
    condition_idx: int,
    condition: str
) -> np.ndarray:
    """Generate realistic cardiac temporal patterns"""
    
    # Base temporal pattern
    time_points = np.linspace(0, 2 * np.pi, seq_length)
    
    # Condition-specific parameters
    if condition_idx == 0:  # Healthy
        amplitude = 1.0
        frequency = 1.0
        noise_level = 0.1
        trend = 0.0
    elif condition_idx == 1:  # Mild Inflammation
        amplitude = 1.2
        frequency = 1.1
        noise_level = 0.2
        trend = 0.05
    elif condition_idx == 2:  # Fibrotic
        amplitude = 0.8
        frequency = 0.9
        noise_level = 0.15
        trend = -0.1
    elif condition_idx == 3:  # Ischemic
        amplitude = 1.5
        frequency = 1.3
        noise_level = 0.3
        trend = 0.1
    else:  # Severe Disease
        amplitude = 2.0
        frequency = 1.5
        noise_level = 0.4
        trend = 0.2
    
    # Generate sequence
    sequence = np.zeros((seq_length, num_features))
    
    for t in range(seq_length):
        # Cardiac cycle pattern
        cardiac_cycle = amplitude * np.sin(frequency * time_points[t])
        
        # Add trend
        trend_component = trend * (t / seq_length)
        
        # Feature-specific patterns
        for f in range(num_features):
            # Gene-specific modulation
            gene_phase = np.random.uniform(0, 2 * np.pi)
            gene_amplitude = np.random.uniform(0.5, 1.5)
            
            # Combine patterns
            value = (
                cardiac_cycle * gene_amplitude * np.cos(gene_phase) +
                trend_component +
                np.random.normal(0, noise_level)
            )
            
            sequence[t, f] = value
    
    return sequence

"""
Recurrent Neural Network Encoder for Temporal Data

This module implements the RNN component of the hybrid framework for modeling
temporal gene expression dynamics during cardiomyocyte differentiation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any


class TemporalRNNEncoder(nn.Module):
    """
    RNN encoder for temporal gene expression data.
    
    Processes time-series gene expression data to learn temporal dynamics
    and predict cardiomyocyte differentiation efficiency and maturation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        rnn_type: str = "LSTM",
        dropout: float = 0.1,
        bidirectional: bool = True,
        use_attention: bool = True,
        use_cell_type_embedding: bool = True,
        num_cell_types: int = 6
    ):
        super(TemporalRNNEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_cell_type_embedding = use_cell_type_embedding
        
        # Cell type embedding
        if self.use_cell_type_embedding:
            self.cell_type_embedding = nn.Embedding(num_cell_types, 32)
            rnn_input_dim = input_dim + 32
        else:
            rnn_input_dim = input_dim
        
        # Input projection
        self.input_projection = nn.Linear(rnn_input_dim, hidden_dim)
        
        # RNN layers
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Calculate RNN output dimension
        rnn_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Temporal attention mechanism
        if self.use_attention:
            self.attention = TemporalAttention(rnn_output_dim)
            attention_output_dim = rnn_output_dim
        else:
            attention_output_dim = rnn_output_dim
        
        # Output layers
        self.dropout_layer = nn.Dropout(dropout)
        self.output_projection = nn.Sequential(
            nn.Linear(attention_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self, 
        x: torch.Tensor, 
        cell_types: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the RNN encoder.
        
        Args:
            x (torch.Tensor): Input sequences [batch_size, seq_len, input_dim]
            cell_types (torch.Tensor, optional): Cell type indices [batch_size, seq_len]
            lengths (torch.Tensor, optional): Sequence lengths [batch_size]
            
        Returns:
            torch.Tensor: Encoded temporal representations [batch_size, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Add cell type embeddings if enabled
        if self.use_cell_type_embedding and cell_types is not None:
            cell_type_embeds = self.cell_type_embedding(cell_types)
            x = torch.cat([x, cell_type_embeds], dim=-1)
        
        # Input projection
        x = self.input_projection(x)
        x = F.relu(x)
        
        # Pack sequences if lengths are provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # RNN forward pass
        if self.rnn_type == "LSTM":
            output, (hidden, cell) = self.rnn(x)
        else:  # GRU
            output, hidden = self.rnn(x)
        
        # Unpack sequences if they were packed
        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        # Apply attention or use final hidden state
        if self.use_attention:
            # Apply temporal attention
            attended_output = self.attention(output, lengths)
        else:
            # Use final hidden state
            if self.bidirectional:
                # Concatenate final forward and backward hidden states
                attended_output = torch.cat([hidden[-2], hidden[-1]], dim=-1)
            else:
                attended_output = hidden[-1]
        
        # Output projection
        attended_output = self.dropout_layer(attended_output)
        output = self.output_projection(attended_output)
        
        return output


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for RNN outputs.
    """
    
    def __init__(self, hidden_dim: int):
        super(TemporalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.attention_layer = nn.Linear(hidden_dim, 1)
        
    def forward(
        self, 
        rnn_outputs: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply temporal attention to RNN outputs.
        
        Args:
            rnn_outputs (torch.Tensor): RNN outputs [batch_size, seq_len, hidden_dim]
            lengths (torch.Tensor, optional): Sequence lengths [batch_size]
            
        Returns:
            torch.Tensor: Attended output [batch_size, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = rnn_outputs.shape
        
        # Calculate attention scores
        attention_scores = self.attention_layer(rnn_outputs).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply length mask if lengths are provided
        if lengths is not None:
            mask = self._create_length_mask(lengths, seq_len, rnn_outputs.device)
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        
        # Softmax attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, seq_len]
        
        # Apply attention to get weighted output
        attended_output = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch_size, 1, seq_len]
            rnn_outputs  # [batch_size, seq_len, hidden_dim]
        ).squeeze(1)  # [batch_size, hidden_dim]
        
        return attended_output
    
    def _create_length_mask(self, lengths: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
        """Create mask for variable-length sequences."""
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, max_len)
        mask = mask < lengths.unsqueeze(1)
        return mask


class TemporalFeatureExtractor:
    """
    Utility class for extracting temporal features from time-series data.
    """
    
    @staticmethod
    def create_time_features(timepoints: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Create temporal features from timepoints.
        
        Args:
            timepoints (np.ndarray): Array of timepoints
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of temporal features
        """
        features = {}
        
        # Normalize timepoints to [0, 1]
        features['time_normalized'] = (timepoints - timepoints.min()) / (timepoints.max() - timepoints.min())
        
        # Sinusoidal encoding
        features['time_sin'] = np.sin(2 * np.pi * features['time_normalized'])
        features['time_cos'] = np.cos(2 * np.pi * features['time_normalized'])
        
        # Polynomial features
        features['time_squared'] = features['time_normalized'] ** 2
        features['time_cubed'] = features['time_normalized'] ** 3
        
        # Days since start
        features['days_since_start'] = timepoints - timepoints.min()
        
        return features
    
    @staticmethod
    def create_differentiation_trajectory(
        gene_expression: np.ndarray,
        timepoints: np.ndarray,
        cell_types: np.ndarray
    ) -> Dict[str, Any]:
        """
        Create differentiation trajectory features.
        
        Args:
            gene_expression (np.ndarray): Gene expression matrix [cells, genes]
            timepoints (np.ndarray): Timepoints for each cell
            cell_types (np.ndarray): Cell type labels
            
        Returns:
            Dict[str, Any]: Trajectory features and metadata
        """
        trajectory_data = {
            'expression_by_time': {},
            'cell_type_progression': {},
            'differentiation_markers': {}
        }
        
        unique_timepoints = np.unique(timepoints)
        unique_cell_types = np.unique(cell_types)
        
        # Expression by timepoint
        for tp in unique_timepoints:
            tp_mask = timepoints == tp
            trajectory_data['expression_by_time'][str(tp)] = {
                'mean_expression': gene_expression[tp_mask].mean(axis=0),
                'std_expression': gene_expression[tp_mask].std(axis=0),
                'n_cells': tp_mask.sum()
            }
        
        # Cell type progression
        for tp in unique_timepoints:
            tp_mask = timepoints == tp
            cell_type_dist = np.bincount(cell_types[tp_mask]) / tp_mask.sum()
            trajectory_data['cell_type_progression'][str(tp)] = cell_type_dist
        
        return trajectory_data


class TemporalDataLoader:
    """
    Data loader for temporal sequences with variable lengths.
    """
    
    def __init__(
        self,
        temporal_data: Dict[str, Any],
        batch_size: int = 32,
        max_seq_length: int = 50,
        shuffle: bool = True
    ):
        self.temporal_data = temporal_data
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.shuffle = shuffle
        
    def create_sequences(self, donor_data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create sequences from donor data.
        
        Args:
            donor_data (Dict[str, Any]): Data for a single donor
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                (sequences, cell_types, lengths)
        """
        timepoints = donor_data['timepoints']
        expressions = donor_data['expressions']
        cell_types = donor_data['cell_types']
        
        # Sort by timepoint
        sort_indices = np.argsort(timepoints)
        timepoints = timepoints[sort_indices]
        expressions = expressions[sort_indices]
        cell_types = cell_types[sort_indices]
        
        # Create sequence tensors
        seq_length = min(len(timepoints), self.max_seq_length)
        
        sequence = torch.tensor(expressions[:seq_length], dtype=torch.float)
        cell_type_seq = torch.tensor(cell_types[:seq_length], dtype=torch.long)
        length = torch.tensor(seq_length, dtype=torch.long)
        
        return sequence, cell_type_seq, length


if __name__ == "__main__":
    # Example usage and testing
    torch.manual_seed(42)
    
    # Create sample temporal data
    batch_size = 16
    seq_len = 10
    input_dim = 500  # Number of genes
    num_cell_types = 6
    
    # Sample sequences with variable lengths
    sequences = torch.randn(batch_size, seq_len, input_dim)
    cell_types = torch.randint(0, num_cell_types, (batch_size, seq_len))
    lengths = torch.randint(5, seq_len + 1, (batch_size,))
    
    print(f"Created temporal data:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Input dimension: {input_dim}")
    print(f"  Cell types: {num_cell_types}")
    
    # Create RNN encoder
    rnn_encoder = TemporalRNNEncoder(
        input_dim=input_dim,
        hidden_dim=256,
        output_dim=128,
        num_layers=2,
        rnn_type="LSTM",
        bidirectional=True,
        use_attention=True,
        use_cell_type_embedding=True,
        num_cell_types=num_cell_types
    )
    
    print(f"\nRNN Encoder:")
    print(f"  Parameters: {sum(p.numel() for p in rnn_encoder.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        output = rnn_encoder(sequences, cell_types, lengths)
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

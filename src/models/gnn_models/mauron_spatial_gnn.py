"""
GNN model for Mauron Visium section graphs.

The model exposes an encoder for self-supervised pretraining, node-level
classification for deconvolution labels, graph-level classification for section
metadata labels, and section/case embedding export.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool


class MauronSpatialGNN(nn.Module):
    """Spatial GraphSAGE encoder with node, graph, and reconstruction heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        num_classes: int = 33,
        num_layers: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim,
                    heads=4,
                    concat=False,
                    edge_dim=4,
                    dropout=dropout,
                )
            )
            self.norms.append(GraphNorm(hidden_dim))

        self.embedding_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.node_classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )
        self.graph_classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )
        self.feature_decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.position_decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def encode(self, data: Data, x_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return node embeddings for a section graph or PyG batch."""

        x = data.x if x_override is None else x_override
        edge_index = data.edge_index
        edge_attr = self._spatial_edge_attr(data)
        batch = getattr(data, "batch", None)

        x = self.input_projection(x)
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x, batch=batch)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual
        return self.embedding_projection(x)

    @staticmethod
    def _spatial_edge_attr(data: Data) -> torch.Tensor:
        """Use real relative spatial coordinates as edge features."""

        row, col = data.edge_index
        delta = data.pos[col] - data.pos[row]
        distance = torch.linalg.norm(delta, dim=1, keepdim=True).clamp_min(1e-6)
        direction = delta / distance
        mean_distance = distance.mean().clamp_min(1e-6)
        scaled_distance = distance / mean_distance
        inverse_distance = mean_distance / distance
        return torch.cat([direction, scaled_distance, inverse_distance], dim=1).float()

    def node_logits(self, data: Data) -> torch.Tensor:
        embeddings = self.encode(data)
        return self.node_classifier(embeddings)

    def graph_logits(self, data: Data) -> torch.Tensor:
        embeddings = self.encode(data)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(embeddings.shape[0], dtype=torch.long, device=embeddings.device)
        graph_embeddings = global_mean_pool(embeddings, batch)
        return self.graph_classifier(graph_embeddings)

    def reconstruct(self, data: Data, x_masked: torch.Tensor) -> torch.Tensor:
        embeddings = self.encode(data, x_override=x_masked)
        return self.feature_decoder(embeddings)

    def reconstruct_position(self, data: Data, x_masked: Optional[torch.Tensor] = None) -> torch.Tensor:
        embeddings = self.encode(data, x_override=x_masked)
        return self.position_decoder(embeddings)

    def section_embedding(self, data: Data) -> torch.Tensor:
        """Return one mean-pooled embedding per graph."""

        embeddings = self.encode(data)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(embeddings.shape[0], dtype=torch.long, device=embeddings.device)
        return global_mean_pool(embeddings, batch)

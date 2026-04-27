#!/usr/bin/env python3
"""
Train an end-to-end Mauron graph-recurrent hybrid.

The hybrid uses the fixed branches:
- spatial branch: edge-aware GATv2 over real Visium coordinates;
- temporal branch: GRU over train-only fetal-age expression prototypes;
- fusion head: predicts spot-level deconvolution labels.

The split is case-grouped and age-bin-stratified, matching the fixed RNN/GNN
baselines. Mauron is cross-sectional, so the recurrent branch models a
developmental age axis, not true same-cell longitudinal tracking.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.mauron_visium_processor import MauronBuildConfig, MauronVisiumGraphDataset  # noqa: E402
from src.models.gnn_models.mauron_spatial_gnn import MauronSpatialGNN  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("mauron_spatial_hybrid")

AGE_BIN_ORDER = ["early_w6_w7_5", "mid_w8_w9", "late_w10_w12"]
RELATED_GROUPS = {
    "cardiomyocyte": ["CM", "aCM", "vCM", "Mat_", "MetAct", "Immat"],
    "fibroblast_epicardial": ["FB", "EPDC"],
    "endothelial": ["EC", "Endoc", "LEC"],
    "smooth_muscle_mural": ["SMC", "MC", "PC", "Peric"],
    "neural": ["NB-N", "SCP", "GC"],
    "immune": ["LyC", "MyC"],
    "excluded_or_other": ["HL_excl", "TMSB10high"],
}


def biological_group(label: str) -> str:
    for group, tokens in RELATED_GROUPS.items():
        if any(token in label for token in tokens):
            return group
    return "other"


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 0.5):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt).pow(self.gamma) * ce).mean()


class MauronGraphRecurrentHybrid(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_bio_groups: int,
        num_age_bins: int,
        chamber_dim: int,
        gnn_hidden_dim: int = 128,
        gnn_embedding_dim: int = 64,
        gnn_layers: int = 3,
        rnn_projection_dim: int = 96,
        rnn_hidden_dim: int = 96,
        dropout: float = 0.25,
        class_gate_weight: float = 0.0,
    ):
        super().__init__()
        self.class_gate_weight = class_gate_weight
        self.gnn = MauronSpatialGNN(
            input_dim=input_dim,
            hidden_dim=gnn_hidden_dim,
            embedding_dim=gnn_embedding_dim,
            num_classes=num_classes,
            num_layers=gnn_layers,
            dropout=dropout,
        )
        self.temporal_project = nn.Sequential(
            nn.Linear(input_dim * 4, rnn_projection_dim),
            nn.LayerNorm(rnn_projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_gru = nn.GRU(
            input_size=rnn_projection_dim,
            hidden_size=rnn_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        temporal_dim = rnn_hidden_dim * 2
        self.temporal_head = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim, num_classes),
        )
        self.age_head = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim, num_age_bins),
        )
        self.chamber_head = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim, chamber_dim),
        )

        fusion_width = 128
        self.graph_fusion_projection = nn.Sequential(
            nn.Linear(gnn_embedding_dim, fusion_width),
            nn.LayerNorm(fusion_width),
            nn.GELU(),
        )
        self.temporal_fusion_projection = nn.Sequential(
            nn.Linear(temporal_dim, fusion_width),
            nn.LayerNorm(fusion_width),
            nn.GELU(),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear((fusion_width * 2) + 4, fusion_width),
            nn.LayerNorm(fusion_width),
            nn.GELU(),
            nn.Linear(fusion_width, fusion_width),
            nn.Sigmoid(),
        )
        fusion_dim = fusion_width + (num_classes * 2) + 4
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )
        self.class_logit_gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_width, num_classes),
            nn.Sigmoid(),
        )
        self.bio_group_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_width, num_bio_groups),
        )

    def temporal_encode(self, sequences: torch.Tensor) -> torch.Tensor:
        z = self.temporal_project(sequences)
        _, hidden = self.temporal_gru(z)
        return torch.cat([hidden[-2], hidden[-1]], dim=1)

    @staticmethod
    def developmental_sequences(x: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        repeated_x = x.unsqueeze(1).expand(-1, prototypes.shape[0], -1)
        repeated_prototypes = prototypes.unsqueeze(0).expand(x.shape[0], -1, -1)
        signed_delta = repeated_x - repeated_prototypes
        abs_delta = signed_delta.abs()
        return torch.cat([repeated_x, repeated_prototypes, signed_delta, abs_delta], dim=2)

    @staticmethod
    def confidence_features(graph_logits: torch.Tensor, temporal_logits: torch.Tensor) -> torch.Tensor:
        num_classes = graph_logits.shape[1]
        log_denom = torch.log(torch.tensor(float(num_classes), device=graph_logits.device)).clamp_min(1e-6)
        graph_prob = F.softmax(graph_logits, dim=1)
        temporal_prob = F.softmax(temporal_logits, dim=1)
        graph_max = graph_prob.max(dim=1, keepdim=True).values
        temporal_max = temporal_prob.max(dim=1, keepdim=True).values
        graph_entropy = -(graph_prob * graph_prob.clamp_min(1e-8).log()).sum(dim=1, keepdim=True) / log_denom
        temporal_entropy = -(temporal_prob * temporal_prob.clamp_min(1e-8).log()).sum(dim=1, keepdim=True) / log_denom
        return torch.cat([graph_max, temporal_max, graph_entropy, temporal_entropy], dim=1)

    def forward(self, graph: Data, prototypes: torch.Tensor) -> Dict[str, torch.Tensor]:
        graph_embeddings = self.gnn.encode(graph)
        graph_logits = self.gnn.node_classifier(graph_embeddings)

        temporal_x = graph.x_temporal if hasattr(graph, "x_temporal") else graph.x
        sequences = self.developmental_sequences(temporal_x, prototypes)
        temporal_embeddings = self.temporal_encode(sequences)
        temporal_logits = self.temporal_head(temporal_embeddings)

        graph_signal = self.graph_fusion_projection(graph_embeddings)
        temporal_signal = self.temporal_fusion_projection(temporal_embeddings)
        confidence = self.confidence_features(graph_logits, temporal_logits)
        gate = self.fusion_gate(torch.cat([graph_signal, temporal_signal, confidence], dim=1))
        gated_signal = gate * graph_signal + (1.0 - gate) * temporal_signal
        fused = torch.cat([gated_signal, graph_logits, temporal_logits, confidence], dim=1)
        fusion_residual = self.fusion_head(fused)
        class_gate = self.class_logit_gate(fused)
        class_gated_logits = class_gate * graph_logits + (1.0 - class_gate) * temporal_logits
        fusion_logits = fusion_residual + (self.class_gate_weight * class_gated_logits)
        return {
            "fusion": fusion_logits,
            "fusion_residual": fusion_residual,
            "graph": graph_logits,
            "temporal": temporal_logits,
            "bio_group": self.bio_group_head(fused),
            "age": self.age_head(temporal_embeddings),
            "chamber": self.chamber_head(temporal_embeddings),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=("data/New/Mauron_spatial_dynamics_part_a/Spatial dynamics of the developing human heart, pa"),
    )
    parser.add_argument("--cache-dir", default="cache/mauron_visium_graphs")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-genes", type=int, default=512)
    parser.add_argument("--split-group", default="case", choices=["case", "code"])
    parser.add_argument("--gnn-hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-embedding-dim", type=int, default=64)
    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument("--rnn-projection-dim", type=int, default=96)
    parser.add_argument("--rnn-hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--class-weight-beta", type=float, default=0.99)
    parser.add_argument("--max-class-weight", type=float, default=3.0)
    parser.add_argument("--focal-gamma", type=float, default=0.5)
    parser.add_argument("--graph-loss-weight", type=float, default=0.25)
    parser.add_argument("--temporal-loss-weight", type=float, default=0.25)
    parser.add_argument("--age-loss-weight", type=float, default=0.05)
    parser.add_argument("--chamber-loss-weight", type=float, default=0.05)
    parser.add_argument("--bio-group-loss-weight", type=float, default=0.15)
    parser.add_argument("--hierarchical-group-logit-weight", type=float, default=0.25)
    parser.add_argument("--groupwise-loss-weight", type=float, default=0.15)
    parser.add_argument("--class-gate-weight", type=float, default=0.0)
    parser.add_argument("--soft-loss-weight", type=float, default=0.0)
    parser.add_argument("--rare-replay-weight", type=float, default=0.0)
    parser.add_argument("--rare-support-threshold", type=int, default=120)
    parser.add_argument("--gnn-checkpoint", default="mauron_spatial_gnn_fresh_case_split/best_mauron_spatial_gnn.pt")
    parser.add_argument("--rnn-checkpoint", default="mauron_developmental_rnn_fresh_case_split/best_mauron_developmental_rnn.pt")
    parser.add_argument("--fusion-warmup-epochs", type=int, default=6)
    parser.add_argument("--freeze-gnn-epochs", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_graphs(graphs: Sequence[Data], split_group: str, seed: int) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    group_to_indices: Dict[str, List[int]] = {}
    group_to_bins: Dict[str, set] = {}
    for idx, graph in enumerate(graphs):
        group = str(getattr(graph, split_group))
        group_to_indices.setdefault(group, []).append(idx)
        group_to_bins.setdefault(group, set()).add(graph.age_bin)

    bin_to_groups: Dict[str, List[str]] = {label: [] for label in AGE_BIN_ORDER}
    for group, bins in group_to_bins.items():
        if len(bins) != 1:
            raise ValueError(f"Group {group!r} spans multiple age bins: {sorted(bins)}")
        bin_to_groups.setdefault(next(iter(bins)), []).append(group)

    split_groups = {"train": [], "val": [], "test": []}
    for age_bin in AGE_BIN_ORDER:
        groups = sorted(bin_to_groups.get(age_bin, []))
        if len(groups) < 3:
            raise ValueError(f"Need at least 3 {split_group} groups for age bin {age_bin!r}.")
        shuffled = np.asarray(groups, dtype=object)
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * 0.15)))
        test_count = max(1, int(round(len(shuffled) * 0.15)))
        if val_count + test_count >= len(shuffled):
            val_count = 1
            test_count = 1
        split_groups["val"].extend(shuffled[:val_count].tolist())
        split_groups["test"].extend(shuffled[val_count : val_count + test_count].tolist())
        split_groups["train"].extend(shuffled[val_count + test_count :].tolist())

    return {
        split_name: sorted(idx for group in groups for idx in group_to_indices[group])
        for split_name, groups in split_groups.items()
    }


def split_summary(graphs: Sequence[Data], splits: Dict[str, List[int]], split_group: str) -> Dict:
    summary = {}
    for name, indices in splits.items():
        selected = [graphs[idx] for idx in indices]
        summary[name] = {
            "num_sections": len(indices),
            "sections": [int(graph.section_id.item()) for graph in selected],
            "codes": [graph.code for graph in selected],
            "cases": sorted({graph.case for graph in selected}),
            "groups": sorted({str(getattr(graph, split_group)) for graph in selected}),
            "age_bins": sorted({graph.age_bin for graph in selected}),
            "ages": sorted({graph.age for graph in selected}),
            "num_spots": int(sum(graph.num_nodes for graph in selected)),
        }
    return summary


def add_temporal_standardized_features(graphs: Sequence[Data], train_indices: Sequence[int]) -> None:
    train_x = torch.cat([graphs[idx].x for idx in train_indices], dim=0)
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-5)
    for graph in graphs:
        graph.x_temporal = ((graph.x - mean) / std).float()


def build_age_prototypes(graphs: Sequence[Data], train_indices: Sequence[int]) -> Tuple[torch.Tensor, List[float], Dict[str, List[int]]]:
    train_ages = sorted({float(graphs[idx].age_weeks.item()) for idx in train_indices})
    prototypes = []
    members: Dict[str, List[int]] = {}
    for age in train_ages:
        age_graphs = [idx for idx in train_indices if float(graphs[idx].age_weeks.item()) == age]
        members[str(age)] = [int(graphs[idx].section_id.item()) for idx in age_graphs]
        prototypes.append(torch.cat([graphs[idx].x_temporal for idx in age_graphs], dim=0).mean(dim=0))
    return torch.stack(prototypes, dim=0).float(), train_ages, members


def attach_soft_deconvolution_targets(graphs: Sequence[Data], data_root: str) -> int:
    deconv_path = Path(data_root) / "7_Metadata" / "W.2023-09-25133054.715121_hl.tsv"
    if not deconv_path.exists():
        logger.warning("Soft deconvolution table not found: %s", deconv_path)
        return 0
    table = pd.read_csv(deconv_path, sep="\t", index_col=0)
    parsed = table.index.to_series().str.extract(r"^(?P<barcode>.+)_(?P<section>[0-9]+)$")
    lookup = {}
    values = table.to_numpy(dtype=np.float32)
    row_sums = np.maximum(values.sum(axis=1, keepdims=True), 1e-8)
    values = values / row_sums
    for row_id, vector in zip(parsed.itertuples(index=False), values):
        if pd.isna(row_id.section) or pd.isna(row_id.barcode):
            continue
        lookup[(int(row_id.section), str(row_id.barcode))] = vector

    attached = 0
    for graph in graphs:
        section = int(graph.section_id.item())
        vectors = []
        missing = 0
        for barcode in graph.barcodes:
            vector = lookup.get((section, str(barcode)))
            if vector is None:
                missing += 1
                one_hot = np.zeros(len(graph.label_names), dtype=np.float32)
                one_hot[int(graph.y[len(vectors)].item())] = 1.0
                vector = one_hot
            vectors.append(vector)
        if missing:
            logger.warning("Soft labels missing for %d spots in section %s; using hard one-hot fallback.", missing, section)
        graph.y_soft = torch.tensor(np.vstack(vectors), dtype=torch.float32)
        attached += graph.num_nodes - missing
    return attached


def load_gnn_checkpoint(model: MauronGraphRecurrentHybrid, checkpoint_path: str) -> bool:
    path = Path(checkpoint_path)
    if not path.exists():
        logger.warning("GNN checkpoint not found, training graph branch from scratch: %s", path)
        return False
    checkpoint = torch.load(path, map_location="cpu")
    model.gnn.load_state_dict(checkpoint["model_state_dict"], strict=True)
    logger.info("Initialized graph branch from %s", path)
    return True


def load_rnn_checkpoint(model: MauronGraphRecurrentHybrid, checkpoint_path: str) -> bool:
    path = Path(checkpoint_path)
    if not path.exists():
        logger.warning("RNN checkpoint not found, training temporal branch from scratch: %s", path)
        return False
    checkpoint = torch.load(path, map_location="cpu")
    source = checkpoint["model_state_dict"]
    mapped = {}
    key_map = {
        "project.": "temporal_project.",
        "gru.": "temporal_gru.",
        "main_classifier.": "temporal_head.",
        "age_classifier.": "age_head.",
        "chamber_classifier.": "chamber_head.",
    }
    for key, value in source.items():
        for prefix, target_prefix in key_map.items():
            if key.startswith(prefix):
                mapped[target_prefix + key[len(prefix) :]] = value
                break
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    unexpected = [key for key in unexpected if key in mapped]
    if unexpected:
        raise RuntimeError(f"Unexpected temporal checkpoint keys: {unexpected}")
    unresolved = [key for key in mapped if key in missing]
    if unresolved:
        raise RuntimeError(f"Could not load temporal checkpoint keys: {unresolved[:8]}")
    logger.info("Initialized temporal branch from %s", path)
    return True


def set_gnn_trainable(model: MauronGraphRecurrentHybrid, trainable: bool) -> None:
    for parameter in model.gnn.parameters():
        parameter.requires_grad = trainable


def set_temporal_trainable(model: MauronGraphRecurrentHybrid, trainable: bool) -> None:
    modules = [
        model.temporal_project,
        model.temporal_gru,
        model.temporal_head,
        model.age_head,
        model.chamber_head,
    ]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = trainable


def effective_class_weights(
    graphs: Sequence[Data],
    train_indices: Sequence[int],
    num_classes: int,
    device: torch.device,
    beta: float,
    max_weight: float,
) -> torch.Tensor:
    labels = torch.cat([graphs[idx].y for idx in train_indices]).cpu().numpy().astype(int)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    effective_num = 1.0 - np.power(beta, counts[present])
    weights[present] = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    weights[present] = weights[present] / np.mean(weights[present])
    weights[present] = np.minimum(weights[present], max_weight)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def rare_class_ids(graphs: Sequence[Data], train_indices: Sequence[int], support_threshold: int) -> torch.Tensor:
    labels = torch.cat([graphs[idx].y for idx in train_indices]).cpu().numpy().astype(int)
    counts = np.bincount(labels, minlength=max(labels.max() + 1, 1))
    rare = np.where((counts > 0) & (counts <= support_threshold))[0].astype(np.int64)
    return torch.tensor(rare, dtype=torch.long)


def soft_target_kl_loss(logits: torch.Tensor, soft_target: torch.Tensor) -> torch.Tensor:
    target = soft_target / soft_target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    log_probs = F.log_softmax(logits, dim=1)
    return F.kl_div(log_probs, target, reduction="batchmean")


def hierarchical_class_logits(
    class_logits: torch.Tensor,
    bio_group_logits: torch.Tensor,
    label_to_group_id: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    if weight <= 0:
        return class_logits
    group_log_probs = F.log_softmax(bio_group_logits, dim=1)
    class_group_log_probs = group_log_probs[:, label_to_group_id.to(class_logits.device)]
    return class_logits + (weight * class_group_log_probs)


def groupwise_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_to_group_id: torch.Tensor,
) -> torch.Tensor:
    label_to_group_id = label_to_group_id.to(labels.device)
    present_groups = torch.unique(label_to_group_id[labels])
    losses = []
    for group_id in present_groups.tolist():
        class_ids = torch.where(label_to_group_id == group_id)[0]
        if class_ids.numel() <= 1:
            continue
        spot_mask = torch.isin(labels, class_ids)
        if not spot_mask.any():
            continue
        local_logits = logits[spot_mask][:, class_ids]
        local_lookup = torch.full(
            (int(label_to_group_id.numel()),),
            fill_value=-1,
            dtype=torch.long,
            device=labels.device,
        )
        local_lookup[class_ids] = torch.arange(class_ids.numel(), device=labels.device)
        local_labels = local_lookup[labels[spot_mask]]
        losses.append(F.cross_entropy(local_logits, local_labels))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def rare_replay_loss(
    criterion: FocalLoss,
    logits: torch.Tensor,
    labels: torch.Tensor,
    rare_ids: torch.Tensor,
) -> torch.Tensor:
    if rare_ids.numel() == 0:
        return logits.sum() * 0.0
    rare_ids = rare_ids.to(labels.device)
    mask = torch.isin(labels, rare_ids)
    if not mask.any():
        return logits.sum() * 0.0
    return criterion(logits[mask], labels[mask])


def age_targets(graph: Data, age_label_to_id: Dict[str, int], device: torch.device) -> torch.Tensor:
    return torch.full((graph.num_nodes,), age_label_to_id[graph.age_bin], dtype=torch.long, device=device)


def chamber_targets(graph: Data, device: torch.device) -> torch.Tensor:
    chamber = graph.chamber_multi_hot.to(device).float()
    return chamber.expand(graph.num_nodes, -1)


def bio_group_targets(graph: Data, label_to_group_id: torch.Tensor, device: torch.device) -> torch.Tensor:
    return label_to_group_id.to(device)[graph.y.long()]


def evaluate(
    model: MauronGraphRecurrentHybrid,
    graphs: Sequence[Data],
    indices: Sequence[int],
    prototypes: torch.Tensor,
    device: torch.device,
    label_to_group_id: torch.Tensor,
    hierarchical_group_logit_weight: float,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    model.eval()
    outputs_by_head = {name: ([], []) for name in ["fusion", "graph", "temporal"]}
    with torch.no_grad():
        for graph_idx in indices:
            graph = graphs[graph_idx].to(device)
            outputs = model(graph, prototypes)
            fusion_logits = hierarchical_class_logits(
                outputs["fusion"],
                outputs["bio_group"],
                label_to_group_id,
                hierarchical_group_logit_weight,
            )
            outputs = {**outputs, "fusion": fusion_logits}
            truth = graph.y.cpu().numpy().astype(int).tolist()
            for head in outputs_by_head:
                pred = outputs[head].argmax(dim=1).cpu().numpy().astype(int).tolist()
                outputs_by_head[head][0].extend(truth)
                outputs_by_head[head][1].extend(pred)
    return {
        head: (np.asarray(true, dtype=int), np.asarray(pred, dtype=int))
        for head, (true, pred) in outputs_by_head.items()
    }


def metrics_for(y_true: np.ndarray, y_pred: np.ndarray, label_names: Sequence[str]) -> Dict:
    labels = list(range(len(label_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "num_examples": int(len(y_true)),
        "label_counts": {str(label): int(count) for label, count in Counter(y_true.tolist()).items()},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=list(label_names),
            output_dict=True,
            zero_division=0,
        ),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu")
    output_dir = Path(args.output_dir or f"mauron_spatial_hybrid_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    build_config = MauronBuildConfig(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        num_genes=args.num_genes,
        k_neighbors=8,
        label_mode="deconv_hl_argmax",
    )
    graphs, metadata = MauronVisiumGraphDataset(build_config).load_or_build(force_rebuild=args.force_rebuild)
    splits = split_graphs(graphs, args.split_group, args.seed)
    soft_attached = attach_soft_deconvolution_targets(graphs, args.data_root)
    add_temporal_standardized_features(graphs, splits["train"])
    prototypes, train_ages, prototype_members = build_age_prototypes(graphs, splits["train"])
    prototypes = prototypes.to(device)

    label_names = metadata["label_names"]
    num_classes = len(label_names)
    bio_group_names = sorted({biological_group(label) for label in label_names})
    bio_group_to_id = {group: idx for idx, group in enumerate(bio_group_names)}
    label_to_group_id = torch.tensor(
        [bio_group_to_id[biological_group(label)] for label in label_names],
        dtype=torch.long,
    )
    age_label_to_id = {label: idx for idx, label in enumerate(AGE_BIN_ORDER)}
    chamber_dim = int(graphs[0].chamber_multi_hot.shape[1])

    model = MauronGraphRecurrentHybrid(
        input_dim=graphs[0].x.shape[1],
        num_classes=num_classes,
        num_bio_groups=len(bio_group_names),
        num_age_bins=len(AGE_BIN_ORDER),
        chamber_dim=chamber_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_embedding_dim=args.gnn_embedding_dim,
        gnn_layers=args.gnn_layers,
        rnn_projection_dim=args.rnn_projection_dim,
        rnn_hidden_dim=args.rnn_hidden_dim,
        dropout=args.dropout,
        class_gate_weight=args.class_gate_weight,
    ).to(device)
    loaded_gnn = load_gnn_checkpoint(model, args.gnn_checkpoint)
    loaded_rnn = load_rnn_checkpoint(model, args.rnn_checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    main_criterion = FocalLoss(
        effective_class_weights(
            graphs,
            splits["train"],
            num_classes,
            device,
            args.class_weight_beta,
            args.max_class_weight,
        ),
        gamma=args.focal_gamma,
    )
    age_criterion = nn.CrossEntropyLoss()
    bio_group_criterion = nn.CrossEntropyLoss()
    chamber_criterion = nn.BCEWithLogitsLoss()
    rare_ids = rare_class_ids(graphs, splits["train"], args.rare_support_threshold)

    logger.info("Output directory: %s", output_dir)
    logger.info("Device: %s", device)
    logger.info("Bio groups: %s", bio_group_names)
    logger.info("Attached soft deconvolution targets for %d spots.", soft_attached)
    logger.info("Rare replay class ids at <=%d train spots: %s", args.rare_support_threshold, rare_ids.tolist())
    logger.info("Train age prototypes: %s", train_ages)
    logger.info("Grouped split summary: %s", json.dumps(split_summary(graphs, splits, args.split_group), indent=2))

    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_val_macro_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        # Stage 1: train only fusion/group heads; stage 2: tune temporal + fusion;
        # stage 3: unfreeze the pretrained GNN with the rest of the hybrid.
        set_temporal_trainable(model, not loaded_rnn or epoch > args.fusion_warmup_epochs)
        set_gnn_trainable(model, not loaded_gnn or epoch > args.freeze_gnn_epochs)
        model.train()
        total_loss = 0.0
        total_fusion_loss = 0.0
        train_order = list(splits["train"])
        random.shuffle(train_order)
        for graph_idx in train_order:
            graph = graphs[graph_idx].to(device)
            outputs = model(graph, prototypes)
            y = graph.y.long()
            hierarchical_logits = hierarchical_class_logits(
                outputs["fusion"],
                outputs["bio_group"],
                label_to_group_id,
                args.hierarchical_group_logit_weight,
            )
            fusion_loss = main_criterion(hierarchical_logits, y)
            graph_loss = main_criterion(outputs["graph"], y)
            temporal_loss = main_criterion(outputs["temporal"], y)
            soft_loss = soft_target_kl_loss(hierarchical_logits, graph.y_soft.to(device).float())
            rare_loss = rare_replay_loss(main_criterion, hierarchical_logits, y, rare_ids)
            bio_group_loss = bio_group_criterion(outputs["bio_group"], bio_group_targets(graph, label_to_group_id, device))
            groupwise_loss = groupwise_classification_loss(hierarchical_logits, y, label_to_group_id)
            age_loss = age_criterion(outputs["age"], age_targets(graph, age_label_to_id, device))
            chamber_loss = chamber_criterion(outputs["chamber"], chamber_targets(graph, device))
            loss = (
                fusion_loss
                + args.graph_loss_weight * graph_loss
                + args.temporal_loss_weight * temporal_loss
                + args.soft_loss_weight * soft_loss
                + args.rare_replay_weight * rare_loss
                + args.bio_group_loss_weight * bio_group_loss
                + args.groupwise_loss_weight * groupwise_loss
                + args.age_loss_weight * age_loss
                + args.chamber_loss_weight * chamber_loss
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
            total_fusion_loss += float(fusion_loss.item())

        val_outputs = evaluate(
            model,
            graphs,
            splits["val"],
            prototypes,
            device,
            label_to_group_id,
            args.hierarchical_group_logit_weight,
        )
        val_metrics = metrics_for(*val_outputs["fusion"], label_names)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_order), 1),
            "train_fusion_loss": total_fusion_loss / max(len(train_order), 1),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(record)
        logger.info(
            "Epoch %03d/%03d - loss %.4f val acc %.3f val F1 %.3f",
            epoch,
            args.epochs,
            record["train_loss"],
            record["val_accuracy"],
            record["val_macro_f1"],
        )
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                logger.info("Early stopping after %d stale epochs.", stale_epochs)
                break

    model.load_state_dict(best_state)
    split_metrics = {}
    branch_metrics = {}
    for split_name, indices in splits.items():
        outputs = evaluate(
            model,
            graphs,
            indices,
            prototypes,
            device,
            label_to_group_id,
            args.hierarchical_group_logit_weight,
        )
        split_metrics[split_name] = metrics_for(*outputs["fusion"], label_names)
        branch_metrics[split_name] = {
            head: metrics_for(y_true, y_pred, label_names)
            for head, (y_true, y_pred) in outputs.items()
        }

    model_path = output_dir / "best_mauron_graph_recurrent_hybrid.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "model_config": {
                "input_dim": int(graphs[0].x.shape[1]),
                "num_classes": num_classes,
                "num_bio_groups": len(bio_group_names),
                "num_age_bins": len(AGE_BIN_ORDER),
                "chamber_dim": chamber_dim,
                "gnn_hidden_dim": args.gnn_hidden_dim,
                "gnn_embedding_dim": args.gnn_embedding_dim,
                "gnn_layers": args.gnn_layers,
                "rnn_projection_dim": args.rnn_projection_dim,
                "rnn_hidden_dim": args.rnn_hidden_dim,
                "dropout": args.dropout,
                "class_gate_weight": args.class_gate_weight,
            },
            "label_names": label_names,
            "bio_group_names": bio_group_names,
            "age_bin_order": AGE_BIN_ORDER,
        },
        model_path,
    )

    results = {
        "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "task": "spot_deconvolution_argmax_label",
        "model": "edge_aware_spatial_gnn_plus_developmental_prototype_gru",
        "source_dataset": "Mauron developing human heart Visium part a",
        "dataset_metadata": metadata,
        "label_names": label_names,
        "bio_group_names": bio_group_names,
        "age_bin_order": AGE_BIN_ORDER,
        "train_age_prototypes": train_ages,
        "prototype_members": prototype_members,
        "soft_deconvolution_targets_attached": int(soft_attached),
        "rare_replay_class_ids": rare_ids.tolist(),
        "split_group": args.split_group,
        "splits": split_summary(graphs, splits, args.split_group),
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "history": history,
        "split_metrics": split_metrics,
        "branch_metrics": branch_metrics,
        "model_path": str(model_path),
        "validation_notes": [
            "Train/validation/test are grouped by held-out case and stratified by fetal age bin.",
            "Fusion, graph-only head, and temporal-only head are all reported on the same examples.",
            "The recurrent branch uses cross-sectional fetal-age prototypes from training cases only.",
        ],
    }
    results_path = output_dir / "mauron_spatial_hybrid_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Saved results to %s", results_path)
    logger.info("Test metrics: %s", json.dumps(split_metrics["test"], indent=2)[:2000])


if __name__ == "__main__":
    main()

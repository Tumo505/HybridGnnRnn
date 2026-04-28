#!/usr/bin/env python3
"""Train a spatial graph adapter on top of finetuned Geneformer spot embeddings.

This is the added spatial-context layer for the foundation model:

    Mauron spot expression -> finetuned Geneformer embedding
    -> coordinate-aware graph adapter -> contextual spatial-time embedding

The adapter is trained with real within-section coordinates and multi-task
supervision. It predicts spot-level deconvolution cell state plus section-level
age bin, continuous developmental progress, and chamber composition. Grouped
splits are by case or code, never by random spots.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("geneformer_spatial_context_adapter")


AGE_BIN_ORDER = ["early_w6_w7_5", "mid_w8_w9", "late_w10_w12"]
CHAMBERS = ["LV", "RV", "LA", "RA"]
FAMILY_ORDER = [
    "cardiomyocyte",
    "fibroblast",
    "endothelial",
    "smooth_muscle",
    "epicardial_mesothelial",
    "valve_mesenchymal",
    "neural",
    "immune",
    "blood",
    "other",
]


def infer_cell_family(cell_state: str, deconv_label: str = "") -> str:
    text = f"{cell_state} {deconv_label}".lower()
    if any(token in text for token in ["acm", "vcm", "cardiomyocyte", "metact"]):
        return "cardiomyocyte"
    if any(token in text for token in ["fb", "fib", "fibro"]):
        return "fibroblast"
    if any(token in text for token in ["ec", "endoc", "endo", "vascular", "vasc", "lec"]):
        return "endothelial"
    if any(token in text for token in ["smc", "smooth"]):
        return "smooth_muscle"
    if any(token in text for token in ["epdc", "epicard", "epc", "mesothelial"]):
        return "epicardial_mesothelial"
    if any(token in text for token in ["valve", "cush", "_mc", " mc"]):
        return "valve_mesenchymal"
    if any(token in text for token in ["neural", "neur", "nb-n", "scp", "gc", "innervation"]):
        return "neural"
    if any(token in text for token in ["immune", "macro", "lymph", "mono"]):
        return "immune"
    if any(token in text for token in ["blood", "ery", "hem", "eryth"]):
        return "blood"
    return "other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spatial-embeddings",
        default="foundation_spatial_geneformer_adapter_inputs/mauron_spatial_finetuned_geneformer_embeddings.npy",
    )
    parser.add_argument(
        "--spatial-metadata",
        default="foundation_spatial_geneformer_adapter_inputs/mauron_spatial_finetuned_geneformer_metadata.csv",
    )
    parser.add_argument("--output-dir", default="foundation_geneformer_spatial_context_adapter")
    parser.add_argument("--output-prefix", default="mauron_geneformer_spatial_adapter")
    parser.add_argument("--split-group", choices=["case", "code"], default="case")
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--node-loss-weight", type=float, default=0.8)
    parser.add_argument("--family-loss-weight", type=float, default=0.8)
    parser.add_argument("--age-loss-weight", type=float, default=0.8)
    parser.add_argument("--age-ordinal-loss-weight", type=float, default=0.35)
    parser.add_argument("--progress-loss-weight", type=float, default=0.5)
    parser.add_argument("--chamber-loss-weight", type=float, default=0.25)
    parser.add_argument("--coordinate-smoothness-weight", type=float, default=0.03)
    parser.add_argument("--neighborhood-consistency-weight", type=float, default=0.05)
    parser.add_argument("--max-class-weight", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--spatial-ablation", choices=["real", "shuffle_pos", "zero_pos"], default="real")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_knn_edges(pos: np.ndarray, k_neighbors: int) -> np.ndarray:
    if pos.shape[0] < 2:
        return np.empty((2, 0), dtype=np.int64)
    n_neighbors = min(k_neighbors + 1, pos.shape[0])
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nbrs.fit(pos)
    indices = nbrs.kneighbors(pos, return_distance=False)
    sources = np.repeat(np.arange(pos.shape[0]), n_neighbors - 1)
    targets = indices[:, 1:].reshape(-1)
    pairs = np.vstack([np.column_stack([sources, targets]), np.column_stack([targets, sources])])
    pairs = np.unique(pairs, axis=0)
    return pairs.T.astype(np.int64)


def chamber_multi_hot(combo: str) -> list[float]:
    parts = set(str(combo).split("+"))
    return [1.0 if chamber in parts else 0.0 for chamber in CHAMBERS]


def normalize_section_pos(df: pd.DataFrame) -> np.ndarray:
    pos = df[["x", "y"]].to_numpy(dtype=np.float32)
    center = pos.mean(axis=0, keepdims=True)
    scale = pos.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return ((pos - center) / scale).astype(np.float32)


def load_graphs(embeddings_path: Path, metadata_path: Path, k_neighbors: int, spatial_ablation: str, seed: int) -> tuple[list[Data], dict]:
    embeddings = np.load(embeddings_path).astype(np.float32)
    metadata = pd.read_csv(metadata_path)
    if len(metadata) != embeddings.shape[0]:
        raise ValueError(f"Metadata rows ({len(metadata)}) do not match embeddings ({embeddings.shape[0]}).")
    required = {"section_id", "code", "case", "age_weeks", "age_bin", "chamber_combo", "x", "y", "deconv_label", "cell_state"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"Spatial metadata is missing columns: {missing}")

    metadata = metadata.copy()
    metadata["section_id"] = metadata["section_id"].astype(int)
    label_keys = sorted(metadata["deconv_label"].astype(str).unique())
    label_to_id = {label: idx for idx, label in enumerate(label_keys)}
    label_names = (
        metadata[["deconv_label", "cell_state"]]
        .astype(str)
        .drop_duplicates()
        .set_index("deconv_label")["cell_state"]
        .reindex(label_keys)
        .fillna(pd.Series(label_keys, index=label_keys))
        .tolist()
    )
    family_to_id = {label: idx for idx, label in enumerate(FAMILY_ORDER)}
    metadata["cell_family"] = [
        infer_cell_family(cell_state, label)
        for cell_state, label in zip(metadata["cell_state"].astype(str), metadata["deconv_label"].astype(str))
    ]
    age_to_id = {label: idx for idx, label in enumerate(AGE_BIN_ORDER)}
    rng = np.random.default_rng(seed)

    graphs: list[Data] = []
    for section_id, df in metadata.groupby("section_id", sort=True):
        idx = df.index.to_numpy()
        pos = normalize_section_pos(df)
        if spatial_ablation == "shuffle_pos":
            pos = pos[rng.permutation(pos.shape[0])]
        elif spatial_ablation == "zero_pos":
            pos = np.zeros_like(pos)
        edge_index = build_knn_edges(pos, k_neighbors)
        age_bin = str(df["age_bin"].iloc[0])
        progress = float(np.clip((float(df["age_weeks"].iloc[0]) - 6.0) / 6.0, 0.0, 1.0))
        graph = Data(
            x=torch.from_numpy(embeddings[idx]),
            pos=torch.from_numpy(pos),
            edge_index=torch.from_numpy(edge_index),
            y=torch.tensor([label_to_id[str(label)] for label in df["deconv_label"]], dtype=torch.long),
            family_y=torch.tensor([family_to_id[infer_cell_family(state, label)] for state, label in zip(df["cell_state"], df["deconv_label"])], dtype=torch.long),
            label_confidence=torch.tensor(df.get("deconv_confidence", pd.Series(1.0, index=df.index)).astype(float).to_numpy(), dtype=torch.float32),
            age_bin_y=torch.tensor([age_to_id[age_bin]], dtype=torch.long),
            progress_y=torch.tensor([progress], dtype=torch.float32),
            chamber_y=torch.tensor([chamber_multi_hot(str(df["chamber_combo"].iloc[0]))], dtype=torch.float32),
        )
        graph.section_id = int(section_id)
        graph.code = str(df["code"].iloc[0])
        graph.case = str(df["case"].iloc[0])
        graph.age_bin = age_bin
        graph.age_weeks = float(df["age_weeks"].iloc[0])
        graph.chamber_combo = str(df["chamber_combo"].iloc[0])
        graph.spot_ids = df["spot_id"].astype(str).tolist() if "spot_id" in df.columns else df.index.astype(str).tolist()
        graphs.append(graph)

    info = {
        "embedding_shape": list(embeddings.shape),
        "num_graphs": len(graphs),
        "num_spots": int(sum(g.num_nodes for g in graphs)),
        "label_keys": label_keys,
        "label_names": label_names,
        "family_names": FAMILY_ORDER,
        "family_counts": metadata["cell_family"].value_counts().to_dict(),
        "age_bin_order": AGE_BIN_ORDER,
        "chambers": CHAMBERS,
    }
    return graphs, info


def split_graphs(graphs: Sequence[Data], split_group: str, seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    group_to_indices: dict[str, list[int]] = {}
    group_to_bins: dict[str, set[str]] = {}
    for idx, graph in enumerate(graphs):
        group = str(getattr(graph, split_group))
        group_to_indices.setdefault(group, []).append(idx)
        group_to_bins.setdefault(group, set()).add(graph.age_bin)

    split_groups = {"train": [], "val": [], "test": []}
    for age_bin in AGE_BIN_ORDER:
        groups = sorted(group for group, bins in group_to_bins.items() if age_bin in bins)
        if not groups:
            continue
        shuffled = np.asarray(groups, dtype=object)
        rng.shuffle(shuffled)
        if len(shuffled) >= 3:
            val_count = max(1, int(round(len(shuffled) * 0.15)))
            test_count = max(1, int(round(len(shuffled) * 0.15)))
            if val_count + test_count >= len(shuffled):
                val_count = 1
                test_count = 1
            split_groups["val"].extend(shuffled[:val_count].tolist())
            split_groups["test"].extend(shuffled[val_count : val_count + test_count].tolist())
            split_groups["train"].extend(shuffled[val_count + test_count :].tolist())
        else:
            split_groups["train"].extend(shuffled.tolist())

    assigned = set(split_groups["train"] + split_groups["val"] + split_groups["test"])
    for group in sorted(set(group_to_indices).difference(assigned)):
        split_groups["train"].append(group)
    return {
        split: sorted(idx for group in groups for idx in group_to_indices[group])
        for split, groups in split_groups.items()
    }


def split_summary(graphs: Sequence[Data], splits: dict[str, list[int]], split_group: str) -> dict:
    summary = {}
    for split, indices in splits.items():
        subset = [graphs[idx] for idx in indices]
        summary[split] = {
            "num_sections": len(subset),
            "num_spots": int(sum(graph.num_nodes for graph in subset)),
            "sections": [graph.section_id for graph in subset],
            "codes": [graph.code for graph in subset],
            "cases": sorted({graph.case for graph in subset}),
            "groups": sorted({str(getattr(graph, split_group)) for graph in subset}),
            "age_bins": dict(Counter(graph.age_bin for graph in subset)),
        }
    return summary


class GeneformerSpatialContextAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_cell_states: int,
        num_families: int,
        num_age_bins: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.dropout = dropout
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.convs = nn.ModuleList(
            [GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=4, dropout=dropout) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([GraphNorm(hidden_dim) for _ in range(num_layers)])
        self.embedding_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.cell_state_head = nn.Linear(embedding_dim, num_cell_states)
        self.family_head = nn.Linear(embedding_dim, num_families)
        self.age_bin_head = nn.Linear(embedding_dim, num_age_bins)
        self.progress_head = nn.Linear(embedding_dim, 1)
        self.chamber_head = nn.Linear(embedding_dim, len(CHAMBERS))

    @staticmethod
    def edge_attr(data: Data) -> torch.Tensor:
        row, col = data.edge_index
        delta = data.pos[col] - data.pos[row]
        distance = torch.linalg.norm(delta, dim=1, keepdim=True).clamp_min(1e-6)
        direction = delta / distance
        mean_distance = distance.mean().clamp_min(1e-6)
        return torch.cat([direction, distance / mean_distance, mean_distance / distance], dim=1).float()

    def encode(self, data: Data) -> torch.Tensor:
        x = self.input_projection(data.x)
        edge_attr = self.edge_attr(data)
        batch = getattr(data, "batch", None)
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, data.edge_index, edge_attr=edge_attr)
            x = norm(x, batch=batch)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual
        return self.embedding_projection(x)

    def forward(self, data: Data) -> dict[str, torch.Tensor]:
        node_z = self.encode(data)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(node_z.shape[0], dtype=torch.long, device=node_z.device)
        graph_z = global_mean_pool(node_z, batch)
        return {
            "node_z": node_z,
            "graph_z": graph_z,
            "cell_state_logits": self.cell_state_head(node_z),
            "family_logits": self.family_head(node_z),
            "age_bin_logits": self.age_bin_head(graph_z),
            "progress": self.progress_head(graph_z).squeeze(1),
            "chamber_logits": self.chamber_head(graph_z),
        }


def to_device(graph: Data, device: torch.device) -> Data:
    return graph.to(device)


def node_class_weights(
    graphs: Sequence[Data],
    indices: Sequence[int],
    num_classes: int,
    max_weight: float,
    device: torch.device,
    attr: str = "y",
) -> torch.Tensor:
    labels = []
    for idx in indices:
        labels.extend(getattr(graphs[idx], attr).cpu().numpy().astype(int).tolist())
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(np.float32)
    weights = np.ones(num_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    weights[present] = np.minimum(weights[present], max_weight)
    weights[~present] = 0.0
    return torch.tensor(weights, dtype=torch.float32, device=device)


def confidence_weighted_ce(logits: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(logits, target, weight=class_weights, reduction="none")
    normalized_conf = confidence.float().clamp(0.05, 1.0)
    normalized_conf = normalized_conf / normalized_conf.mean().clamp_min(1e-6)
    return (loss * normalized_conf).mean()


def ordinal_age_loss(age_logits: torch.Tensor, age_target: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(age_logits, dim=1)
    values = torch.arange(age_logits.shape[1], dtype=probs.dtype, device=probs.device)
    expected = (probs * values).sum(dim=1) / max(age_logits.shape[1] - 1, 1)
    target = age_target.float() / max(age_logits.shape[1] - 1, 1)
    return F.smooth_l1_loss(expected, target, beta=0.15)


def graph_smoothness_loss(node_z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return node_z.new_tensor(0.0)
    row, col = edge_index
    return F.smooth_l1_loss(node_z[row], node_z[col])


def neighborhood_consistency_loss(logits: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return logits.new_tensor(0.0)
    row, col = edge_index
    log_probs = torch.log_softmax(logits, dim=1)
    probs = torch.softmax(logits, dim=1)
    return 0.5 * (
        F.kl_div(log_probs[row], probs[col], reduction="batchmean")
        + F.kl_div(log_probs[col], probs[row], reduction="batchmean")
    )


@dataclass
class EvalResult:
    metrics: dict
    node_predictions: pd.DataFrame
    section_predictions: pd.DataFrame


def evaluate(
    model: GeneformerSpatialContextAdapter,
    graphs: Sequence[Data],
    indices: Sequence[int],
    device: torch.device,
    label_names: Sequence[str],
    family_names: Sequence[str],
) -> EvalResult:
    model.eval()
    node_rows = []
    section_rows = []
    with torch.no_grad():
        for idx in indices:
            graph = to_device(graphs[idx].clone(), device)
            out = model(graph)
            pred_node = out["cell_state_logits"].argmax(dim=1).cpu().numpy()
            pred_family = out["family_logits"].argmax(dim=1).cpu().numpy()
            true_node = graph.y.cpu().numpy()
            true_family = graph.family_y.cpu().numpy()
            for spot_id, y_true, y_pred, fam_true, fam_pred in zip(graph.spot_ids, true_node, pred_node, true_family, pred_family):
                node_rows.append(
                    {
                        "section_id": graph.section_id,
                        "code": graph.code,
                        "case": graph.case,
                        "spot_id": spot_id,
                        "true_cell_state_id": int(y_true),
                        "pred_cell_state_id": int(y_pred),
                        "true_cell_state": label_names[int(y_true)],
                        "pred_cell_state": label_names[int(y_pred)],
                        "true_family_id": int(fam_true),
                        "pred_family_id": int(fam_pred),
                        "true_family": family_names[int(fam_true)],
                        "pred_family": family_names[int(fam_pred)],
                    }
                )
            pred_age = int(out["age_bin_logits"].argmax(dim=1).cpu().item())
            pred_progress = float(out["progress"].cpu().item())
            pred_chambers = (torch.sigmoid(out["chamber_logits"]).cpu().numpy()[0] >= 0.5).astype(int)
            section_rows.append(
                {
                    "section_id": graph.section_id,
                    "code": graph.code,
                    "case": graph.case,
                    "split": getattr(graph, "split", "unknown"),
                    "age_bin": graph.age_bin,
                    "true_age_bin_id": int(graph.age_bin_y.cpu().item()),
                    "pred_age_bin_id": pred_age,
                    "pred_age_bin": AGE_BIN_ORDER[pred_age],
                    "age_weeks": float(graph.age_weeks),
                    "true_progress": float(graph.progress_y.cpu().item()),
                    "pred_progress": pred_progress,
                    "chamber_combo": graph.chamber_combo,
                    "pred_chamber_combo": "+".join([c for c, keep in zip(CHAMBERS, pred_chambers) if keep]) or "unknown",
                }
            )
    node_pred = pd.DataFrame(node_rows)
    section_pred = pd.DataFrame(section_rows)
    metrics = {
        "num_sections": int(len(section_pred)),
        "num_spots": int(len(node_pred)),
    }
    if not node_pred.empty:
        metrics.update(
            {
                "cell_state_accuracy": float(accuracy_score(node_pred["true_cell_state_id"], node_pred["pred_cell_state_id"])),
                "cell_state_macro_f1": float(f1_score(node_pred["true_cell_state_id"], node_pred["pred_cell_state_id"], average="macro", zero_division=0)),
                "family_accuracy": float(accuracy_score(node_pred["true_family_id"], node_pred["pred_family_id"])),
                "family_macro_f1": float(f1_score(node_pred["true_family_id"], node_pred["pred_family_id"], average="macro", zero_division=0)),
            }
        )
    if not section_pred.empty:
        metrics.update(
            {
                "age_bin_accuracy": float(accuracy_score(section_pred["true_age_bin_id"], section_pred["pred_age_bin_id"])),
                "age_bin_macro_f1": float(f1_score(section_pred["true_age_bin_id"], section_pred["pred_age_bin_id"], average="macro", zero_division=0)),
                "progress_mae": float(mean_absolute_error(section_pred["true_progress"], section_pred["pred_progress"])),
            }
        )
    return EvalResult(metrics=metrics, node_predictions=node_pred, section_predictions=section_pred)


def export_embeddings(
    model: GeneformerSpatialContextAdapter,
    graphs: Sequence[Data],
    output_dir: Path,
    output_prefix: str,
    device: torch.device,
) -> dict:
    model.eval()
    spot_embeddings = []
    spot_rows = []
    section_embeddings = []
    section_rows = []
    with torch.no_grad():
        for graph in graphs:
            device_graph = to_device(graph.clone(), device)
            out = model(device_graph)
            node_z = out["node_z"].cpu().numpy().astype(np.float32)
            graph_z = out["graph_z"].cpu().numpy().astype(np.float32)
            spot_embeddings.append(node_z)
            section_embeddings.append(graph_z)
            section_rows.append(
                    {
                        "section_id": graph.section_id,
                        "code": graph.code,
                        "case": graph.case,
                        "split": getattr(graph, "split", "unknown"),
                        "age_bin": graph.age_bin,
                        "age_weeks": graph.age_weeks,
                        "chamber_combo": graph.chamber_combo,
                }
            )
            for spot_id in graph.spot_ids:
                spot_rows.append(
                    {
                        "section_id": graph.section_id,
                        "code": graph.code,
                        "case": graph.case,
                        "split": getattr(graph, "split", "unknown"),
                        "spot_id": spot_id,
                        "age_bin": graph.age_bin,
                        "age_weeks": graph.age_weeks,
                        "chamber_combo": graph.chamber_combo,
                    }
                )
    spot_array = np.vstack(spot_embeddings)
    section_array = np.vstack(section_embeddings)
    np.save(output_dir / f"{output_prefix}_contextual_spot_embeddings.npy", spot_array)
    np.save(output_dir / f"{output_prefix}_contextual_section_embeddings.npy", section_array)
    pd.DataFrame(spot_rows).to_csv(output_dir / f"{output_prefix}_contextual_spot_metadata.csv", index=False)
    pd.DataFrame(section_rows).to_csv(output_dir / f"{output_prefix}_contextual_section_metadata.csv", index=False)
    return {
        "contextual_spot_embeddings": str(output_dir / f"{output_prefix}_contextual_spot_embeddings.npy"),
        "contextual_spot_metadata": str(output_dir / f"{output_prefix}_contextual_spot_metadata.csv"),
        "contextual_section_embeddings": str(output_dir / f"{output_prefix}_contextual_section_embeddings.npy"),
        "contextual_section_metadata": str(output_dir / f"{output_prefix}_contextual_section_metadata.csv"),
        "spot_embedding_shape": list(spot_array.shape),
        "section_embedding_shape": list(section_array.shape),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graphs, info = load_graphs(
        Path(args.spatial_embeddings),
        Path(args.spatial_metadata),
        args.k_neighbors,
        args.spatial_ablation,
        args.seed,
    )
    splits = split_graphs(graphs, args.split_group, args.seed)
    for split_name, indices in splits.items():
        for graph_idx in indices:
            graphs[graph_idx].split = split_name
    summary_splits = split_summary(graphs, splits, args.split_group)
    logger.info("Split summary: %s", json.dumps(summary_splits, indent=2))

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = GeneformerSpatialContextAdapter(
        input_dim=int(graphs[0].x.shape[1]),
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_cell_states=len(info["label_keys"]),
        num_families=len(info["family_names"]),
        num_age_bins=len(AGE_BIN_ORDER),
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    fine_weights = node_class_weights(graphs, splits["train"], len(info["label_keys"]), args.max_class_weight, device, attr="y")
    family_weights = node_class_weights(graphs, splits["train"], len(info["family_names"]), args.max_class_weight, device, attr="family_y")
    ce_age = nn.CrossEntropyLoss()
    bce_chamber = nn.BCEWithLogitsLoss()
    regression = nn.SmoothL1Loss(beta=0.08)

    best_score = -float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(splits["train"])
        losses = []
        for idx in splits["train"]:
            graph = to_device(graphs[idx].clone(), device)
            optimizer.zero_grad(set_to_none=True)
            out = model(graph)
            node_loss = confidence_weighted_ce(out["cell_state_logits"], graph.y, fine_weights, graph.label_confidence)
            family_loss = confidence_weighted_ce(out["family_logits"], graph.family_y, family_weights, graph.label_confidence)
            age_loss = ce_age(out["age_bin_logits"], graph.age_bin_y)
            age_ordinal = ordinal_age_loss(out["age_bin_logits"], graph.age_bin_y)
            progress_loss = regression(out["progress"], graph.progress_y)
            chamber_loss = bce_chamber(out["chamber_logits"], graph.chamber_y)
            smooth_loss = graph_smoothness_loss(out["node_z"], graph.edge_index)
            consistency_loss = neighborhood_consistency_loss(out["family_logits"], graph.edge_index)
            loss = (
                args.node_loss_weight * node_loss
                + args.family_loss_weight * family_loss
                + args.age_loss_weight * age_loss
                + args.age_ordinal_loss_weight * age_ordinal
                + args.progress_loss_weight * progress_loss
                + args.chamber_loss_weight * chamber_loss
                + args.coordinate_smoothness_weight * smooth_loss
                + args.neighborhood_consistency_weight * consistency_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val = evaluate(model, graphs, splits["val"], device, info["label_names"], info["family_names"])
        val_score = (
            0.5 * val.metrics.get("cell_state_macro_f1", 0.0)
            + val.metrics.get("family_macro_f1", 0.0)
            + val.metrics.get("age_bin_macro_f1", 0.0)
            - val.metrics.get("progress_mae", 1.0)
        )
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_metrics": val.metrics, "val_score": float(val_score)}
        history.append(row)
        if epoch == 1 or epoch % 10 == 0:
            logger.info("Epoch %03d/%03d %s", epoch, args.epochs, json.dumps(row))
        if val_score > best_score:
            best_score = float(val_score)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                logger.info("Early stopping after %d stale epochs.", stale)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_eval = evaluate(model, graphs, splits["train"], device, info["label_names"], info["family_names"])
    val_eval = evaluate(model, graphs, splits["val"], device, info["label_names"], info["family_names"])
    test_eval = evaluate(model, graphs, splits["test"], device, info["label_names"], info["family_names"])
    exports = export_embeddings(model, graphs, output_dir, args.output_prefix, device)

    train_eval.node_predictions.to_csv(output_dir / f"{args.output_prefix}_train_node_predictions.csv", index=False)
    val_eval.node_predictions.to_csv(output_dir / f"{args.output_prefix}_val_node_predictions.csv", index=False)
    test_eval.node_predictions.to_csv(output_dir / f"{args.output_prefix}_test_node_predictions.csv", index=False)
    train_eval.section_predictions.to_csv(output_dir / f"{args.output_prefix}_train_section_predictions.csv", index=False)
    val_eval.section_predictions.to_csv(output_dir / f"{args.output_prefix}_val_section_predictions.csv", index=False)
    test_eval.section_predictions.to_csv(output_dir / f"{args.output_prefix}_test_section_predictions.csv", index=False)
    if not test_eval.node_predictions.empty:
        report = classification_report(
            test_eval.node_predictions["true_cell_state_id"],
            test_eval.node_predictions["pred_cell_state_id"],
            labels=list(range(len(info["label_names"]))),
            target_names=info["label_names"],
            output_dict=True,
            zero_division=0,
        )
        family_report = classification_report(
            test_eval.node_predictions["true_family_id"],
            test_eval.node_predictions["pred_family_id"],
            labels=list(range(len(info["family_names"]))),
            target_names=info["family_names"],
            output_dict=True,
            zero_division=0,
        )
    else:
        report = {}
        family_report = {}

    summary = {
        "model": "finetuned_geneformer_spatial_context_adapter",
        "important_caveat": "This models inferred spatial-time context from static spatial sections; it is not true longitudinal tracking of identical cells.",
        "inputs": {
            "spatial_embeddings": args.spatial_embeddings,
            "spatial_metadata": args.spatial_metadata,
            "spatial_ablation": args.spatial_ablation,
        },
        "graph_info": info,
        "split_group": args.split_group,
        "splits": summary_splits,
        "train_metrics": train_eval.metrics,
        "val_metrics": val_eval.metrics,
        "test_metrics": test_eval.metrics,
        "test_classification_report": report,
        "test_family_classification_report": family_report,
        "exports": exports,
        "history": history,
    }
    (output_dir / f"{args.output_prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), output_dir / f"{args.output_prefix}_model.pt")
    print(
        json.dumps(
            {
                "important_caveat": summary["important_caveat"],
                "val_metrics": val_eval.metrics,
                "test_metrics": test_eval.metrics,
                "exports": exports,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

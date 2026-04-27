#!/usr/bin/env python3
"""
Train a Mauron Visium spatial GNN from raw section graphs.

This pipeline intentionally avoids the older opaque cached graph. It:
1. builds one graph per tissue section from raw Visium matrices and coordinates;
2. uses real labels from deconvolution argmax or section metadata;
3. splits by Case or Code, never random spots;
4. pretrains the encoder with masked feature reconstruction on all sections;
5. fine-tunes supervised on grouped training sections; and
6. exports section-level and case-level GNN embeddings for temporal alignment.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.mauron_visium_processor import (  # noqa: E402
    MauronBuildConfig,
    MauronVisiumGraphDataset,
)
from src.models.gnn_models.mauron_spatial_gnn import MauronSpatialGNN  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("mauron_spatial_gnn")

AGE_BIN_ORDER = ["early_w6_w7_5", "mid_w8_w9", "late_w10_w12"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=(
            "data/New/Mauron_spatial_dynamics_part_a/"
            "Spatial dynamics of the developing human heart, pa"
        ),
        help="Mauron dataset root containing 2_Visium_spaceranger_data_ST and 7_Metadata.",
    )
    parser.add_argument("--cache-dir", default="cache/mauron_visium_graphs")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--label-mode", default="deconv_hl_argmax", choices=sorted(MauronVisiumGraphDataset.VALID_LABEL_MODES))
    parser.add_argument("--split-group", default="case", choices=["case", "code"])
    parser.add_argument("--num-genes", type=int, default=512)
    parser.add_argument("--min-gene-spots", type=int, default=25)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--max-sections", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=80)
    parser.add_argument("--mask-rate", type=float, default=0.20)
    parser.add_argument("--spatial-pretrain-weight", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight-beta", type=float, default=0.99)
    parser.add_argument("--max-class-weight", type=float, default=3.0)
    parser.add_argument("--focal-gamma", type=float, default=0.5)
    parser.add_argument(
        "--spatial-ablation",
        default="real",
        choices=["real", "shuffle_pos", "zero_pos"],
        help="Control whether real spatial coordinates are used in edge features.",
    )
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_device(graph: Data, device: torch.device) -> Data:
    return graph.to(device)


def apply_spatial_ablation(graphs: Sequence[Data], mode: str, seed: int) -> None:
    if mode == "real":
        return
    rng = np.random.default_rng(seed)
    for graph in graphs:
        if mode == "shuffle_pos":
            order = torch.from_numpy(rng.permutation(graph.num_nodes).astype(np.int64))
            graph.pos = graph.pos[order].clone()
        elif mode == "zero_pos":
            graph.pos = torch.zeros_like(graph.pos)
        else:
            raise ValueError(f"Unknown spatial ablation mode: {mode}")


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
        age_bin = next(iter(bins))
        bin_to_groups.setdefault(age_bin, []).append(group)

    split_groups = {"train": [], "val": [], "test": []}
    for age_bin in AGE_BIN_ORDER:
        groups = sorted(bin_to_groups.get(age_bin, []))
        if len(groups) < 3:
            raise ValueError(
                f"Need at least 3 {split_group} groups for age bin {age_bin!r}; found {len(groups)}."
            )
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
        split_name: sorted(
            idx
            for group in groups
            for idx in group_to_indices[group]
        )
        for split_name, groups in split_groups.items()
    }


def split_summary(graphs: Sequence[Data], splits: Dict[str, List[int]], split_group: str) -> Dict:
    summary = {}
    for split_name, indices in splits.items():
        split_graphs_ = [graphs[idx] for idx in indices]
        summary[split_name] = {
            "num_sections": len(indices),
            "sections": [int(graph.section_id.item()) for graph in split_graphs_],
            "codes": [graph.code for graph in split_graphs_],
            "cases": sorted({graph.case for graph in split_graphs_}),
            "groups": sorted({str(getattr(graph, split_group)) for graph in split_graphs_}),
            "age_bins": sorted({graph.age_bin for graph in split_graphs_}),
            "ages": sorted({graph.age for graph in split_graphs_}),
            "num_spots": int(sum(graph.num_nodes for graph in split_graphs_)),
        }
    return summary


def graph_task_level(graphs: Sequence[Data]) -> str:
    levels = {getattr(graph, "task_level", "node") for graph in graphs}
    if len(levels) != 1:
        raise ValueError(f"Mixed task levels are not supported: {levels}")
    return levels.pop()


def num_classes_from_graphs(graphs: Sequence[Data], metadata: Dict) -> int:
    if metadata.get("label_names"):
        return len(metadata["label_names"])
    max_y = max(int(graph.y.max().item()) for graph in graphs)
    max_graph_y = max(int(graph.graph_y.max().item()) for graph in graphs)
    return max(max_y, max_graph_y) + 1


def class_weights(graphs: Sequence[Data], indices: Sequence[int], task_level: str, num_classes: int, device: torch.device) -> torch.Tensor:
    labels: List[int] = []
    for idx in indices:
        graph = graphs[idx]
        if task_level == "node":
            labels.extend(graph.y.cpu().numpy().astype(int).tolist())
        else:
            labels.append(int(graph.graph_y.item()))

    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.from_numpy(weights).to(device)


def effective_class_weights(
    graphs: Sequence[Data],
    indices: Sequence[int],
    task_level: str,
    num_classes: int,
    device: torch.device,
    beta: float,
    max_weight: float,
) -> torch.Tensor:
    labels: List[int] = []
    for idx in indices:
        graph = graphs[idx]
        if task_level == "node":
            labels.extend(graph.y.cpu().numpy().astype(int).tolist())
        else:
            labels.append(int(graph.graph_y.item()))

    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    effective_num = 1.0 - np.power(beta, counts[present])
    weights[present] = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    weights[present] = weights[present] / np.mean(weights[present])
    weights[present] = np.minimum(weights[present], max_weight)
    return torch.from_numpy(weights).to(device)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt).pow(self.gamma) * ce).mean()


def pretrain(
    model: MauronSpatialGNN,
    graphs: Sequence[Data],
    device: torch.device,
    epochs: int,
    mask_rate: float,
    spatial_weight: float,
    learning_rate: float,
    weight_decay: float,
) -> List[float]:
    if epochs <= 0:
        return []
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        random_order = list(range(len(graphs)))
        random.shuffle(random_order)
        for idx in random_order:
            graph = to_device(graphs[idx], device)
            x = graph.x
            mask = torch.rand_like(x) < mask_rate
            if not mask.any():
                continue
            x_masked = x.masked_fill(mask, 0.0)
            prediction = model.reconstruct(graph, x_masked)
            feature_loss = F.mse_loss(prediction[mask], x[mask])
            pos = graph.pos
            pos_target = (pos - pos.mean(dim=0, keepdim=True)) / pos.std(dim=0, keepdim=True).clamp_min(1e-6)
            pos_prediction = model.reconstruct_position(graph, x_masked)
            position_loss = F.mse_loss(pos_prediction, pos_target)
            loss = feature_loss + spatial_weight * position_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
        mean_loss = total_loss / max(len(graphs), 1)
        history.append(mean_loss)
        logger.info("Pretrain epoch %03d/%03d - masked MSE %.4f", epoch, epochs, mean_loss)
    return history


def supervised_loss(
    model: MauronSpatialGNN,
    graph: Data,
    task_level: str,
    criterion: torch.nn.Module,
) -> torch.Tensor:
    if task_level == "node":
        logits = model.node_logits(graph)
        return criterion(logits, graph.y)
    logits = model.graph_logits(graph)
    return criterion(logits, graph.graph_y)


def predict_graphs(
    model: MauronSpatialGNN,
    graphs: Sequence[Data],
    indices: Sequence[int],
    task_level: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for idx in indices:
            graph = to_device(graphs[idx], device)
            if task_level == "node":
                logits = model.node_logits(graph)
                pred = logits.argmax(dim=1).cpu().numpy().astype(int)
                truth = graph.y.cpu().numpy().astype(int)
                y_pred.extend(pred.tolist())
                y_true.extend(truth.tolist())
            else:
                logits = model.graph_logits(graph)
                pred = int(logits.argmax(dim=1).item())
                truth = int(graph.graph_y.item())
                y_pred.append(pred)
                y_true.append(truth)
    return np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)


def metrics_for(y_true: np.ndarray, y_pred: np.ndarray, label_names: Sequence[str]) -> Dict:
    if y_true.size == 0:
        return {"accuracy": None, "macro_f1": None, "num_examples": 0}
    labels = list(range(len(label_names))) if label_names else sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    target_names = list(label_names) if label_names else [str(label) for label in labels]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "num_examples": int(y_true.size),
        "label_counts": {str(k): int(v) for k, v in Counter(y_true.tolist()).items()},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        ),
    }


def fine_tune(
    model: MauronSpatialGNN,
    graphs: Sequence[Data],
    splits: Dict[str, List[int]],
    task_level: str,
    label_names: Sequence[str],
    device: torch.device,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    class_weight_beta: float,
    max_class_weight: float,
    focal_gamma: float,
) -> Tuple[Dict, Dict[str, torch.Tensor]]:
    weights = effective_class_weights(
        graphs,
        splits["train"],
        task_level,
        len(label_names),
        device,
        class_weight_beta,
        max_class_weight,
    )
    criterion = FocalLoss(weights, focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_val_f1 = -1.0
    stale_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        train_order = list(splits["train"])
        random.shuffle(train_order)
        for idx in train_order:
            graph = to_device(graphs[idx], device)
            loss = supervised_loss(model, graph, task_level, criterion)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())

        train_true, train_pred = predict_graphs(model, graphs, splits["train"], task_level, device)
        val_true, val_pred = predict_graphs(model, graphs, splits["val"], task_level, device)
        train_metrics = metrics_for(train_true, train_pred, label_names)
        val_metrics = metrics_for(val_true, val_pred, label_names)
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(splits["train"]), 1),
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(epoch_record)
        logger.info(
            "Fine-tune epoch %03d/%03d - loss %.4f train acc %.3f val acc %.3f val F1 %.3f",
            epoch,
            epochs,
            epoch_record["train_loss"],
            train_metrics["accuracy"] or 0.0,
            val_metrics["accuracy"] or 0.0,
            val_metrics["macro_f1"] or 0.0,
        )

        current_val = val_metrics["macro_f1"] if val_metrics["macro_f1"] is not None else -1.0
        if current_val > best_val_f1:
            best_val_f1 = current_val
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                logger.info("Early stopping after %d stale epochs.", stale_epochs)
                break

    model.load_state_dict(best_state)
    split_metrics = {}
    for split_name, indices in splits.items():
        y_true, y_pred = predict_graphs(model, graphs, indices, task_level, device)
        split_metrics[split_name] = metrics_for(y_true, y_pred, label_names)

    return {"history": history, "split_metrics": split_metrics, "best_val_macro_f1": best_val_f1}, best_state


def export_embeddings(
    model: MauronSpatialGNN,
    graphs: Sequence[Data],
    splits: Dict[str, List[int]],
    output_dir: Path,
    device: torch.device,
) -> Dict:
    model.eval()
    section_records = []
    section_embeddings = []
    split_by_index = {idx: name for name, values in splits.items() for idx in values}

    with torch.no_grad():
        for idx, graph in enumerate(graphs):
            graph_device = to_device(graph, device)
            embedding = model.section_embedding(graph_device).squeeze(0).cpu().numpy().astype(np.float32)
            section_embeddings.append(embedding)
            section_records.append(
                {
                    "graph_index": idx,
                    "split": split_by_index.get(idx, "unsplit"),
                    "section": int(graph.section_id.item()),
                    "code": graph.code,
                    "case": graph.case,
                    "age": graph.age,
                    "age_weeks": float(graph.age_weeks.item()),
                    "age_bin": graph.age_bin,
                    "chamber_combo": graph.chamber_combo,
                    "num_spots": int(graph.num_nodes),
                }
            )

    section_array = np.vstack(section_embeddings)
    section_metadata = pd.DataFrame(section_records)
    np.save(output_dir / "section_embeddings.npy", section_array)
    section_metadata.to_csv(output_dir / "section_embedding_metadata.csv", index=False)

    case_embeddings = []
    case_records = []
    for case, indices in section_metadata.groupby("case").groups.items():
        idx = np.asarray(list(indices), dtype=int)
        case_embeddings.append(section_array[idx].mean(axis=0))
        subset = section_metadata.iloc[idx]
        case_records.append(
            {
                "case": case,
                "num_sections": int(len(idx)),
                "sections": ";".join(str(value) for value in subset["section"].tolist()),
                "codes": ";".join(subset["code"].tolist()),
                "age_bins": ";".join(sorted(set(subset["age_bin"].tolist()))),
                "chamber_combos": ";".join(sorted(set(subset["chamber_combo"].tolist()))),
                "splits": ";".join(sorted(set(subset["split"].tolist()))),
            }
        )
    case_array = np.vstack(case_embeddings)
    case_metadata = pd.DataFrame(case_records)
    np.save(output_dir / "case_embeddings.npy", case_array)
    case_metadata.to_csv(output_dir / "case_embedding_metadata.csv", index=False)

    return {
        "section_embeddings": {
            "path": str(output_dir / "section_embeddings.npy"),
            "shape": list(section_array.shape),
            "metadata_path": str(output_dir / "section_embedding_metadata.csv"),
        },
        "case_embeddings": {
            "path": str(output_dir / "case_embeddings.npy"),
            "shape": list(case_array.shape),
            "metadata_path": str(output_dir / "case_embedding_metadata.csv"),
        },
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"mauron_spatial_gnn_results_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Output directory: %s", output_dir)
    logger.info("Device: %s", device)

    build_config = MauronBuildConfig(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        num_genes=args.num_genes,
        k_neighbors=args.k_neighbors,
        label_mode=args.label_mode,
        min_gene_spots=args.min_gene_spots,
        max_sections=args.max_sections,
    )
    dataset = MauronVisiumGraphDataset(build_config)
    graphs, metadata = dataset.load_or_build(force_rebuild=args.force_rebuild)
    apply_spatial_ablation(graphs, args.spatial_ablation, args.seed)
    task_level = graph_task_level(graphs)
    label_names = metadata["label_names"] or metadata["graph_label_names"]
    num_classes = num_classes_from_graphs(graphs, metadata)
    if len(label_names) != num_classes:
        label_names = [str(idx) for idx in range(num_classes)]

    splits = split_graphs(graphs, args.split_group, args.seed)
    logger.info("Grouped split summary: %s", json.dumps(split_summary(graphs, splits, args.split_group), indent=2))

    model = MauronSpatialGNN(
        input_dim=graphs[0].x.shape[1],
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_classes=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    pretrain_history = pretrain(
        model=model,
        graphs=graphs,
        device=device,
        epochs=args.pretrain_epochs,
        mask_rate=args.mask_rate,
        spatial_weight=args.spatial_pretrain_weight,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    finetune_results, best_state = fine_tune(
        model=model,
        graphs=graphs,
        splits=splits,
        task_level=task_level,
        label_names=label_names,
        device=device,
        epochs=args.finetune_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        class_weight_beta=args.class_weight_beta,
        max_class_weight=args.max_class_weight,
        focal_gamma=args.focal_gamma,
    )

    model_path = output_dir / "best_mauron_spatial_gnn.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "model_config": {
                "input_dim": graphs[0].x.shape[1],
                "hidden_dim": args.hidden_dim,
                "embedding_dim": args.embedding_dim,
                "num_classes": num_classes,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "edge_features": "direction_x,direction_y,scaled_distance,inverse_distance",
                "spatial_pretrain_weight": args.spatial_pretrain_weight,
            },
            "label_names": label_names,
            "task_level": task_level,
        },
        model_path,
    )
    embedding_manifest = export_embeddings(model, graphs, splits, output_dir, device)

    results = {
        "created_at": timestamp,
        "script": str(Path(__file__).resolve()),
        "args": vars(args),
        "dataset_metadata": metadata,
        "task_level": task_level,
        "num_classes": num_classes,
        "label_names": label_names,
        "split_group": args.split_group,
        "splits": split_summary(graphs, splits, args.split_group),
        "pretrain_history": pretrain_history,
        "finetune": finetune_results,
        "model_path": str(model_path),
        "embeddings": embedding_manifest,
        "validation_notes": [
            "Supervised metrics are evaluated on held-out sections grouped by Case or Code.",
            "No random spot split is used.",
            "Self-supervised pretraining uses all sections and no labels.",
        ],
    }
    results_path = output_dir / "mauron_spatial_gnn_results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    logger.info("Saved results to %s", results_path)
    logger.info("Test metrics: %s", json.dumps(finetune_results["split_metrics"]["test"], indent=2)[:2000])


if __name__ == "__main__":
    main()

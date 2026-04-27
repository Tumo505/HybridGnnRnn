#!/usr/bin/env python3
"""
Train a developmental RNN on Mauron Visium spots.

This RNN uses Mauron's fetal developmental axis rather than spatial-neighborhood
order. It is still cross-sectional, not same-heart longitudinal tracking.

For each spot, the input sequence is ordered by fetal age prototypes computed
from training cases only. At each time step, the model sees how the spot's
expression relates to the train-set prototype for that fetal age. By default,
the target is the same spot-level deconvolution label used by the fresh GNN.
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.mauron_visium_processor import MauronBuildConfig, MauronVisiumGraphDataset  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("mauron_developmental_rnn")

AGE_BIN_ORDER = ["early_w6_w7_5", "mid_w8_w9", "late_w10_w12"]


class DevelopmentalSpotDataset(Dataset):
    def __init__(
        self,
        graphs: Sequence,
        graph_indices: Sequence[int],
        prototypes: torch.Tensor,
        target: str,
        age_label_to_id: Dict[str, int],
    ):
        self.graphs = graphs
        self.graph_indices = list(graph_indices)
        self.prototypes = prototypes
        self.target = target
        self.age_label_to_id = age_label_to_id
        self.lookup: List[Tuple[int, int]] = []
        for graph_idx in self.graph_indices:
            for node_idx in range(graphs[graph_idx].num_nodes):
                self.lookup.append((graph_idx, node_idx))

    def __len__(self) -> int:
        return len(self.lookup)

    def __getitem__(self, index: int):
        graph_idx, node_idx = self.lookup[index]
        graph = self.graphs[graph_idx]
        x = graph.x[node_idx]
        repeated_x = x.unsqueeze(0).expand_as(self.prototypes)
        signed_delta = x.unsqueeze(0) - self.prototypes
        abs_delta = signed_delta.abs()
        sequence = torch.cat([repeated_x, self.prototypes, signed_delta, abs_delta], dim=1)
        if self.target == "age_bin":
            label = self.age_label_to_id[graph.age_bin]
        else:
            label = int(graph.y[node_idx].item())
        age_label = self.age_label_to_id[graph.age_bin]
        chamber = graph.chamber_multi_hot.squeeze(0).float()
        return (
            sequence,
            torch.tensor(label, dtype=torch.long),
            torch.tensor(age_label, dtype=torch.long),
            chamber,
        )


class DevelopmentalPrototypeGRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_age_bins: int,
        chamber_dim: int,
        projection_dim: int = 96,
        hidden_dim: int = 96,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(projection_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.main_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )
        self.age_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_age_bins),
        )
        self.chamber_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, chamber_dim),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.project(x)
        _, hidden = self.gru(z)
        final = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return {
            "main": self.main_classifier(final),
            "age": self.age_classifier(final),
            "chamber": self.chamber_classifier(final),
        }


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 1.5):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt).pow(self.gamma) * ce).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=(
            "data/New/Mauron_spatial_dynamics_part_a/"
            "Spatial dynamics of the developing human heart, pa"
        ),
    )
    parser.add_argument("--cache-dir", default="cache/mauron_visium_graphs")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-genes", type=int, default=512)
    parser.add_argument("--target", default="deconv_label", choices=["deconv_label", "age_bin"])
    parser.add_argument("--split-group", default="case", choices=["case", "code"])
    parser.add_argument("--projection-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--class-weight-beta", type=float, default=0.999)
    parser.add_argument("--max-class-weight", type=float, default=8.0)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--age-loss-weight", type=float, default=0.15)
    parser.add_argument("--chamber-loss-weight", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_graphs(graphs: Sequence, split_group: str, seed: int) -> Dict[str, List[int]]:
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


def split_summary(graphs: Sequence, splits: Dict[str, List[int]], split_group: str) -> Dict:
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


def standardize_from_train(graphs: Sequence, train_indices: Sequence[int]) -> None:
    train_x = torch.cat([graphs[idx].x for idx in train_indices], dim=0)
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-5)
    for graph in graphs:
        graph.x = ((graph.x - mean) / std).float()


def build_age_prototypes(graphs: Sequence, train_indices: Sequence[int]) -> Tuple[torch.Tensor, List[float], Dict[str, List[int]]]:
    train_ages = sorted({float(graphs[idx].age_weeks.item()) for idx in train_indices})
    prototypes = []
    members: Dict[str, List[int]] = {}
    for age in train_ages:
        age_graphs = [idx for idx in train_indices if float(graphs[idx].age_weeks.item()) == age]
        members[str(age)] = [int(graphs[idx].section_id.item()) for idx in age_graphs]
        age_x = torch.cat([graphs[idx].x for idx in age_graphs], dim=0)
        prototypes.append(age_x.mean(dim=0))
    return torch.stack(prototypes, dim=0).float(), train_ages, members


def class_weights(
    graphs: Sequence,
    train_indices: Sequence[int],
    target: str,
    num_classes: int,
    age_label_to_id: Dict[str, int],
    device: torch.device,
    beta: float,
    max_weight: float,
) -> torch.Tensor:
    labels = []
    for idx in train_indices:
        if target == "age_bin":
            labels.extend([age_label_to_id[graphs[idx].age_bin]] * graphs[idx].num_nodes)
        else:
            labels.extend(graphs[idx].y.cpu().numpy().astype(int).tolist())
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    effective_num = 1.0 - np.power(beta, counts[present])
    weights[present] = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    weights[present] = weights[present] / np.mean(weights[present])
    weights[present] = np.minimum(weights[present], max_weight)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for sequence, label, _age_label, _chamber in loader:
            logits = model(sequence.to(device).float())["main"]
            pred = logits.argmax(dim=1).cpu().numpy().astype(int)
            y_pred.extend(pred.tolist())
            y_true.extend(label.numpy().astype(int).tolist())
    return np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)


def metrics_for(y_true: np.ndarray, y_pred: np.ndarray, label_names: Sequence[str]) -> Dict:
    labels = list(range(len(label_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "num_examples": int(len(y_true)),
        "label_counts": {label_names[label]: int(count) for label, count in Counter(y_true.tolist()).items()},
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
    output_dir = Path(args.output_dir or f"mauron_developmental_rnn_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    build_config = MauronBuildConfig(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        num_genes=args.num_genes,
        k_neighbors=8,
        label_mode="deconv_hl_argmax",
    )
    dataset = MauronVisiumGraphDataset(build_config)
    graphs, metadata = dataset.load_or_build(force_rebuild=args.force_rebuild)
    splits = split_graphs(graphs, args.split_group, args.seed)
    standardize_from_train(graphs, splits["train"])
    prototypes, train_ages, prototype_members = build_age_prototypes(graphs, splits["train"])
    age_label_to_id = {label: idx for idx, label in enumerate(AGE_BIN_ORDER)}
    if args.target == "age_bin":
        label_names = AGE_BIN_ORDER
    else:
        label_names = list(graphs[0].label_names)
    num_classes = len(label_names)
    chamber_dim = int(graphs[0].chamber_multi_hot.shape[1])

    datasets = {
        split_name: DevelopmentalSpotDataset(graphs, indices, prototypes, args.target, age_label_to_id)
        for split_name, indices in splits.items()
    }
    loaders = {
        split_name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split_name == "train"),
            num_workers=0,
        )
        for split_name, dataset in datasets.items()
    }

    model = DevelopmentalPrototypeGRU(
        input_dim=prototypes.shape[1] * 4,
        num_classes=num_classes,
        num_age_bins=len(AGE_BIN_ORDER),
        chamber_dim=chamber_dim,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    main_criterion = FocalLoss(
        alpha=class_weights(
            graphs,
            splits["train"],
            args.target,
            num_classes,
            age_label_to_id,
            device,
            args.class_weight_beta,
            args.max_class_weight,
        ),
        gamma=args.focal_gamma,
    )
    age_criterion = nn.CrossEntropyLoss()
    chamber_criterion = nn.BCEWithLogitsLoss()

    logger.info("Output directory: %s", output_dir)
    logger.info("Device: %s", device)
    logger.info("Target: %s (%d classes)", args.target, num_classes)
    logger.info(
        "Loss: focal gamma %.2f, effective class beta %.4f, aux age %.2f, aux chamber %.2f",
        args.focal_gamma,
        args.class_weight_beta,
        args.age_loss_weight,
        args.chamber_loss_weight,
    )
    logger.info("Train age prototypes: %s", train_ages)
    logger.info("Grouped split summary: %s", json.dumps(split_summary(graphs, splits, args.split_group), indent=2))

    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_val_macro_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_main_loss = 0.0
        total_age_loss = 0.0
        total_chamber_loss = 0.0
        batches = 0
        for sequence, label, age_label, chamber in loaders["train"]:
            sequence = sequence.to(device).float()
            label = label.to(device).long()
            age_label = age_label.to(device).long()
            chamber = chamber.to(device).float()
            optimizer.zero_grad()
            outputs = model(sequence)
            main_loss = main_criterion(outputs["main"], label)
            age_loss = age_criterion(outputs["age"], age_label)
            chamber_loss = chamber_criterion(outputs["chamber"], chamber)
            loss = main_loss + args.age_loss_weight * age_loss + args.chamber_loss_weight * chamber_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
            total_main_loss += float(main_loss.item())
            total_age_loss += float(age_loss.item())
            total_chamber_loss += float(chamber_loss.item())
            batches += 1

        val_true, val_pred = evaluate(model, loaders["val"], device)
        val_metrics = metrics_for(val_true, val_pred, label_names)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(batches, 1),
            "train_main_loss": total_main_loss / max(batches, 1),
            "train_age_loss": total_age_loss / max(batches, 1),
            "train_chamber_loss": total_chamber_loss / max(batches, 1),
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
    for split_name, loader in loaders.items():
        true, pred = evaluate(model, loader, device)
        split_metrics[split_name] = metrics_for(true, pred, label_names)

    model_path = output_dir / "best_mauron_developmental_rnn.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "model_config": {
                "input_dim": int(prototypes.shape[1] * 4),
                "sequence_length": int(prototypes.shape[0]),
                "num_classes": num_classes,
                "num_age_bins": len(AGE_BIN_ORDER),
                "chamber_dim": chamber_dim,
                "projection_dim": args.projection_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
            },
            "class_names": list(label_names),
            "age_bin_order": AGE_BIN_ORDER,
        },
        model_path,
    )

    results = {
        "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "task": f"developmental_temporal_rnn_{args.target}_classification",
        "model": "spot_level_developmental_age_prototype_multitask_gru",
        "source_dataset": "Mauron developing human heart Visium part a",
        "input_strategy": (
            "Each spot is a chronological sequence of expression differences to train-only fetal-age prototypes. "
            "The default target is the same deconvolution label task as the fresh GNN. "
            "Training uses focal loss with capped effective-number class weights plus auxiliary age-bin and chamber supervision. "
            "This uses cross-sectional developmental age, not true same-case longitudinal tracking."
        ),
        "dataset_metadata": metadata,
        "age_bin_order": AGE_BIN_ORDER,
        "label_names": list(label_names),
        "train_age_prototypes": train_ages,
        "prototype_members": prototype_members,
        "split_group": args.split_group,
        "splits": split_summary(graphs, splits, args.split_group),
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "history": history,
        "split_metrics": split_metrics,
        "model_path": str(model_path),
    }
    results_path = output_dir / "mauron_developmental_rnn_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Saved results to %s", results_path)
    logger.info("Test metrics: %s", json.dumps(split_metrics["test"], indent=2)[:2000])


if __name__ == "__main__":
    main()

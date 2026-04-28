#!/usr/bin/env python3
"""Fair temporal baselines for the multi-atlas Geneformer benchmark.

The foundation model is trained on Geneformer-tokenized GSE175634 + GSE202398
cells. This script reuses the exact same split and labels for lightweight
baselines:

- rnn: token embedding + bi-GRU over the ranked gene-token sequence
- gnn: expression-neighborhood graph context from token-hash features
- hybrid: RNN sequence encoder fused with graph-neighborhood context

The GNN here is intentionally an expression-neighborhood GNN, not a spatial
GNN, because GSE175634 and GSE202398 are scRNA-seq datasets without real
spatial coordinates.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, Dataset as TorchDataset, WeightedRandomSampler

from datasets import concatenate_datasets, load_from_disk

from train_geneformer_multiatlas_multitask import (
    DAY_ORDER,
    DAY_TO_NUMERIC,
    STATE_ORDER,
    STAGE_MAP,
    set_seed,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("multiatlas_baselines")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["rnn", "gnn", "hybrid"], default="rnn")
    parser.add_argument("--gse175634-tokenized", default="foundation_model_data/geneformer/tokenized/gse175634_geneformer.dataset")
    parser.add_argument(
        "--gse202398-tokenized",
        default="foundation_model_data/geneformer_external_unseen/gse202398_tokenized/gse202398_filtered_cellplex_matched_days.dataset",
    )
    parser.add_argument("--output-dir", default="foundation_multiatlas_baselines")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--holdout-gse202398-field", choices=["line_id", "sample_id"], default="line_id")
    parser.add_argument("--holdout-gse202398-value", default="LMNA")
    parser.add_argument("--gse175634-val-line-fraction", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--token-embedding-dim", type=int, default=96)
    parser.add_argument("--rnn-hidden-dim", type=int, default=128)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--graph-hidden-dim", type=int, default=192)
    parser.add_argument("--graph-neighbors", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--day-loss-weight", type=float, default=1.0)
    parser.add_argument("--stage-loss-weight", type=float, default=1.0)
    parser.add_argument("--state-loss-weight", type=float, default=0.5)
    parser.add_argument("--regression-loss-weight", type=float, default=0.6)
    parser.add_argument("--ordinal-loss-weight", type=float, default=0.5)
    parser.add_argument("--domain-loss-weight", type=float, default=0.0)
    parser.add_argument("--class-weighting", choices=["none", "inverse_sqrt"], default="inverse_sqrt")
    parser.add_argument("--balance-sampler", action="store_true")
    parser.add_argument("--gse202398-sampler-boost", type=float, default=3.0)
    parser.add_argument("--max-gse175634-train-cells", type=int, default=None)
    parser.add_argument("--max-gse202398-train-cells", type=int, default=None)
    parser.add_argument("--max-train-cells", type=int, default=None)
    parser.add_argument("--max-eval-cells", type=int, default=None)
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


@dataclass
class BaselineBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    graph_features: torch.Tensor
    graph_neighbor_features: torch.Tensor
    day_label: torch.Tensor
    stage_label: torch.Tensor
    state_label: torch.Tensor
    day_regression: torch.Tensor
    domain_label: torch.Tensor
    metadata: dict[str, list[Any]]


def deterministic_line_split(lines: list[str], val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    lines = sorted(set(str(line) for line in lines))
    rng = random.Random(seed)
    rng.shuffle(lines)
    n_val = max(1, int(round(len(lines) * val_fraction))) if lines else 0
    return set(lines[n_val:]), set(lines[:n_val])


def select_indices(indices: np.ndarray, max_cells: int | None, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if max_cells is None or len(indices) <= max_cells:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_cells, replace=False))


def label_maps() -> dict[str, dict[int, str]]:
    stage_order = list(dict.fromkeys(STAGE_MAP.values()))
    return {
        "day": {idx: label for idx, label in enumerate(DAY_ORDER)},
        "stage": {idx: label for idx, label in enumerate(stage_order)},
        "state": {idx: label for idx, label in enumerate(STATE_ORDER)},
        "domain": {0: "GSE175634", 1: "GSE202398"},
    }


def add_labels(row: dict[str, Any], source: str) -> dict[str, Any]:
    day_to_id = {label: idx for idx, label in enumerate(DAY_ORDER)}
    stage_to_id = {label: idx for idx, label in enumerate(list(dict.fromkeys(STAGE_MAP.values())))}
    state_to_id = {label: idx for idx, label in enumerate(STATE_ORDER)}
    day = str(row.get("diffday", "unknown"))
    day_num = int(str(day).replace("day", "")) if str(day).startswith("day") else -1
    state = str(row.get("cell_state", "UNK"))
    row["source_dataset"] = source
    row["day_label"] = day_to_id.get(day, -100)
    row["stage_text"] = STAGE_MAP.get(day, "unknown")
    row["stage_label"] = stage_to_id.get(row["stage_text"], -100)
    row["state_label"] = state_to_id.get(state, state_to_id["UNK"])
    row["day_regression"] = float(np.clip(day_num / 15.0, 0.0, 1.0)) if day_num >= 0 else -1.0
    row["domain_label"] = 0 if source == "GSE175634" else 1
    return row


def prepare_splits_fast(args: argparse.Namespace) -> tuple[Any, Any, Any, dict]:
    gse175 = load_from_disk(args.gse175634_tokenized)
    gse202 = load_from_disk(args.gse202398_tokenized)
    allowed_days = set(DAY_ORDER)

    gse175_days = np.asarray(gse175["diffday"], dtype=object)
    gse175_lines = np.asarray(gse175["line_id"], dtype=object)
    gse175_valid = np.flatnonzero(np.isin(gse175_days, list(allowed_days)))
    train_lines, val_lines = deterministic_line_split([str(gse175_lines[idx]) for idx in gse175_valid], args.gse175634_val_line_fraction, args.seed)
    gse175_train_idx = gse175_valid[np.isin(gse175_lines[gse175_valid].astype(str), list(train_lines))]
    gse175_val_idx = gse175_valid[np.isin(gse175_lines[gse175_valid].astype(str), list(val_lines))]

    gse202_days = np.asarray(gse202["diffday"], dtype=object)
    holdout_values = np.asarray(gse202[args.holdout_gse202398_field], dtype=object).astype(str)
    gse202_valid = np.flatnonzero(np.isin(gse202_days, list(allowed_days)))
    holdout_value = str(args.holdout_gse202398_value)
    gse202_test_idx = gse202_valid[holdout_values[gse202_valid] == holdout_value]
    gse202_train_idx = gse202_valid[holdout_values[gse202_valid] != holdout_value]
    if len(gse202_test_idx) == 0:
        raise ValueError(f"No GSE202398 cells matched {args.holdout_gse202398_field}={holdout_value!r}")
    if len(gse202_train_idx) == 0:
        raise ValueError(f"Holding out {args.holdout_gse202398_field}={holdout_value!r} leaves no GSE202398 training cells.")

    gse175_train_idx = select_indices(gse175_train_idx, args.max_gse175634_train_cells, args.seed + 10)
    gse202_train_idx = select_indices(gse202_train_idx, args.max_gse202398_train_cells, args.seed + 20)
    gse175_val_idx = select_indices(gse175_val_idx, args.max_eval_cells, args.seed + 1)
    gse202_test_idx = select_indices(gse202_test_idx, args.max_eval_cells, args.seed + 2)

    gse175_train = gse175.select(gse175_train_idx.tolist()).map(lambda row: add_labels(row, "GSE175634"), desc="Label GSE175634 train")
    gse202_train = gse202.select(gse202_train_idx.tolist()).map(lambda row: add_labels(row, "GSE202398"), desc="Label GSE202398 train")
    train = concatenate_datasets([gse175_train, gse202_train]).shuffle(seed=args.seed)
    if args.max_train_cells is not None and len(train) > args.max_train_cells:
        train = train.select(select_indices(np.arange(len(train)), args.max_train_cells, args.seed).tolist())
    val = gse175.select(gse175_val_idx.tolist()).map(lambda row: add_labels(row, "GSE175634"), desc="Label GSE175634 val")
    test = gse202.select(gse202_test_idx.tolist()).map(lambda row: add_labels(row, "GSE202398"), desc="Label GSE202398 test")

    maps = label_maps()
    split_summary = {
        "train_cells": len(train),
        "val_cells_gse175634_lines": len(val),
        "test_cells_gse202398_holdout": len(test),
        "gse202398_holdout": {"field": args.holdout_gse202398_field, "value": holdout_value},
        "gse175634_train_lines": sorted(train_lines),
        "gse175634_val_lines": sorted(val_lines),
        "train_day_counts": pd.Series(train["diffday"]).value_counts().to_dict(),
        "test_day_counts": pd.Series(test["diffday"]).value_counts().to_dict(),
        "train_source_counts": pd.Series(train["source_dataset"]).value_counts().to_dict(),
        "test_source_counts": pd.Series(test["source_dataset"]).value_counts().to_dict(),
        "label_maps": maps,
    }
    return train, val, test, split_summary


class IndexedCellDataset(TorchDataset):
    def __init__(self, hf_dataset: Any, graph_features: np.ndarray, neighbor_features: np.ndarray):
        self.hf_dataset = hf_dataset
        self.graph_features = graph_features.astype(np.float32, copy=False)
        self.neighbor_features = neighbor_features.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = dict(self.hf_dataset[int(idx)])
        row["_row_index"] = int(idx)
        return row


def collate_baseline(examples: list[dict[str, Any]], graph_features: np.ndarray, neighbor_features: np.ndarray, max_length: int, vocab_size: int) -> BaselineBatch:
    lengths = [min(len(example["input_ids"]), max_length) for example in examples]
    max_len = max(lengths)
    input_ids = torch.zeros((len(examples), max_len), dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.float32)
    for row_idx, example in enumerate(examples):
        ids = torch.tensor(example["input_ids"][: lengths[row_idx]], dtype=torch.long).clamp_(0, vocab_size - 1)
        input_ids[row_idx, : lengths[row_idx]] = ids
        attention_mask[row_idx, : lengths[row_idx]] = 1.0
    source_to_domain = {"GSE175634": 0, "GSE202398": 1}
    row_indices = [int(example["_row_index"]) for example in examples]
    metadata_cols = ["source_dataset", "line_id", "sample_id", "diffday", "cell_state", "stage_text"]
    return BaselineBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        graph_features=torch.tensor(graph_features[row_indices], dtype=torch.float32),
        graph_neighbor_features=torch.tensor(neighbor_features[row_indices], dtype=torch.float32),
        day_label=torch.tensor([example["day_label"] for example in examples], dtype=torch.long),
        stage_label=torch.tensor([example["stage_label"] for example in examples], dtype=torch.long),
        state_label=torch.tensor([example["state_label"] for example in examples], dtype=torch.long),
        day_regression=torch.tensor([example["day_regression"] for example in examples], dtype=torch.float32),
        domain_label=torch.tensor([source_to_domain[str(example["source_dataset"])] for example in examples], dtype=torch.long),
        metadata={col: [example.get(col) for example in examples] for col in metadata_cols},
    )


def token_hash_features(dataset: Any, feature_dim: int, max_length: int) -> np.ndarray:
    features = np.zeros((len(dataset), feature_dim), dtype=np.float32)
    for idx, row in enumerate(dataset):
        ids = np.asarray(row["input_ids"][:max_length], dtype=np.int64)
        if ids.size == 0:
            continue
        bins = np.mod(ids, feature_dim)
        np.add.at(features[idx], bins, 1.0)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-6)


def graph_neighbor_features(train_features: np.ndarray, target_features: np.ndarray, n_neighbors: int) -> np.ndarray:
    n_neighbors = min(max(1, n_neighbors), len(train_features))
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn_model.fit(train_features)
    indices = nn_model.kneighbors(target_features, return_distance=False)
    return train_features[indices].mean(axis=1).astype(np.float32)


def class_weights_from_dataset(dataset: Any, column: str, num_classes: int, mode: str, device: torch.device) -> torch.Tensor | None:
    if mode == "none":
        return None
    labels = np.asarray(dataset[column], dtype=np.int64)
    labels = labels[labels >= 0]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def sampler_weights_from_dataset(dataset: Any, gse202398_boost: float) -> list[float]:
    day_counts = pd.Series(dataset["day_label"]).value_counts().to_dict()
    sources = dataset["source_dataset"]
    labels = dataset["day_label"]
    weights = []
    for source, label in zip(sources, labels):
        day_weight = 1.0 / math.sqrt(float(day_counts.get(label, 1)))
        source_weight = gse202398_boost if source == "GSE202398" else 1.0
        weights.append(day_weight * source_weight)
    return weights


def move_batch(batch: BaselineBatch, device: torch.device) -> BaselineBatch:
    return BaselineBatch(
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        graph_features=batch.graph_features.to(device),
        graph_neighbor_features=batch.graph_neighbor_features.to(device),
        day_label=batch.day_label.to(device),
        stage_label=batch.stage_label.to(device),
        state_label=batch.state_label.to(device),
        day_regression=batch.day_regression.to(device),
        domain_label=batch.domain_label.to(device),
        metadata=batch.metadata,
    )


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


class PredictionHeads(nn.Module):
    def __init__(self, hidden_dim: int, num_day: int, num_stage: int, num_state: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.day_head = nn.Linear(hidden_dim, num_day)
        self.stage_head = nn.Linear(hidden_dim, num_stage)
        self.state_head = nn.Linear(hidden_dim, num_state)
        self.regression_head = nn.Linear(hidden_dim, 1)
        self.domain_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, embedding: torch.Tensor, grl_lambda: float) -> dict[str, torch.Tensor]:
        embedding = self.dropout(embedding)
        reversed_embedding = GradientReverse.apply(embedding, grl_lambda)
        return {
            "embedding": embedding,
            "day_logits": self.day_head(embedding),
            "stage_logits": self.stage_head(embedding),
            "state_logits": self.state_head(embedding),
            "day_value": self.regression_head(embedding).squeeze(-1),
            "domain_logits": self.domain_head(reversed_embedding),
        }


class RNNBaseline(nn.Module):
    def __init__(self, args: argparse.Namespace, num_day: int, num_stage: int, num_state: int):
        super().__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.token_embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(
            args.token_embedding_dim,
            args.rnn_hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=1,
            dropout=0.0,
        )
        self.proj = nn.Sequential(nn.LayerNorm(args.rnn_hidden_dim * 2), nn.Linear(args.rnn_hidden_dim * 2, args.rnn_hidden_dim * 2), nn.ReLU())
        self.heads = PredictionHeads(args.rnn_hidden_dim * 2, num_day, num_stage, num_state, args.dropout)

    def encode(self, batch: BaselineBatch) -> torch.Tensor:
        embedded = self.embedding(batch.input_ids)
        output, _ = self.rnn(embedded)
        mask = batch.attention_mask.unsqueeze(-1)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.proj(pooled)

    def forward(self, batch: BaselineBatch, grl_lambda: float) -> dict[str, torch.Tensor]:
        return self.heads(self.encode(batch), grl_lambda)


class GraphContextEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor, neighbor_features: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([features, neighbor_features, features - neighbor_features], dim=1))


class GNNBaseline(nn.Module):
    def __init__(self, args: argparse.Namespace, num_day: int, num_stage: int, num_state: int):
        super().__init__()
        self.encoder = GraphContextEncoder(args.feature_dim, args.graph_hidden_dim, args.dropout)
        self.heads = PredictionHeads(args.graph_hidden_dim, num_day, num_stage, num_state, args.dropout)

    def forward(self, batch: BaselineBatch, grl_lambda: float) -> dict[str, torch.Tensor]:
        return self.heads(self.encoder(batch.graph_features, batch.graph_neighbor_features), grl_lambda)


class HybridBaseline(nn.Module):
    def __init__(self, args: argparse.Namespace, num_day: int, num_stage: int, num_state: int):
        super().__init__()
        self.rnn_encoder = RNNBaseline(args, num_day, num_stage, num_state)
        self.graph_encoder = GraphContextEncoder(args.feature_dim, args.graph_hidden_dim, args.dropout)
        fusion_dim = args.rnn_hidden_dim * 2 + args.graph_hidden_dim
        self.fusion = nn.Sequential(nn.Linear(fusion_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU())
        self.heads = PredictionHeads(fusion_dim, num_day, num_stage, num_state, args.dropout)

    def forward(self, batch: BaselineBatch, grl_lambda: float) -> dict[str, torch.Tensor]:
        rnn_embedding = self.rnn_encoder.encode(batch)
        graph_embedding = self.graph_encoder(batch.graph_features, batch.graph_neighbor_features)
        return self.heads(self.fusion(torch.cat([rnn_embedding, graph_embedding], dim=1)), grl_lambda)


def compute_loss(outputs: dict[str, torch.Tensor], batch: BaselineBatch, losses: dict[str, nn.Module], args: argparse.Namespace, day_values: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    day_loss = losses["day"](outputs["day_logits"], batch.day_label)
    stage_loss = losses["stage"](outputs["stage_logits"], batch.stage_label)
    state_loss = losses["state"](outputs["state_logits"], batch.state_label)
    reg_loss = losses["regression"](outputs["day_value"], batch.day_regression)
    probs = torch.softmax(outputs["day_logits"], dim=1)
    expected_day = (probs * day_values.to(probs.device)).sum(dim=1)
    ordinal_loss = losses["regression"](expected_day, batch.day_regression)
    domain_loss = losses["domain"](outputs["domain_logits"], batch.domain_label)
    total = (
        args.day_loss_weight * day_loss
        + args.stage_loss_weight * stage_loss
        + args.state_loss_weight * state_loss
        + args.regression_loss_weight * reg_loss
        + args.ordinal_loss_weight * ordinal_loss
        + args.domain_loss_weight * domain_loss
    )
    return total, {
        "day_loss": float(day_loss.detach().cpu()),
        "stage_loss": float(stage_loss.detach().cpu()),
        "state_loss": float(state_loss.detach().cpu()),
        "regression_loss": float(reg_loss.detach().cpu()),
        "ordinal_loss": float(ordinal_loss.detach().cpu()),
        "domain_loss": float(domain_loss.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }


def evaluate(model: nn.Module, loader: DataLoader, losses: dict[str, nn.Module], args: argparse.Namespace, device: torch.device, label_maps: dict, split_name: str) -> tuple[dict, pd.DataFrame]:
    model.eval()
    day_values = torch.tensor([DAY_TO_NUMERIC[label] / 15.0 for label in DAY_ORDER], dtype=torch.float32, device=device)
    rows = []
    loss_rows = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(batch, grl_lambda=0.0)
            _, loss_parts = compute_loss(outputs, batch, losses, args, day_values)
            loss_rows.append(loss_parts)
            day_pred = outputs["day_logits"].argmax(dim=1).detach().cpu().numpy()
            stage_pred = outputs["stage_logits"].argmax(dim=1).detach().cpu().numpy()
            state_pred = outputs["state_logits"].argmax(dim=1).detach().cpu().numpy()
            domain_pred = outputs["domain_logits"].argmax(dim=1).detach().cpu().numpy()
            reg_pred = outputs["day_value"].detach().cpu().numpy()
            for idx in range(len(day_pred)):
                rows.append(
                    {
                        "split": split_name,
                        "source_dataset": batch.metadata["source_dataset"][idx],
                        "line_id": batch.metadata["line_id"][idx],
                        "sample_id": batch.metadata["sample_id"][idx],
                        "true_day": label_maps["day"][int(batch.day_label[idx].detach().cpu())],
                        "pred_day": label_maps["day"][int(day_pred[idx])],
                        "true_stage": label_maps["stage"][int(batch.stage_label[idx].detach().cpu())],
                        "pred_stage": label_maps["stage"][int(stage_pred[idx])],
                        "true_state": label_maps["state"][int(batch.state_label[idx].detach().cpu())],
                        "pred_state": label_maps["state"][int(state_pred[idx])],
                        "true_day_value": float(batch.day_regression[idx].detach().cpu()),
                        "pred_day_value": float(reg_pred[idx]),
                        "true_domain": label_maps["domain"][int(batch.domain_label[idx].detach().cpu())],
                        "pred_domain": label_maps["domain"][int(domain_pred[idx])],
                    }
                )
    pred_df = pd.DataFrame(rows)
    metrics = {
        "split": split_name,
        "num_cells": int(len(pred_df)),
        "loss": float(pd.DataFrame(loss_rows)["total_loss"].mean()) if loss_rows else 0.0,
        "day_accuracy": float(accuracy_score(pred_df["true_day"], pred_df["pred_day"])),
        "day_macro_f1": float(f1_score(pred_df["true_day"], pred_df["pred_day"], average="macro", zero_division=0)),
        "stage_accuracy": float(accuracy_score(pred_df["true_stage"], pred_df["pred_stage"])),
        "stage_macro_f1": float(f1_score(pred_df["true_stage"], pred_df["pred_stage"], average="macro", zero_division=0)),
        "state_accuracy": float(accuracy_score(pred_df["true_state"], pred_df["pred_state"])),
        "state_macro_f1": float(f1_score(pred_df["true_state"], pred_df["pred_state"], average="macro", zero_division=0)),
        "day_regression_mae_days": float(mean_absolute_error(pred_df["true_day_value"], pred_df["pred_day_value"]) * 15.0),
        "domain_accuracy": float(accuracy_score(pred_df["true_domain"], pred_df["pred_domain"])),
    }
    return metrics, pred_df


def write_eval_outputs(output_dir: Path, prefix: str, split_name: str, metrics: dict, pred_df: pd.DataFrame) -> None:
    pred_df.to_csv(output_dir / f"{prefix}_{split_name}_predictions.csv", index=False)
    for name, true_col, pred_col in [("day", "true_day", "pred_day"), ("stage", "true_stage", "pred_stage"), ("state", "true_state", "pred_state")]:
        labels = sorted(set(pred_df[true_col]).union(set(pred_df[pred_col])))
        report = classification_report(pred_df[true_col], pred_df[pred_col], labels=labels, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(output_dir / f"{prefix}_{split_name}_{name}_classification_report.csv")


def make_loaders(args: argparse.Namespace, train_ds: Any, val_ds: Any, test_ds: Any) -> tuple[DataLoader, DataLoader, DataLoader]:
    logger.info("Building token-hash features for expression-neighborhood graph context.")
    train_features = token_hash_features(train_ds, args.feature_dim, args.max_length)
    val_features = token_hash_features(val_ds, args.feature_dim, args.max_length)
    test_features = token_hash_features(test_ds, args.feature_dim, args.max_length)
    train_neighbors = graph_neighbor_features(train_features, train_features, args.graph_neighbors)
    val_neighbors = graph_neighbor_features(train_features, val_features, args.graph_neighbors)
    test_neighbors = graph_neighbor_features(train_features, test_features, args.graph_neighbors)

    train_torch = IndexedCellDataset(train_ds, train_features, train_neighbors)
    val_torch = IndexedCellDataset(val_ds, val_features, val_neighbors)
    test_torch = IndexedCellDataset(test_ds, test_features, test_neighbors)
    train_collate = lambda examples: collate_baseline(examples, train_features, train_neighbors, args.max_length, args.vocab_size)
    val_collate = lambda examples: collate_baseline(examples, val_features, val_neighbors, args.max_length, args.vocab_size)
    test_collate = lambda examples: collate_baseline(examples, test_features, test_neighbors, args.max_length, args.vocab_size)

    sampler = None
    shuffle = True
    if args.balance_sampler:
        sampler = WeightedRandomSampler(sampler_weights_from_dataset(train_ds, args.gse202398_sampler_boost), num_samples=len(train_ds), replacement=True)
        shuffle = False
    train_loader = DataLoader(train_torch, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler, collate_fn=train_collate)
    val_loader = DataLoader(val_torch, batch_size=args.eval_batch_size, shuffle=False, collate_fn=val_collate)
    test_loader = DataLoader(test_torch, batch_size=args.eval_batch_size, shuffle=False, collate_fn=test_collate)
    return train_loader, val_loader, test_loader


def build_model(args: argparse.Namespace, label_maps: dict) -> nn.Module:
    num_day = len(label_maps["day"])
    num_stage = len(label_maps["stage"])
    num_state = len(label_maps["state"])
    if args.model == "rnn":
        return RNNBaseline(args, num_day, num_stage, num_state)
    if args.model == "gnn":
        return GNNBaseline(args, num_day, num_stage, num_state)
    return HybridBaseline(args, num_day, num_stage, num_state)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or f"{args.model}_gse175634_gse202398_holdout_{str(args.holdout_gse202398_value).lower()}"

    train_ds, val_ds, test_ds, split_summary = prepare_splits_fast(args)
    split_summary["baseline_model"] = args.model
    split_summary["gnn_note"] = "Expression-neighborhood graph from token-hash features; not spatial coordinates."
    label_maps = split_summary["label_maps"]
    (output_dir / f"{prefix}_split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
    logger.info("Split summary: %s", json.dumps(split_summary, indent=2)[:4000])

    train_loader, val_loader, test_loader = make_loaders(args, train_ds, val_ds, test_ds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    model = build_model(args, label_maps).to(device)
    losses = {
        "day": nn.CrossEntropyLoss(weight=class_weights_from_dataset(train_ds, "day_label", len(label_maps["day"]), args.class_weighting, device)),
        "stage": nn.CrossEntropyLoss(weight=class_weights_from_dataset(train_ds, "stage_label", len(label_maps["stage"]), args.class_weighting, device)),
        "state": nn.CrossEntropyLoss(weight=class_weights_from_dataset(train_ds, "state_label", len(label_maps["state"]), args.class_weighting, device)),
        "domain": nn.CrossEntropyLoss(),
        "regression": nn.SmoothL1Loss(beta=0.08),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    day_values = torch.tensor([DAY_TO_NUMERIC[label] / 15.0 for label in DAY_ORDER], dtype=torch.float32, device=device)

    best_val_loss = float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch, grl_lambda=1.0)
            loss, loss_parts = compute_loss(outputs, batch, losses, args, day_values)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running.append(loss_parts)
        val_metrics, val_pred = evaluate(model, val_loader, losses, args, device, label_maps, "gse175634_val_lines")
        test_metrics, test_pred = evaluate(model, test_loader, losses, args, device, label_maps, "gse202398_holdout")
        history_row = {
            "epoch": epoch,
            "train_loss": float(pd.DataFrame(running)["total_loss"].mean()) if running else 0.0,
            "val": val_metrics,
            "test": test_metrics,
        }
        history.append(history_row)
        logger.info("Epoch %03d/%03d %s", epoch, args.epochs, json.dumps(history_row))
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
            write_eval_outputs(output_dir, prefix, "gse175634_val_lines", val_metrics, val_pred)
            write_eval_outputs(output_dir, prefix, "gse202398_holdout", test_metrics, test_pred)
        else:
            stale += 1
            if stale >= args.patience:
                logger.info("Early stopping after %d stale epochs.", stale)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_val_metrics, final_val_pred = evaluate(model, val_loader, losses, args, device, label_maps, "gse175634_val_lines")
    final_test_metrics, final_test_pred = evaluate(model, test_loader, losses, args, device, label_maps, "gse202398_holdout")
    write_eval_outputs(output_dir, prefix, "final_gse175634_val_lines", final_val_metrics, final_val_pred)
    write_eval_outputs(output_dir, prefix, "final_gse202398_holdout", final_test_metrics, final_test_pred)
    summary = {
        "model": args.model,
        "history": history,
        "final_val_metrics": final_val_metrics,
        "final_test_metrics": final_test_metrics,
        "split_summary": split_summary,
        "architecture_note": {
            "rnn": "Token embedding plus bi-GRU over ranked gene-token sequences.",
            "gnn": "Expression-neighborhood graph context from token-hash features and train-set nearest neighbors.",
            "hybrid": "RNN sequence encoder fused with expression-neighborhood graph context.",
        }[args.model],
    }
    (output_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"final_val_metrics": final_val_metrics, "final_test_metrics": final_test_metrics}, indent=2))


if __name__ == "__main__":
    main()

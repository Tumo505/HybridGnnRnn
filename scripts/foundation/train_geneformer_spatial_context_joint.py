#!/usr/bin/env python3
"""Joint LoRA-style fine-tuning of Geneformer plus the spatial context adapter.

This is the end-to-end variant of the spatial foundation model. Instead of
training only from cached Geneformer embeddings, it samples spots from each
Mauron tissue section, runs the fine-tuned Geneformer encoder with LoRA modules
enabled in the final transformer layers, then trains the coordinate-aware graph
adapter on those pooled embeddings.

The implementation is deliberately parameter efficient:

- load the previously fine-tuned multi-atlas Geneformer checkpoint;
- freeze the base Geneformer parameters;
- add LoRA modules to query/value projections in the last N encoder layers;
- train LoRA parameters plus the spatial graph adapter.

This is still inferred spatial-time modelling, not longitudinal tracking.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "foundation"))

from train_geneformer_multiatlas_multitask import GeneformerMultiTaskModel, set_seed  # noqa: E402
from train_geneformer_spatial_context_adapter import (  # noqa: E402
    AGE_BIN_ORDER,
    CHAMBERS,
    FAMILY_ORDER,
    GeneformerSpatialContextAdapter,
    chamber_multi_hot,
    confidence_weighted_ce,
    infer_cell_family,
    neighborhood_consistency_loss,
    ordinal_age_loss,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("geneformer_spatial_context_joint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenized-dataset",
        default="foundation_spatial_geneformer_adapter_inputs/tokenized/mauron_spatial_geneformer.dataset",
    )
    parser.add_argument("--model-dir", default="models/geneformer/Geneformer/Geneformer-V1-10M")
    parser.add_argument(
        "--checkpoint",
        default="foundation_geneformer_multiatlas_multitask/gse175634_gse202398_holdout_lmna_best_model/pytorch_model_multitask.bin",
    )
    parser.add_argument(
        "--label-maps",
        default="foundation_geneformer_multiatlas_multitask/gse175634_gse202398_holdout_lmna_best_model/label_maps.json",
    )
    parser.add_argument("--output-dir", default="foundation_geneformer_spatial_context_joint_lora")
    parser.add_argument("--output-prefix", default="mauron_geneformer_spatial_joint_lora")
    parser.add_argument(
        "--adapter-checkpoint",
        default=None,
        help="Optional spatial adapter state dict to warm-start from the improved frozen-embedding adapter.",
    )
    parser.add_argument("--split-group", choices=["case", "code"], default="case")
    parser.add_argument("--max-sections", type=int, default=None)
    parser.add_argument("--max-spots-per-section", type=int, default=512)
    parser.add_argument("--eval-spots-per-section", type=int, default=768)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lora-learning-rate", type=float, default=None)
    parser.add_argument("--adapter-learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-layers", type=int, default=2)
    parser.add_argument("--node-loss-weight", type=float, default=0.8)
    parser.add_argument("--family-loss-weight", type=float, default=0.8)
    parser.add_argument("--age-loss-weight", type=float, default=0.8)
    parser.add_argument("--age-ordinal-loss-weight", type=float, default=0.35)
    parser.add_argument("--progress-loss-weight", type=float, default=0.5)
    parser.add_argument("--chamber-loss-weight", type=float, default=0.25)
    parser.add_argument("--neighborhood-consistency-weight", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--freeze-adapter-epochs", type=int, default=0)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class LoRALinear(nn.Module):
    """Low-rank trainable update around a frozen Linear layer."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.rank = rank
        self.scaling = alpha / max(rank, 1)
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def apply_lora_to_geneformer(model: GeneformerMultiTaskModel, rank: int, alpha: float, dropout: float, num_layers: int) -> list[str]:
    for param in model.parameters():
        param.requires_grad = False
    encoder_layers = list(getattr(getattr(model.encoder, "encoder", None), "layer", []))
    if not encoder_layers:
        raise RuntimeError("Could not find BERT encoder layers for LoRA insertion.")
    changed = []
    for layer_idx, layer in enumerate(encoder_layers[-num_layers:]):
        self_attn = getattr(getattr(layer, "attention", None), "self", None)
        if self_attn is None:
            continue
        for name in ["query", "value"]:
            module = getattr(self_attn, name, None)
            if isinstance(module, nn.Linear):
                setattr(self_attn, name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
                changed.append(f"encoder.layer.{len(encoder_layers) - num_layers + layer_idx}.attention.self.{name}")
    if not changed:
        raise RuntimeError("No query/value Linear modules were replaced with LoRA.")
    return changed


def trainable_parameter_summary(model: nn.Module) -> dict:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total": int(total), "trainable": int(trainable), "trainable_fraction": float(trainable / max(total, 1))}


def set_adapter_trainable(adapter: nn.Module, trainable: bool) -> None:
    for param in adapter.parameters():
        param.requires_grad = trainable


def load_adapter_checkpoint(adapter: nn.Module, checkpoint_path: str | None) -> dict:
    if checkpoint_path is None:
        return {"loaded": False}
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "adapter_state_dict" in state:
        state = state["adapter_state_dict"]
    current = adapter.state_dict()
    compatible = {}
    skipped_shape = {}
    for key, value in state.items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            compatible[key] = value
        elif key in current:
            skipped_shape[key] = {"checkpoint": list(value.shape), "current": list(current[key].shape)}
    missing, unexpected = adapter.load_state_dict(compatible, strict=False)
    return {
        "loaded": True,
        "path": str(path),
        "loaded_keys": len(compatible),
        "skipped_shape_mismatch": skipped_shape,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def lora_parameters(model: GeneformerMultiTaskModel) -> list[nn.Parameter]:
    return [param for name, param in model.named_parameters() if "lora_" in name and param.requires_grad]


def collate_token_rows(rows: list[dict[str, Any]], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = [min(len(row["input_ids"]), max_length) for row in rows]
    max_len = max(lengths)
    input_ids = torch.zeros((len(rows), max_len), dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    for i, row in enumerate(rows):
        ids = torch.tensor(row["input_ids"][: lengths[i]], dtype=torch.long)
        input_ids[i, : lengths[i]] = ids
        attention_mask[i, : lengths[i]] = 1
    return input_ids, attention_mask


def metadata_frame(dataset) -> pd.DataFrame:
    columns = [
        "section_id",
        "code",
        "case",
        "age_weeks",
        "age_bin",
        "chamber_combo",
        "x",
        "y",
        "deconv_label",
        "cell_state",
        "deconv_confidence",
        "spot_id",
    ]
    present = [col for col in columns if col in dataset.column_names]
    df = pd.DataFrame({col: dataset[col] for col in present})
    required = set(columns[:-1]).difference(df.columns)
    if required:
        raise ValueError(f"Tokenized spatial dataset is missing columns: {sorted(required)}")
    df["row_index"] = np.arange(len(df))
    df["section_id"] = df["section_id"].astype(int)
    df["age_weeks"] = df["age_weeks"].astype(float)
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    df["deconv_confidence"] = df["deconv_confidence"].astype(float)
    df["cell_family"] = [
        infer_cell_family(state, label)
        for state, label in zip(df["cell_state"].astype(str), df["deconv_label"].astype(str))
    ]
    return df


def split_sections(meta: pd.DataFrame, split_group: str, seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    section_meta = meta.groupby("section_id").agg(
        code=("code", "first"),
        case=("case", "first"),
        age_bin=("age_bin", "first"),
    )
    group_col = section_meta[split_group].astype(str)
    group_to_sections = {
        group: section_meta.index[group_col == group].astype(int).tolist()
        for group in sorted(group_col.unique())
    }
    group_to_bins = {
        group: set(section_meta.loc[sections, "age_bin"].astype(str))
        for group, sections in group_to_sections.items()
    }
    split_groups = {"train": [], "val": [], "test": []}
    for age_bin in AGE_BIN_ORDER:
        groups = sorted(group for group, bins in group_to_bins.items() if age_bin in bins)
        shuffled = np.asarray(groups, dtype=object)
        rng.shuffle(shuffled)
        if len(shuffled) >= 3:
            split_groups["val"].extend(shuffled[:1].tolist())
            split_groups["test"].extend(shuffled[1:2].tolist())
            split_groups["train"].extend(shuffled[2:].tolist())
        else:
            split_groups["train"].extend(shuffled.tolist())
    if not split_groups["val"] or not split_groups["test"]:
        groups = sorted(group_to_sections)
        rng.shuffle(groups)
        if len(groups) >= 3:
            split_groups = {"val": [groups[0]], "test": [groups[1]], "train": groups[2:]}
        elif len(groups) == 2:
            split_groups = {"val": [groups[0]], "test": [groups[1]], "train": groups}
        elif len(groups) == 1:
            split_groups = {"val": [], "test": [], "train": groups}

    return {
        split: sorted({section for group in groups for section in group_to_sections[group]})
        for split, groups in split_groups.items()
    }


def normalize_pos(pos: np.ndarray) -> np.ndarray:
    center = pos.mean(axis=0, keepdims=True)
    scale = pos.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return ((pos - center) / scale).astype(np.float32)


def knn_edges(pos: np.ndarray, k: int) -> np.ndarray:
    if len(pos) < 2:
        return np.empty((2, 0), dtype=np.int64)
    n_neighbors = min(k + 1, len(pos))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(pos)
    indices = nbrs.kneighbors(pos, return_distance=False)
    src = np.repeat(np.arange(len(pos)), n_neighbors - 1)
    dst = indices[:, 1:].reshape(-1)
    pairs = np.vstack([np.column_stack([src, dst]), np.column_stack([dst, src])])
    return np.unique(pairs, axis=0).T.astype(np.int64)


@dataclass
class LabelMaps:
    fine: dict[str, int]
    family: dict[str, int]
    age: dict[str, int]


def build_graph_from_rows(section_df: pd.DataFrame, embeddings: torch.Tensor, label_maps: LabelMaps, k: int, device: torch.device) -> Data:
    pos = normalize_pos(section_df[["x", "y"]].to_numpy(dtype=np.float32))
    progress = float(np.clip((float(section_df["age_weeks"].iloc[0]) - 6.0) / 6.0, 0.0, 1.0))
    graph = Data(
        x=embeddings,
        pos=torch.tensor(pos, dtype=torch.float32, device=device),
        edge_index=torch.tensor(knn_edges(pos, k), dtype=torch.long, device=device),
        y=torch.tensor([label_maps.fine[str(v)] for v in section_df["deconv_label"]], dtype=torch.long, device=device),
        family_y=torch.tensor([label_maps.family[str(v)] for v in section_df["cell_family"]], dtype=torch.long, device=device),
        label_confidence=torch.tensor(section_df["deconv_confidence"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device),
        age_bin_y=torch.tensor([label_maps.age[str(section_df["age_bin"].iloc[0])]], dtype=torch.long, device=device),
        progress_y=torch.tensor([progress], dtype=torch.float32, device=device),
        chamber_y=torch.tensor([chamber_multi_hot(str(section_df["chamber_combo"].iloc[0]))], dtype=torch.float32, device=device),
    )
    graph.section_id = int(section_df["section_id"].iloc[0])
    graph.code = str(section_df["code"].iloc[0])
    graph.case = str(section_df["case"].iloc[0])
    graph.age_bin = str(section_df["age_bin"].iloc[0])
    return graph


def class_weights(meta: pd.DataFrame, sections: Iterable[int], column: str, labels: list[str], device: torch.device) -> torch.Tensor:
    subset = meta[meta["section_id"].isin(set(sections))]
    counts = subset[column].astype(str).value_counts().to_dict()
    values = np.ones(len(labels), dtype=np.float32)
    total = sum(counts.values())
    present = 0
    for i, label in enumerate(labels):
        count = counts.get(label, 0)
        if count > 0:
            present += 1
            values[i] = total / max(count, 1)
        else:
            values[i] = 0.0
    if present:
        values = values / np.mean(values[values > 0])
    values = np.minimum(values, 5.0)
    return torch.tensor(values, dtype=torch.float32, device=device)


def sample_section(meta: pd.DataFrame, section_id: int, max_spots: int, seed: int) -> pd.DataFrame:
    df = meta[meta["section_id"] == section_id]
    if len(df) <= max_spots:
        return df.copy()
    return df.sample(n=max_spots, random_state=seed).sort_index().copy()


def run_section(
    dataset,
    meta: pd.DataFrame,
    section_id: int,
    max_spots: int,
    seed: int,
    geneformer: GeneformerMultiTaskModel,
    adapter: GeneformerSpatialContextAdapter,
    label_maps: LabelMaps,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], Data]:
    section_df = sample_section(meta, section_id, max_spots, seed)
    rows = [dataset[int(i)] for i in section_df["row_index"]]
    input_ids, attention_mask = collate_token_rows(rows, args.max_length)
    pooled = geneformer.pooled(input_ids.to(device), attention_mask.to(device))
    graph = build_graph_from_rows(section_df, pooled, label_maps, args.k_neighbors, device)
    return adapter(graph), graph


def evaluate(
    dataset,
    meta: pd.DataFrame,
    sections: list[int],
    geneformer: GeneformerMultiTaskModel,
    adapter: GeneformerSpatialContextAdapter,
    label_maps: LabelMaps,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    geneformer.eval()
    adapter.eval()
    node_true = []
    node_pred = []
    family_true = []
    family_pred = []
    age_true = []
    age_pred = []
    progress_true = []
    progress_pred = []
    with torch.no_grad():
        for section_id in sections:
            out, graph = run_section(dataset, meta, section_id, args.eval_spots_per_section, args.seed + section_id, geneformer, adapter, label_maps, args, device)
            node_true.extend(graph.y.cpu().tolist())
            node_pred.extend(out["cell_state_logits"].argmax(dim=1).cpu().tolist())
            family_true.extend(graph.family_y.cpu().tolist())
            family_pred.extend(out["family_logits"].argmax(dim=1).cpu().tolist())
            age_true.append(int(graph.age_bin_y.cpu().item()))
            age_pred.append(int(out["age_bin_logits"].argmax(dim=1).cpu().item()))
            progress_true.append(float(graph.progress_y.cpu().item()))
            progress_pred.append(float(out["progress"].cpu().item()))
    return {
        "num_sections": int(len(sections)),
        "num_sampled_spots": int(len(node_true)),
        "cell_state_accuracy": float(accuracy_score(node_true, node_pred)) if node_true else 0.0,
        "cell_state_macro_f1": float(f1_score(node_true, node_pred, average="macro", zero_division=0)) if node_true else 0.0,
        "family_accuracy": float(accuracy_score(family_true, family_pred)) if family_true else 0.0,
        "family_macro_f1": float(f1_score(family_true, family_pred, average="macro", zero_division=0)) if family_true else 0.0,
        "age_bin_accuracy": float(accuracy_score(age_true, age_pred)) if age_true else 0.0,
        "age_bin_macro_f1": float(f1_score(age_true, age_pred, average="macro", zero_division=0)) if age_true else 0.0,
        "progress_mae": float(mean_absolute_error(progress_true, progress_pred)) if progress_true else 0.0,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    dataset = load_from_disk(args.tokenized_dataset)
    meta = metadata_frame(dataset)
    if args.max_sections is not None:
        keep_sections = sorted(meta["section_id"].unique())[: args.max_sections]
        meta = meta[meta["section_id"].isin(keep_sections)].copy()

    fine_labels = sorted(meta["deconv_label"].astype(str).unique())
    label_maps = LabelMaps(
        fine={label: i for i, label in enumerate(fine_labels)},
        family={label: i for i, label in enumerate(FAMILY_ORDER)},
        age={label: i for i, label in enumerate(AGE_BIN_ORDER)},
    )
    splits = split_sections(meta, args.split_group, args.seed)

    raw_label_maps = json.loads(Path(args.label_maps).read_text(encoding="utf-8"))
    raw_label_maps = {key: {int(k): v for k, v in value.items()} for key, value in raw_label_maps.items()}
    geneformer = GeneformerMultiTaskModel(
        args.model_dir,
        num_day=len(raw_label_maps["day"]),
        num_stage=len(raw_label_maps["stage"]),
        num_state=len(raw_label_maps["state"]),
        num_domain=len(raw_label_maps["domain"]),
        dropout=0.0,
        grl_lambda=0.0,
    )
    geneformer.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    lora_targets = apply_lora_to_geneformer(geneformer, args.lora_rank, args.lora_alpha, args.lora_dropout, args.lora_layers)
    geneformer.to(device)

    adapter = GeneformerSpatialContextAdapter(
        input_dim=256,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_cell_states=len(fine_labels),
        num_families=len(FAMILY_ORDER),
        num_age_bins=len(AGE_BIN_ORDER),
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    adapter_load = load_adapter_checkpoint(adapter, args.adapter_checkpoint)
    lora_lr = args.lora_learning_rate or args.learning_rate
    adapter_lr = args.adapter_learning_rate or args.learning_rate
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters(geneformer), "lr": lora_lr},
            {"params": [param for param in adapter.parameters() if param.requires_grad], "lr": adapter_lr},
        ],
        weight_decay=args.weight_decay,
    )
    fine_weights = class_weights(meta, splits["train"], "deconv_label", fine_labels, device)
    family_weights = class_weights(meta, splits["train"], "cell_family", FAMILY_ORDER, device)
    ce_age = nn.CrossEntropyLoss()
    bce_chamber = nn.BCEWithLogitsLoss()
    regression = nn.SmoothL1Loss(beta=0.08)

    summary = {
        "model": "joint_geneformer_lora_spatial_context_adapter",
        "important_caveat": "This is inferred spatial-time modelling; it is not true longitudinal tracking.",
        "lora_targets": lora_targets,
        "adapter_warm_start": adapter_load,
        "learning_rates": {"lora": lora_lr, "adapter": adapter_lr},
        "geneformer_params": trainable_parameter_summary(geneformer),
        "adapter_params": trainable_parameter_summary(adapter),
        "splits": {name: [int(x) for x in values] for name, values in splits.items()},
    }
    logger.info("Joint LoRA setup: %s", json.dumps(summary, indent=2))

    best_state = None
    best_score = -float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        geneformer.train()
        adapter.train()
        if args.freeze_adapter_epochs > 0:
            set_adapter_trainable(adapter, epoch > args.freeze_adapter_epochs)
        train_sections = splits["train"][:]
        random.shuffle(train_sections)
        losses = []
        for section_id in train_sections:
            optimizer.zero_grad(set_to_none=True)
            out, graph = run_section(dataset, meta, section_id, args.max_spots_per_section, args.seed + epoch + section_id, geneformer, adapter, label_maps, args, device)
            node_loss = confidence_weighted_ce(out["cell_state_logits"], graph.y, fine_weights, graph.label_confidence)
            family_loss = confidence_weighted_ce(out["family_logits"], graph.family_y, family_weights, graph.label_confidence)
            age_loss = ce_age(out["age_bin_logits"], graph.age_bin_y)
            age_ordinal = ordinal_age_loss(out["age_bin_logits"], graph.age_bin_y)
            progress_loss = regression(out["progress"], graph.progress_y)
            chamber_loss = bce_chamber(out["chamber_logits"], graph.chamber_y)
            consistency = neighborhood_consistency_loss(out["family_logits"], graph.edge_index)
            loss = (
                args.node_loss_weight * node_loss
                + args.family_loss_weight * family_loss
                + args.age_loss_weight * age_loss
                + args.age_ordinal_loss_weight * age_ordinal
                + args.progress_loss_weight * progress_loss
                + args.chamber_loss_weight * chamber_loss
                + args.neighborhood_consistency_weight * consistency
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in list(geneformer.parameters()) + list(adapter.parameters()) if p.requires_grad], 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate(dataset, meta, splits["val"], geneformer, adapter, label_maps, args, device)
        val_score = val_metrics["family_macro_f1"] + val_metrics["age_bin_macro_f1"] - val_metrics["progress_mae"]
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_metrics": val_metrics, "val_score": float(val_score)}
        history.append(row)
        logger.info("Epoch %03d/%03d %s", epoch, args.epochs, json.dumps(row))
        (output_dir / f"{args.output_prefix}_progress.json").write_text(
            json.dumps({"last_completed_epoch": epoch, "history": history}, indent=2),
            encoding="utf-8",
        )
        if args.save_every_epoch:
            torch.save(
                {
                    "epoch": epoch,
                    "geneformer_lora_state_dict": {key: value.cpu() for key, value in geneformer.state_dict().items() if "lora_" in key},
                    "adapter_state_dict": adapter.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                output_dir / f"{args.output_prefix}_epoch{epoch:03d}_checkpoint.pt",
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if val_score > best_score:
            best_score = val_score
            best_state = {
                "geneformer": {key: value.detach().cpu().clone() for key, value in geneformer.state_dict().items() if "lora_" in key},
                "adapter": {key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()},
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                logger.info("Early stopping after %d stale epochs.", stale)
                break

    if best_state is not None:
        adapter.load_state_dict(best_state["adapter"])
        current = geneformer.state_dict()
        current.update(best_state["geneformer"])
        geneformer.load_state_dict(current)

    final = {
        **summary,
        "train_metrics": evaluate(dataset, meta, splits["train"], geneformer, adapter, label_maps, args, device),
        "val_metrics": evaluate(dataset, meta, splits["val"], geneformer, adapter, label_maps, args, device),
        "test_metrics": evaluate(dataset, meta, splits["test"], geneformer, adapter, label_maps, args, device),
        "history": history,
        "fine_labels": fine_labels,
        "family_labels": FAMILY_ORDER,
    }
    (output_dir / f"{args.output_prefix}_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    torch.save(
        {
            "geneformer_lora_state_dict": {key: value.cpu() for key, value in geneformer.state_dict().items() if "lora_" in key},
            "adapter_state_dict": adapter.state_dict(),
            "label_maps": {
                "fine": label_maps.fine,
                "family": label_maps.family,
                "age": label_maps.age,
            },
            "args": vars(args),
        },
        output_dir / f"{args.output_prefix}_checkpoint.pt",
    )
    print(json.dumps({key: final[key] for key in ["important_caveat", "val_metrics", "test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Multi-atlas, multi-task Geneformer fine-tuning for trajectory generalisation.

This trainer is designed for the GSE175634 + GSE202398 setting:

- train on GSE175634 plus part of GSE202398
- hold out one GSE202398 line or run for external validation
- predict exact day and broader biological trajectory stage
- add state supervision and continuous ordinal day regression
- add a domain-adversarial head so the encoder cannot solve the task only by
  recognizing the dataset/batch

The script intentionally avoids Geneformer's single-target Classifier wrapper
because we need multiple heads and a gradient-reversal domain objective.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoConfig, AutoModel, get_cosine_schedule_with_warmup


os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("geneformer_multiatlas_multitask")

DAY_ORDER = ["day0", "day1", "day3", "day5", "day7", "day11", "day15"]
DAY_TO_NUMERIC = {label: int(label.replace("day", "")) for label in DAY_ORDER}

STAGE_MAP = {
    "day0": "early_ipsc",
    "day1": "early_mesoderm",
    "day3": "cardiac_mesoderm",
    "day5": "cardiac_progenitor",
    "day7": "cardiac_progenitor",
    "day11": "cardiomyocyte_maturation",
    "day15": "cardiomyocyte_maturation",
}

STATE_ORDER = ["IPSC", "MES", "CMES", "PROG", "CM", "CF", "UNK"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gse175634-tokenized", default="foundation_model_data/geneformer/tokenized/gse175634_geneformer.dataset")
    parser.add_argument(
        "--gse202398-tokenized",
        default="foundation_model_data/geneformer_external_unseen/gse202398_tokenized/gse202398_filtered_cellplex_matched_days.dataset",
    )
    parser.add_argument("--model-dir", default="models/geneformer/Geneformer/Geneformer-V1-10M")
    parser.add_argument("--output-dir", default="foundation_geneformer_multiatlas_multitask")
    parser.add_argument("--output-prefix", default="gse175634_gse202398_holdout_lmna")
    parser.add_argument("--holdout-gse202398-field", choices=["line_id", "sample_id"], default="line_id")
    parser.add_argument("--holdout-gse202398-value", default="LMNA")
    parser.add_argument("--gse175634-val-line-fraction", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--freeze-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--day-loss-weight", type=float, default=1.0)
    parser.add_argument("--stage-loss-weight", type=float, default=1.0)
    parser.add_argument("--state-loss-weight", type=float, default=0.5)
    parser.add_argument("--regression-loss-weight", type=float, default=0.6)
    parser.add_argument("--ordinal-loss-weight", type=float, default=0.5)
    parser.add_argument("--domain-loss-weight", type=float, default=0.2)
    parser.add_argument("--domain-grl-lambda", type=float, default=1.0)
    parser.add_argument("--class-weighting", choices=["none", "inverse_sqrt"], default="inverse_sqrt")
    parser.add_argument("--balance-sampler", action="store_true", help="Use weighted sampling to boost GSE202398 and rare days.")
    parser.add_argument("--gse202398-sampler-boost", type=float, default=3.0)
    parser.add_argument("--max-gse175634-train-cells", type=int, default=None)
    parser.add_argument("--max-gse202398-train-cells", type=int, default=None)
    parser.add_argument("--max-train-cells", type=int, default=None, help="Subsample training cells for smoke tests.")
    parser.add_argument("--max-eval-cells", type=int, default=None, help="Subsample validation/test cells for smoke tests.")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_day_numeric(day: str) -> int:
    text = str(day).lower().replace("day", "")
    try:
        return int(text)
    except ValueError:
        return -1


def add_source_and_labels(dataset: Dataset, source: str, domain: str) -> Dataset:
    day_to_id = {label: idx for idx, label in enumerate(DAY_ORDER)}
    stage_order = sorted(set(STAGE_MAP.values()), key=list(STAGE_MAP.values()).index)
    stage_to_id = {label: idx for idx, label in enumerate(stage_order)}
    state_to_id = {label: idx for idx, label in enumerate(STATE_ORDER)}

    def mapper(example: dict[str, Any]) -> dict[str, Any]:
        day = str(example.get("diffday", "unknown"))
        day_num = parse_day_numeric(day)
        state = str(example.get("cell_state", "UNK"))
        stage = STAGE_MAP.get(day, "unknown")
        example["source_dataset"] = source
        example["domain_label_text"] = domain
        example["day_label"] = day_to_id.get(day, -100)
        example["stage_label"] = stage_to_id.get(stage, -100)
        example["state_label"] = state_to_id.get(state, state_to_id["UNK"])
        example["day_regression"] = float(np.clip(day_num / 15.0, 0.0, 1.0)) if day_num >= 0 else -1.0
        example["stage_text"] = stage
        return example

    return dataset.map(mapper, desc=f"Adding labels for {source}")


def deterministic_line_split(lines: list[str], val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    lines = sorted(set(str(line) for line in lines))
    rng = random.Random(seed)
    rng.shuffle(lines)
    n_val = max(1, int(round(len(lines) * val_fraction))) if lines else 0
    val_lines = set(lines[:n_val])
    train_lines = set(lines[n_val:])
    return train_lines, val_lines


def select_max(dataset: Dataset, max_cells: int | None, seed: int) -> Dataset:
    if max_cells is None or len(dataset) <= max_cells:
        return dataset
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), size=max_cells, replace=False))
    return dataset.select(indices.tolist())


def prepare_splits(args: argparse.Namespace) -> tuple[Dataset, Dataset, Dataset, dict]:
    gse175 = add_source_and_labels(load_from_disk(args.gse175634_tokenized), "GSE175634", "GSE175634")
    gse202 = add_source_and_labels(load_from_disk(args.gse202398_tokenized), "GSE202398", "GSE202398")

    allowed_days = set(DAY_ORDER)
    gse175 = gse175.filter(lambda row: row["diffday"] in allowed_days and row["day_label"] >= 0)
    gse202 = gse202.filter(lambda row: row["diffday"] in allowed_days and row["day_label"] >= 0)

    holdout_field = args.holdout_gse202398_field
    holdout_value = str(args.holdout_gse202398_value)
    gse202_test = gse202.filter(lambda row: str(row[holdout_field]) == holdout_value)
    gse202_train = gse202.filter(lambda row: str(row[holdout_field]) != holdout_value)
    if len(gse202_test) == 0:
        raise ValueError(f"No GSE202398 cells matched {holdout_field}={holdout_value!r}")
    if len(gse202_train) == 0:
        raise ValueError(f"Holding out {holdout_field}={holdout_value!r} leaves no GSE202398 training cells.")

    gse175_lines = sorted(set(str(line) for line in gse175["line_id"]))
    gse175_train_lines, gse175_val_lines = deterministic_line_split(gse175_lines, args.gse175634_val_line_fraction, args.seed)
    gse175_train = gse175.filter(lambda row: str(row["line_id"]) in gse175_train_lines)
    gse175_val = gse175.filter(lambda row: str(row["line_id"]) in gse175_val_lines)

    gse175_train = select_max(gse175_train.shuffle(seed=args.seed + 10), args.max_gse175634_train_cells, args.seed + 10)
    gse202_train = select_max(gse202_train.shuffle(seed=args.seed + 20), args.max_gse202398_train_cells, args.seed + 20)
    train = concatenate_datasets([gse175_train, gse202_train]).shuffle(seed=args.seed)
    val = gse175_val.shuffle(seed=args.seed)
    test = gse202_test.shuffle(seed=args.seed)
    train = select_max(train, args.max_train_cells, args.seed)
    val = select_max(val, args.max_eval_cells, args.seed + 1)
    test = select_max(test, args.max_eval_cells, args.seed + 2)

    label_maps = {
        "day": {idx: label for idx, label in enumerate(DAY_ORDER)},
        "stage": {idx: label for idx, label in enumerate(sorted(set(STAGE_MAP.values()), key=list(STAGE_MAP.values()).index))},
        "state": {idx: label for idx, label in enumerate(STATE_ORDER)},
        "domain": {0: "GSE175634", 1: "GSE202398"},
    }
    split_summary = {
        "train_cells": len(train),
        "val_cells_gse175634_lines": len(val),
        "test_cells_gse202398_holdout": len(test),
        "gse202398_holdout": {"field": holdout_field, "value": holdout_value},
        "gse175634_train_lines": sorted(gse175_train_lines),
        "gse175634_val_lines": sorted(gse175_val_lines),
        "train_day_counts": pd.Series(train["diffday"]).value_counts().to_dict(),
        "test_day_counts": pd.Series(test["diffday"]).value_counts().to_dict(),
        "train_source_counts": pd.Series(train["source_dataset"]).value_counts().to_dict(),
        "test_source_counts": pd.Series(test["source_dataset"]).value_counts().to_dict(),
        "label_maps": label_maps,
    }
    return train, val, test, split_summary


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    day_label: torch.Tensor
    stage_label: torch.Tensor
    state_label: torch.Tensor
    day_regression: torch.Tensor
    domain_label: torch.Tensor
    metadata: dict[str, list[Any]]


def collate_examples(examples: list[dict[str, Any]], max_length: int, pad_token_id: int = 0) -> Batch:
    lengths = [min(len(example["input_ids"]), max_length) for example in examples]
    max_len = max(lengths)
    input_ids = torch.full((len(examples), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
    for row, example in enumerate(examples):
        ids = torch.tensor(example["input_ids"][: lengths[row]], dtype=torch.long)
        input_ids[row, : lengths[row]] = ids
        attention_mask[row, : lengths[row]] = 1
    source_to_domain = {"GSE175634": 0, "GSE202398": 1}
    metadata_cols = ["source_dataset", "line_id", "sample_id", "diffday", "cell_state", "stage_text"]
    metadata = {col: [example.get(col) for example in examples] for col in metadata_cols}
    return Batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        day_label=torch.tensor([example["day_label"] for example in examples], dtype=torch.long),
        stage_label=torch.tensor([example["stage_label"] for example in examples], dtype=torch.long),
        state_label=torch.tensor([example["state_label"] for example in examples], dtype=torch.long),
        day_regression=torch.tensor([example["day_regression"] for example in examples], dtype=torch.float32),
        domain_label=torch.tensor([source_to_domain[str(example["source_dataset"])] for example in examples], dtype=torch.long),
        metadata=metadata,
    )


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


class GeneformerMultiTaskModel(nn.Module):
    def __init__(self, model_dir: str, num_day: int, num_stage: int, num_state: int, num_domain: int, dropout: float, grl_lambda: float):
        super().__init__()
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
        self.encoder = AutoModel.from_pretrained(model_dir, config=config, trust_remote_code=False)
        hidden = int(config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.day_head = nn.Linear(hidden, num_day)
        self.stage_head = nn.Linear(hidden, num_stage)
        self.state_head = nn.Linear(hidden, num_state)
        self.regression_head = nn.Linear(hidden, 1)
        self.domain_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, num_domain))
        self.grl_lambda = grl_lambda

    def pooled(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).type_as(hidden)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = self.dropout(self.pooled(input_ids, attention_mask))
        reversed_pooled = GradientReverse.apply(pooled, self.grl_lambda)
        return {
            "day_logits": self.day_head(pooled),
            "stage_logits": self.stage_head(pooled),
            "state_logits": self.state_head(pooled),
            "day_value": self.regression_head(pooled).squeeze(-1),
            "domain_logits": self.domain_head(reversed_pooled),
        }


def freeze_layers(model: GeneformerMultiTaskModel, num_layers: int) -> None:
    if num_layers <= 0:
        return
    embeddings = getattr(model.encoder, "embeddings", None)
    if embeddings is not None:
        for param in embeddings.parameters():
            param.requires_grad = False
    encoder_layers = getattr(getattr(model.encoder, "encoder", None), "layer", [])
    for layer in list(encoder_layers)[:num_layers]:
        for param in layer.parameters():
            param.requires_grad = False


def class_weights(dataset: Dataset, column: str, num_classes: int, mode: str, device: torch.device) -> torch.Tensor | None:
    if mode == "none":
        return None
    labels = np.array(dataset[column])
    labels = labels[labels >= 0]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def sampler_weights(dataset: Dataset, gse202398_boost: float) -> list[float]:
    day_counts = pd.Series(dataset["day_label"]).value_counts().to_dict()
    weights = []
    for row in dataset:
        day_weight = 1.0 / math.sqrt(float(day_counts.get(row["day_label"], 1)))
        source_weight = gse202398_boost if row["source_dataset"] == "GSE202398" else 1.0
        weights.append(day_weight * source_weight)
    return weights


def move_batch(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        day_label=batch.day_label.to(device),
        stage_label=batch.stage_label.to(device),
        state_label=batch.state_label.to(device),
        day_regression=batch.day_regression.to(device),
        domain_label=batch.domain_label.to(device),
        metadata=batch.metadata,
    )


def compute_loss(outputs: dict[str, torch.Tensor], batch: Batch, losses: dict[str, nn.Module], args: argparse.Namespace, day_values: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
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


def evaluate(model: GeneformerMultiTaskModel, loader: DataLoader, losses: dict[str, nn.Module], args: argparse.Namespace, device: torch.device, label_maps: dict, split_name: str) -> tuple[dict, pd.DataFrame]:
    model.eval()
    day_values = torch.tensor([DAY_TO_NUMERIC[label] / 15.0 for label in DAY_ORDER], dtype=torch.float32, device=device)
    rows = []
    loss_rows = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(batch.input_ids, batch.attention_mask)
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
    for target in [("day", "true_day", "pred_day"), ("stage", "true_stage", "pred_stage"), ("state", "true_state", "pred_state")]:
        name, true_col, pred_col = target
        labels = sorted(set(pred_df[true_col]).union(set(pred_df[pred_col])))
        report = classification_report(pred_df[true_col], pred_df[pred_col], labels=labels, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(output_dir / f"{prefix}_{split_name}_{name}_classification_report.csv")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, split_summary = prepare_splits(args)
    label_maps = split_summary["label_maps"]
    (output_dir / f"{args.output_prefix}_split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
    logger.info("Split summary: %s", json.dumps(split_summary, indent=2)[:4000])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    model = GeneformerMultiTaskModel(
        args.model_dir,
        num_day=len(label_maps["day"]),
        num_stage=len(label_maps["stage"]),
        num_state=len(label_maps["state"]),
        num_domain=len(label_maps["domain"]),
        dropout=args.dropout,
        grl_lambda=args.domain_grl_lambda,
    )
    freeze_layers(model, args.freeze_layers)
    model.to(device)

    losses = {
        "day": nn.CrossEntropyLoss(weight=class_weights(train_ds, "day_label", len(label_maps["day"]), args.class_weighting, device)),
        "stage": nn.CrossEntropyLoss(weight=class_weights(train_ds, "stage_label", len(label_maps["stage"]), args.class_weighting, device)),
        "state": nn.CrossEntropyLoss(weight=class_weights(train_ds, "state_label", len(label_maps["state"]), args.class_weighting, device)),
        "domain": nn.CrossEntropyLoss(),
        "regression": nn.SmoothL1Loss(beta=0.08),
    }

    collate = lambda examples: collate_examples(examples, args.max_length, pad_token_id=0)
    sampler = None
    shuffle = True
    if args.balance_sampler:
        sampler = WeightedRandomSampler(sampler_weights(train_ds, args.gse202398_sampler_boost), num_samples=len(train_ds), replacement=True)
        shuffle = False
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=shuffle, sampler=sampler, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / max(args.gradient_accumulation_steps, 1))
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.fp16 and device.type == "cuda"))
    day_values = torch.tensor([DAY_TO_NUMERIC[label] / 15.0 for label in DAY_ORDER], dtype=torch.float32, device=device)

    history = []
    best_val_loss = float("inf")
    best_state = None
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = []
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            with torch.cuda.amp.autocast(enabled=bool(args.fp16 and device.type == "cuda")):
                outputs = model(batch.input_ids, batch.attention_mask)
                loss, loss_parts = compute_loss(outputs, batch, losses, args, day_values)
                loss = loss / max(args.gradient_accumulation_steps, 1)
            scaler.scale(loss).backward()
            running.append(loss_parts)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if step % 200 == 0:
                logger.info("Epoch %d step %d/%d loss %.4f", epoch, step, len(train_loader), pd.DataFrame(running).tail(200)["total_loss"].mean())

        val_metrics, val_pred = evaluate(model, val_loader, losses, args, device, label_maps, "gse175634_val_lines")
        test_metrics, test_pred = evaluate(model, test_loader, losses, args, device, label_maps, "gse202398_holdout")
        history_row = {"epoch": epoch, "train_loss": float(pd.DataFrame(running)["total_loss"].mean()), "val": val_metrics, "test": test_metrics}
        history.append(history_row)
        logger.info("Epoch %d validation: %s", epoch, json.dumps(history_row, indent=2))
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            write_eval_outputs(output_dir, args.output_prefix, "gse175634_val_lines", val_metrics, val_pred)
            write_eval_outputs(output_dir, args.output_prefix, "gse202398_holdout", test_metrics, test_pred)

    if best_state is not None:
        model.load_state_dict(best_state)
    final_val_metrics, final_val_pred = evaluate(model, val_loader, losses, args, device, label_maps, "gse175634_val_lines")
    final_test_metrics, final_test_pred = evaluate(model, test_loader, losses, args, device, label_maps, "gse202398_holdout")
    write_eval_outputs(output_dir, args.output_prefix, "final_gse175634_val_lines", final_val_metrics, final_val_pred)
    write_eval_outputs(output_dir, args.output_prefix, "final_gse202398_holdout", final_test_metrics, final_test_pred)

    final_summary = {"history": history, "final_val_metrics": final_val_metrics, "final_test_metrics": final_test_metrics, "split_summary": split_summary}
    (output_dir / f"{args.output_prefix}_summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    if args.save_model:
        model_dir = output_dir / f"{args.output_prefix}_best_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_dir / "pytorch_model_multitask.bin")
        (model_dir / "label_maps.json").write_text(json.dumps(label_maps, indent=2), encoding="utf-8")
    print(json.dumps({"final_val_metrics": final_val_metrics, "final_test_metrics": final_test_metrics}, indent=2))


if __name__ == "__main__":
    main()

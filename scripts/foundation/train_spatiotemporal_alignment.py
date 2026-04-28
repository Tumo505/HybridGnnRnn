#!/usr/bin/env python3
"""Weakly supervised spatial-temporal alignment for inferred 4D modelling.

This script does not claim true observed 4D tracking. It aligns:

- temporal Geneformer cell embeddings from differentiating scRNA-seq
- spatial GNN section embeddings from Mauron Visium sections

into a shared latent space using coarse developmental progress supervision.
The resulting model can ask whether spatial sections retrieve temporally
compatible single-cell states, and whether temporal trajectory structure is
consistent with spatial fetal-age structure.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from train_geneformer_multiatlas_multitask import DAY_ORDER, DAY_TO_NUMERIC, STAGE_MAP, deterministic_line_split, set_seed


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("spatiotemporal_alignment")

STAGE_ORDER = list(dict.fromkeys(STAGE_MAP.values()))
AGE_BIN_ORDER = ["early_w6_w7_5", "mid_w8_w9", "late_w10_w12"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-embeddings", default="foundation_geneformer_embeddings/gse175634_day_lineheldout_full_embeddings.csv")
    parser.add_argument("--temporal-metadata", default=None, help="Metadata CSV for .npy temporal embeddings.")
    parser.add_argument("--temporal-tokenized", default="foundation_model_data/geneformer/tokenized/gse175634_geneformer.dataset")
    parser.add_argument("--spatial-embeddings", default="mauron_spatial_gnn_fresh_case_split/section_embeddings.npy")
    parser.add_argument("--spatial-metadata", default="mauron_spatial_gnn_fresh_case_split/section_embedding_metadata.csv")
    parser.add_argument("--output-dir", default="foundation_spatiotemporal_alignment")
    parser.add_argument("--output-prefix", default="gse175634_mauron_alignment")
    parser.add_argument("--max-temporal-cells", type=int, default=100000)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.12)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--temporal-loss-weight", type=float, default=1.0)
    parser.add_argument("--spatial-loss-weight", type=float, default=1.0)
    parser.add_argument("--progress-loss-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=250)
    return parser.parse_args()


def temporal_progress_bin(day: str) -> str:
    day_num = DAY_TO_NUMERIC[str(day)]
    if day_num <= 3:
        return "early_w6_w7_5"
    if day_num <= 7:
        return "mid_w8_w9"
    return "late_w10_w12"


def read_temporal_embeddings(path: Path, max_cells: int | None) -> np.ndarray:
    logger.info("Reading temporal embeddings from %s", path)
    if path.suffix.lower() == ".npy":
        embeddings = np.load(path).astype(np.float32)
        return embeddings[:max_cells] if max_cells is not None else embeddings
    df = pd.read_csv(path, nrows=max_cells)
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.drop(columns=[df.columns[0]])
    return df.to_numpy(dtype=np.float32)


def prepare_temporal(args: argparse.Namespace) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings = read_temporal_embeddings(Path(args.temporal_embeddings), args.max_temporal_cells)
    if args.temporal_metadata:
        metadata = pd.read_csv(args.temporal_metadata).iloc[: len(embeddings)].copy()
        metadata["row_index"] = np.arange(len(metadata))
        if "stage_text" in metadata.columns and "stage" not in metadata.columns:
            metadata["stage"] = metadata["stage_text"]
    else:
        ds = load_from_disk(args.temporal_tokenized)
        ds = ds.select(range(len(embeddings)))
        metadata = pd.DataFrame(
            {
                "row_index": np.arange(len(ds)),
                "diffday": ds["diffday"],
                "line_id": ds["line_id"],
                "sample_id": ds["sample_id"],
                "cell_state": ds["cell_state"],
            }
        )
    metadata = metadata[metadata["diffday"].isin(DAY_ORDER)].reset_index(drop=True)
    embeddings = embeddings[metadata["row_index"].to_numpy()]
    metadata["day_label"] = metadata["diffday"].map({label: idx for idx, label in enumerate(DAY_ORDER)}).astype(int)
    metadata["stage"] = metadata["diffday"].map(STAGE_MAP)
    metadata["stage_label"] = metadata["stage"].map({label: idx for idx, label in enumerate(STAGE_ORDER)}).astype(int)
    metadata["progress"] = metadata["diffday"].map(lambda x: DAY_TO_NUMERIC[str(x)] / 15.0).astype(float)
    metadata["progress_bin"] = metadata["diffday"].map(temporal_progress_bin)
    metadata["progress_bin_label"] = metadata["progress_bin"].map({label: idx for idx, label in enumerate(AGE_BIN_ORDER)}).astype(int)
    train_lines, val_lines = deterministic_line_split(sorted(metadata["line_id"].astype(str).unique()), 0.12, args.seed)
    metadata["split"] = np.where(metadata["line_id"].astype(str).isin(val_lines), "val", "train")
    return embeddings, metadata


def prepare_spatial(args: argparse.Namespace) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings = np.load(args.spatial_embeddings).astype(np.float32)
    metadata = pd.read_csv(args.spatial_metadata)
    if "section" not in metadata.columns and "section_id" in metadata.columns:
        metadata["section"] = metadata["section_id"]
    metadata["age_bin_label"] = metadata["age_bin"].map({label: idx for idx, label in enumerate(AGE_BIN_ORDER)}).astype(int)
    min_week, max_week = 6.0, 12.0
    metadata["progress"] = ((metadata["age_weeks"].astype(float) - min_week) / (max_week - min_week)).clip(0.0, 1.0)
    metadata["progress_bin"] = metadata["age_bin"]
    metadata["progress_bin_label"] = metadata["age_bin_label"]
    return embeddings, metadata


class TemporalDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, metadata: pd.DataFrame):
        self.embeddings = embeddings.astype(np.float32, copy=False)
        self.metadata = metadata.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.metadata.iloc[int(idx)]
        return {
            "x": self.embeddings[int(idx)],
            "day_label": int(row.day_label),
            "stage_label": int(row.stage_label),
            "progress": float(row.progress),
            "progress_bin_label": int(row.progress_bin_label),
            "row_index": int(row.row_index),
        }


@dataclass
class Batch:
    x: torch.Tensor
    day_label: torch.Tensor
    stage_label: torch.Tensor
    progress: torch.Tensor
    progress_bin_label: torch.Tensor
    row_index: torch.Tensor


def collate_temporal(rows: list[dict[str, Any]]) -> Batch:
    return Batch(
        x=torch.tensor(np.stack([row["x"] for row in rows]), dtype=torch.float32),
        day_label=torch.tensor([row["day_label"] for row in rows], dtype=torch.long),
        stage_label=torch.tensor([row["stage_label"] for row in rows], dtype=torch.long),
        progress=torch.tensor([row["progress"] for row in rows], dtype=torch.float32),
        progress_bin_label=torch.tensor([row["progress_bin_label"] for row in rows], dtype=torch.long),
        row_index=torch.tensor([row["row_index"] for row in rows], dtype=torch.long),
    )


class ModalityProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(x), dim=1)


class AlignmentModel(nn.Module):
    def __init__(self, temporal_dim: int, spatial_dim: int, hidden_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.temporal_projector = ModalityProjector(temporal_dim, hidden_dim, latent_dim, dropout)
        self.spatial_projector = ModalityProjector(spatial_dim, hidden_dim, latent_dim, dropout)
        self.day_head = nn.Linear(latent_dim, len(DAY_ORDER))
        self.stage_head = nn.Linear(latent_dim, len(STAGE_ORDER))
        self.temporal_progress_head = nn.Linear(latent_dim, 1)
        self.age_bin_head = nn.Linear(latent_dim, len(AGE_BIN_ORDER))
        self.spatial_progress_head = nn.Linear(latent_dim, 1)

    def temporal(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.temporal_projector(x)
        return {
            "z": z,
            "day_logits": self.day_head(z),
            "stage_logits": self.stage_head(z),
            "progress": self.temporal_progress_head(z).squeeze(1),
        }

    def spatial(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.spatial_projector(x)
        return {
            "z": z,
            "age_bin_logits": self.age_bin_head(z),
            "progress": self.spatial_progress_head(z).squeeze(1),
        }


def cross_modal_contrastive(z_t: torch.Tensor, bin_t: torch.Tensor, z_s: torch.Tensor, bin_s: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = z_t @ z_s.T / temperature
    positives = (bin_t[:, None] == bin_s[None, :]).float()
    positives = positives / positives.sum(dim=1, keepdim=True).clamp(min=1.0)
    loss_t = -(positives * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
    positives_s = positives.T
    positives_s = positives_s / positives_s.sum(dim=1, keepdim=True).clamp(min=1.0)
    loss_s = -(positives_s * torch.log_softmax(logits.T, dim=1)).sum(dim=1).mean()
    return 0.5 * (loss_t + loss_s)


def evaluate_temporal(model: AlignmentModel, x: np.ndarray, metadata: pd.DataFrame, device: torch.device) -> dict:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(metadata), 2048):
            xb = torch.tensor(x[start : start + 2048], dtype=torch.float32, device=device)
            out = model.temporal(xb)
            preds.append(
                pd.DataFrame(
                    {
                        "pred_day_label": out["day_logits"].argmax(dim=1).cpu().numpy(),
                        "pred_stage_label": out["stage_logits"].argmax(dim=1).cpu().numpy(),
                        "pred_progress": out["progress"].cpu().numpy(),
                    }
                )
            )
    pred = pd.concat(preds, ignore_index=True)
    return {
        "num_cells": int(len(metadata)),
        "day_accuracy": float(accuracy_score(metadata["day_label"], pred["pred_day_label"])),
        "day_macro_f1": float(f1_score(metadata["day_label"], pred["pred_day_label"], average="macro", zero_division=0)),
        "stage_accuracy": float(accuracy_score(metadata["stage_label"], pred["pred_stage_label"])),
        "stage_macro_f1": float(f1_score(metadata["stage_label"], pred["pred_stage_label"], average="macro", zero_division=0)),
        "progress_mae": float(mean_absolute_error(metadata["progress"], pred["pred_progress"])),
    }


def evaluate_spatial(model: AlignmentModel, x: np.ndarray, metadata: pd.DataFrame, device: torch.device) -> tuple[dict, pd.DataFrame]:
    model.eval()
    with torch.no_grad():
        out = model.spatial(torch.tensor(x, dtype=torch.float32, device=device))
    pred = metadata.copy()
    pred["pred_age_bin_label"] = out["age_bin_logits"].argmax(dim=1).cpu().numpy()
    pred["pred_age_bin"] = pred["pred_age_bin_label"].map({idx: label for idx, label in enumerate(AGE_BIN_ORDER)})
    pred["pred_progress"] = out["progress"].cpu().numpy()
    metrics = {
        "num_sections": int(len(metadata)),
        "age_bin_accuracy": float(accuracy_score(pred["age_bin_label"], pred["pred_age_bin_label"])),
        "age_bin_macro_f1": float(f1_score(pred["age_bin_label"], pred["pred_age_bin_label"], average="macro", zero_division=0)),
        "progress_mae": float(mean_absolute_error(pred["progress"], pred["pred_progress"])),
    }
    return metrics, pred


def cross_modal_retrieval(
    model: AlignmentModel,
    temporal_x: np.ndarray,
    temporal_meta: pd.DataFrame,
    spatial_x: np.ndarray,
    spatial_meta: pd.DataFrame,
    device: torch.device,
    top_k: int,
) -> tuple[dict, pd.DataFrame]:
    model.eval()
    temporal_z = []
    with torch.no_grad():
        for start in range(0, len(temporal_meta), 4096):
            temporal_z.append(model.temporal(torch.tensor(temporal_x[start : start + 4096], dtype=torch.float32, device=device))["z"].cpu().numpy())
        spatial_z = model.spatial(torch.tensor(spatial_x, dtype=torch.float32, device=device))["z"].cpu().numpy()
    temporal_z_np = np.vstack(temporal_z)
    similarities = spatial_z @ temporal_z_np.T
    rows = []
    for idx, row in spatial_meta.reset_index(drop=True).iterrows():
        k = min(top_k, similarities.shape[1])
        top_idx = np.argpartition(-similarities[idx], kth=k - 1)[:k]
        retrieved = temporal_meta.iloc[top_idx]
        retrieved_progress = float(retrieved["progress"].mean())
        retrieved_bin = AGE_BIN_ORDER[int(np.clip(np.round(retrieved_progress * (len(AGE_BIN_ORDER) - 1)), 0, len(AGE_BIN_ORDER) - 1))]
        rows.append(
            {
                "section": row["section"],
                "code": row["code"],
                "case": row["case"],
                "split": row["split"],
                "age_bin": row["age_bin"],
                "age_weeks": row["age_weeks"],
                "true_progress": row["progress"],
                "retrieved_temporal_progress": retrieved_progress,
                "retrieved_progress_bin": retrieved_bin,
                "top_day_counts": json.dumps(retrieved["diffday"].value_counts().to_dict()),
                "top_stage_counts": json.dumps(retrieved["stage"].value_counts().to_dict()),
            }
        )
    retrieval = pd.DataFrame(rows)
    metrics = {
        "num_sections": int(len(retrieval)),
        "top_k": int(top_k),
        "retrieval_progress_mae": float(mean_absolute_error(retrieval["true_progress"], retrieval["retrieved_temporal_progress"])),
        "retrieval_bin_accuracy": float(accuracy_score(retrieval["age_bin"], retrieval["retrieved_progress_bin"])),
        "retrieval_bin_macro_f1": float(f1_score(retrieval["age_bin"], retrieval["retrieved_progress_bin"], average="macro", zero_division=0)),
    }
    return metrics, retrieval


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temporal_x, temporal_meta = prepare_temporal(args)
    spatial_x, spatial_meta = prepare_spatial(args)
    train_temporal_mask = temporal_meta["split"] == "train"
    val_temporal_mask = temporal_meta["split"] == "val"
    train_spatial_mask = spatial_meta["split"] == "train"
    val_spatial_mask = spatial_meta["split"] == "val"
    test_spatial_mask = spatial_meta["split"] == "test"

    train_temporal = TemporalDataset(temporal_x[train_temporal_mask.to_numpy()], temporal_meta[train_temporal_mask].reset_index(drop=True))
    day_counts = train_temporal.metadata["day_label"].value_counts().to_dict()
    sample_weights = [1.0 / np.sqrt(day_counts[int(label)]) for label in train_temporal.metadata["day_label"]]
    loader = DataLoader(
        train_temporal,
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True),
        collate_fn=collate_temporal,
    )

    spatial_train_x = torch.tensor(spatial_x[train_spatial_mask.to_numpy()], dtype=torch.float32)
    spatial_train_age = torch.tensor(spatial_meta.loc[train_spatial_mask, "age_bin_label"].to_numpy(), dtype=torch.long)
    spatial_train_progress = torch.tensor(spatial_meta.loc[train_spatial_mask, "progress"].to_numpy(), dtype=torch.float32)
    spatial_train_bin = torch.tensor(spatial_meta.loc[train_spatial_mask, "progress_bin_label"].to_numpy(), dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlignmentModel(temporal_x.shape[1], spatial_x.shape[1], args.hidden_dim, args.latent_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()
    mse = nn.SmoothL1Loss(beta=0.08)

    spatial_train_x = spatial_train_x.to(device)
    spatial_train_age = spatial_train_age.to(device)
    spatial_train_progress = spatial_train_progress.to(device)
    spatial_train_bin = spatial_train_bin.to(device)

    best_val_score = -float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = Batch(
                x=batch.x.to(device),
                day_label=batch.day_label.to(device),
                stage_label=batch.stage_label.to(device),
                progress=batch.progress.to(device),
                progress_bin_label=batch.progress_bin_label.to(device),
                row_index=batch.row_index.to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            temporal_out = model.temporal(batch.x)
            spatial_out = model.spatial(spatial_train_x)
            temporal_loss = ce(temporal_out["day_logits"], batch.day_label) + ce(temporal_out["stage_logits"], batch.stage_label)
            temporal_progress_loss = mse(temporal_out["progress"], batch.progress)
            spatial_loss = ce(spatial_out["age_bin_logits"], spatial_train_age)
            spatial_progress_loss = mse(spatial_out["progress"], spatial_train_progress)
            contrastive_loss = cross_modal_contrastive(
                temporal_out["z"],
                batch.progress_bin_label,
                spatial_out["z"],
                spatial_train_bin,
                args.temperature,
            )
            loss = (
                args.temporal_loss_weight * temporal_loss
                + args.spatial_loss_weight * spatial_loss
                + args.progress_loss_weight * (temporal_progress_loss + spatial_progress_loss)
                + args.contrastive_weight * contrastive_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_temporal_metrics = evaluate_temporal(model, temporal_x[val_temporal_mask.to_numpy()], temporal_meta[val_temporal_mask].reset_index(drop=True), device)
        val_spatial_metrics, _ = evaluate_spatial(model, spatial_x[val_spatial_mask.to_numpy()], spatial_meta[val_spatial_mask].reset_index(drop=True), device)
        val_score = val_temporal_metrics["stage_macro_f1"] + val_spatial_metrics["age_bin_macro_f1"] - val_spatial_metrics["progress_mae"]
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_temporal": val_temporal_metrics,
            "val_spatial": val_spatial_metrics,
            "val_score": float(val_score),
        }
        history.append(row)
        if epoch == 1 or epoch % 10 == 0:
            logger.info("Epoch %03d/%03d %s", epoch, args.epochs, json.dumps(row))
        if val_score > best_val_score:
            best_val_score = val_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                logger.info("Early stopping after %d stale epochs.", stale)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_temporal = evaluate_temporal(model, temporal_x[val_temporal_mask.to_numpy()], temporal_meta[val_temporal_mask].reset_index(drop=True), device)
    final_val_spatial, val_spatial_pred = evaluate_spatial(model, spatial_x[val_spatial_mask.to_numpy()], spatial_meta[val_spatial_mask].reset_index(drop=True), device)
    final_test_spatial, test_spatial_pred = evaluate_spatial(model, spatial_x[test_spatial_mask.to_numpy()], spatial_meta[test_spatial_mask].reset_index(drop=True), device)
    retrieval_all, retrieval_df = cross_modal_retrieval(
        model,
        temporal_x,
        temporal_meta.reset_index(drop=True),
        spatial_x,
        spatial_meta.reset_index(drop=True),
        device,
        args.top_k,
    )
    retrieval_test, retrieval_test_df = cross_modal_retrieval(
        model,
        temporal_x,
        temporal_meta.reset_index(drop=True),
        spatial_x[test_spatial_mask.to_numpy()],
        spatial_meta[test_spatial_mask].reset_index(drop=True),
        device,
        args.top_k,
    )

    summary = {
        "model": "weakly_supervised_spatiotemporal_alignment",
        "important_caveat": "This is inferred 4D alignment, not true longitudinal tracking of identical cells or tissue regions.",
        "inputs": {
            "temporal_embeddings": args.temporal_embeddings,
            "temporal_cells_loaded": int(len(temporal_meta)),
            "spatial_embeddings": args.spatial_embeddings,
            "spatial_sections": int(len(spatial_meta)),
        },
        "label_bridge": {
            "temporal_progress": "diffday / 15",
            "spatial_progress": "(fetal_week - 6) / 6",
            "coarse_alignment_bins": AGE_BIN_ORDER,
        },
        "final_val_temporal_metrics": final_val_temporal,
        "final_val_spatial_metrics": final_val_spatial,
        "final_test_spatial_metrics": final_test_spatial,
        "cross_modal_retrieval_all_sections": retrieval_all,
        "cross_modal_retrieval_test_sections": retrieval_test,
        "history": history,
    }
    val_spatial_pred.to_csv(output_dir / f"{args.output_prefix}_val_spatial_predictions.csv", index=False)
    test_spatial_pred.to_csv(output_dir / f"{args.output_prefix}_test_spatial_predictions.csv", index=False)
    retrieval_df.to_csv(output_dir / f"{args.output_prefix}_cross_modal_retrieval_all_sections.csv", index=False)
    retrieval_test_df.to_csv(output_dir / f"{args.output_prefix}_cross_modal_retrieval_test_sections.csv", index=False)
    (output_dir / f"{args.output_prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), output_dir / f"{args.output_prefix}_alignment_model.pt")
    print(json.dumps({key: summary[key] for key in ["important_caveat", "final_val_temporal_metrics", "final_val_spatial_metrics", "final_test_spatial_metrics", "cross_modal_retrieval_test_sections"]}, indent=2))


if __name__ == "__main__":
    main()

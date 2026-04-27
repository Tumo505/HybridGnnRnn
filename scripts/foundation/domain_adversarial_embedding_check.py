#!/usr/bin/env python3
"""Check whether embeddings separate biology from donor/batch/domain signal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--h5ad", default="foundation_model_data/geneformer/gse175634_geneformer.h5ad")
    parser.add_argument("--target", choices=["diffday", "cell_state"], default="cell_state")
    parser.add_argument("--domain-col", default="line_id")
    parser.add_argument("--output-dir", default="foundation_validation/domain_adversarial")
    parser.add_argument("--prefix", default="domain_adversarial_check")
    parser.add_argument("--max-cells", type=int, default=80000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lambda-domain", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class DomainAdversarialMLP(nn.Module):
    def __init__(self, input_dim: int, target_classes: int, domain_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.ReLU())
        self.target_head = nn.Linear(128, target_classes)
        self.domain_head = nn.Linear(128, domain_classes)

    def forward(self, x, alpha: float):
        z = self.encoder(x)
        target_logits = self.target_head(z)
        domain_logits = self.domain_head(GradientReverse.apply(z, alpha))
        return target_logits, domain_logits


def load_embeddings(path: Path) -> np.ndarray:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    feature_cols = [col for col in columns if col != "Unnamed: 0"]
    return pd.read_csv(path, usecols=feature_cols).to_numpy(dtype=np.float32)


def cap_indices(y: np.ndarray, max_cells: int, seed: int) -> np.ndarray:
    if len(y) <= max_cells:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    selected = []
    per_class = max(1, max_cells // len(np.unique(y)))
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        selected.extend(rng.choice(idx, size=min(per_class, len(idx)), replace=False).tolist())
    if len(selected) < max_cells:
        rest = np.setdiff1d(np.arange(len(y)), np.array(selected), assume_unique=False)
        selected.extend(rng.choice(rest, size=min(max_cells - len(selected), len(rest)), replace=False).tolist())
    return np.array(sorted(selected), dtype=int)


def evaluate(model, x, y_target, y_domain, device) -> dict:
    model.eval()
    with torch.no_grad():
        target_logits, domain_logits = model(torch.tensor(x, dtype=torch.float32, device=device), alpha=0.0)
    target_pred = target_logits.argmax(dim=1).cpu().numpy()
    domain_pred = domain_logits.argmax(dim=1).cpu().numpy()
    return {
        "target_accuracy": float(accuracy_score(y_target, target_pred)),
        "target_macro_f1": float(f1_score(y_target, target_pred, average="macro", zero_division=0)),
        "domain_accuracy": float(accuracy_score(y_domain, domain_pred)),
        "domain_macro_f1": float(f1_score(y_domain, domain_pred, average="macro", zero_division=0)),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x = load_embeddings(Path(args.embeddings))
    obs = ad.read_h5ad(args.h5ad, backed="r").obs
    target_raw = obs[args.target].astype(str).to_numpy()
    domain_raw = obs[args.domain_col].astype(str).to_numpy()
    idx = cap_indices(target_raw, args.max_cells, args.seed)
    x = x[idx]
    target_raw = target_raw[idx]
    domain_raw = domain_raw[idx]
    x = StandardScaler().fit_transform(x).astype(np.float32)
    target_encoder = LabelEncoder()
    domain_encoder = LabelEncoder()
    y_target = target_encoder.fit_transform(target_raw)
    y_domain = domain_encoder.fit_transform(domain_raw)

    train_idx, test_idx = train_test_split(
        np.arange(len(x)),
        test_size=0.2,
        random_state=args.seed,
        stratify=y_domain,
    )
    train_ds = TensorDataset(
        torch.tensor(x[train_idx], dtype=torch.float32),
        torch.tensor(y_target[train_idx], dtype=torch.long),
        torch.tensor(y_domain[train_idx], dtype=torch.long),
    )
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DomainAdversarialMLP(x.shape[1], len(target_encoder.classes_), len(domain_encoder.classes_)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        alpha = args.lambda_domain * float(epoch + 1) / float(args.epochs)
        for xb, yb, db in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            db = db.to(device)
            opt.zero_grad()
            target_logits, domain_logits = model(xb, alpha=alpha)
            loss = loss_fn(target_logits, yb) + loss_fn(domain_logits, db)
            loss.backward()
            opt.step()

    train_metrics = evaluate(model, x[train_idx], y_target[train_idx], y_domain[train_idx], device)
    random_domain_test_metrics = evaluate(model, x[test_idx], y_target[test_idx], y_domain[test_idx], device)

    group_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    _, heldout_line_idx = next(group_splitter.split(x, y_target, groups=domain_raw))
    heldout_line_metrics = evaluate(model, x[heldout_line_idx], y_target[heldout_line_idx], y_domain[heldout_line_idx], device)
    summary = {
        "target": args.target,
        "domain_col": args.domain_col,
        "num_cells": int(len(x)),
        "num_target_classes": int(len(target_encoder.classes_)),
        "num_domain_classes": int(len(domain_encoder.classes_)),
        "train_metrics": train_metrics,
        "random_domain_test_metrics": random_domain_test_metrics,
        "heldout_line_probe_metrics": heldout_line_metrics,
        "interpretation": (
            "High random-domain test accuracy means embeddings retain donor/batch signal. "
            "Held-out-line domain accuracy is expected to be poor because those exact line labels were unseen."
        ),
    }
    (output_dir / f"{args.prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

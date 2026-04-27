#!/usr/bin/env python3
"""Fast repeated held-out-line CV on exported Geneformer embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--h5ad", default="foundation_model_data/geneformer/gse175634_geneformer.h5ad")
    parser.add_argument("--target", choices=["diffday", "cell_state"], default="cell_state")
    parser.add_argument("--group-col", default="line_id")
    parser.add_argument("--output-dir", default="foundation_validation")
    parser.add_argument("--prefix", default="embedding_group_cv")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-cells", type=int, default=None, help="Optional stratified-ish cell cap for faster smoke tests.")
    return parser.parse_args()


def load_embeddings(path: Path) -> np.ndarray:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    feature_cols = [col for col in columns if col != "Unnamed: 0"]
    return pd.read_csv(path, usecols=feature_cols).to_numpy(dtype=np.float32)


def maybe_cap_indices(y: np.ndarray, max_cells: int | None, seed: int) -> np.ndarray:
    if max_cells is None or len(y) <= max_cells:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    selected = []
    labels = np.unique(y)
    per_label = max(1, max_cells // len(labels))
    for label in labels:
        label_idx = np.flatnonzero(y == label)
        take = min(per_label, len(label_idx))
        selected.extend(rng.choice(label_idx, size=take, replace=False).tolist())
    remaining = max_cells - len(selected)
    if remaining > 0:
        pool = np.setdiff1d(np.arange(len(y)), np.array(selected), assume_unique=False)
        selected.extend(rng.choice(pool, size=min(remaining, len(pool)), replace=False).tolist())
    return np.array(sorted(selected), dtype=int)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x = load_embeddings(Path(args.embeddings))
    obs = ad.read_h5ad(args.h5ad, backed="r").obs
    if len(obs) != x.shape[0]:
        raise ValueError(f"Embedding rows ({x.shape[0]}) do not match h5ad obs rows ({len(obs)}).")
    y_raw = obs[args.target].astype(str).to_numpy()
    groups = obs[args.group_col].astype(str).to_numpy()
    idx = maybe_cap_indices(y_raw, args.max_cells, args.seed)
    x = x[idx]
    y_raw = y_raw[idx]
    groups = groups[idx]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    splitter = GroupShuffleSplit(n_splits=args.repeats, test_size=args.test_size, random_state=args.seed)
    split_records = []
    aggregate_cm = np.zeros((len(encoder.classes_), len(encoder.classes_)), dtype=np.int64)

    for split_idx, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), start=1):
        clf = make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                max_iter=1000,
                tol=1e-3,
                random_state=args.seed + split_idx,
            ),
        )
        clf.fit(x[train_idx], y[train_idx])
        pred = clf.predict(x[test_idx])
        aggregate_cm += confusion_matrix(y[test_idx], pred, labels=np.arange(len(encoder.classes_)))
        split_records.append(
            {
                "split": split_idx,
                "num_train_cells": int(len(train_idx)),
                "num_test_cells": int(len(test_idx)),
                "train_lines": ",".join(sorted(set(groups[train_idx]))),
                "test_lines": ",".join(sorted(set(groups[test_idx]))),
                "accuracy": float(accuracy_score(y[test_idx], pred)),
                "macro_f1": float(f1_score(y[test_idx], pred, average="macro", zero_division=0)),
            }
        )

    split_df = pd.DataFrame(split_records)
    split_df.to_csv(output_dir / f"{args.prefix}_splits.csv", index=False)
    cm_df = pd.DataFrame(aggregate_cm, index=encoder.classes_, columns=encoder.classes_)
    cm_df.to_csv(output_dir / f"{args.prefix}_aggregate_confusion_matrix.csv")
    summary = {
        "target": args.target,
        "num_cells": int(len(y)),
        "num_lines": int(len(set(groups))),
        "repeats": int(args.repeats),
        "mean_accuracy": float(split_df["accuracy"].mean()),
        "std_accuracy": float(split_df["accuracy"].std(ddof=0)),
        "mean_macro_f1": float(split_df["macro_f1"].mean()),
        "std_macro_f1": float(split_df["macro_f1"].std(ddof=0)),
    }
    (output_dir / f"{args.prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

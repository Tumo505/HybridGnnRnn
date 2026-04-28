#!/usr/bin/env python3
"""Evaluate external Geneformer predictions against recovered external labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="CSV from predict_geneformer_classifier.py")
    parser.add_argument("--target-column", required=True, help="Column containing external label, e.g. diffday or cell_state")
    parser.add_argument("--output-dir", default="foundation_validation/external_unseen")
    parser.add_argument("--prefix", default="external_eval")
    parser.add_argument("--prediction-column", default="pred_class")
    parser.add_argument("--accepted-column", default="accepted_by_confidence")
    parser.add_argument("--compatible-only", action="store_true", help="Drop OOD/unknown/unmapped target labels before scoring.")
    parser.add_argument("--drop-target", action="append", default=[], help="Target label to drop. Can be passed multiple times.")
    return parser.parse_args()


def clean_target_mask(series: pd.Series) -> pd.Series:
    text = series.astype(str)
    upper = text.str.upper()
    return (
        text.notna()
        & ~upper.isin({"", "NAN", "NA", "UNKNOWN", "UNK", "-1", "OOD_UNMAPPED", "SPATIAL_SPOT_UNLABELED"})
        & ~text.str.startswith("week", na=False)
    )


def summarize(df: pd.DataFrame, target_col: str, pred_col: str) -> dict:
    labels = sorted(set(df[target_col].astype(str)).union(set(df[pred_col].astype(str))))
    target_labels = sorted(set(df[target_col].astype(str)))
    report = classification_report(df[target_col], df[pred_col], labels=labels, output_dict=True, zero_division=0)
    return {
        "num_cells": int(len(df)),
        "accuracy": float(accuracy_score(df[target_col], df[pred_col])),
        "macro_f1": float(f1_score(df[target_col], df[pred_col], labels=labels, average="macro", zero_division=0)),
        "target_class_macro_f1": float(
            f1_score(df[target_col], df[pred_col], labels=target_labels, average="macro", zero_division=0)
        ),
        "target_class_weighted_f1": float(
            f1_score(df[target_col], df[pred_col], labels=target_labels, average="weighted", zero_division=0)
        ),
        "target_counts": df[target_col].value_counts().to_dict(),
        "prediction_counts": df[pred_col].value_counts().to_dict(),
        "classification_report": report,
    }


def write_confusion(df: pd.DataFrame, target_col: str, pred_col: str, output_dir: Path, prefix: str) -> None:
    labels = sorted(set(df[target_col].astype(str)).union(set(df[pred_col].astype(str))))
    cm = confusion_matrix(df[target_col], df[pred_col], labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_dir / f"{prefix}_confusion_matrix.csv")
    top_confusions = (
        df.loc[df[target_col].astype(str) != df[pred_col].astype(str)]
        .groupby([target_col, pred_col])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    top_confusions.to_csv(output_dir / f"{prefix}_top_confusions.csv", index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.predictions)
    if args.target_column not in df.columns:
        raise ValueError(f"{args.predictions} has no target column {args.target_column!r}")
    if args.prediction_column not in df.columns:
        raise ValueError(f"{args.predictions} has no prediction column {args.prediction_column!r}")

    df[args.target_column] = df[args.target_column].astype(str)
    df[args.prediction_column] = df[args.prediction_column].astype(str)
    for value in args.drop_target:
        df = df.loc[df[args.target_column] != value].copy()
    if args.compatible_only:
        df = df.loc[clean_target_mask(df[args.target_column])].copy()

    summary = {
        "predictions": args.predictions,
        "target_column": args.target_column,
        "prediction_column": args.prediction_column,
        "compatible_only": bool(args.compatible_only),
        "dropped_targets": args.drop_target,
        "all_scored": summarize(df, args.target_column, args.prediction_column) if len(df) else None,
    }
    write_confusion(df, args.target_column, args.prediction_column, output_dir, args.prefix)

    if args.accepted_column in df.columns:
        accepted = df.loc[df[args.accepted_column].astype(str).str.lower().isin({"true", "1"})].copy()
        rejected = df.loc[~df.index.isin(accepted.index)].copy()
        summary["accepted"] = summarize(accepted, args.target_column, args.prediction_column) if len(accepted) else None
        summary["accepted_fraction"] = float(len(accepted) / len(df)) if len(df) else 0.0
        summary["rejected_count"] = int(len(rejected))
        accepted.to_csv(output_dir / f"{args.prefix}_accepted_predictions.csv", index=False)

    df.to_csv(output_dir / f"{args.prefix}_scored_predictions.csv", index=False)
    summary_path = output_dir / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

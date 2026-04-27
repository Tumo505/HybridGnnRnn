#!/usr/bin/env python3
"""Stronger validation reports for Geneformer trajectory classifiers."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


MARKER_SETS = {
    "IPSC": ["POU5F1", "NANOG", "SOX2", "LIN28A", "DPPA4"],
    "MES": ["T", "MIXL1", "MESP1", "EOMES", "TBX6"],
    "CMES": ["MESP1", "NKX2-5", "ISL1", "HAND1", "TBX5"],
    "PROG": ["ISL1", "NKX2-5", "TBX5", "HAND2", "MEF2C"],
    "CM": ["TNNT2", "MYH6", "MYH7", "TNNI3", "ACTN2", "MYL2", "MYL7", "PLN"],
    "CF": ["COL1A1", "COL1A2", "DCN", "LUM", "POSTN", "VIM"],
}

EXPECTED_STATE_MARKERS = {
    "IPSC": {"IPSC"},
    "MES": {"MES"},
    "CMES": {"CMES", "PROG", "MES"},
    "PROG": {"PROG", "CMES", "CM"},
    "CM": {"CM"},
    "CF": {"CF"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dict", help="Geneformer *_pred_dict.pkl file.")
    parser.add_argument("--id-class-dict", help="Geneformer *_id_class_dict.pkl file.")
    parser.add_argument("--h5ad", default=None, help="Optional Geneformer-ready h5ad for marker-gene sanity checks.")
    parser.add_argument("--output-dir", default="foundation_validation")
    parser.add_argument("--prefix", default="geneformer_validation")
    return parser.parse_args()


def load_pickle(path: str | Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def as_list(values):
    if values is None:
        return []
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def prediction_frame(pred_dict: dict, id_to_class: dict[int, str]) -> pd.DataFrame:
    pred_ids = as_list(pred_dict.get("pred_ids"))
    label_ids = as_list(pred_dict.get("label_ids"))
    metadata = pred_dict.get("prediction_metadata") or {}
    meta_df = pd.DataFrame(metadata)
    rows = []
    for idx, (label_id, pred_id) in enumerate(zip(label_ids, pred_ids)):
        if str(pred_id).lower() == "tie":
            continue
        try:
            label_int = int(label_id)
            pred_int = int(pred_id)
        except (TypeError, ValueError):
            continue
        rows.append((idx, label_int, pred_int))
    df = pd.DataFrame(rows, columns=["prediction_index", "y_true_id", "y_pred_id"])
    if not meta_df.empty:
        meta_df = meta_df.reset_index(drop=True)
        df = df.join(meta_df.iloc[df["prediction_index"].to_numpy()].reset_index(drop=True))
    df["y_true"] = df["y_true_id"].map(id_to_class).fillna(df["y_true_id"].astype(str))
    df["y_pred"] = df["y_pred_id"].map(id_to_class).fillna(df["y_pred_id"].astype(str))
    return df


def group_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    records = []
    for group, sub in df.groupby(group_col, dropna=False):
        labels = sorted(set(sub["y_true"]).union(set(sub["y_pred"])))
        records.append(
            {
                group_col: group,
                "num_cells": int(len(sub)),
                "accuracy": float(accuracy_score(sub["y_true"], sub["y_pred"])),
                "macro_f1": float(f1_score(sub["y_true"], sub["y_pred"], labels=labels, average="macro", zero_division=0)),
                "class_count": int(sub["y_true"].nunique()),
            }
        )
    return pd.DataFrame(records).sort_values(["accuracy", "num_cells"], ascending=[True, False])


def write_prediction_reports(df: pd.DataFrame, id_to_class: dict[int, str], output_dir: Path, prefix: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [id_to_class[key] for key in sorted(id_to_class)]
    labels = [label for label in labels if label in set(df["y_true"]).union(set(df["y_pred"]))]

    report = classification_report(df["y_true"], df["y_pred"], labels=labels, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_dir / f"{prefix}_classification_report.csv")

    cm = confusion_matrix(df["y_true"], df["y_pred"], labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(output_dir / f"{prefix}_confusion_matrix.csv")

    top_confusions = (
        df.loc[df["y_true"] != df["y_pred"]]
        .groupby(["y_true", "y_pred"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    top_confusions.to_csv(output_dir / f"{prefix}_top_confusions.csv", index=False)

    if "line_id" in df.columns:
        group_metrics(df, "line_id").to_csv(output_dir / f"{prefix}_per_line_metrics.csv", index=False)
    if "sample_id" in df.columns:
        group_metrics(df, "sample_id").to_csv(output_dir / f"{prefix}_per_sample_metrics.csv", index=False)

    summary = {
        "num_predictions": int(len(df)),
        "accuracy": float(accuracy_score(df["y_true"], df["y_pred"])),
        "macro_f1": float(f1_score(df["y_true"], df["y_pred"], labels=labels, average="macro", zero_division=0)),
        "label_counts": df["y_true"].value_counts().to_dict(),
        "prediction_counts": df["y_pred"].value_counts().to_dict(),
        "top_confusions": top_confusions.head(20).to_dict(orient="records"),
    }
    return summary


def marker_sanity(h5ad_path: Path, output_dir: Path, prefix: str) -> dict:
    import anndata as ad
    from scipy import sparse

    adata = ad.read_h5ad(h5ad_path)
    gene_names = adata.var.get("gene_name", pd.Series(adata.var_names, index=adata.var_names)).astype(str)
    uppercase_to_index = {name.upper(): idx for idx, name in enumerate(gene_names)}

    coverage_records = []
    score_columns = {}
    for set_name, genes in MARKER_SETS.items():
        indices = [uppercase_to_index[gene.upper()] for gene in genes if gene.upper() in uppercase_to_index]
        coverage_records.append(
            {"marker_set": set_name, "requested_genes": len(genes), "found_genes": len(indices), "found_fraction": len(indices) / len(genes)}
        )
        if indices:
            expr = adata.X[:, indices]
            if sparse.issparse(expr):
                expr = expr.toarray()
            score_columns[set_name] = np.log1p(np.asarray(expr, dtype=np.float32)).mean(axis=1)
        else:
            score_columns[set_name] = np.full(adata.n_obs, np.nan, dtype=np.float32)

    score_df = pd.DataFrame(score_columns, index=adata.obs_names)
    for column in ["cell_state", "diffday", "line_id", "sample_id"]:
        if column in adata.obs:
            score_df[column] = adata.obs[column].astype(str).to_numpy()
    coverage_df = pd.DataFrame(coverage_records)
    coverage_df.to_csv(output_dir / f"{prefix}_marker_gene_coverage.csv", index=False)

    summary: dict[str, object] = {"marker_gene_coverage": coverage_df.to_dict(orient="records")}
    marker_cols = list(MARKER_SETS)
    if "cell_state" in score_df.columns:
        by_state = score_df.groupby("cell_state")[marker_cols].mean().sort_index()
        by_state.to_csv(output_dir / f"{prefix}_marker_scores_by_state.csv")
        state_calls = []
        for state, row in by_state.iterrows():
            best_set = str(row.idxmax())
            expected = EXPECTED_STATE_MARKERS.get(str(state), set())
            state_calls.append(
                {
                    "cell_state": str(state),
                    "top_marker_set": best_set,
                    "expected_marker_sets": sorted(expected),
                    "passes_expected_marker_check": bool(not expected or best_set in expected),
                }
            )
        pd.DataFrame(state_calls).to_csv(output_dir / f"{prefix}_marker_state_checks.csv", index=False)
        summary["state_marker_checks"] = state_calls
    if "diffday" in score_df.columns:
        by_day = score_df.groupby("diffday")[marker_cols].mean().sort_index()
        by_day.to_csv(output_dir / f"{prefix}_marker_scores_by_day.csv")
        summary["day_marker_top_sets"] = {str(day): str(row.idxmax()) for day, row in by_day.iterrows()}
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {}

    if args.pred_dict and args.id_class_dict:
        id_to_class = {int(key): value for key, value in load_pickle(args.id_class_dict).items()}
        pred_dict = load_pickle(args.pred_dict)
        df = prediction_frame(pred_dict, id_to_class)
        df.to_csv(output_dir / f"{args.prefix}_predictions.csv", index=False)
        summary["prediction_validation"] = write_prediction_reports(df, id_to_class, output_dir, args.prefix)

    if args.h5ad:
        summary["marker_sanity"] = marker_sanity(Path(args.h5ad), output_dir, args.prefix)

    summary_path = output_dir / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate biological diagnostics for Mauron RNN/GNN result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


RELATED_GROUPS = {
    "cardiomyocyte": ["CM", "aCM", "vCM", "Mat_", "MetAct", "Immat"],
    "fibroblast_epicardial": ["FB", "EPDC"],
    "endothelial": ["EC", "Endoc", "LEC"],
    "smooth_muscle_mural": ["SMC", "MC", "PC", "Peric"],
    "neural": ["NB-N", "SCP", "GC"],
    "immune": ["LyC", "MyC"],
    "excluded_or_other": ["HL_excl", "TMSB10high"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rnn-results", default="mauron_developmental_rnn_fresh_case_split/mauron_developmental_rnn_results.json")
    parser.add_argument("--gnn-results", default="mauron_spatial_gnn_fresh_case_split/mauron_spatial_gnn_results.json")
    parser.add_argument("--hybrid-results", default=None)
    parser.add_argument("--gnn-ablation-results", default=None)
    parser.add_argument("--output-dir", default="mauron_biological_diagnostics")
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_metrics(payload: Dict) -> Dict:
    if "split_metrics" in payload:
        return payload["split_metrics"]["test"]
    return payload["finetune"]["split_metrics"]["test"]


def validation_metrics(payload: Dict) -> Dict:
    if "split_metrics" in payload:
        return payload["split_metrics"]["val"]
    return payload["finetune"]["split_metrics"]["val"]


def label_names(payload: Dict) -> List[str]:
    names = payload.get("label_names") or payload.get("class_names")
    if names:
        return list(names)
    metadata = payload.get("dataset_metadata", {})
    return list(metadata.get("label_names") or metadata.get("graph_label_names") or [])


def biological_group(label: str) -> str:
    for group, tokens in RELATED_GROUPS.items():
        if any(token in label for token in tokens):
            return group
    return "other"


def per_class_table(model_name: str, payload: Dict) -> pd.DataFrame:
    metrics = test_metrics(payload)
    report = metrics["classification_report"]
    records = []
    for label, values in report.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        if not isinstance(values, dict) or "support" not in values:
            continue
        support = int(values.get("support", 0))
        if support <= 0:
            status = "not_present_in_test"
        elif values.get("f1-score", 0.0) >= 0.70:
            status = "strong"
        elif values.get("f1-score", 0.0) >= 0.40:
            status = "moderate"
        elif values.get("f1-score", 0.0) >= 0.20:
            status = "weak"
        else:
            status = "poor"
        records.append(
            {
                "model": model_name,
                "cell_type": label,
                "biological_group": biological_group(label),
                "support": support,
                "precision": float(values.get("precision", 0.0)),
                "recall": float(values.get("recall", 0.0)),
                "f1": float(values.get("f1-score", 0.0)),
                "status": status,
            }
        )
    return pd.DataFrame(records).sort_values(["model", "f1", "support"], ascending=[True, False, False])


def confusion_table(model_name: str, payload: Dict) -> pd.DataFrame:
    metrics = test_metrics(payload)
    matrix = metrics.get("confusion_matrix")
    names = label_names(payload)
    if matrix is None or not names:
        return pd.DataFrame()
    cm = np.asarray(matrix, dtype=np.int64)
    if cm.shape[0] != len(names):
        names = names[: cm.shape[0]]
    rows = []
    for true_idx, true_label in enumerate(names):
        total = int(cm[true_idx].sum())
        if total == 0:
            continue
        for pred_idx, pred_label in enumerate(names):
            if true_idx == pred_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            if count == 0:
                continue
            rows.append(
                {
                    "model": model_name,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "true_group": biological_group(true_label),
                    "predicted_group": biological_group(pred_label),
                    "count": count,
                    "fraction_of_true": count / total,
                    "within_related_group": biological_group(true_label) == biological_group(pred_label),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["count", "fraction_of_true"], ascending=False)


def group_confusion_table(model_name: str, payload: Dict) -> pd.DataFrame:
    metrics = test_metrics(payload)
    matrix = metrics.get("confusion_matrix")
    names = label_names(payload)
    if matrix is None or not names:
        return pd.DataFrame()
    cm = np.asarray(matrix, dtype=np.int64)
    if cm.shape[0] != len(names):
        names = names[: cm.shape[0]]
    groups = sorted({biological_group(name) for name in names})
    group_to_id = {group: idx for idx, group in enumerate(groups)}
    agg = np.zeros((len(groups), len(groups)), dtype=np.int64)
    for true_idx, true_label in enumerate(names):
        for pred_idx, pred_label in enumerate(names):
            agg[group_to_id[biological_group(true_label)], group_to_id[biological_group(pred_label)]] += cm[true_idx, pred_idx]
    rows = []
    for true_group, true_id in group_to_id.items():
        total = int(agg[true_id].sum())
        if total == 0:
            continue
        for pred_group, pred_id in group_to_id.items():
            rows.append(
                {
                    "model": model_name,
                    "true_group": true_group,
                    "predicted_group": pred_group,
                    "count": int(agg[true_id, pred_id]),
                    "fraction_of_true": float(agg[true_id, pred_id] / total),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "true_group", "count"], ascending=[True, True, False])


def model_summary(model_name: str, payload: Dict) -> Dict:
    val = validation_metrics(payload)
    test = test_metrics(payload)
    args = payload.get("args", {})
    return {
        "model": model_name,
        "spatial_ablation": args.get("spatial_ablation", ""),
        "val_accuracy": val.get("accuracy"),
        "val_macro_f1": val.get("macro_f1"),
        "test_accuracy": test.get("accuracy"),
        "test_macro_f1": test.get("macro_f1"),
        "test_weighted_f1": test.get("weighted_f1"),
        "num_test_examples": test.get("num_examples"),
    }


def write_markdown(
    output_dir: Path,
    summary: pd.DataFrame,
    per_class: pd.DataFrame,
    confusions: pd.DataFrame,
    group_confusions: pd.DataFrame,
) -> None:
    lines = [
        "# Mauron Biological Diagnostics",
        "",
        "## Model Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Strongest Cell Types",
        "",
        per_class.sort_values("f1", ascending=False).head(15).to_markdown(index=False),
        "",
        "## Weakest Present Cell Types",
        "",
        per_class[per_class["support"] > 0].sort_values(["f1", "support"], ascending=[True, False]).head(15).to_markdown(index=False),
        "",
    ]
    if not confusions.empty:
        lines.extend(
            [
                "## Top Related-Population Confusions",
                "",
                confusions.head(25).to_markdown(index=False),
                "",
            ]
        )
    if not group_confusions.empty:
        lines.extend(
            [
                "## Aggregated Biological Group Confusions",
                "",
                group_confusions.to_markdown(index=False),
                "",
            ]
        )
    (output_dir / "biological_diagnostics_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads = []
    result_paths = [("RNN", args.rnn_results), ("GNN", args.gnn_results)]
    if args.hybrid_results:
        result_paths.append(("Hybrid", args.hybrid_results))
    if args.gnn_ablation_results:
        result_paths.append(("GNN_ablation", args.gnn_ablation_results))
    for model_name, path_text in result_paths:
        path = Path(path_text)
        if path.exists():
            payloads.append((model_name, load_json(path)))

    summary = pd.DataFrame([model_summary(name, payload) for name, payload in payloads])
    per_class = pd.concat([per_class_table(name, payload) for name, payload in payloads], ignore_index=True)
    confusion_frames = [confusion_table(name, payload) for name, payload in payloads]
    confusions = pd.concat([frame for frame in confusion_frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in confusion_frames) else pd.DataFrame()
    group_frames = [group_confusion_table(name, payload) for name, payload in payloads]
    group_confusions = pd.concat([frame for frame in group_frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in group_frames) else pd.DataFrame()

    summary.to_csv(output_dir / "model_summary.csv", index=False)
    per_class.to_csv(output_dir / "per_class_biological_review.csv", index=False)
    if not confusions.empty:
        confusions.to_csv(output_dir / "top_cell_type_confusions.csv", index=False)
    if not group_confusions.empty:
        group_confusions.to_csv(output_dir / "biological_group_confusions.csv", index=False)
    write_markdown(output_dir, summary, per_class, confusions, group_confusions)
    print(f"Saved biological diagnostics to {output_dir}")


if __name__ == "__main__":
    main()

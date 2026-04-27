#!/usr/bin/env python3
"""Re-evaluate a saved Mauron GNN checkpoint and refresh split metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.train_mauron_spatial_gnn import (  # noqa: E402
    apply_spatial_ablation,
    metrics_for,
    predict_graphs,
    split_graphs,
)
from src.data_processing.mauron_visium_processor import MauronBuildConfig, MauronVisiumGraphDataset  # noqa: E402
from src.models.gnn_models.mauron_spatial_gnn import MauronSpatialGNN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="mauron_spatial_gnn_fresh_case_split/mauron_spatial_gnn_results.json")
    parser.add_argument("--update", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    train_args = payload["args"]

    build_config = MauronBuildConfig(
        data_root=train_args["data_root"],
        cache_dir=train_args["cache_dir"],
        num_genes=train_args["num_genes"],
        k_neighbors=train_args["k_neighbors"],
        label_mode=train_args["label_mode"],
        min_gene_spots=train_args["min_gene_spots"],
        max_sections=train_args.get("max_sections"),
    )
    graphs, metadata = MauronVisiumGraphDataset(build_config).load_or_build(force_rebuild=False)
    apply_spatial_ablation(graphs, train_args.get("spatial_ablation", "real"), train_args["seed"])
    splits = split_graphs(graphs, train_args["split_group"], train_args["seed"])
    label_names = metadata["label_names"] or metadata["graph_label_names"]
    model_config = payload["model_path"]
    checkpoint = torch.load(model_config, map_location="cpu")
    cfg = checkpoint["model_config"]
    model = MauronSpatialGNN(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        embedding_dim=cfg["embedding_dim"],
        num_classes=cfg["num_classes"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    split_metrics = {}
    for split_name, indices in splits.items():
        y_true, y_pred = predict_graphs(model, graphs, indices, payload["task_level"], torch.device("cpu"))
        split_metrics[split_name] = metrics_for(y_true, y_pred, label_names)

    if args.update:
        payload["finetune"]["split_metrics"] = split_metrics
        results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Updated {results_path}")
    else:
        print(json.dumps(split_metrics["test"], indent=2)[:2000])


if __name__ == "__main__":
    main()

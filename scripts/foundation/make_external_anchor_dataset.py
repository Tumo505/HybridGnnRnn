#!/usr/bin/env python3
"""Combine GSE175634 with QC-passed external iPSC anchors for Geneformer fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import concatenate_datasets, load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-dataset", default="foundation_model_data/geneformer/tokenized/gse175634_geneformer.dataset")
    parser.add_argument(
        "--external-dataset",
        default="foundation_model_data/geneformer_external_gse130731_topcells/tokenized/gse130731_ips_topcells_geneformer.dataset",
    )
    parser.add_argument("--output-dir", default="foundation_model_data/geneformer_anchor_augmented")
    parser.add_argument("--output-prefix", default="gse175634_plus_gse130731_ips_anchor")
    parser.add_argument("--external-max-cells", type=int, default=None)
    return parser.parse_args()


def mark_external_anchor(example: dict) -> dict:
    example["cell_state"] = "IPSC"
    example["diffday"] = "day0"
    example["diffday_numeric"] = 0
    example["line_id"] = f"external_gse130731_{example.get('sample_id', 'ips')}"
    example["sample_id"] = f"external_gse130731_{example.get('sample_id', 'ips')}"
    example["dpt_pseudotime"] = 0.0
    return example


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    internal = load_from_disk(args.internal_dataset)
    external = load_from_disk(args.external_dataset)
    if args.external_max_cells is not None:
        external = external.select(range(min(args.external_max_cells, len(external))))
    external = external.map(mark_external_anchor)
    keep_columns = [col for col in internal.column_names if col in external.column_names]
    combined = concatenate_datasets([internal.select_columns(keep_columns), external.select_columns(keep_columns)])
    output_path = output_dir / f"{args.output_prefix}.dataset"
    combined.save_to_disk(str(output_path))
    summary = {
        "output_dataset": str(output_path),
        "internal_cells": int(len(internal)),
        "external_anchor_cells": int(len(external)),
        "combined_cells": int(len(combined)),
        "external_anchor_label": {"cell_state": "IPSC", "diffday": "day0"},
        "note": "External anchors are QC-passed high-count GSE130731 iPS cells; use only as an early-state/domain anchor.",
    }
    (output_dir / f"{args.output_prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

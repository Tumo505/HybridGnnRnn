#!/usr/bin/env python3
"""Prepare Mauron Visium spots as Geneformer-ready h5ad input.

The previously fine-tuned Geneformer operates on tokenized expression profiles.
This script converts Mauron spatial transcriptomics spots into the same input
format while preserving spatial metadata needed by the downstream graph adapter:

- section/case identifiers
- real Visium coordinates
- fetal age bin/progress
- chamber combination
- deconvolution argmax cell-state labels

The output is an h5ad file with raw counts, ``var["ensembl_id"]``, and
``obs["n_counts"]`` so it can be tokenized by Geneformer's tokenizer.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Iterable

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.mauron_visium_processor import (  # noqa: E402
    MauronBuildConfig,
    MauronVisiumGraphDataset,
    _chamber_combo,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("prepare_mauron_spatial_geneformer_input")


AGE_BIN_ORDER = ["early_w6_w7_5", "mid_w8_w9", "late_w10_w12"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=(
            "data/New/Mauron_spatial_dynamics_part_a/"
            "Spatial dynamics of the developing human heart, pa"
        ),
    )
    parser.add_argument("--output-dir", default="foundation_spatial_geneformer_adapter_inputs")
    parser.add_argument("--output-prefix", default="mauron_spatial_geneformer")
    parser.add_argument("--max-spots-per-section", type=int, default=None)
    parser.add_argument("--max-total-spots", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_10x_h5_with_features(path: Path) -> tuple[sparse.csc_matrix, list[str], list[str], list[str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        matrix = sparse.csc_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=tuple(group["shape"][:]),
        )
        barcodes = [value.decode("utf-8") for value in group["barcodes"][:]]
        features = group["features"]
        gene_ids = [value.decode("utf-8").split(".")[0] for value in features["id"][:]]
        gene_names = [value.decode("utf-8") for value in features["name"][:]]
    return matrix, barcodes, gene_ids, gene_names


def load_hl_deconvolution(dataset: MauronVisiumGraphDataset) -> tuple[dict[tuple[int, str], tuple[str, float]], dict[str, str]]:
    table = pd.read_csv(dataset.hl_deconv_path, sep="\t", index_col=0)
    argmax_labels = table.idxmax(axis=1).astype(str)
    confidence = table.max(axis=1).astype(float)
    parsed = table.index.to_series().str.extract(r"^(?P<barcode>.+)_(?P<section>[0-9]+)$")
    lookup: dict[tuple[int, str], tuple[str, float]] = {}
    for parsed_row, label, score in zip(parsed.itertuples(index=False), argmax_labels.to_numpy(), confidence.to_numpy()):
        if pd.isna(parsed_row.section) or pd.isna(parsed_row.barcode):
            continue
        lookup[(int(parsed_row.section), str(parsed_row.barcode))] = (str(label), float(score))

    label_name_map = {label: label for label in table.columns.astype(str)}
    if dataset.hl_annotation_path.exists():
        annot = pd.read_csv(dataset.hl_annotation_path, sep=";")
        if {"cluster", "cell_type"}.issubset(annot.columns):
            for _, row in annot.iterrows():
                cluster = str(row["cluster"]).replace("_", "-")
                label_name_map[cluster] = str(row["cell_type"])
                label_name_map[cluster.replace("-", "_")] = str(row["cell_type"])
    return lookup, label_name_map


def choose_indices(indices: Iterable[int], max_items: int | None, rng: random.Random) -> list[int]:
    selected = list(indices)
    if max_items is not None and len(selected) > max_items:
        selected = sorted(rng.sample(selected, max_items))
    return selected


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = MauronBuildConfig(data_root=args.data_root, num_genes=512)
    dataset = MauronVisiumGraphDataset(config)
    sections = dataset._read_section_metadata()
    deconv_lookup, label_name_map = load_hl_deconvolution(dataset)

    matrices: list[sparse.csr_matrix] = []
    obs_rows: list[dict] = []
    reference_gene_ids: list[str] | None = None
    reference_gene_names: list[str] | None = None

    total_spots = 0
    for _, section_row in sections.iterrows():
        section = int(section_row["Section"])
        code = str(section_row["Code"])
        matrix, barcodes, gene_ids, gene_names = read_10x_h5_with_features(dataset._matrix_path(code))
        if reference_gene_ids is None:
            reference_gene_ids = gene_ids
            reference_gene_names = gene_names
        elif gene_ids != reference_gene_ids:
            raise ValueError(f"Gene order differs in section {code}.")

        positions = dataset._read_tissue_positions(code)
        barcode_to_col = {barcode: idx for idx, barcode in enumerate(barcodes)}
        positions = positions[positions["barcode"].isin(barcode_to_col)].copy()
        positions["has_deconv"] = positions["barcode"].map(lambda barcode: (section, str(barcode)) in deconv_lookup)
        positions = positions[positions["has_deconv"]].copy()
        if positions.empty:
            logger.warning("Skipping section %s (%s): no deconvolution-labelled tissue spots.", section, code)
            continue

        kept_position_rows = choose_indices(positions.index.tolist(), args.max_spots_per_section, rng)
        positions = positions.loc[kept_position_rows].copy()
        if args.max_total_spots is not None:
            remaining = args.max_total_spots - total_spots
            if remaining <= 0:
                break
            positions = positions.iloc[:remaining].copy()

        col_indices = [barcode_to_col[str(barcode)] for barcode in positions["barcode"]]
        matrices.append(matrix[:, col_indices].T.tocsr())
        progress = float((float(section_row["age_weeks"]) - 6.0) / 6.0)
        progress = float(np.clip(progress, 0.0, 1.0))
        chamber_combo = _chamber_combo(section_row)
        for _, pos_row in positions.iterrows():
            label_key, label_conf = deconv_lookup[(section, str(pos_row["barcode"]))]
            obs_rows.append(
                {
                    "cell": f"{code}_{pos_row['barcode']}",
                    "spot_id": f"{code}_{pos_row['barcode']}",
                    "barcode": str(pos_row["barcode"]),
                    "section_id": section,
                    "code": code,
                    "sample_id": code,
                    "case": str(section_row["Case"]),
                    "line_id": str(section_row["Case"]),
                    "age": str(section_row["Age"]),
                    "age_weeks": float(section_row["age_weeks"]),
                    "age_bin": str(section_row["age_bin"]),
                    "spatial_progress": progress,
                    "chamber_combo": chamber_combo,
                    "x": float(pos_row["pxl_col"]),
                    "y": float(pos_row["pxl_row"]),
                    "array_row": int(pos_row["array_row"]),
                    "array_col": int(pos_row["array_col"]),
                    "deconv_label": label_key,
                    "cell_state": label_name_map.get(label_key, label_key),
                    "deconv_confidence": float(label_conf),
                    "source_dataset": "MauronVisium",
                    "diffday": str(section_row["age_bin"]),
                    "diffday_numeric": float(section_row["age_weeks"]),
                    "dpt_pseudotime_numeric": progress,
                }
            )
        total_spots += len(positions)
        logger.info("Prepared %s spots from section %s (%s).", len(positions), section, code)

    if not matrices or reference_gene_ids is None or reference_gene_names is None:
        raise RuntimeError("No Mauron spatial spots were prepared.")

    x = sparse.vstack(matrices, format="csr")
    obs = pd.DataFrame(obs_rows).set_index("cell", drop=False)
    obs["n_counts"] = np.asarray(x.sum(axis=1)).ravel().astype(np.float32)
    keep = obs["n_counts"].to_numpy() > 0
    x = x[keep]
    obs = obs.loc[keep].copy()

    var = pd.DataFrame(
        {
            "ensembl_id": reference_gene_ids,
            "gene_name": reference_gene_names,
        },
        index=pd.Index(reference_gene_ids, name="ensembl_id_index"),
    )
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.var_names_make_unique()

    output_path = output_dir / f"{args.output_prefix}.h5ad"
    adata.write_h5ad(output_path)
    summary = {
        "output_h5ad": str(output_path),
        "shape": list(adata.shape),
        "sections": int(obs["section_id"].nunique()),
        "cases": int(obs["case"].nunique()),
        "age_bin_counts": obs["age_bin"].value_counts().to_dict(),
        "cell_state_counts_top20": obs["cell_state"].value_counts().head(20).to_dict(),
        "important_caveat": "Mauron Visium spots are spatial snapshots; this prepares inferred spatial-time inputs, not longitudinally tracked cells.",
    }
    (output_dir / f"{args.output_prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

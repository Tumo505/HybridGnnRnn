#!/usr/bin/env python3
"""Prepare independent external datasets for Geneformer validation.

The primary goal is to create manuscript-grade external validation inputs
without accidentally using training/anchor data:

- GSE202398: filtered 10x H5 matrices from an independent hiPSC cardiac
  differentiation atlas. CellPlex antibody tags in the H5 are used to infer
  line and day labels.
- Asp developmental heart: filtered human developmental heart scRNA-seq and
  spatial transcriptomics metadata. The scRNA-seq data are used for conservative
  external state validation where labels map cleanly to our broad states.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import pickle
import re
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("prepare_external_validation_inputs")

TRAINED_DAY_CLASSES = {"day0", "day1", "day3", "day5", "day7", "day11", "day15"}

ASP_STATE_MAP = {
    "Ventricular cardiomyocytes": "CM",
    "Atrial cardiomyocytes": "CM",
    "Myoz2-enriched cardiomyocytes": "CM",
    "Fibroblast-like": "CF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["gse202398", "asp_scrna", "asp_spatial"], required=True)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default="foundation_model_data/geneformer_external_unseen")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--matched-days-only", action="store_true", help="For GSE202398, retain only classes present in the trained day classifier.")
    parser.add_argument("--min-hashtag-count", type=int, default=5)
    parser.add_argument("--min-hashtag-margin", type=float, default=1.5)
    parser.add_argument("--gene-id-map", default=None, help="Optional gene_name/ensembl_id mapping for Asp scRNA gene symbols.")
    parser.add_argument("--allow-gene-symbols-as-ids", action="store_true")
    return parser.parse_args()


def clean_ensembl(values: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(values, dtype="string").str.replace(r"\.\d+$", "", regex=True)


def read_gene_map(path: str | None) -> dict[str, str]:
    if path is None:
        try:
            import geneformer

            bundled_map = Path(geneformer.__file__).resolve().parent / "gene_name_id_dict_gc104M.pkl"
            if bundled_map.exists():
                with open(bundled_map, "rb") as handle:
                    mapping = pickle.load(handle)
                return {str(k): str(v).replace(".0", "") for k, v in mapping.items()}
        except Exception as exc:
            logger.warning("Could not load Geneformer bundled gene-name map: %s", exc)
        return {}
    map_path = Path(path)
    sep = "\t" if map_path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(map_path, sep=sep)
    required = {"gene_name", "ensembl_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{map_path} is missing columns: {sorted(missing)}")
    df = df.dropna(subset=["gene_name", "ensembl_id"])
    return dict(zip(df["gene_name"].astype(str), clean_ensembl(df["ensembl_id"]).astype(str)))


def parse_cmo_label(label: str) -> tuple[str, str]:
    match = re.match(r"(?P<line>[^_]+)_DAY(?P<day>\d+)$", str(label).strip())
    if not match:
        return "unknown", "unknown"
    line = match.group("line")
    day = f"day{int(match.group('day'))}"
    return line, day


def day_to_numeric(day: str) -> int:
    match = re.search(r"(\d+)", str(day))
    return int(match.group(1)) if match else -1


def heuristic_state_from_day(day: str) -> str:
    numeric = day_to_numeric(day)
    if numeric == 0:
        return "IPSC"
    if numeric in {1, 2}:
        return "MES"
    if numeric in {3, 4}:
        return "CMES"
    if numeric in {5, 6, 7}:
        return "PROG"
    if numeric >= 11:
        return "CM"
    return "unknown"


def read_10x_h5_filtered(path: Path, min_hashtag_count: int, min_hashtag_margin: float) -> tuple[ad.AnnData, dict]:
    logger.info("Reading %s", path.name)
    with h5py.File(path, "r") as handle:
        matrix_group = handle["matrix"]
        shape = tuple(int(x) for x in matrix_group["shape"][()])
        matrix = sparse.csc_matrix(
            (
                matrix_group["data"][()],
                matrix_group["indices"][()],
                matrix_group["indptr"][()],
            ),
            shape=shape,
            dtype=np.int32,
        )
        barcodes = [x.decode("utf-8", errors="replace") for x in matrix_group["barcodes"][()]]
        feature_names = [x.decode("utf-8", errors="replace") for x in matrix_group["features"]["name"][()]]
        feature_ids = [x.decode("utf-8", errors="replace") for x in matrix_group["features"]["id"][()]]
        feature_types = [x.decode("utf-8", errors="replace") for x in matrix_group["features"]["feature_type"][()]]

    gene_idx = np.array([idx for idx, value in enumerate(feature_types) if value == "Gene Expression"], dtype=int)
    cmo_idx = np.array([idx for idx, value in enumerate(feature_types) if value == "Antibody Capture"], dtype=int)
    if len(cmo_idx) == 0:
        raise ValueError(f"{path} does not contain Antibody Capture features for CellPlex demultiplexing.")

    gene_matrix = matrix[gene_idx, :].transpose().tocsr()
    cmo_matrix = matrix[cmo_idx, :].transpose().tocsr()
    cmo_dense = cmo_matrix.toarray()
    best_idx = np.argmax(cmo_dense, axis=1)
    sorted_counts = np.sort(cmo_dense, axis=1)
    best_count = sorted_counts[:, -1]
    second_count = sorted_counts[:, -2] if cmo_dense.shape[1] > 1 else np.zeros_like(best_count)
    margin = best_count / np.maximum(second_count, 1)
    keep = (best_count >= min_hashtag_count) & (margin >= min_hashtag_margin)

    cmo_labels = [feature_names[cmo_idx[idx]] for idx in best_idx]
    line_day = [parse_cmo_label(label) for label in cmo_labels]
    line = [value[0] for value in line_day]
    day = [value[1] for value in line_day]
    run_id = path.name.replace("_filtered_feature_bc_matrix.h5", "")
    obs = pd.DataFrame(index=[f"{run_id}_{barcode}" for barcode in barcodes])
    obs["cell"] = obs.index.astype(str)
    obs["barcode"] = barcodes
    obs["sample_id"] = run_id
    obs["source_dataset"] = "GSE202398"
    obs["line_id"] = line
    obs["external_cellplex_tag"] = cmo_labels
    obs["external_hashtag_best_count"] = best_count.astype(np.float32)
    obs["external_hashtag_second_count"] = second_count.astype(np.float32)
    obs["external_hashtag_margin"] = margin.astype(np.float32)
    obs["diffday"] = day
    obs["diffday_numeric"] = np.array([day_to_numeric(value) for value in day], dtype=np.int16)
    obs["dpt_pseudotime_numeric"] = np.clip(obs["diffday_numeric"].to_numpy(dtype=np.float32) / 30.0, 0.0, 1.0)
    obs["cell_state"] = [heuristic_state_from_day(value) for value in day]
    obs["external_state_is_heuristic"] = True
    obs["filter_pass"] = 1
    obs["n_counts"] = np.asarray(gene_matrix.sum(axis=1)).ravel().astype(np.float32)

    keep = keep & (obs["n_counts"].to_numpy() > 0)
    gene_matrix = gene_matrix[keep]
    obs = obs.loc[keep].copy()

    var = pd.DataFrame(index=clean_ensembl(np.array(feature_ids)[gene_idx]).astype(str))
    var["ensembl_id"] = var.index.astype(str)
    var["gene_name"] = np.array(feature_names)[gene_idx]
    adata = ad.AnnData(X=gene_matrix, obs=obs, var=var)
    adata.var_names_make_unique()

    summary = {
        "file": str(path),
        "raw_filtered_barcodes": int(len(barcodes)),
        "retained_confident_cells": int(adata.n_obs),
        "cmo_tags": obs["external_cellplex_tag"].value_counts().to_dict(),
        "days": obs["diffday"].value_counts().sort_index().to_dict(),
        "lines": obs["line_id"].value_counts().to_dict(),
    }
    return adata, summary


def prepare_gse202398(args: argparse.Namespace) -> tuple[ad.AnnData, dict]:
    input_dir = Path(args.input_dir or "data/GSE202398_extracted")
    files = sorted(input_dir.glob("*_filtered_feature_bc_matrix.h5"))
    if not files:
        raise FileNotFoundError(f"No filtered 10x H5 files found in {input_dir}")
    adatas = []
    file_summaries = []
    for path in files:
        adata, summary = read_10x_h5_filtered(path, args.min_hashtag_count, args.min_hashtag_margin)
        adatas.append(adata)
        file_summaries.append(summary)
    combined = ad.concat(adatas, axis=0, join="outer", merge="same", index_unique=None)
    if args.matched_days_only:
        keep = combined.obs["diffday"].astype(str).isin(TRAINED_DAY_CLASSES).to_numpy()
        combined = combined[keep].copy()
    if args.max_cells is not None and combined.n_obs > args.max_cells:
        order = np.argsort(-combined.obs["n_counts"].to_numpy())[: args.max_cells]
        combined = combined[sorted(order)].copy()
    summary = {
        "dataset": "GSE202398",
        "source_role": "independent_external_validation",
        "input_dir": str(input_dir),
        "uses_filtered_matrices_only": True,
        "demultiplexing": {
            "method": "CellPlex antibody-capture argmax with count and margin filters",
            "min_hashtag_count": args.min_hashtag_count,
            "min_hashtag_margin": args.min_hashtag_margin,
        },
        "matched_days_only": bool(args.matched_days_only),
        "shape": [int(combined.n_obs), int(combined.n_vars)],
        "day_counts": combined.obs["diffday"].value_counts().sort_index().to_dict(),
        "line_counts": combined.obs["line_id"].value_counts().to_dict(),
        "state_counts_heuristic": combined.obs["cell_state"].value_counts().to_dict(),
        "files": file_summaries,
    }
    return combined, summary


def asp_base_dir(input_dir: str | None) -> Path:
    if input_dir:
        return Path(input_dir)
    return Path("data/New/Asp_developmental_heart_mendeley")


def find_one(base: Path, name: str) -> Path:
    matches = [p for p in base.rglob(name) if "__MACOSX" not in str(p)]
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {base}")
    return matches[0]


def prepare_asp_scrna(args: argparse.Namespace) -> tuple[ad.AnnData, dict]:
    base = asp_base_dir(args.input_dir)
    matrix_path = find_one(base, "all_cells_count_matrix_filtered.tsv.gz")
    meta_path = find_one(base, "all_cells_meta_data_filtered.tsv.gz")
    logger.info("Reading Asp scRNA-seq metadata from %s", meta_path)
    obs = pd.read_csv(meta_path, sep="\t", compression="gzip", index_col=0)
    obs.index = obs.index.astype(str)

    logger.info("Reading Asp scRNA-seq matrix from %s", matrix_path)
    matrix_df = pd.read_csv(matrix_path, sep="\t", compression="gzip", index_col=0)
    matrix_df.index = matrix_df.index.astype(str)
    cells = [cell for cell in matrix_df.columns.astype(str) if cell in obs.index]
    obs = obs.loc[cells].copy()
    matrix_df = matrix_df.loc[:, cells]

    gene_map = read_gene_map(args.gene_id_map)
    gene_names = matrix_df.index.astype(str).to_series(index=matrix_df.index)
    mapped = gene_names.map(gene_map)
    if mapped.isna().any():
        if not args.allow_gene_symbols_as_ids:
            logger.warning(
                "Dropping %d Asp scRNA genes without Ensembl mapping. Use --allow-gene-symbols-as-ids only for plumbing tests.",
                int(mapped.isna().sum()),
            )
            keep_genes = mapped.notna().to_numpy()
            matrix_df = matrix_df.iloc[keep_genes, :]
            gene_names = gene_names.iloc[keep_genes]
            mapped = mapped.iloc[keep_genes]
        else:
            mapped = mapped.fillna(gene_names)

    matrix = sparse.csr_matrix(matrix_df.transpose().to_numpy(dtype=np.int32))
    obs["cell"] = obs.index.astype(str)
    obs["sample_id"] = obs.get("experiment", "Asp_scRNA").astype(str).replace("nan", "unknown")
    obs["source_dataset"] = "Asp_developmental_heart_mendeley_scRNA"
    obs["line_id"] = obs["sample_id"].astype(str)
    cleaned_celltype = obs["celltype"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    obs["external_celltype"] = cleaned_celltype
    obs["cell_state"] = cleaned_celltype.map(ASP_STATE_MAP).fillna("OOD_UNMAPPED")
    obs["external_state_is_direct_label_map"] = obs["cell_state"] != "OOD_UNMAPPED"
    obs["diffday"] = "unknown"
    obs["diffday_numeric"] = np.int16(-1)
    obs["dpt_pseudotime_numeric"] = np.float32(0.0)
    obs["filter_pass"] = 1
    obs["n_counts"] = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)

    keep = obs["n_counts"].to_numpy() > 0
    matrix = matrix[keep]
    obs = obs.loc[keep].copy()
    if args.max_cells is not None and matrix.shape[0] > args.max_cells:
        order = np.argsort(-obs["n_counts"].to_numpy())[: args.max_cells]
        order = np.sort(order)
        matrix = matrix[order]
        obs = obs.iloc[order].copy()

    var = pd.DataFrame(index=clean_ensembl(mapped.astype(str)).to_numpy())
    var["ensembl_id"] = var.index.astype(str)
    var["gene_name"] = gene_names.to_numpy()
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.var_names_make_unique()
    summary = {
        "dataset": "Asp_developmental_heart_mendeley_scRNA",
        "source_role": "independent_external_validation",
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "matrix_path": str(matrix_path),
        "metadata_path": str(meta_path),
        "celltype_counts": obs["external_celltype"].value_counts().to_dict(),
        "mapped_state_counts": obs["cell_state"].value_counts().to_dict(),
        "compatible_state_cells": int((obs["cell_state"] != "OOD_UNMAPPED").sum()),
        "validation_use": "state-level compatible-subset validation for CM/CF plus OOD rejection for unmapped classes",
    }
    return adata, summary


def prepare_asp_spatial(args: argparse.Namespace) -> tuple[ad.AnnData, dict]:
    base = asp_base_dir(args.input_dir)
    matrix_path = find_one(base, "filtered_matrix.tsv.gz")
    meta_path = find_one(base, "meta_data.tsv.gz")
    logger.info("Reading Asp spatial metadata from %s", meta_path)
    obs = pd.read_csv(meta_path, sep="\t", compression="gzip", index_col=0)
    obs.index = obs.index.astype(str)
    logger.info("Reading Asp spatial matrix from %s", matrix_path)
    matrix_df = pd.read_csv(matrix_path, sep="\t", compression="gzip", index_col=0)
    matrix_df.index = clean_ensembl(matrix_df.index).astype(str)
    spots = [spot for spot in matrix_df.columns.astype(str) if spot in obs.index]
    obs = obs.loc[spots].copy()
    matrix_df = matrix_df.loc[:, spots]
    matrix = sparse.csr_matrix(matrix_df.transpose().to_numpy(dtype=np.int32))
    obs["cell"] = obs.index.astype(str)
    obs["sample_id"] = obs["Sample"].astype(str)
    obs["source_dataset"] = "Asp_developmental_heart_mendeley_spatial"
    obs["line_id"] = obs["Sample"].astype(str)
    obs["diffday"] = "week" + obs["weeks"].astype(str)
    obs["diffday_numeric"] = obs["weeks"].astype(int) * 7
    obs["dpt_pseudotime_numeric"] = np.clip(obs["diffday_numeric"].to_numpy(dtype=np.float32) / 63.0, 0.0, 1.0)
    obs["cell_state"] = "spatial_spot_unlabeled"
    obs["filter_pass"] = 1
    obs["n_counts"] = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
    obs["array_x"] = obs["new_x"].astype(float)
    obs["array_y"] = obs["new_y"].astype(float)
    var = pd.DataFrame(index=matrix_df.index.astype(str))
    var["ensembl_id"] = var.index.astype(str)
    var["gene_name"] = var.index.astype(str)
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.var_names_make_unique()
    summary = {
        "dataset": "Asp_developmental_heart_mendeley_spatial",
        "source_role": "independent_external_spatial_ood_projection",
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "matrix_path": str(matrix_path),
        "metadata_path": str(meta_path),
        "week_counts": obs["weeks"].value_counts().sort_index().to_dict(),
        "sample_counts": obs["Sample"].value_counts().to_dict(),
        "has_real_coordinates": True,
        "validation_use": "spatial/OOD projection and graph construction; not direct supervised accuracy against Mauron labels without label harmonization",
    }
    return adata, summary


def write_outputs(adata: ad.AnnData, summary: dict, output_dir: Path, output_prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    h5ad_path = output_dir / f"{output_prefix}.h5ad"
    summary_path = output_dir / f"{output_prefix}_summary.json"
    adata.write_h5ad(h5ad_path, compression="gzip")
    summary["output_h5ad"] = str(h5ad_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", h5ad_path)
    logger.info("Wrote %s", summary_path)
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    default_prefix = {
        "gse202398": "gse202398_filtered_cellplex_geneformer",
        "asp_scrna": "asp_developmental_heart_scrna_geneformer",
        "asp_spatial": "asp_developmental_heart_spatial_geneformer",
    }[args.dataset]
    output_prefix = args.output_prefix or default_prefix
    if args.dataset == "gse202398":
        adata, summary = prepare_gse202398(args)
    elif args.dataset == "asp_scrna":
        adata, summary = prepare_asp_scrna(args)
    else:
        adata, summary = prepare_asp_spatial(args)
    write_outputs(adata, summary, Path(args.output_dir), output_prefix)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Prepare local cardiac datasets for Geneformer tokenization.

Geneformer expects raw-count single-cell data in h5ad/loom format with:
- var["ensembl_id"]
- obs["n_counts"]

This script first targets GSE175634, our temporal iPSC-cardiomyocyte
differentiation dataset. The local GSE175634 files contain gene symbols, so the
script maps them to Ensembl IDs with Geneformer's bundled dictionary or an
optional user-provided map.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import io, sparse


os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("prepare_geneformer_input")

DAY_ORDER = ["day0", "day1", "day3", "day5", "day7", "day11", "day15"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["gse175634", "gse130731_10x_ips"], default="gse175634")
    parser.add_argument("--input-dir", default="data/GSE175634_temporal_data")
    parser.add_argument("--output-dir", default="foundation_model_data/geneformer")
    parser.add_argument("--output-prefix", default="gse175634_geneformer")
    parser.add_argument(
        "--gene-id-map",
        default=None,
        help="TSV/CSV with gene_name and ensembl_id columns. Defaults to Geneformer's bundled gene-name map if installed.",
    )
    parser.add_argument("--max-cells", type=int, default=None, help="Optional smoke-test cell cap.")
    parser.add_argument(
        "--max-cells-per-sample",
        type=int,
        default=2000,
        help="Cell cap per sample for external 10x MatrixMarket datasets.",
    )
    parser.add_argument(
        "--allow-gene-symbols-as-ids",
        action="store_true",
        help="Write gene symbols into var['ensembl_id'] when no map is available. This is only for plumbing tests.",
    )
    return parser.parse_args()


def read_gene_map(path: str | None) -> dict[str, str]:
    if path is None:
        try:
            import geneformer

            geneformer_root = Path(geneformer.__file__).resolve().parent
            bundled_map = geneformer_root / "gene_name_id_dict_gc104M.pkl"
            if bundled_map.exists():
                with open(bundled_map, "rb") as handle:
                    return pickle.load(handle)
        except Exception as exc:
            logger.warning("Could not load Geneformer's bundled gene-name map: %s", exc)
        return {}
    map_path = Path(path)
    if not map_path.exists():
        raise FileNotFoundError(map_path)
    sep = "\t" if map_path.suffix.lower() in {".tsv", ".txt"} else ","
    mapping_df = pd.read_csv(map_path, sep=sep)
    required = {"gene_name", "ensembl_id"}
    missing = required.difference(mapping_df.columns)
    if missing:
        raise ValueError(f"{map_path} is missing columns: {sorted(missing)}")
    mapping_df = mapping_df.dropna(subset=["gene_name", "ensembl_id"])
    mapping_df["gene_name"] = mapping_df["gene_name"].astype(str)
    mapping_df["ensembl_id"] = mapping_df["ensembl_id"].astype(str)
    return dict(zip(mapping_df["gene_name"], mapping_df["ensembl_id"]))


def numeric_day(value: str) -> int:
    text = str(value).lower().replace("day", "")
    return int(text)


def read_matrix_market_subset_columns(path: Path, max_columns: int) -> sparse.csr_matrix:
    rows = []
    cols = []
    data = []
    n_rows = None
    n_cols = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("%"):
                continue
            n_rows, n_cols, _ = map(int, line.strip().split())
            break
        if n_rows is None or n_cols is None:
            raise ValueError(f"Could not read MatrixMarket shape from {path}")
        kept_columns = min(max_columns, n_cols)
        for line in handle:
            row_text, col_text, value_text = line.strip().split()
            col = int(col_text)
            if col > kept_columns:
                break
            rows.append(int(row_text) - 1)
            cols.append(col - 1)
            data.append(int(value_text))
    return sparse.coo_matrix((data, (rows, cols)), shape=(n_rows, kept_columns), dtype=np.int32).tocsr()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def read_10x_matrix_dir(matrix_dir: Path, max_cells: int | None) -> ad.AnnData:
    genes_path = matrix_dir / "genes.tsv"
    if not genes_path.exists():
        genes_path = matrix_dir / "features.tsv"
    barcodes_path = matrix_dir / "barcodes.tsv"
    matrix_path = matrix_dir / "matrix.mtx"
    if not matrix_path.exists():
        matrix_path = matrix_dir / "matrix.mtx.gz"
    for path in [genes_path, barcodes_path, matrix_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    genes = pd.read_csv(genes_path, sep="\t", header=None)
    if genes.shape[1] < 2:
        raise ValueError(f"{genes_path} should contain Ensembl IDs and gene names.")
    genes = genes.iloc[:, :2].copy()
    genes.columns = ["ensembl_id", "gene_name"]
    genes["ensembl_id"] = genes["ensembl_id"].astype(str).str.replace(r"\.\d+$", "", regex=True)
    genes["gene_name"] = genes["gene_name"].astype(str)

    barcodes = pd.read_csv(barcodes_path, sep="\t", header=None)[0].astype(str)
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    n_rows = None
    n_cols = None
    selected_columns = None
    with open_text(matrix_path) as handle:
        for line in handle:
            if line.startswith("%"):
                continue
            n_rows, n_cols, _ = map(int, line.strip().split())
            break
        if n_rows is None or n_cols is None:
            raise ValueError(f"Could not read MatrixMarket shape from {matrix_path}")
        if max_cells is not None and max_cells < n_cols:
            logger.info("Selecting top %d count barcodes from %s.", max_cells, matrix_dir.name)
            column_sums = np.zeros(int(n_cols), dtype=np.float64)
            for line in handle:
                _, col_text, value_text = line.strip().split()
                column_sums[int(col_text) - 1] += float(value_text)
            selected_columns = np.flatnonzero(column_sums > 0)
            if len(selected_columns) > max_cells:
                top = np.argpartition(column_sums[selected_columns], -max_cells)[-max_cells:]
                selected_columns = selected_columns[top]
            selected_columns = np.array(sorted(selected_columns), dtype=int)
        else:
            selected_columns = np.arange(int(n_cols), dtype=int)

    selected_set = set(int(col) for col in selected_columns)
    column_remap = {int(col): idx for idx, col in enumerate(selected_columns)}
    with open_text(matrix_path) as handle:
        for line in handle:
            if line.startswith("%"):
                continue
            n_rows, n_cols, _ = map(int, line.strip().split())
            break
        for line in handle:
            row_text, col_text, value_text = line.strip().split()
            col = int(col_text) - 1
            if col not in selected_set:
                continue
            rows.append(int(row_text) - 1)
            cols.append(column_remap[col])
            data.append(int(value_text))
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(int(n_rows), len(selected_columns)), dtype=np.int32).tocsr()
    matrix = matrix.transpose().tocsr()
    barcodes = barcodes.iloc[selected_columns].reset_index(drop=True)

    sample_id = matrix_dir.name.split(".")[0]
    obs = pd.DataFrame(index=[f"{sample_id}_{barcode}" for barcode in barcodes.iloc[: matrix.shape[0]]])
    obs["cell"] = obs.index.astype(str)
    obs["sample_id"] = sample_id
    obs["line_id"] = sample_id
    obs["external_label"] = sample_id
    obs["cell_state"] = "IPSC"
    obs["diffday"] = "day0"
    obs["diffday_numeric"] = np.int16(0)
    obs["dpt_pseudotime_numeric"] = np.float32(0.0)
    obs["filter_pass"] = 1
    obs["n_counts"] = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
    keep = obs["n_counts"].to_numpy() > 0
    matrix = matrix[keep]
    obs = obs.loc[keep].copy()

    var = pd.DataFrame(
        {
            "gene_name": genes["gene_name"].to_numpy(),
            "ensembl_id": genes["ensembl_id"].to_numpy(),
        }
    )
    var.index = var["ensembl_id"].astype(str)
    return ad.AnnData(X=matrix, obs=obs, var=var)


def build_gse175634_h5ad(args: argparse.Namespace) -> Path:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts_path = input_dir / "GSE175634_cell_counts.mtx.gz"
    genes_path = input_dir / "GSE175634_gene_indices_counts.tsv.gz"
    cells_path = input_dir / "GSE175634_cell_indices.tsv.gz"
    metadata_path = input_dir / "GSE175634_cell_metadata.tsv.gz"
    for path in [counts_path, genes_path, cells_path, metadata_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    logger.info("Reading GSE175634 metadata.")
    genes = pd.read_csv(genes_path, sep="\t")
    cells = pd.read_csv(cells_path, sep="\t")
    obs = pd.read_csv(metadata_path, sep="\t")

    if args.max_cells is not None:
        logger.info("Using first %d cells for smoke-test output.", args.max_cells)
        cells = cells.iloc[: args.max_cells].copy()
        obs = obs.iloc[: args.max_cells].copy()

    if not cells["cell_name"].equals(obs["cell"]):
        obs = cells.merge(obs, left_on="cell_name", right_on="cell", how="left")
        if obs["diffday"].isna().any():
            raise ValueError("Could not align GSE175634 cell indices to metadata.")
    else:
        obs = obs.copy()

    gene_map = read_gene_map(args.gene_id_map)
    genes["gene_name"] = genes["gene_name"].astype(str)
    if gene_map:
        genes["ensembl_id"] = genes["gene_name"].map(gene_map)
        mapped = int(genes["ensembl_id"].notna().sum())
        logger.info("Mapped %d/%d gene symbols to Ensembl IDs.", mapped, len(genes))
        keep_genes = genes["ensembl_id"].notna().to_numpy()
    elif args.allow_gene_symbols_as_ids:
        logger.warning("Using gene symbols as var['ensembl_id']; Geneformer tokenization will not be biologically valid.")
        genes["ensembl_id"] = genes["gene_name"]
        keep_genes = np.ones(len(genes), dtype=bool)
    else:
        raise ValueError(
            "GSE175634 stores gene symbols, but Geneformer requires Ensembl IDs. "
            "Provide --gene-id-map or use --allow-gene-symbols-as-ids only for a smoke test."
        )

    logger.info("Reading sparse raw count matrix.")
    if args.max_cells is not None:
        matrix = read_matrix_market_subset_columns(counts_path, args.max_cells).tocsc()
    else:
        matrix = io.mmread(counts_path).tocsc()
    matrix = matrix[keep_genes, :].transpose().tocsr()
    genes = genes.loc[keep_genes].reset_index(drop=True)

    obs["diffday"] = pd.Categorical(obs["diffday"], categories=DAY_ORDER, ordered=True)
    obs["diffday_numeric"] = obs["diffday"].astype(str).map(numeric_day).astype(np.int16)
    obs["cell_state"] = obs["type"].astype(str)
    obs["line_id"] = obs["individual"].astype(str)
    obs["sample_id"] = obs["sample"].astype(str)
    obs["filter_pass"] = 1
    obs["n_counts"] = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
    obs["dpt_pseudotime_numeric"] = pd.to_numeric(obs["dpt_pseudotime"], errors="coerce")
    obs.index = obs["cell"].astype(str)

    var = pd.DataFrame(
        {
            "gene_name": genes["gene_name"].astype(str).to_numpy(),
            "ensembl_id": genes["ensembl_id"].astype(str).to_numpy(),
        }
    )
    var.index = var["ensembl_id"]
    adata = ad.AnnData(X=sparse.csr_matrix(matrix), obs=obs, var=var)
    adata.uns["source_dataset"] = "GSE175634"
    adata.uns["trajectory_task"] = "human_iPSC_cardiomyocyte_differentiation"
    adata.uns["day_order"] = DAY_ORDER

    output_path = output_dir / f"{args.output_prefix}.h5ad"
    logger.info("Writing %s with shape %s.", output_path, adata.shape)
    adata.write_h5ad(output_path, compression="gzip")

    summary = {
        "output_h5ad": str(output_path),
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "day_counts": obs["diffday"].astype(str).value_counts().to_dict(),
        "state_counts": obs["cell_state"].value_counts().to_dict(),
        "line_count": int(obs["line_id"].nunique()),
        "gene_id_source": "gene_name_to_ensembl_map" if gene_map else "gene_symbols_smoke_test",
        "mapped_gene_count": int(adata.n_vars),
    }
    summary_path = output_dir / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote summary to %s.", summary_path)
    return output_path


def build_gse130731_10x_ips_h5ad(args: argparse.Namespace) -> Path:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dirs = sorted(path for path in input_dir.glob("iPS_*.barcodes.genes.matrix") if path.is_dir())
    if not sample_dirs:
        raise FileNotFoundError(f"No iPS_*.barcodes.genes.matrix directories found under {input_dir}")

    logger.info("Reading %d GSE130731 iPS 10x MatrixMarket samples.", len(sample_dirs))
    adatas = [read_10x_matrix_dir(path, args.max_cells_per_sample) for path in sample_dirs]
    adata = ad.concat(adatas, join="inner", label="source_sample", keys=[a.obs["sample_id"].iloc[0] for a in adatas])
    adata.var["ensembl_id"] = adata.var.index.astype(str)
    adata.var["gene_name"] = adatas[0].var.loc[adata.var.index, "gene_name"].astype(str).to_numpy()
    adata.uns["source_dataset"] = "GSE130731"
    adata.uns["trajectory_task"] = "external_iPSC_sanity_check"
    adata.uns["external_expected_behavior"] = {
        "cell_state": "IPSC",
        "diffday": "day0_or_other_early_day",
        "note": "This external dataset is not a cardiac differentiation benchmark; labels encode an expected iPSC sanity check.",
    }

    output_path = output_dir / f"{args.output_prefix}.h5ad"
    logger.info("Writing %s with shape %s.", output_path, adata.shape)
    adata.write_h5ad(output_path, compression="gzip")

    summary = {
        "output_h5ad": str(output_path),
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "sample_counts": adata.obs["sample_id"].value_counts().to_dict(),
        "expected_state": "IPSC",
        "expected_day": "day0_or_early",
        "mapped_gene_count": int(adata.n_vars),
        "external_test_type": "iPSC sanity/projection test, not full labeled cardiac trajectory accuracy",
    }
    summary_path = output_dir / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote summary to %s.", summary_path)
    return output_path


def main() -> None:
    args = parse_args()
    if args.dataset == "gse175634":
        build_gse175634_h5ad(args)
    elif args.dataset == "gse130731_10x_ips":
        build_gse130731_10x_ips_h5ad(args)


if __name__ == "__main__":
    main()

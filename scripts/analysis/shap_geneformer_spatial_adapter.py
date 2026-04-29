#!/usr/bin/env python3
"""SHAP explainability for the Geneformer spatial adapter.

The final spatial adapter consumes Geneformer embeddings, which are not directly
gene-interpretable. This script therefore trains a compact tree surrogate on the
same Mauron spot-level gene expression matrix to explain either the adapter's
predicted labels or the biological labels on held-out spots. SHAP values from
the surrogate provide gene-level attributions, while surrogate fidelity reports
how well those attributions approximate the adapter decision surface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder


MARKER_SETS = {
    "pluripotency": ["POU5F1", "NANOG", "SOX2", "LIN28A", "DPPA4"],
    "mesoderm": ["T", "MIXL1", "MESP1", "EOMES", "TBX6"],
    "cardiac_mesoderm": ["MESP1", "NKX2-5", "ISL1", "HAND1", "TBX5"],
    "cardiac_progenitor": ["ISL1", "NKX2-5", "TBX5", "HAND2", "MEF2C"],
    "cardiomyocyte": ["TNNT2", "MYH6", "MYH7", "TNNI3", "ACTN2", "MYL2", "MYL7", "PLN"],
    "ion_calcium": ["SCN5A", "KCNH2", "CACNA1C", "RYR2", "PLN", "ATP2A2"],
    "fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "POSTN", "VIM"],
    "endothelial": ["PECAM1", "VWF", "KDR", "CDH5", "ENG"],
    "epicardial": ["WT1", "TBX18", "TCF21", "ALDH1A2"],
    "smooth_muscle": ["ACTA2", "TAGLN", "MYH11", "CNN1"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", default="foundation_spatial_geneformer_adapter_inputs/mauron_spatial_geneformer.h5ad")
    parser.add_argument("--adapter-dir", default="foundation_geneformer_spatial_context_adapter_improved")
    parser.add_argument("--adapter-prefix", default="mauron_geneformer_spatial_adapter_improved")
    parser.add_argument("--output-dir", default="foundation_validation/shap_geneformer_spatial_adapter")
    parser.add_argument(
        "--target",
        choices=["pred_family", "true_family", "pred_cell_state", "true_cell_state"],
        default="pred_family",
        help="Use pred_* to explain the adapter decision surface; use true_* to explain labels.",
    )
    parser.add_argument("--top-genes", type=int, default=512)
    parser.add_argument("--max-train", type=int, default=2500)
    parser.add_argument("--max-explain", type=int, default=500)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="mauron_geneformer_spatial_adapter_shap")
    return parser.parse_args()


def load_predictions(adapter_dir: Path, adapter_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = adapter_dir / f"{adapter_prefix}_train_node_predictions.csv"
    test_path = adapter_dir / f"{adapter_prefix}_test_node_predictions.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing adapter prediction CSVs under {adapter_dir}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train["split"] = "train"
    test["split"] = "test"
    return train, test


def target_column(target: str) -> str:
    return {
        "pred_family": "pred_family",
        "true_family": "true_family",
        "pred_cell_state": "pred_cell_state",
        "true_cell_state": "true_cell_state",
    }[target]


def stratified_sample(df: pd.DataFrame, label_col: str, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.copy()
    rng = np.random.default_rng(seed)
    labels = sorted(df[label_col].astype(str).unique())
    per_label = max(1, max_rows // max(1, len(labels)))
    selected: list[int] = []
    for label in labels:
        idx = df.index[df[label_col].astype(str) == label].to_numpy()
        take = min(len(idx), per_label)
        if take:
            selected.extend(rng.choice(idx, size=take, replace=False).tolist())
    remaining = max_rows - len(selected)
    if remaining > 0:
        pool = np.setdiff1d(df.index.to_numpy(), np.asarray(selected, dtype=int), assume_unique=False)
        if len(pool):
            selected.extend(rng.choice(pool, size=min(remaining, len(pool)), replace=False).tolist())
    return df.loc[sorted(selected)].copy()


def index_predictions(adata: ad.AnnData, predictions: pd.DataFrame) -> pd.DataFrame:
    obs = adata.obs.copy()
    if "spot_id" not in obs.columns:
        raise ValueError("h5ad obs must contain a spot_id column.")
    spot_to_row = pd.Series(np.arange(len(obs), dtype=np.int64), index=obs["spot_id"].astype(str))
    pred = predictions.copy()
    pred["spot_id"] = pred["spot_id"].astype(str)
    pred["adata_row"] = pred["spot_id"].map(spot_to_row)
    pred = pred.dropna(subset=["adata_row"]).copy()
    pred["adata_row"] = pred["adata_row"].astype(np.int64)
    return pred


def as_dense_float(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def read_rows(adata: ad.AnnData, rows: Iterable[int], gene_indices: np.ndarray | slice) -> np.ndarray:
    rows = np.asarray(list(rows), dtype=np.int64)
    order = np.argsort(rows)
    sorted_rows = rows[order]
    # Backed sparse AnnData can fail on simultaneous row and column fancy
    # indexing in some SciPy/h5py combinations. Row slicing first keeps the
    # temporary matrix modest because SHAP runs are sampled.
    x_all = adata[sorted_rows, :].X
    if sparse.issparse(x_all):
        x = x_all[:, gene_indices].toarray()
    else:
        x = np.asarray(x_all[:, gene_indices])
    x = as_dense_float(x)
    inverse = np.argsort(order)
    return np.log1p(x[inverse])


def select_variable_genes(adata: ad.AnnData, rows: np.ndarray, top_genes: int) -> np.ndarray:
    x = adata[np.sort(rows), :].X
    if sparse.issparse(x):
        means = np.asarray(x.mean(axis=0)).ravel()
        sq_means = np.asarray(x.power(2).mean(axis=0)).ravel()
    else:
        dense = np.asarray(x, dtype=np.float32)
        means = dense.mean(axis=0)
        sq_means = np.square(dense).mean(axis=0)
    variances = np.maximum(sq_means - np.square(means), 0.0)
    expressed = means > 0
    scores = np.where(expressed, variances, -1.0)
    top = np.argsort(scores)[::-1][: min(top_genes, scores.shape[0])]
    return np.sort(top.astype(np.int64))


def gene_names(adata: ad.AnnData, gene_indices: np.ndarray) -> list[str]:
    var = adata.var.iloc[gene_indices]
    if "gene_name" in var.columns:
        names = var["gene_name"].astype(str).tolist()
    else:
        names = var.index.astype(str).tolist()
    return [name if name and name.lower() != "nan" else str(idx) for name, idx in zip(names, var.index)]


def normalize_shap_values(values, num_classes: int) -> np.ndarray:
    if isinstance(values, list):
        return np.stack(values, axis=0)
    arr = np.asarray(values)
    if arr.ndim == 3 and arr.shape[-1] == num_classes:
        return np.moveaxis(arr, -1, 0)
    if arr.ndim == 3 and arr.shape[0] == num_classes:
        return arr
    if arr.ndim == 2:
        return arr[None, :, :]
    raise ValueError(f"Unexpected SHAP value shape: {arr.shape}")


def marker_lookup() -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    for set_name, genes in MARKER_SETS.items():
        for gene in genes:
            lookup.setdefault(gene.upper(), []).append(set_name)
    return lookup


def write_plots(global_df: pd.DataFrame, class_df: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    top = global_df.head(25).iloc[::-1]
    plt.figure(figsize=(8, 7))
    plt.barh(top["gene"], top["mean_abs_shap"], color="#5b7fb0")
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Gene")
    plt.title("Global SHAP gene importance")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_global_top_genes.png", dpi=220)
    plt.close()

    pivot = (
        class_df.groupby(["class", "gene"], as_index=False)["mean_abs_shap"].mean()
        .sort_values(["class", "mean_abs_shap"], ascending=[True, False])
        .groupby("class")
        .head(10)
    )
    if not pivot.empty:
        matrix = pivot.pivot_table(index="gene", columns="class", values="mean_abs_shap", fill_value=0.0)
        plt.figure(figsize=(max(8, 1.2 * matrix.shape[1]), max(7, 0.28 * matrix.shape[0])))
        plt.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
        plt.colorbar(label="Mean absolute SHAP value")
        plt.xticks(np.arange(matrix.shape[1]), matrix.columns, rotation=45, ha="right")
        plt.yticks(np.arange(matrix.shape[0]), matrix.index)
        plt.title("Class-specific SHAP top genes")
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_class_top_genes_heatmap.png", dpi=220)
        plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import shap

    adapter_dir = Path(args.adapter_dir)
    train_pred, test_pred = load_predictions(adapter_dir, args.adapter_prefix)
    label_col = target_column(args.target)

    adata = ad.read_h5ad(args.h5ad, backed="r")
    train_pred = index_predictions(adata, train_pred)
    test_pred = index_predictions(adata, test_pred)
    train_sample = stratified_sample(train_pred, label_col, args.max_train, args.seed)
    explain_sample = stratified_sample(test_pred, label_col, args.max_explain, args.seed + 1)

    gene_idx = select_variable_genes(adata, train_sample["adata_row"].to_numpy(), args.top_genes)
    genes = gene_names(adata, gene_idx)
    x_train = read_rows(adata, train_sample["adata_row"], gene_idx)
    x_explain = read_rows(adata, explain_sample["adata_row"], gene_idx)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_sample[label_col].astype(str))
    y_explain = encoder.transform(explain_sample[label_col].astype(str))

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2,
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_explain)
    surrogate_metrics = {
        "surrogate_target": args.target,
        "surrogate_accuracy": float(accuracy_score(y_explain, pred)),
        "surrogate_macro_f1": float(f1_score(y_explain, pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_explain,
            pred,
            labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_,
            output_dict=True,
            zero_division=0,
        ),
    }

    explainer = shap.TreeExplainer(clf)
    shap_values = normalize_shap_values(explainer.shap_values(x_explain, check_additivity=False), len(encoder.classes_))
    mean_abs = np.abs(shap_values).mean(axis=(0, 1))
    lookup = marker_lookup()
    global_df = pd.DataFrame(
        {
            "gene": genes,
            "mean_abs_shap": mean_abs,
            "marker_sets": [";".join(lookup.get(gene.upper(), [])) for gene in genes],
        }
    ).sort_values("mean_abs_shap", ascending=False)
    global_df.to_csv(output_dir / f"{args.prefix}_global_gene_importance.csv", index=False)

    class_rows = []
    for class_idx, class_name in enumerate(encoder.classes_):
        class_mean = np.abs(shap_values[class_idx]).mean(axis=0)
        top_order = np.argsort(class_mean)[::-1][:50]
        for rank, gene_pos in enumerate(top_order, start=1):
            gene = genes[gene_pos]
            class_rows.append(
                {
                    "class": class_name,
                    "rank": rank,
                    "gene": gene,
                    "mean_abs_shap": float(class_mean[gene_pos]),
                    "marker_sets": ";".join(lookup.get(gene.upper(), [])),
                }
            )
    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(output_dir / f"{args.prefix}_class_gene_importance.csv", index=False)

    marker_df = global_df[global_df["marker_sets"] != ""].copy()
    marker_df["global_rank"] = np.arange(1, len(global_df) + 1)[global_df["marker_sets"] != ""]
    marker_df.to_csv(output_dir / f"{args.prefix}_marker_overlap.csv", index=False)
    explain_sample.to_csv(output_dir / f"{args.prefix}_explained_spots.csv", index=False)
    write_plots(global_df, class_df, output_dir, args.prefix)

    summary = {
        "method": "TreeExplainer SHAP on a RandomForest surrogate trained on Mauron spot-level gene expression.",
        "interpretation_caveat": (
            "The final Geneformer spatial adapter is graph/embedding based. These SHAP values are gene-level "
            "surrogate explanations of the selected target, not direct gradients through the transformer."
        ),
        "h5ad": str(args.h5ad),
        "adapter_dir": str(args.adapter_dir),
        "adapter_prefix": args.adapter_prefix,
        "target": args.target,
        "num_train_spots": int(len(train_sample)),
        "num_explained_test_spots": int(len(explain_sample)),
        "num_selected_genes": int(len(genes)),
        "classes": encoder.classes_.tolist(),
        "surrogate_metrics": surrogate_metrics,
        "top_global_genes": global_df.head(25).to_dict(orient="records"),
        "top_marker_overlaps": marker_df.head(25).to_dict(orient="records"),
        "outputs": {
            "global_gene_importance": str(output_dir / f"{args.prefix}_global_gene_importance.csv"),
            "class_gene_importance": str(output_dir / f"{args.prefix}_class_gene_importance.csv"),
            "marker_overlap": str(output_dir / f"{args.prefix}_marker_overlap.csv"),
            "explained_spots": str(output_dir / f"{args.prefix}_explained_spots.csv"),
            "global_top_genes_plot": str(output_dir / f"{args.prefix}_global_top_genes.png"),
            "class_top_genes_heatmap": str(output_dir / f"{args.prefix}_class_top_genes_heatmap.png"),
        },
    }
    (output_dir / f"{args.prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["target", "num_train_spots", "num_explained_test_spots", "surrogate_metrics", "outputs"]}, indent=2))


if __name__ == "__main__":
    main()

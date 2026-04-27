"""
Mauron Visium spatial graph processor.

This processor builds one PyTorch Geometric graph per tissue section from the
raw Mauron developing human heart Visium data. It uses real spot coordinates,
section/case metadata, and deconvolution argmax labels instead of the older
opaque cached graph.
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree

import h5py
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


CHAMBERS = ("LV", "RV", "LA", "RA")


@dataclass(frozen=True)
class MauronBuildConfig:
    """Configuration used to turn raw Mauron Visium files into section graphs."""

    data_root: str = (
        "data/New/Mauron_spatial_dynamics_part_a/"
        "Spatial dynamics of the developing human heart, pa"
    )
    cache_dir: str = "cache/mauron_visium_graphs"
    num_genes: int = 512
    k_neighbors: int = 8
    label_mode: str = "deconv_hl_argmax"
    min_gene_spots: int = 25
    max_sections: Optional[int] = None

    def cache_key(self) -> str:
        max_sections = "all" if self.max_sections is None else str(self.max_sections)
        return (
            f"mauron_visium_{self.label_mode}_genes{self.num_genes}_"
            f"k{self.k_neighbors}_minspots{self.min_gene_spots}_sections{max_sections}.pt"
        )


class MauronVisiumGraphDataset:
    """Builds and loads Mauron Visium section graphs.

    Each element is one tissue section graph with:
    - ``x``: log-normalized selected gene expression per tissue spot.
    - ``pos``: real Visium pixel coordinates.
    - ``edge_index``: spatial kNN graph built within that section only.
    - ``y``: node labels for deconvolution argmax modes, or repeated graph
      labels for graph-level modes.
    - metadata fields for ``section_id``, ``code``, ``case``, ``age``,
      ``chamber_combo``, and grouped train/validation/test splitting.
    """

    VALID_LABEL_MODES = {
        "deconv_hl_argmax",
        "deconv_dl_argmax",
        "chamber_combo",
        "age_bin",
    }

    def __init__(self, config: Optional[MauronBuildConfig] = None):
        self.config = config or MauronBuildConfig()
        if self.config.label_mode not in self.VALID_LABEL_MODES:
            raise ValueError(
                f"Unsupported label_mode={self.config.label_mode!r}. "
                f"Choose one of {sorted(self.VALID_LABEL_MODES)}."
            )

        self.root = Path(self.config.data_root)
        self.cache_dir = Path(self.config.cache_dir)
        self.visium_dir = self.root / "2_Visium_spaceranger_data_ST"
        self.metadata_dir = self.root / "7_Metadata"
        self.section_metadata_path = (
            self.metadata_dir / "HDCA_heart_ST_sections_overview_chambers_present.xlsx"
        )
        self.hl_deconv_path = self.metadata_dir / "W.2023-09-25133054.715121_hl.tsv"
        self.dl_deconv_path = self.metadata_dir / "W.2023-10-28105405.918149_dl.tsv"
        self.hl_annotation_path = self.metadata_dir / "HDCA_heart_SC_annotations_HL_240115.csv"
        self.st_annotation_path = self.metadata_dir / "HDCA_heart_ST_annotations.csv"

    @property
    def cache_path(self) -> Path:
        return self.cache_dir / self.config.cache_key()

    def load_or_build(self, force_rebuild: bool = False) -> Tuple[List[Data], Dict]:
        """Load cached section graphs or build them from raw files."""

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.cache_path.exists() and not force_rebuild:
            logger.info("Loading Mauron section graphs from %s", self.cache_path)
            try:
                payload = torch.load(self.cache_path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(self.cache_path, map_location="cpu")
            return payload["graphs"], payload["metadata"]

        graphs, metadata = self.build()
        torch.save({"graphs": graphs, "metadata": metadata}, self.cache_path)
        self._write_audit(metadata)
        logger.info("Saved Mauron section graph cache to %s", self.cache_path)
        return graphs, metadata

    def build(self) -> Tuple[List[Data], Dict]:
        """Build all configured section graphs from raw Visium and metadata files."""

        self._validate_paths()
        sections = self._read_section_metadata()
        if self.config.max_sections is not None:
            sections = sections.iloc[: self.config.max_sections].copy()

        selected_gene_indices, selected_gene_names = self._select_genes(sections)
        label_lookup, label_names = self._load_label_lookup(sections)
        graph_label_names = self._graph_label_names(sections)
        graph_label_to_id = {name: idx for idx, name in enumerate(graph_label_names)}

        graphs: List[Data] = []
        skipped: List[Dict] = []
        for _, row in sections.iterrows():
            try:
                graph = self._build_section_graph(
                    row=row,
                    selected_gene_indices=selected_gene_indices,
                    selected_gene_names=selected_gene_names,
                    label_lookup=label_lookup,
                    label_names=label_names,
                    graph_label_to_id=graph_label_to_id,
                )
                graphs.append(graph)
                logger.info(
                    "Built section %s (%s): %d spots, %d edges",
                    int(row["Section"]),
                    row["Code"],
                    graph.num_nodes,
                    graph.edge_index.shape[1],
                )
            except Exception as exc:
                skipped.append(
                    {
                        "section": int(row["Section"]),
                        "code": str(row["Code"]),
                        "reason": repr(exc),
                    }
                )
                logger.warning("Skipping section %s (%s): %s", row["Section"], row["Code"], exc)

        if not graphs:
            raise RuntimeError("No Mauron Visium graphs were built.")

        metadata = {
            "config": asdict(self.config),
            "cache_path": str(self.cache_path),
            "num_graphs": len(graphs),
            "num_sections_requested": int(len(sections)),
            "skipped_sections": skipped,
            "feature_names": selected_gene_names,
            "label_mode": self.config.label_mode,
            "label_names": label_names,
            "graph_label_names": graph_label_names,
            "section_summary": [self._graph_summary(g) for g in graphs],
            "notes": [
                "One graph is built per tissue section.",
                "Edges are within-section spatial kNN edges from real Visium coordinates.",
                "Grouped evaluation should split by case or code, never random spots.",
            ],
        }
        return graphs, metadata

    def _validate_paths(self) -> None:
        required = [
            self.root,
            self.visium_dir,
            self.metadata_dir,
            self.section_metadata_path,
            self.hl_deconv_path,
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"Required Mauron file/folder not found: {path}")

    def _read_section_metadata(self) -> pd.DataFrame:
        df = _read_xlsx_with_fallback(self.section_metadata_path)
        required = {"Section", "Code", "Age", "Case", *CHAMBERS}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"Section metadata is missing columns: {missing}")

        df = df.dropna(subset=["Section", "Code", "Age", "Case"]).copy()
        df["Section"] = df["Section"].astype(int)
        df["Code"] = df["Code"].astype(str)
        df["Age"] = df["Age"].astype(str)
        df["Case"] = df["Case"].astype(str)
        df["age_weeks"] = df["Age"].map(_parse_age_weeks)
        df["age_bin"] = df["age_weeks"].map(_age_bin)
        df["chamber_combo"] = df.apply(_chamber_combo, axis=1)
        df = df.sort_values("Section").reset_index(drop=True)

        existing_codes = {p.name for p in self.visium_dir.iterdir() if p.is_dir()}
        df = df[df["Code"].isin(existing_codes)].reset_index(drop=True)
        if df.empty:
            raise RuntimeError("No section metadata rows matched Visium section folders.")
        return df

    def _select_genes(self, sections: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        sums = None
        sums_sq = None
        detected = None
        total_spots = 0
        gene_names: Optional[List[str]] = None

        for _, row in sections.iterrows():
            matrix, _, genes = self._read_10x_h5(self._matrix_path(row["Code"]))
            if gene_names is None:
                gene_names = genes
                n_genes = len(genes)
                sums = np.zeros(n_genes, dtype=np.float64)
                sums_sq = np.zeros(n_genes, dtype=np.float64)
                detected = np.zeros(n_genes, dtype=np.int64)
            elif gene_names != genes:
                raise ValueError(f"Gene order differs in section {row['Code']}")

            sums += np.asarray(matrix.sum(axis=1)).ravel()
            squared = matrix.copy()
            squared.data = squared.data.astype(np.float64) ** 2
            sums_sq += np.asarray(squared.sum(axis=1)).ravel()
            detected += matrix.getnnz(axis=1)
            total_spots += matrix.shape[1]

        assert gene_names is not None
        assert sums is not None and sums_sq is not None and detected is not None

        mean = sums / max(total_spots, 1)
        variance = (sums_sq / max(total_spots, 1)) - (mean**2)
        allowed = np.array([_is_informative_gene(name) for name in gene_names])
        allowed &= detected >= self.config.min_gene_spots
        candidate_indices = np.where(allowed)[0]
        if len(candidate_indices) < self.config.num_genes:
            logger.warning(
                "Only %d genes passed filters; requested %d.",
                len(candidate_indices),
                self.config.num_genes,
            )
        ranked = candidate_indices[np.argsort(variance[candidate_indices])[::-1]]
        selected = ranked[: self.config.num_genes]
        selected_names = [gene_names[i] for i in selected]
        return selected.astype(np.int64), selected_names

    def _load_label_lookup(
        self, sections: pd.DataFrame
    ) -> Tuple[Dict[Tuple[int, str], Tuple[int, float]], List[str]]:
        if self.config.label_mode == "deconv_hl_argmax":
            deconv_path = self.hl_deconv_path
            annotation_path = self.hl_annotation_path
            table = pd.read_csv(deconv_path, sep="\t", index_col=0)
            label_keys = list(table.columns)
            label_names = _load_hl_label_names(annotation_path, label_keys)
        elif self.config.label_mode == "deconv_dl_argmax":
            if not self.dl_deconv_path.exists():
                raise FileNotFoundError(f"Detailed deconvolution table not found: {self.dl_deconv_path}")
            table = pd.read_csv(self.dl_deconv_path, sep="\t", index_col=0)
            label_keys = list(table.columns)
            label_names = label_keys
        else:
            return {}, self._graph_label_names(sections)

        label_to_id = {label: idx for idx, label in enumerate(label_keys)}
        argmax_labels = table.idxmax(axis=1)
        argmax_confidence = table.max(axis=1)
        parsed = table.index.to_series().str.extract(r"^(?P<barcode>.+)_(?P<section>[0-9]+)$")
        lookup: Dict[Tuple[int, str], Tuple[int, float]] = {}
        for row_id, label_key, confidence in zip(
            parsed.itertuples(index=False), argmax_labels.to_numpy(), argmax_confidence.to_numpy()
        ):
            if pd.isna(row_id.section) or pd.isna(row_id.barcode):
                continue
            lookup[(int(row_id.section), str(row_id.barcode))] = (
                int(label_to_id[str(label_key)]),
                float(confidence),
            )
        return lookup, label_names

    def _graph_label_names(self, sections: pd.DataFrame) -> List[str]:
        if self.config.label_mode == "age_bin":
            return sorted(sections["age_bin"].dropna().astype(str).unique().tolist())
        if self.config.label_mode == "chamber_combo":
            return sorted(sections["chamber_combo"].dropna().astype(str).unique().tolist())
        return []

    def _build_section_graph(
        self,
        row: pd.Series,
        selected_gene_indices: np.ndarray,
        selected_gene_names: Sequence[str],
        label_lookup: Dict[Tuple[int, str], Tuple[int, float]],
        label_names: Sequence[str],
        graph_label_to_id: Dict[str, int],
    ) -> Data:
        section = int(row["Section"])
        code = str(row["Code"])
        case = str(row["Case"])
        age = str(row["Age"])
        age_bin = str(row["age_bin"])
        chamber_combo = str(row["chamber_combo"])

        matrix, barcodes, _ = self._read_10x_h5(self._matrix_path(code))
        positions = self._read_tissue_positions(code)
        barcode_to_col = {barcode: idx for idx, barcode in enumerate(barcodes)}
        positions = positions[positions["barcode"].isin(barcode_to_col)].copy()
        if positions.empty:
            raise RuntimeError("No tissue position barcodes matched expression matrix barcodes.")
        if self.config.label_mode.startswith("deconv_"):
            has_label = positions["barcode"].map(lambda barcode: (section, str(barcode)) in label_lookup)
            missing_labels = int((~has_label).sum())
            if missing_labels:
                logger.warning(
                    "Dropping %d spots without deconvolution labels from section %s (%s).",
                    missing_labels,
                    section,
                    code,
                )
                positions = positions[has_label].copy()
            if positions.empty:
                raise RuntimeError("No tissue spots have deconvolution labels after filtering.")

        col_indices = np.array([barcode_to_col[barcode] for barcode in positions["barcode"]], dtype=np.int64)
        expr = matrix[selected_gene_indices, :][:, col_indices].T.tocsr()
        x = _log_normalize_dense(expr)
        pos = positions[["pxl_row", "pxl_col"]].to_numpy(dtype=np.float32)
        edge_index = _build_spatial_knn(pos, self.config.k_neighbors)

        graph_label_name = age_bin if self.config.label_mode == "age_bin" else chamber_combo
        graph_y = graph_label_to_id.get(graph_label_name, -1)

        if self.config.label_mode.startswith("deconv_"):
            y, label_confidence = self._node_deconv_labels(section, positions["barcode"], label_lookup)
            task_level = "node"
        else:
            y = np.full(positions.shape[0], graph_y, dtype=np.int64)
            label_confidence = np.ones(positions.shape[0], dtype=np.float32)
            task_level = "graph"

        data = Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_index),
            pos=torch.from_numpy(pos),
            y=torch.from_numpy(y),
            label_confidence=torch.from_numpy(label_confidence),
            graph_y=torch.tensor([graph_y], dtype=torch.long),
            chamber_multi_hot=torch.tensor([_chamber_multi_hot(row)], dtype=torch.float32),
            section_id=torch.tensor([section], dtype=torch.long),
            age_weeks=torch.tensor([float(row["age_weeks"])], dtype=torch.float32),
        )
        data.code = code
        data.case = case
        data.age = age
        data.age_bin = age_bin
        data.chamber_combo = chamber_combo
        data.task_level = task_level
        data.barcodes = positions["barcode"].astype(str).tolist()
        data.feature_names = list(selected_gene_names)
        data.label_names = list(label_names)
        return data

    def _node_deconv_labels(
        self,
        section: int,
        barcodes: Iterable[str],
        label_lookup: Dict[Tuple[int, str], Tuple[int, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        labels = []
        confidence = []
        for barcode in barcodes:
            label, score = label_lookup.get((section, str(barcode)), (-1, 0.0))
            labels.append(label)
            confidence.append(score)
        y = np.asarray(labels, dtype=np.int64)
        conf = np.asarray(confidence, dtype=np.float32)
        if np.any(y < 0):
            missing = int(np.sum(y < 0))
            raise RuntimeError(f"{missing} tissue spots are missing deconvolution labels.")
        return y, conf

    def _matrix_path(self, code: str) -> Path:
        return self.visium_dir / str(code) / "filtered_feature_bc_matrix.h5"

    def _read_tissue_positions(self, code: str) -> pd.DataFrame:
        path = self.visium_dir / str(code) / "spatial" / "tissue_positions_list.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, header=None)
        df.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
        df = df[df["in_tissue"] == 1].copy()
        return df.reset_index(drop=True)

    @staticmethod
    def _read_10x_h5(path: Path) -> Tuple[sparse.csc_matrix, List[str], List[str]]:
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            group = handle["matrix"]
            data = group["data"][:]
            indices = group["indices"][:]
            indptr = group["indptr"][:]
            shape = tuple(group["shape"][:])
            barcodes = [value.decode("utf-8") for value in group["barcodes"][:]]
            genes = [value.decode("utf-8") for value in group["features"]["name"][:]]
        matrix = sparse.csc_matrix((data, indices, indptr), shape=shape)
        return matrix, barcodes, genes

    def _write_audit(self, metadata: Dict) -> None:
        audit_path = self.cache_path.with_suffix(".json")
        compact = dict(metadata)
        compact["feature_names"] = metadata["feature_names"][:50]
        with audit_path.open("w", encoding="utf-8") as handle:
            json.dump(compact, handle, indent=2)

    @staticmethod
    def _graph_summary(graph: Data) -> Dict:
        label_counts = torch.bincount(graph.y.cpu(), minlength=max(int(graph.y.max().item()) + 1, 1))
        return {
            "section": int(graph.section_id.item()),
            "code": graph.code,
            "case": graph.case,
            "age": graph.age,
            "age_bin": graph.age_bin,
            "chamber_combo": graph.chamber_combo,
            "num_spots": int(graph.num_nodes),
            "num_edges": int(graph.edge_index.shape[1]),
            "task_level": graph.task_level,
            "label_counts": label_counts.tolist(),
        }


def _parse_age_weeks(age: str) -> float:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(age))
    if not match:
        raise ValueError(f"Could not parse fetal age from {age!r}")
    return float(match.group(1))


def _age_bin(age_weeks: float) -> str:
    if age_weeks <= 7.5:
        return "early_w6_w7_5"
    if age_weeks <= 9:
        return "mid_w8_w9"
    return "late_w10_w12"


def _chamber_combo(row: pd.Series) -> str:
    present = []
    for chamber in CHAMBERS:
        value = str(row.get(chamber, "")).lower()
        if "x" in value:
            present.append(chamber)
    return "+".join(present) if present else "unknown"


def _chamber_multi_hot(row: pd.Series) -> List[float]:
    combo = _chamber_combo(row).split("+")
    return [1.0 if chamber in combo else 0.0 for chamber in CHAMBERS]


def _is_informative_gene(gene: str) -> bool:
    upper = gene.upper()
    if upper.startswith("MT-"):
        return False
    if upper.startswith("RPL") or upper.startswith("RPS"):
        return False
    if upper in {"MALAT1", "XIST"}:
        return False
    return True


def _log_normalize_dense(matrix: sparse.csr_matrix, scale_factor: float = 1e4) -> np.ndarray:
    matrix = matrix.astype(np.float32).tocsr(copy=True)
    library_size = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.divide(
        scale_factor,
        library_size,
        out=np.zeros_like(library_size, dtype=np.float32),
        where=library_size > 0,
    )
    matrix = sparse.diags(scale).dot(matrix)
    matrix.data = np.log1p(matrix.data)
    return matrix.toarray().astype(np.float32)


def _build_spatial_knn(pos: np.ndarray, k_neighbors: int) -> np.ndarray:
    n_nodes = pos.shape[0]
    if n_nodes < 2:
        return np.empty((2, 0), dtype=np.int64)
    n_neighbors = min(k_neighbors + 1, n_nodes)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nbrs.fit(pos)
    indices = nbrs.kneighbors(pos, return_distance=False)
    sources = np.repeat(np.arange(n_nodes), n_neighbors - 1)
    targets = indices[:, 1:].reshape(-1)
    edge_pairs = np.vstack([sources, targets]).T
    reverse_pairs = edge_pairs[:, ::-1]
    all_pairs = np.vstack([edge_pairs, reverse_pairs])
    all_pairs = np.unique(all_pairs, axis=0)
    return all_pairs.T.astype(np.int64)


def _load_hl_label_names(annotation_path: Path, label_keys: Sequence[str]) -> List[str]:
    if not annotation_path.exists():
        return [str(key) for key in label_keys]
    annotation = pd.read_csv(annotation_path, sep=";")
    mapping = {
        str(row["cluster"]).replace("_", "-"): str(row["cell_type"])
        for _, row in annotation.iterrows()
        if "cluster" in row and "cell_type" in row
    }
    names = []
    for key in label_keys:
        normalized = str(key).replace("_", "-")
        names.append(mapping.get(normalized, str(key)))
    return names


def _read_xlsx_with_fallback(path: Path) -> pd.DataFrame:
    """Read the simple Mauron metadata workbook without requiring openpyxl.

    Pandas normally delegates .xlsx parsing to openpyxl. The project venv used
    for training may not have that optional dependency, so this falls back to a
    small OOXML reader that is sufficient for the first-sheet metadata table.
    """

    try:
        return pd.read_excel(path)
    except ImportError as exc:
        if "openpyxl" not in str(exc):
            raise
        logger.warning("openpyxl is not installed; using built-in XLSX metadata reader.")

    rows = _read_first_xlsx_sheet(path)
    if not rows:
        raise RuntimeError(f"No rows found in workbook: {path}")
    header = [str(value) if value is not None else "" for value in rows[0]]
    values = rows[1:]
    width = len(header)
    normalized = []
    for row in values:
        padded = list(row[:width]) + [None] * max(0, width - len(row))
        normalized.append([np.nan if value == "" else value for value in padded])
    return pd.DataFrame(normalized, columns=header)


def _read_first_xlsx_sheet(path: Path) -> List[List[object]]:
    namespace = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }

    with zipfile.ZipFile(path) as archive:
        shared_strings = _xlsx_shared_strings(archive, namespace)
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        first_sheet = workbook.find("main:sheets/main:sheet", namespace)
        if first_sheet is None:
            return []
        relation_id = first_sheet.attrib[
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        ]
        rels = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in rels.findall("pkgrel:Relationship", namespace):
            if rel.attrib.get("Id") == relation_id:
                target = rel.attrib["Target"]
                break
        if target is None:
            raise RuntimeError(f"Could not resolve first worksheet relationship in {path}")
        sheet_path = "xl/" + target.lstrip("/")
        sheet = ElementTree.fromstring(archive.read(sheet_path))

    parsed_rows: List[List[object]] = []
    for row in sheet.findall(".//main:sheetData/main:row", namespace):
        cells: List[object] = []
        for cell in row.findall("main:c", namespace):
            cell_ref = cell.attrib.get("r", "")
            col_idx = _xlsx_column_index(cell_ref)
            while len(cells) < col_idx:
                cells.append(None)
            cells.append(_xlsx_cell_value(cell, shared_strings, namespace))
        parsed_rows.append(cells)
    return parsed_rows


def _xlsx_shared_strings(archive: zipfile.ZipFile, namespace: Dict[str, str]) -> List[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    values = []
    for item in root.findall("main:si", namespace):
        text_parts = [node.text or "" for node in item.findall(".//main:t", namespace)]
        values.append("".join(text_parts))
    return values


def _xlsx_column_index(cell_ref: str) -> int:
    letters = "".join(char for char in cell_ref if char.isalpha())
    if not letters:
        return 0
    index = 0
    for char in letters:
        index = index * 26 + (ord(char.upper()) - ord("A") + 1)
    return index - 1


def _xlsx_cell_value(
    cell: ElementTree.Element,
    shared_strings: Sequence[str],
    namespace: Dict[str, str],
) -> object:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//main:t", namespace))

    value_node = cell.find("main:v", namespace)
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "s":
        return shared_strings[int(raw)]
    if cell_type in {"str", "b"}:
        return raw
    try:
        numeric = float(raw)
        return int(numeric) if numeric.is_integer() else numeric
    except ValueError:
        return raw

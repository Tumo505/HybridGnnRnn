#!/usr/bin/env python3
"""Tokenize Mauron spatial Geneformer h5ad files while preserving coordinates."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("tokenize_mauron_spatial_geneformer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="foundation_spatial_geneformer_adapter_inputs")
    parser.add_argument("--output-dir", default="foundation_spatial_geneformer_adapter_inputs/tokenized")
    parser.add_argument("--output-prefix", default="mauron_spatial_geneformer")
    parser.add_argument("--model-version", choices=["V1", "V2"], default="V1")
    parser.add_argument("--nproc", type=int, default=0, help="Use 0 for single-process mode on Windows.")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument(
        "--geneformer-source",
        default="models/geneformer/Geneformer",
        help="Local Geneformer source checkout to add to PYTHONPATH if the package is not installed.",
    )
    return parser.parse_args()


def import_tokenizer(geneformer_source: str):
    try:
        from geneformer import TranscriptomeTokenizer

        return TranscriptomeTokenizer
    except ImportError:
        source = Path(geneformer_source)
        if source.exists():
            sys.path.insert(0, str(source.resolve()))
            from geneformer import TranscriptomeTokenizer

            return TranscriptomeTokenizer
        raise


def main() -> None:
    args = parse_args()
    TranscriptomeTokenizer = import_tokenizer(args.geneformer_source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    custom_attrs = {
        "spot_id": "spot_id",
        "barcode": "barcode",
        "section_id": "section_id",
        "code": "code",
        "case": "case",
        "line_id": "line_id",
        "sample_id": "sample_id",
        "age": "age",
        "age_weeks": "age_weeks",
        "age_bin": "age_bin",
        "spatial_progress": "spatial_progress",
        "chamber_combo": "chamber_combo",
        "x": "x",
        "y": "y",
        "array_row": "array_row",
        "array_col": "array_col",
        "deconv_label": "deconv_label",
        "cell_state": "cell_state",
        "deconv_confidence": "deconv_confidence",
        "source_dataset": "source_dataset",
        "diffday": "diffday",
        "diffday_numeric": "diffday_numeric",
        "dpt_pseudotime_numeric": "dpt_pseudotime",
    }
    nproc = None if args.nproc == 0 else args.nproc
    tokenizer = TranscriptomeTokenizer(
        custom_attr_name_dict=custom_attrs,
        nproc=nproc,
        chunk_size=args.chunk_size,
        model_version=args.model_version,
    )
    logger.info("Tokenizing Mauron spatial h5ad files in %s.", args.input_dir)
    tokenizer.tokenize_data(args.input_dir, output_dir, args.output_prefix, file_format="h5ad")


if __name__ == "__main__":
    main()

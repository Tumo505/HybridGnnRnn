#!/usr/bin/env python3
"""Tokenize Geneformer-ready h5ad files into Geneformer .dataset format."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("tokenize_geneformer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="foundation_model_data/geneformer")
    parser.add_argument("--output-dir", default="foundation_model_data/geneformer/tokenized")
    parser.add_argument("--output-prefix", default="gse175634_geneformer")
    parser.add_argument("--model-version", choices=["V1", "V2"], default="V1")
    parser.add_argument("--nproc", type=int, default=4, help="Use 0 for single-process mode on restricted Windows shells.")
    parser.add_argument("--chunk-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from geneformer import TranscriptomeTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Geneformer is not installed. Install the optional foundation stack and Geneformer first. "
            "See foundation_model_data/README_geneformer_pipeline.md after running this setup."
        ) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    custom_attrs = {
        "diffday": "diffday",
        "diffday_numeric": "diffday_numeric",
        "cell_state": "cell_state",
        "line_id": "line_id",
        "sample_id": "sample_id",
        "dpt_pseudotime_numeric": "dpt_pseudotime",
    }
    nproc = None if args.nproc == 0 else args.nproc
    tokenizer = TranscriptomeTokenizer(
        custom_attr_name_dict=custom_attrs,
        nproc=nproc,
        chunk_size=args.chunk_size,
        model_version=args.model_version,
    )
    logger.info("Tokenizing h5ad files in %s.", args.input_dir)
    tokenizer.tokenize_data(args.input_dir, output_dir, args.output_prefix, file_format="h5ad")


if __name__ == "__main__":
    main()

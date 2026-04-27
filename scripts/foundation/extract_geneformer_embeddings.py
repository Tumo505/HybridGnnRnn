#!/usr/bin/env python3
"""Extract Geneformer cell/spot embeddings for alignment with Mauron spatial outputs."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("extract_geneformer_embeddings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenized-dataset", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default="foundation_geneformer_embeddings")
    parser.add_argument("--output-prefix", default="geneformer_embeddings")
    parser.add_argument("--model-type", choices=["Pretrained", "CellClassifier"], default="CellClassifier")
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--model-version", choices=["V1", "V2"], default="V1")
    parser.add_argument("--emb-layer", type=int, default=-1)
    parser.add_argument("--summary-stat", default=None, choices=[None, "mean", "median", "exact_mean", "exact_median"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-cells", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from geneformer import EmbExtractor
    except ImportError as exc:
        raise SystemExit("Geneformer is not installed. Install it before extracting embeddings.") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = EmbExtractor(
        model_type=args.model_type,
        num_classes=args.num_classes,
        emb_mode="cell",
        max_ncells=args.max_cells,
        emb_layer=args.emb_layer,
        forward_batch_size=args.batch_size,
        summary_stat=args.summary_stat,
        model_version=args.model_version,
    )
    logger.info("Extracting embeddings from %s.", args.tokenized_dataset)
    extractor.extract_embs(
        model_directory=args.model_dir,
        input_data_file=args.tokenized_dataset,
        output_directory=output_dir,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()

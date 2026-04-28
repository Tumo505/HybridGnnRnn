#!/usr/bin/env python3
"""Extract metadata-aligned embeddings from the fine-tuned multi-task Geneformer.

Geneformer's EmbExtractor writes embeddings without the metadata we need for
downstream spatial-temporal alignment. This extractor keeps row order explicit
and saves:

- embeddings as a NumPy array
- matching per-cell metadata as CSV
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from train_geneformer_multiatlas_multitask import GeneformerMultiTaskModel, add_source_and_labels, collate_examples, set_seed


os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("extract_multitask_geneformer_embeddings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenized-dataset", default="foundation_model_data/geneformer/tokenized/gse175634_geneformer.dataset")
    parser.add_argument("--model-dir", default="models/geneformer/Geneformer/Geneformer-V1-10M")
    parser.add_argument(
        "--checkpoint",
        default="foundation_geneformer_multiatlas_multitask/gse175634_gse202398_holdout_lmna_best_model/pytorch_model_multitask.bin",
    )
    parser.add_argument(
        "--label-maps",
        default="foundation_geneformer_multiatlas_multitask/gse175634_gse202398_holdout_lmna_best_model/label_maps.json",
    )
    parser.add_argument("--output-dir", default="foundation_spatiotemporal_alignment_inputs")
    parser.add_argument("--output-prefix", default="gse175634_multitask_geneformer")
    parser.add_argument("--max-cells", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--shuffle", action="store_true")
    return parser.parse_args()


def select_dataset(dataset: Any, max_cells: int | None, seed: int, shuffle: bool) -> Any:
    if max_cells is None or len(dataset) <= max_cells:
        return dataset.shuffle(seed=seed) if shuffle else dataset
    if shuffle:
        rng = random.Random(seed)
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        return dataset.select(sorted(indices[:max_cells]))
    return dataset.select(range(max_cells))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_maps_raw = json.loads(Path(args.label_maps).read_text(encoding="utf-8"))
    label_maps = {key: {int(k): v for k, v in value.items()} for key, value in label_maps_raw.items()}
    model = GeneformerMultiTaskModel(
        args.model_dir,
        num_day=len(label_maps["day"]),
        num_stage=len(label_maps["stage"]),
        num_state=len(label_maps["state"]),
        num_domain=len(label_maps["domain"]),
        dropout=args.dropout,
        grl_lambda=0.0,
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = add_source_and_labels(load_from_disk(args.tokenized_dataset), "GSE175634", "GSE175634")
    dataset = select_dataset(dataset, args.max_cells, args.seed, args.shuffle)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda rows: collate_examples(rows, args.max_length))
    embeddings = []
    metadata_rows = []
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            pooled = model.pooled(batch.input_ids.to(device), batch.attention_mask.to(device)).detach().cpu().numpy()
            embeddings.append(pooled.astype(np.float32))
            for idx in range(len(batch.metadata["diffday"])):
                metadata_rows.append(
                    {
                        "row_number": len(metadata_rows),
                        "source_dataset": batch.metadata["source_dataset"][idx],
                        "line_id": batch.metadata["line_id"][idx],
                        "sample_id": batch.metadata["sample_id"][idx],
                        "diffday": batch.metadata["diffday"][idx],
                        "cell_state": batch.metadata["cell_state"][idx],
                        "stage_text": batch.metadata["stage_text"][idx],
                    }
                )
            if step % 250 == 0:
                logger.info("Extracted %d cells.", len(metadata_rows))

    embedding_array = np.vstack(embeddings)
    np.save(output_dir / f"{args.output_prefix}_embeddings.npy", embedding_array)
    pd.DataFrame(metadata_rows).to_csv(output_dir / f"{args.output_prefix}_metadata.csv", index=False)
    summary = {
        "tokenized_dataset": args.tokenized_dataset,
        "checkpoint": args.checkpoint,
        "embeddings": str(output_dir / f"{args.output_prefix}_embeddings.npy"),
        "metadata": str(output_dir / f"{args.output_prefix}_metadata.csv"),
        "shape": list(embedding_array.shape),
        "device": str(device),
    }
    (output_dir / f"{args.output_prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract finetuned Geneformer embeddings for Mauron spatial spots."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "foundation"))

from train_geneformer_multiatlas_multitask import GeneformerMultiTaskModel, set_seed  # noqa: E402


os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("extract_mauron_spatial_geneformer_embeddings")


DEFAULT_METADATA_COLUMNS = [
    "spot_id",
    "barcode",
    "section_id",
    "code",
    "case",
    "line_id",
    "sample_id",
    "age",
    "age_weeks",
    "age_bin",
    "spatial_progress",
    "chamber_combo",
    "x",
    "y",
    "array_row",
    "array_col",
    "deconv_label",
    "cell_state",
    "deconv_confidence",
    "source_dataset",
    "diffday",
    "diffday_numeric",
    "dpt_pseudotime",
]


@dataclass
class TokenBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    metadata: dict[str, list[Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenized-dataset",
        default="foundation_spatial_geneformer_adapter_inputs/tokenized/mauron_spatial_geneformer.dataset",
    )
    parser.add_argument("--model-dir", default="models/geneformer/Geneformer/Geneformer-V1-10M")
    parser.add_argument(
        "--checkpoint",
        default="foundation_geneformer_multiatlas_multitask/gse175634_gse202398_holdout_lmna_best_model/pytorch_model_multitask.bin",
    )
    parser.add_argument(
        "--label-maps",
        default="foundation_geneformer_multiatlas_multitask/gse175634_gse202398_holdout_lmna_best_model/label_maps.json",
    )
    parser.add_argument("--output-dir", default="foundation_spatial_geneformer_adapter_inputs")
    parser.add_argument("--output-prefix", default="mauron_spatial_finetuned_geneformer")
    parser.add_argument("--max-spots", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser.parse_args()


def collate_tokens(examples: list[dict[str, Any]], max_length: int, metadata_columns: list[str]) -> TokenBatch:
    lengths = [min(len(example["input_ids"]), max_length) for example in examples]
    max_len = max(lengths)
    input_ids = torch.zeros((len(examples), max_len), dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
    for row, example in enumerate(examples):
        ids = torch.tensor(example["input_ids"][: lengths[row]], dtype=torch.long)
        input_ids[row, : lengths[row]] = ids
        attention_mask[row, : lengths[row]] = 1
    metadata = {col: [example.get(col) for example in examples] for col in metadata_columns}
    return TokenBatch(input_ids=input_ids, attention_mask=attention_mask, metadata=metadata)


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

    dataset = load_from_disk(args.tokenized_dataset)
    if args.max_spots is not None and len(dataset) > args.max_spots:
        dataset = dataset.select(range(args.max_spots))
    metadata_columns = [col for col in DEFAULT_METADATA_COLUMNS if col in dataset.column_names]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_tokens(rows, args.max_length, metadata_columns),
    )

    embeddings = []
    metadata_rows = []
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            pooled = model.pooled(batch.input_ids.to(device), batch.attention_mask.to(device)).detach().cpu().numpy()
            embeddings.append(pooled.astype(np.float32))
            for idx in range(pooled.shape[0]):
                row = {"row_number": len(metadata_rows)}
                row.update({col: batch.metadata[col][idx] for col in metadata_columns})
                metadata_rows.append(row)
            if step % 100 == 0:
                logger.info("Extracted embeddings for %d spatial spots.", len(metadata_rows))

    embedding_array = np.vstack(embeddings)
    metadata = pd.DataFrame(metadata_rows)
    np.save(output_dir / f"{args.output_prefix}_embeddings.npy", embedding_array)
    metadata.to_csv(output_dir / f"{args.output_prefix}_metadata.csv", index=False)
    summary = {
        "tokenized_dataset": args.tokenized_dataset,
        "checkpoint": args.checkpoint,
        "embeddings": str(output_dir / f"{args.output_prefix}_embeddings.npy"),
        "metadata": str(output_dir / f"{args.output_prefix}_metadata.csv"),
        "shape": list(embedding_array.shape),
        "metadata_columns": metadata_columns,
        "device": str(device),
    }
    (output_dir / f"{args.output_prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

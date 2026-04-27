#!/usr/bin/env python3
"""Run a fine-tuned Geneformer classifier on unlabeled or externally labeled tokenized data."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenized-dataset", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--id-class-dict", required=True)
    parser.add_argument("--output-dir", default="foundation_external_validation")
    parser.add_argument("--prefix", default="external_predictions")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--expected-class", default=None, help="Optional class expected for sanity testing, e.g. IPSC.")
    parser.add_argument("--calibration-json", default=None, help="Optional calibration JSON from calibrate_geneformer_confidence.py.")
    return parser.parse_args()


def load_id_map(path: str | Path) -> dict[int, str]:
    with open(path, "rb") as handle:
        return {int(key): value for key, value in pickle.load(handle).items()}


def load_calibration(path: str | None) -> tuple[float, float | None]:
    if not path:
        return 1.0, None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    threshold = float(payload["threshold"]) if "threshold" in payload else None
    return float(payload.get("temperature", 1.0)), threshold


def collate_input_ids(examples: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    lengths = [len(example["input_ids"]) for example in examples]
    max_len = max(lengths)
    input_ids = torch.full((len(examples), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
    for row, example in enumerate(examples):
        ids = torch.tensor(example["input_ids"], dtype=torch.long)
        input_ids[row, : len(ids)] = ids
        attention_mask[row, : len(ids)] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    id_to_class = load_id_map(args.id_class_dict)
    temperature, reject_threshold = load_calibration(args.calibration_json)

    dataset = load_from_disk(args.tokenized_dataset)
    if args.max_cells is not None:
        dataset = dataset.select(range(min(args.max_cells, len(dataset))))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()
    pad_token_id = int(getattr(model.config, "pad_token_id", 0) or 0)

    metadata_cols = [col for col in ["line_id", "sample_id", "diffday", "diffday_numeric", "cell_state", "dpt_pseudotime"] if col in dataset.column_names]
    pred_ids = []
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(dataset), args.batch_size):
            end = min(start + args.batch_size, len(dataset))
            batch_examples = [dataset[idx] for idx in range(start, end)]
            batch = collate_input_ids(batch_examples, pad_token_id)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            probabilities.append(probs)
            pred_ids.extend(np.argmax(probs, axis=1).tolist())

    pred_classes = [id_to_class.get(int(pred_id), str(pred_id)) for pred_id in pred_ids]
    out = pd.DataFrame({"pred_id": pred_ids, "pred_class": pred_classes})
    probs = np.vstack(probabilities) if probabilities else np.empty((0, 0))
    if probs.size:
        out["max_probability"] = probs.max(axis=1)
        out["accepted_by_confidence"] = True if reject_threshold is None else out["max_probability"] >= reject_threshold
        out["final_class"] = np.where(out["accepted_by_confidence"], out["pred_class"], "REJECT_OOD_OR_LOW_CONFIDENCE")
    for col in metadata_cols:
        out[col] = dataset[col]
    out.to_csv(output_dir / f"{args.prefix}_predictions.csv", index=False)

    class_col = "final_class" if "final_class" in out.columns else "pred_class"
    distribution = out[class_col].value_counts(normalize=True).rename("fraction").reset_index()
    distribution.columns = ["pred_class", "fraction"]
    counts = out[class_col].value_counts().rename("count").reset_index()
    counts.columns = ["pred_class", "count"]
    distribution = distribution.merge(counts, on="pred_class")
    distribution.to_csv(output_dir / f"{args.prefix}_prediction_distribution.csv", index=False)

    if "sample_id" in out.columns:
        per_sample = out.groupby(["sample_id", class_col]).size().reset_index(name="count")
        per_sample["fraction"] = per_sample["count"] / per_sample.groupby("sample_id")["count"].transform("sum")
        per_sample.to_csv(output_dir / f"{args.prefix}_per_sample_distribution.csv", index=False)

    summary = {
        "num_cells": int(len(out)),
        "model_dir": args.model_dir,
        "tokenized_dataset": args.tokenized_dataset,
        "temperature": temperature,
        "reject_threshold": reject_threshold,
        "prediction_distribution": distribution.to_dict(orient="records"),
    }
    if probs.size:
        summary["mean_max_probability"] = float(probs.max(axis=1).mean())
    if args.expected_class:
        summary["expected_class"] = args.expected_class
        summary["expected_class_fraction"] = float((out["pred_class"] == args.expected_class).mean())
    if reject_threshold is not None and "accepted_by_confidence" in out.columns:
        summary["accepted_fraction"] = float(out["accepted_by_confidence"].mean())
        summary["rejected_fraction"] = float((~out["accepted_by_confidence"]).mean())

    (output_dir / f"{args.prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

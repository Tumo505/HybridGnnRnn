#!/usr/bin/env python3
"""
Fine-tune Geneformer for temporal cardiomyocyte differentiation tasks.

This is the foundation-model replacement for the RNN-side trajectory learner.
Use held-out line/donor splits whenever possible; random cell splits are not
acceptable for final claims.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path


os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("train_geneformer_trajectory")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenized-dataset", default="foundation_model_data/geneformer/tokenized/gse175634_geneformer.dataset")
    parser.add_argument("--model-dir", default="models/geneformer/geneformer-10M")
    parser.add_argument("--output-dir", default="foundation_geneformer_gse175634_day_classifier")
    parser.add_argument("--output-prefix", default="gse175634_day")
    parser.add_argument("--target", choices=["diffday", "cell_state"], default="diffday")
    parser.add_argument("--model-version", choices=["V1", "V2"], default="V1")
    parser.add_argument("--freeze-layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--max-cells-per-class", type=int, default=None)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument("--train-lines", nargs="*", default=None)
    parser.add_argument("--heldout-lines", nargs="*", default=None)
    parser.add_argument("--eval-lines", nargs="*", default=None)
    parser.add_argument("--random-cell-split", action="store_true", help="Only for plumbing smoke tests; final runs should split by line.")
    parser.add_argument("--evaluate-saved-model-dir", default=None, help="Evaluate an already fine-tuned checkpoint/directory instead of training again.")
    return parser.parse_args()


def serialize_metrics(metrics: dict) -> dict:
    serializable = {}
    for key, value in metrics.items():
        if hasattr(value, "to_dict"):
            serializable[key] = value.to_dict()
        elif hasattr(value, "tolist"):
            serializable[key] = value.tolist()
        else:
            serializable[key] = value
    return serializable


def main() -> None:
    args = parse_args()
    try:
        from geneformer import Classifier
    except ImportError as exc:
        raise SystemExit("Geneformer is not installed. Install it before fine-tuning.") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_prefix = f"{args.output_prefix}_prepared"

    training_args = {
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "warmup_steps": 500,
        "weight_decay": 0.001,
        "logging_steps": 100,
        "save_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "fp16": True,
        "report_to": "none",
    }
    stratify_col = None if args.random_cell_split else args.target
    classifier = Classifier(
        classifier="cell",
        cell_state_dict={"state_key": args.target, "states": "all"},
        training_args=training_args,
        freeze_layers=args.freeze_layers,
        num_crossval_splits=1,
        split_sizes={"train": 0.9, "valid": 0.1, "test": 0.0},
        stratify_splits_col=stratify_col,
        max_ncells=args.max_cells,
        max_ncells_per_class=args.max_cells_per_class,
        forward_batch_size=args.eval_batch_size,
        model_version=args.model_version,
        nproc=args.nproc,
        ngpu=args.ngpu,
    )

    split_id_dict = None
    if args.heldout_lines:
        if not args.train_lines:
            raise ValueError("When using --heldout-lines, also provide explicit --train-lines.")
        split_id_dict = {"attr_key": "line_id", "train": args.train_lines, "test": args.heldout_lines}

    logger.info("Preparing labeled Geneformer dataset for target %s.", args.target)
    prepared_dataset = output_dir / f"{prepared_prefix}_labeled.dataset"
    id_class_dict_file = output_dir / f"{prepared_prefix}_id_class_dict.pkl"
    if not prepared_dataset.exists() or not id_class_dict_file.exists():
        classifier.prepare_data(
            input_data_file=args.tokenized_dataset,
            output_directory=output_dir,
            output_prefix=prepared_prefix,
            split_id_dict=split_id_dict,
            attr_to_split=None if args.random_cell_split or split_id_dict is not None else "line_id",
            attr_to_balance=None if args.random_cell_split or split_id_dict is not None else [args.target],
        )
    else:
        logger.info("Reusing prepared dataset at %s.", prepared_dataset)

    validate_split_id_dict = None
    if args.eval_lines:
        if not args.train_lines:
            raise ValueError("When using --eval-lines, also provide explicit --train-lines.")
        validate_split_id_dict = {"attr_key": "line_id", "train": args.train_lines, "eval": args.eval_lines}

    if args.evaluate_saved_model_dir:
        from datasets import load_from_disk

        eval_dataset = output_dir / f"{args.output_prefix}_eval_lines.dataset"
        if not eval_dataset.exists():
            data = load_from_disk(str(prepared_dataset))
            if args.eval_lines:
                eval_lines = set(args.eval_lines)
                data = data.filter(lambda example: example["line_id"] in eval_lines)
            else:
                logger.warning(
                    "No --eval-lines supplied; evaluating the saved model on the full prepared dataset. "
                    "Use this only for external labeled/sanity datasets, not final held-out-line claims."
                )
            data.save_to_disk(str(eval_dataset))
        eval_features = load_from_disk(str(eval_dataset)).features
        predict_metadata = [
            column
            for column in ["line_id", "sample_id", "diffday", "diffday_numeric", "cell_state", "dpt_pseudotime"]
            if column in eval_features
        ]
        metrics = classifier.evaluate_saved_model(
            model_directory=args.evaluate_saved_model_dir,
            id_class_dict_file=str(id_class_dict_file),
            test_data_file=str(eval_dataset),
            output_directory=str(output_dir),
            output_prefix=args.output_prefix,
            predict=True,
            predict_metadata=predict_metadata,
        )
        metrics_path = output_dir / f"{args.output_prefix}_metrics.json"
        metrics_path.write_text(json.dumps(serialize_metrics(metrics), indent=2, default=str), encoding="utf-8")
        logger.info("Saved metrics to %s.", metrics_path)
        return

    logger.info("Fine-tuning Geneformer from %s.", args.model_dir)
    metrics = classifier.validate(
        model_directory=args.model_dir,
        prepared_input_data_file=str(prepared_dataset),
        id_class_dict_file=str(id_class_dict_file),
        output_directory=str(output_dir),
        output_prefix=args.output_prefix,
        split_id_dict=validate_split_id_dict,
        predict_eval=True,
    )
    metrics_path = output_dir / f"{args.output_prefix}_metrics.json"
    metrics_path.write_text(json.dumps(serialize_metrics(metrics), indent=2, default=str), encoding="utf-8")
    logger.info("Saved metrics to %s.", metrics_path)


if __name__ == "__main__":
    main()

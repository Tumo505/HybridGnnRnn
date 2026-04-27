#!/usr/bin/env python3
"""Temperature calibration and confidence rejection thresholds for Geneformer classifiers."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dict", required=True)
    parser.add_argument("--id-class-dict", required=True)
    parser.add_argument("--output-dir", default="foundation_validation/calibration")
    parser.add_argument("--prefix", default="geneformer_calibration")
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--min-coverage", type=float, default=0.20)
    return parser.parse_args()


def load_pickle(path: str | Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / max(temperature, 1e-6)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (conf > low) & (conf <= high)
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=75, line_search_fn="strong_wolfe")
    loss_fn = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = loss_fn(logits_t / temperature, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).detach().cpu().item())


def choose_threshold(conf: np.ndarray, correct: np.ndarray, target_precision: float, min_coverage: float) -> dict:
    candidates = np.unique(np.quantile(conf, np.linspace(0.0, 1.0, 501)))
    best = None
    best_fallback = None
    for threshold in candidates:
        keep = conf >= threshold
        coverage = float(keep.mean())
        if coverage < min_coverage or not np.any(keep):
            continue
        precision = float(correct[keep].mean())
        candidate = {"threshold": float(threshold), "accepted_accuracy": precision, "coverage": coverage}
        if precision >= target_precision:
            if best is None or candidate["coverage"] > best["coverage"]:
                best = candidate
        if best_fallback is None or (
            candidate["accepted_accuracy"],
            candidate["coverage"],
        ) > (
            best_fallback["accepted_accuracy"],
            best_fallback["coverage"],
        ):
            best_fallback = candidate
    if best is None:
        best = best_fallback
    if best is None:
        threshold = float(np.quantile(conf, 1.0 - min_coverage))
        keep = conf >= threshold
        best = {"threshold": threshold, "accepted_accuracy": float(correct[keep].mean()), "coverage": float(keep.mean())}
    return best


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dict = load_pickle(args.pred_dict)
    id_to_class = {int(key): value for key, value in load_pickle(args.id_class_dict).items()}

    raw_pred_ids = pred_dict["pred_ids"]
    keep = np.asarray([str(x).lower() != "tie" for x in raw_pred_ids], dtype=bool)
    logits = np.asarray(pred_dict["predictions"], dtype=np.float64)[keep]
    labels = np.asarray(pred_dict["label_ids"], dtype=np.int64)[keep]
    pred_ids = np.asarray([int(x) for x, use in zip(raw_pred_ids, keep) if use], dtype=np.int64)
    temperature = fit_temperature(logits, labels)
    raw_probs = softmax(logits, 1.0)
    calibrated_probs = softmax(logits, temperature)
    raw_conf = raw_probs.max(axis=1)
    calibrated_conf = calibrated_probs.max(axis=1)
    calibrated_pred = calibrated_probs.argmax(axis=1)
    correct = calibrated_pred == labels
    threshold = choose_threshold(calibrated_conf, correct, args.target_precision, args.min_coverage)
    accepted = calibrated_conf >= threshold["threshold"]

    summary = {
        "temperature": temperature,
        "threshold": threshold["threshold"],
        "target_precision": args.target_precision,
        "min_coverage": args.min_coverage,
        "raw_accuracy": float(accuracy_score(labels, pred_ids)),
        "calibrated_accuracy": float(accuracy_score(labels, calibrated_pred)),
        "calibrated_macro_f1": float(f1_score(labels, calibrated_pred, average="macro", zero_division=0)),
        "raw_ece": expected_calibration_error(raw_conf, pred_ids == labels),
        "calibrated_ece": expected_calibration_error(calibrated_conf, correct),
        "accepted_coverage": float(accepted.mean()),
        "accepted_accuracy": float(correct[accepted].mean()) if np.any(accepted) else 0.0,
        "rejected_count": int((~accepted).sum()),
        "accepted_count": int(accepted.sum()),
        "dropped_tie_predictions": int((~keep).sum()),
        "classes": id_to_class,
    }
    (output_dir / f"{args.prefix}_calibration.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

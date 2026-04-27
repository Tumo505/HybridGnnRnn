#!/usr/bin/env python3
"""Create or run repeated held-out-line Geneformer validation plans."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenized-dataset", default="foundation_model_data/geneformer/tokenized/gse175634_geneformer.dataset")
    parser.add_argument("--model-dir", default="models/geneformer/Geneformer/Geneformer-V1-10M")
    parser.add_argument("--output-dir", default="foundation_repeated_line_cv")
    parser.add_argument("--targets", nargs="+", choices=["diffday", "cell_state"], default=["diffday", "cell_state"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--heldout-lines", type=int, default=3)
    parser.add_argument("--leave-one-line-out", action="store_true", help="Generate one fold per held-out line.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-cells-per-class", type=int, default=15000)
    parser.add_argument("--freeze-layers", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--nproc", type=int, default=0)
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument("--run", action="store_true", help="Run the generated commands sequentially. Default only writes the plan.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose metrics JSON already exists.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap for resumable long runs.")
    return parser.parse_args()


def line_summary(dataset_path: str) -> pd.DataFrame:
    data = load_from_disk(dataset_path)
    cols = [col for col in ["line_id", "diffday", "cell_state"] if col in data.column_names]
    frame = pd.DataFrame({col: data[col] for col in cols})
    return frame.groupby("line_id").agg(num_cells=("line_id", "size")).reset_index()


def make_folds(lines: list[str], repeats: int, heldout_count: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    folds = []
    for repeat in range(1, repeats + 1):
        shuffled = np.array(lines, dtype=object)
        rng.shuffle(shuffled)
        start = 0
        for fold_id in range(1, max(2, len(lines) // heldout_count) + 1):
            heldout = shuffled[start : start + heldout_count].tolist()
            if len(heldout) < heldout_count:
                heldout = shuffled[:heldout_count].tolist()
            train = [line for line in lines if line not in set(heldout)]
            folds.append({"repeat": repeat, "fold": fold_id, "train_lines": train, "eval_lines": heldout})
            start += heldout_count
            if start >= len(lines):
                break
    return folds


def make_leave_one_line_folds(lines: list[str]) -> list[dict]:
    folds = []
    for fold_id, heldout in enumerate(lines, start=1):
        folds.append({"repeat": 1, "fold": fold_id, "train_lines": [line for line in lines if line != heldout], "eval_lines": [heldout]})
    return folds


def command_for(args: argparse.Namespace, target: str, fold: dict, output_prefix: str) -> list[str]:
    return [
        "python",
        "scripts\\foundation\\train_geneformer_trajectory.py",
        "--tokenized-dataset",
        args.tokenized_dataset,
        "--model-dir",
        args.model_dir,
        "--output-dir",
        str(Path(args.output_dir) / output_prefix),
        "--output-prefix",
        output_prefix,
        "--target",
        target,
        "--model-version",
        "V1",
        "--freeze-layers",
        str(args.freeze_layers),
        "--epochs",
        str(args.epochs),
        "--train-batch-size",
        str(args.train_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--max-cells-per-class",
        str(args.max_cells_per_class),
        "--nproc",
        str(args.nproc),
        "--ngpu",
        str(args.ngpu),
        "--train-lines",
        *fold["train_lines"],
        "--eval-lines",
        *fold["eval_lines"],
    ]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = line_summary(args.tokenized_dataset)
    lines = sorted(summary["line_id"].astype(str).tolist())
    folds = make_leave_one_line_folds(lines) if args.leave_one_line_out else make_folds(lines, args.repeats, args.heldout_lines, args.seed)

    plan = {"line_summary": summary.to_dict(orient="records"), "runs": []}
    ps_lines = []
    executed = 0
    for target in args.targets:
        for fold in folds:
            prefix = f"gse175634_{target}_linecv_r{fold['repeat']}_f{fold['fold']}"
            cmd = command_for(args, target, fold, prefix)
            run_dir = Path(args.output_dir) / prefix
            metrics_path = run_dir / f"{prefix}_metrics.json"
            plan["runs"].append(
                {
                    "target": target,
                    "output_prefix": prefix,
                    "metrics_path": str(metrics_path),
                    **fold,
                    "command": cmd,
                }
            )
            ps_lines.append(" ".join(f'"{part}"' if " " in part else part for part in cmd))
            if args.run:
                if args.skip_existing and metrics_path.exists():
                    continue
                if args.max_runs is not None and executed >= args.max_runs:
                    continue
                subprocess.run(cmd, check=True)
                executed += 1

    (output_dir / "fold_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (output_dir / "run_repeated_geneformer_cv.ps1").write_text("\n".join(ps_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "num_runs": len(plan["runs"]), "run_now": bool(args.run), "executed": executed}, indent=2))


if __name__ == "__main__":
    main()

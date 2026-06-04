#!/usr/bin/env python
"""Summarize training runs into run-level and config-level CSVs.

Reads runs/<name>/metrics.jsonl + config.json for all completed runs
and writes results/run_summary.csv + results/orth_summary.csv.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from src.paths import ROOT, read_jsonl

ORTH_ORDER = ["adamw", "vanilla", "manual", "fast", "polar_express"]


def _orth_label(orth: str) -> str:
    return {
        "adamw": "AdamW",
        "vanilla": "Vanilla",
        "manual": "Manual",
        "fast": "Fast",
        "polar_express": "Polar Express",
    }.get(orth, orth)


def _auc(points: list[tuple[float, float]]) -> float | None:
    points = sorted(points)
    if len(points) < 2:
        return None
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        total += (x1 - x0) * (y0 + y1) / 2
    span = points[-1][0] - points[0][0]
    return total / span if span > 0 else None


def summarize_run(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.json"
    if not metrics_path.exists():
        return None
    rows = read_jsonl(metrics_path)
    if not rows:
        return None

    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    vals = [row for row in rows if "val/loss" in row]
    specs = [row for row in rows if "spec/sample_count" in row]
    final_row = rows[-1]
    final_val = vals[-1] if vals else {}
    final_spec = specs[-1] if specs else {}

    val_token_points = [
        (float(row["val/global_train_tokens"]), float(row["val/loss"]))
        for row in vals
        if "val/global_train_tokens" in row
    ]

    orth = config.get("orthogonalizer_type")
    name = config.get("run_name", run_dir.name)
    return {
        "run": str(run_dir.relative_to(ROOT)),
        "name": name,
        "orthogonalizer_type": orth,
        "orth_label": _orth_label(orth),
        "schedule": config.get("orth_schedule_name"),
        "final_val_loss": final_val.get("val/loss"),
        "best_val_loss": min((row["val/loss"] for row in vals), default=None),
        "val_auc_tokens": _auc(val_token_points),
        "benchmark_wall_clock_s": final_row.get("benchmark/wall_clock_s"),
        "spec_update_orth_error": final_spec.get("spec/update_orth_error"),
        "peak_allocated_mb": final_row.get("memory/peak_allocated_mb"),
        "status": final_row.get("status", "running_or_incomplete"),
    }


def summarize_groups(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        orth = row.get("orthogonalizer_type")
        if orth:
            grouped[orth].append(row)

    summary: list[dict] = []
    for orth in ORTH_ORDER:
        group_rows = grouped.get(orth, [])
        if not group_rows:
            continue

        def _collect(key: str) -> list[float]:
            values = []
            for row in group_rows:
                value = row.get(key)
                if value is not None:
                    values.append(float(value))
            return values

        val_losses = _collect("final_val_loss")
        orth_errors = _collect("spec_update_orth_error")
        statuses = [row.get("status") for row in group_rows]

        summary.append({
            "orthogonalizer_type": orth,
            "orth_label": _orth_label(orth),
            "run_count": len(group_rows),
            "completed_count": sum(s == "completed" for s in statuses),
            "final_val_loss": mean(val_losses) if val_losses else None,
            "spec_update_orth_error": mean(orth_errors) if orth_errors else None,
        })
    return summary


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--print-top", type=int, default=0)
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()

    runs_dir = (ROOT / "runs").resolve()
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [path.parent for path in runs_dir.rglob("metrics.jsonl")]
    rows = [r for r in (summarize_run(d) for d in sorted(run_dirs)) if r is not None]
    rows.sort(key=lambda row: (
        ORTH_ORDER.index(row["orthogonalizer_type"])
        if row["orthogonalizer_type"] in ORTH_ORDER else 999,
        row["name"],
    ))
    orth_rows = summarize_groups(rows)

    run_csv = out_dir / "run_summary.csv"
    orth_csv = out_dir / "orth_summary.csv"
    if not rows and run_csv.exists() and not args.allow_empty:
        print(f"No run metrics found; preserved existing {run_csv}")
        return 0

    run_fields = [
        "run", "name", "orthogonalizer_type", "orth_label",
        "schedule", "final_val_loss", "best_val_loss",
        "val_auc_tokens", "benchmark_wall_clock_s",
        "spec_update_orth_error", "peak_allocated_mb", "status",
    ]
    with run_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows(rows)

    orth_fields = [
        "orthogonalizer_type", "orth_label", "run_count", "completed_count",
        "final_val_loss", "spec_update_orth_error",
    ]
    with orth_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=orth_fields)
        writer.writeheader()
        writer.writerows(orth_rows)

    print(f"Wrote {run_csv}")
    print(f"Wrote {orth_csv}")

    if args.print_top > 0:
        completed = [row for row in rows if row.get("final_val_loss") is not None]
        completed.sort(key=lambda row: float(row["final_val_loss"]))
        print("Top runs by final validation loss:")
        print("orth,name,final_val_loss,status")
        for row in completed[:args.print_top]:
            print(",".join([
                _fmt(row["orth_label"]),
                _fmt(row["name"]),
                _fmt(row["final_val_loss"]),
                _fmt(row["status"]),
            ]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

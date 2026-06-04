#!/usr/bin/env python
"""Summarize fixed 5x3 experiment runs into run-level and config-level CSVs."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from src.paths import ROOT, read_jsonl

ORTH_ORDER = ["adamw", "vanilla", "manual", "fast", "polar_express"]


def auc(points: list[tuple[float, float]]) -> float | None:
    points = sorted(points)
    if len(points) < 2:
        return None
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        total += (x1 - x0) * (y0 + y1) / 2
    span = points[-1][0] - points[0][0]
    return total / span if span > 0 else None


def orth_label(orth: str) -> str:
    return {
        "adamw": "AdamW",
        "vanilla": "Vanilla",
        "manual": "Manual",
        "fast": "Fast",
        "polar_express": "Polar Express",
    }.get(orth, orth)


def parse_seed(name: str) -> int | None:
    if "seed" not in name:
        return None
    try:
        return int(name.rsplit("seed", 1)[1])
    except ValueError:
        return None


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
    trains = [row for row in rows if "train/throughput_tokens_per_sec" in row]
    specs = [row for row in rows if "spec/sample_count" in row]
    final_row = rows[-1]
    final_val = vals[-1] if vals else {}
    final_train = trains[-1] if trains else {}
    final_spec = specs[-1] if specs else {}

    val_token_points = [
        (float(row["val/global_train_tokens"]), float(row["val/loss"]))
        for row in vals
        if "val/global_train_tokens" in row
    ]
    val_wall_points = [
        (float(row["val/global_wall_time_s"]), float(row["val/loss"]))
        for row in vals
        if "val/global_wall_time_s" in row
    ]

    orth = config.get("orthogonalizer_type")
    name = config.get("run_name", run_dir.name)
    return {
        "run": str(run_dir.relative_to(ROOT)),
        "name": name,
        "seed": parse_seed(name),
        "orthogonalizer_type": orth,
        "orth_label": orth_label(orth),
        "schedule": config.get("orth_schedule_name"),
        "lr_mul": config.get("lr_mul"),
        "T_ns": config.get("T_ns"),
        "fast_steps": config.get("fast_steps"),
        "stable_steps": config.get("stable_steps"),
        "pe_T": config.get("pe_T"),
        "pe_lower_bound": config.get("pe_lower_bound"),
        "train_token_budget": config.get("train_token_budget"),
        "final_tokens": final_val.get("val/global_train_tokens") or final_train.get("train/tokens"),
        "final_val_loss": final_val.get("val/loss"),
        "best_val_loss": min((row["val/loss"] for row in vals), default=None),
        "val_auc_tokens": auc(val_token_points),
        "val_auc_wall": auc(val_wall_points),
        "throughput_tokens_per_sec": final_train.get("train/throughput_tokens_per_sec"),
        "step_time_ms": final_train.get("train/step_time_ms"),
        "grad_norm": final_train.get("train/grad_norm"),
        "spec_update_orth_error": final_spec.get("spec/update_orth_error"),
        "spec_update_sv_std": final_spec.get("spec/update_sv_std"),
        "spec_update_stable_rank": final_spec.get("spec/update_stable_rank"),
        "spec_update_svd_entropy": final_spec.get("spec/update_svd_entropy"),
        "spec_momentum_orth_error": final_spec.get("spec/momentum_orth_error"),
        "wall_elapsed_s": final_val.get("val/global_wall_time_s") or final_train.get("wall/elapsed_s"),
        "peak_allocated_mb": final_row.get("memory/peak_allocated_mb"),
        "peak_reserved_mb": final_row.get("memory/peak_reserved_mb"),
        "status": final_row.get("status", "running_or_incomplete"),
    }


def summarize_groups(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        orth = row.get("orthogonalizer_type")
        if orth:
            grouped[orth].append(row)

    summary = []
    for orth in ORTH_ORDER:
        group_rows = grouped.get(orth, [])
        if not group_rows:
            continue

        def collect(key: str) -> list[float]:
            values = []
            for row in group_rows:
                value = row.get(key)
                if value is not None:
                    values.append(float(value))
            return values

        val_losses = collect("final_val_loss")
        auc_tokens = collect("val_auc_tokens")
        throughput = collect("throughput_tokens_per_sec")
        step_time = collect("step_time_ms")
        orth_error = collect("spec_update_orth_error")
        wall_time = collect("wall_elapsed_s")
        statuses = [row.get("status") for row in group_rows]

        summary.append({
            "orthogonalizer_type": orth,
            "orth_label": orth_label(orth),
            "run_count": len(group_rows),
            "completed_count": sum(status == "completed" for status in statuses),
            "final_val_loss_mean": mean(val_losses) if val_losses else None,
            "final_val_loss_std": stdev(val_losses) if len(val_losses) > 1 else 0.0 if val_losses else None,
            "best_val_loss_mean": mean(collect("best_val_loss")) if collect("best_val_loss") else None,
            "val_auc_tokens_mean": mean(auc_tokens) if auc_tokens else None,
            "throughput_tokens_per_sec_mean": mean(throughput) if throughput else None,
            "step_time_ms_mean": mean(step_time) if step_time else None,
            "spec_update_orth_error_mean": mean(orth_error) if orth_error else None,
            "wall_elapsed_s_mean": mean(wall_time) if wall_time else None,
        })
    return summary


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_top(rows: list[dict], count: int) -> None:
    if count <= 0:
        return
    completed = [row for row in rows if row.get("final_val_loss") is not None]
    completed.sort(key=lambda row: (float(row["final_val_loss"]), row["name"]))
    print("Top runs by final validation loss:")
    print("orth,name,seed,final_val_loss,throughput_tokens_per_sec,status")
    for row in completed[:count]:
        print(",".join([
            fmt(row["orth_label"]),
            fmt(row["name"]),
            fmt(row["seed"]),
            fmt(row["final_val_loss"]),
            fmt(row["throughput_tokens_per_sec"]),
            fmt(row["status"]),
        ]))


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
    rows = [row for row in (summarize_run(run_dir) for run_dir in sorted(run_dirs)) if row is not None]
    rows.sort(key=lambda row: (
        ORTH_ORDER.index(row["orthogonalizer_type"]) if row["orthogonalizer_type"] in ORTH_ORDER else 999,
        row["seed"] if row["seed"] is not None else 999,
        row["name"],
    ))
    orth_rows = summarize_groups(rows)

    run_csv = out_dir / "run_summary.csv"
    orth_csv = out_dir / "orth_summary.csv"
    if not rows and run_csv.exists() and not args.allow_empty:
        print(f"No run metrics found; preserved existing {run_csv}")
        return 0

    run_fields = [
        "run", "name", "seed", "orthogonalizer_type", "orth_label", "schedule", "lr_mul",
        "T_ns", "fast_steps", "stable_steps", "pe_T", "pe_lower_bound",
        "train_token_budget", "final_tokens", "final_val_loss", "best_val_loss",
        "val_auc_tokens", "val_auc_wall", "throughput_tokens_per_sec", "step_time_ms",
        "grad_norm", "spec_update_orth_error", "spec_update_sv_std", "spec_update_stable_rank",
        "spec_update_svd_entropy", "spec_momentum_orth_error", "wall_elapsed_s",
        "peak_allocated_mb", "peak_reserved_mb", "status",
    ]
    with run_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows(rows)

    orth_fields = [
        "orthogonalizer_type", "orth_label", "run_count", "completed_count",
        "final_val_loss_mean", "final_val_loss_std", "best_val_loss_mean",
        "val_auc_tokens_mean", "throughput_tokens_per_sec_mean", "step_time_ms_mean",
        "spec_update_orth_error_mean", "wall_elapsed_s_mean",
    ]
    with orth_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=orth_fields)
        writer.writeheader()
        writer.writerows(orth_rows)

    print(f"Wrote {run_csv}")
    print(f"Wrote {orth_csv}")
    print_top(rows, args.print_top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

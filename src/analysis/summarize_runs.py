#!/usr/bin/env python
"""Summarize run JSONL logs produced by the training program."""


import argparse
import csv
import json
from pathlib import Path

from src.utils import ROOT, read_jsonl


def auc(points: list[tuple[float, float]]) -> float | None:
    points = sorted(points)
    if len(points) < 2:
        return None
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        total += (x1 - x0) * (y0 + y1) / 2
    denom = points[-1][0] - points[0][0]
    return total / denom if denom > 0 else None


def summarize_run(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.json"
    if not metrics_path.exists():
        return None
    rows = read_jsonl(metrics_path)
    if not rows:
        return None
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    vals = [r for r in rows if "val/loss" in r]
    trains = [r for r in rows if "train/throughput_tokens_per_sec" in r]
    specs = [r for r in rows if "spec/sample_count" in r]
    final_val = vals[-1] if vals else {}
    final_train = trains[-1] if trains else {}
    final_spec = specs[-1] if specs else {}
    final_row = rows[-1]
    val_token_points = [
        (float(r["val/global_train_tokens"]), float(r["val/loss"]))
        for r in vals
        if "val/global_train_tokens" in r
    ]
    val_wall_points = [
        (float(r["val/global_wall_time_s"]), float(r["val/loss"]))
        for r in vals
        if "val/global_wall_time_s" in r
    ]
    return {
        "run": str(run_dir.relative_to(ROOT)),
        "group": config.get("wandb_group", run_dir.parent.name),
        "name": config.get("run_name", run_dir.name),
        "orthogonalizer_type": config.get("orthogonalizer_type"),
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
        "best_val_loss": min((r["val/loss"] for r in vals), default=None),
        "val_auc_tokens": auc(val_token_points),
        "val_auc_wall": auc(val_wall_points),
        "throughput_tokens_per_sec": final_train.get("train/throughput_tokens_per_sec"),
        "grad_norm": final_train.get("train/grad_norm"),
        "spec_update_orth_error": final_spec.get("spec/update_orth_error"),
        "spec_update_sv_std": final_spec.get("spec/update_sv_std"),
        "spec_update_stable_rank": final_spec.get("spec/update_stable_rank"),
        "spec_update_svd_entropy": final_spec.get("spec/update_svd_entropy"),
        "spec_momentum_sv_std": final_spec.get("spec/momentum_sv_std"),
        "spec_momentum_orth_error": final_spec.get("spec/momentum_orth_error"),
        "spec_momentum_stable_rank": final_spec.get("spec/momentum_stable_rank"),
        "wall_elapsed_s": final_val.get("val/global_wall_time_s") or final_train.get("wall/elapsed_s"),
        "peak_allocated_mb": final_row.get("memory/peak_allocated_mb"),
        "peak_reserved_mb": final_row.get("memory/peak_reserved_mb"),
        "status": final_row.get("status", "running_or_incomplete"),
    }


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_top(rows: list[dict], count: int) -> None:
    if count <= 0:
        return
    if not rows:
        print("No run metrics found.")
        return
    print("Top runs by final validation loss:")
    print("group,name,schedule,lr_mul,final_val_loss,throughput_tokens_per_sec,status")
    for row in rows[:count]:
        print(
            ",".join(
                [
                    fmt(row["group"]),
                    fmt(row["name"]),
                    fmt(row["schedule"]),
                    fmt(row["lr_mul"]),
                    fmt(row["final_val_loss"]),
                    fmt(row["throughput_tokens_per_sec"]),
                    fmt(row["status"]),
                ]
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--print-top", type=int, default=0)
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()

    runs_dir = (ROOT / args.runs_dir).resolve()
    run_dirs = [p.parent for p in runs_dir.rglob("metrics.jsonl")]
    rows = [r for r in (summarize_run(p) for p in sorted(run_dirs)) if r is not None]
    rows.sort(key=lambda r: (r["final_val_loss"] is None, r["final_val_loss"] or 999, r["name"]))

    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "run", "group", "name", "orthogonalizer_type", "schedule", "lr_mul",
        "T_ns", "fast_steps", "stable_steps", "pe_T", "pe_lower_bound",
        "train_token_budget", "final_tokens", "final_val_loss", "best_val_loss",
        "val_auc_tokens", "val_auc_wall", "throughput_tokens_per_sec", "wall_elapsed_s",
        "grad_norm", "spec_update_orth_error", "spec_update_sv_std", "spec_update_stable_rank",
        "spec_update_svd_entropy", "spec_momentum_sv_std", "spec_momentum_orth_error",
        "spec_momentum_stable_rank", "peak_allocated_mb", "peak_reserved_mb", "status",
    ]
    csv_path = out_dir / "run_summary.csv"
    if not rows and csv_path.exists() and not args.allow_empty:
        print(f"No run metrics found; preserved existing {csv_path}")
        return 0
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {csv_path}")
    print_top(rows, args.print_top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

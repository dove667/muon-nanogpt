import csv
import json
from pathlib import Path

from src.paths import ROOT, read_jsonl

ORTHOGONALIZER_ORDER = ["adamw", "vanilla", "manual", "fast", "polar_express"]


def _auc(points: list[tuple[float, float]]) -> float | None:
    points = sorted(points)
    if len(points) < 2:
        return None
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        total += (x1 - x0) * (y0 + y1) / 2
    span = points[-1][0] - points[0][0]
    return total / span if span > 0 else None


def _detect_mode(rows: list[dict]) -> str:
    has_benchmark = any("benchmark/wall_clock_s" in row for row in rows)
    has_spectral = any("spec/sample_count" in row for row in rows)
    if has_benchmark and has_spectral:
        raise ValueError("Mixed benchmark+spectrum runs are not supported.")
    if has_benchmark:
        return "benchmark"
    if has_spectral:
        return "spectral"
    return "train"


def _summarize_single_run(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    rows = read_jsonl(metrics_path)
    if not rows:
        return None

    config = json.loads(config_path.read_text(encoding="utf-8"))
    vals = [row for row in rows if "val/loss" in row]
    specs = [row for row in rows if "spec/sample_count" in row]
    final_row = rows[-1]
    final_val = vals[-1] if vals else {}
    final_spec = specs[-1] if specs else {}
    mode = _detect_mode(rows)

    val_token_points = [
        (float(row["val/global_train_tokens"]), float(row["val/loss"]))
        for row in vals
        if "val/global_train_tokens" in row
    ]

    orthogonalizer_type = config["orthogonalizer_type"]
    run_name = config["run_name"]
    return {
        "run": str(run_dir.relative_to(ROOT)),
        "name": run_name,
        "mode": mode,
        "orthogonalizer_type": orthogonalizer_type,
        "schedule": config.get("orth_schedule_name"),
        "final_val_loss": final_val.get("val/loss"),
        "best_val_loss": min((row["val/loss"] for row in vals), default=None),
        "val_auc_tokens": _auc(val_token_points),
        "benchmark_wall_clock_s": final_row.get("benchmark/wall_clock_s"),
        "spec_update_orth_error": final_spec.get("spec/update_orth_error"),
        "peak_allocated_mb": final_row.get("memory/peak_allocated_mb"),
        "status": final_row.get("status", "running_or_incomplete"),
    }


def summarize_run(runs_dir: Path) -> list[dict]:
    run_dirs = sorted(path.parent for path in runs_dir.rglob("metrics.jsonl"))
    run_summaries = [row for row in (_summarize_single_run(run_dir) for run_dir in run_dirs) if row is not None]
    if not run_summaries:
        return []

    run_summaries.sort(key=lambda row: (
        ORTHOGONALIZER_ORDER.index(row["orthogonalizer_type"])
        if row["orthogonalizer_type"] in ORTHOGONALIZER_ORDER else 999,
        row["mode"],
        row["name"],
    ))

    runs_by_type_and_mode: dict[str, dict[str, dict]] = {}
    for run_summary in run_summaries:
        orthogonalizer_type = run_summary["orthogonalizer_type"]
        run_mode = run_summary["mode"]
        runs_by_mode = runs_by_type_and_mode.setdefault(orthogonalizer_type, {})
        if run_mode in runs_by_mode:
            raise ValueError(
                "Duplicate "
                f"{run_mode} run for orthogonalizer_type={orthogonalizer_type}: "
                f"{runs_by_mode[run_mode]['run']} and {run_summary['run']}"
            )
        runs_by_mode[run_mode] = run_summary

    summary_rows: list[dict] = []
    for orthogonalizer_type in ORTHOGONALIZER_ORDER:
        runs_by_mode = runs_by_type_and_mode.get(orthogonalizer_type)
        if not runs_by_mode:
            continue
        train_run = runs_by_mode.get("train", {})
        benchmark_run = runs_by_mode.get("benchmark", {})
        spectral_run = runs_by_mode.get("spectral", {})
        summary_rows.append({
            "orthogonalizer_type": orthogonalizer_type,
            "train_run": train_run.get("run"),
            "train_name": train_run.get("name"),
            "train_schedule": train_run.get("schedule"),
            "train_final_val_loss": train_run.get("final_val_loss"),
            "train_best_val_loss": train_run.get("best_val_loss"),
            "train_val_auc_tokens": train_run.get("val_auc_tokens"),
            "train_peak_allocated_mb": train_run.get("peak_allocated_mb"),
            "train_status": train_run.get("status"),
            "benchmark_run": benchmark_run.get("run"),
            "benchmark_name": benchmark_run.get("name"),
            "benchmark_schedule": benchmark_run.get("schedule"),
            "benchmark_wall_clock_s": benchmark_run.get("benchmark_wall_clock_s"),
            "benchmark_peak_allocated_mb": benchmark_run.get("peak_allocated_mb"),
            "benchmark_status": benchmark_run.get("status"),
            "spectral_run": spectral_run.get("run"),
            "spectral_name": spectral_run.get("name"),
            "spectral_schedule": spectral_run.get("schedule"),
            "spectral_update_orth_error": spectral_run.get("spec_update_orth_error"),
            "spectral_peak_allocated_mb": spectral_run.get("peak_allocated_mb"),
            "spectral_status": spectral_run.get("status"),
        })
    return summary_rows


def main() -> int:
    runs_dir = (ROOT / "runs").resolve()
    out_dir = (ROOT / "results").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not runs_dir.exists():
        print(f"No runs directory found: {runs_dir}")
        return 0

    summary_rows = summarize_run(runs_dir)
    if not summary_rows:
        print(f"No run metrics found under {runs_dir}")
        return 0
    summary_csv = out_dir / "summary.csv"

    summary_fields = [
        "orthogonalizer_type",
        "train_run", "train_name", "train_schedule",
        "train_final_val_loss", "train_best_val_loss", "train_val_auc_tokens",
        "train_peak_allocated_mb", "train_status",
        "benchmark_run", "benchmark_name", "benchmark_schedule",
        "benchmark_wall_clock_s", "benchmark_peak_allocated_mb", "benchmark_status",
        "spectral_run", "spectral_name", "spectral_schedule",
        "spectral_update_orth_error", "spectral_peak_allocated_mb", "spectral_status",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

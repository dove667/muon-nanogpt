#!/usr/bin/env python
"""Plot validation loss curves for all completed runs.

Reads runs/<name>/metrics.jsonl and outputs results/figures/*.png.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from src.paths import ROOT, read_jsonl

ORTH_ORDER = ["adamw", "vanilla", "manual", "fast", "polar_express"]
ORTH_LABEL = {
    "adamw": "AdamW",
    "vanilla": "Vanilla",
    "manual": "Manual",
    "fast": "Fast",
    "polar_express": "Polar Express",
}
ORTH_COLOR = {
    "adamw": "#4c78a8",
    "vanilla": "#72b7b2",
    "manual": "#f58518",
    "fast": "#e45756",
    "polar_express": "#54a24b",
}


def _load_runs(runs_dir: Path, orths: set[str] | None) -> list[dict]:
    runs = []
    for metrics_path in sorted(runs_dir.rglob("metrics.jsonl")):
        run_dir = metrics_path.parent
        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        orth = config.get("orthogonalizer_type")
        if orths and orth not in orths:
            continue
        name = config.get("run_name", run_dir.name)
        runs.append({
            "name": name,
            "orth": orth,
            "rows": read_jsonl(metrics_path),
        })
    return runs


def _val_points(run: dict, x_key: str) -> list[tuple[float, float]]:
    return [
        (float(row[x_key]), float(row["val/loss"]))
        for row in run["rows"]
        if x_key in row and "val/loss" in row
    ]


def _final_metric(run: dict, key: str) -> float | None:
    for row in run["rows"]:
        if key in row:
            return float(row[key])
    return None


def plot_val_loss_vs_tokens(runs: list[dict], out_dir: Path) -> None:
    """One val/loss curve per orth vs train tokens."""
    import matplotlib.pyplot as plt

    grouped: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        grouped.setdefault(run.get("orth", ""), []).append(run)

    plt.figure(figsize=(10, 6))
    any_series = False
    for orth in ORTH_ORDER:
        orth_runs = grouped.get(orth, [])
        by_x: dict[float, list[float]] = defaultdict(list)
        for run in orth_runs:
            for x, y in _val_points(run, "val/global_train_tokens"):
                by_x[x].append(y)
        if not by_x:
            continue
        xs = sorted(by_x)
        ys = [mean(by_x[x]) for x in xs]
        any_series = True
        plt.plot(xs, ys, color=ORTH_COLOR[orth], linewidth=2.2, label=ORTH_LABEL[orth])
    if not any_series:
        plt.close()
        return
    plt.xlabel("train tokens")
    plt.ylabel("val/loss")
    plt.title("Validation loss by train tokens")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "val_loss_vs_tokens.png", dpi=180)
    plt.close()


def plot_val_loss_vs_wall_time(runs: list[dict], out_dir: Path) -> None:
    """One val/loss curve per orth vs wall time (only if benchmark data exists)."""
    import matplotlib.pyplot as plt

    grouped: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        grouped.setdefault(run.get("orth", ""), []).append(run)

    plt.figure(figsize=(10, 6))
    any_series = False
    for orth in ORTH_ORDER:
        orth_runs = grouped.get(orth, [])
        by_x: dict[float, list[float]] = defaultdict(list)
        for run in orth_runs:
            for x, y in _val_points(run, "val/global_wall_time_s"):
                by_x[x].append(y)
        if not by_x:
            continue
        xs = sorted(by_x)
        ys = [mean(by_x[x]) for x in xs]
        any_series = True
        plt.plot(xs, ys, color=ORTH_COLOR[orth], linewidth=2.2, label=ORTH_LABEL[orth])
    if not any_series:
        plt.close()
        return
    plt.xlabel("wall time (s)")
    plt.ylabel("val/loss")
    plt.title("Validation loss by wall time")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "val_loss_vs_wall.png", dpi=180)
    plt.close()


def plot_final_val_loss_bars(runs: list[dict], out_dir: Path) -> None:
    """Bar chart of final val/loss per orth."""
    import matplotlib.pyplot as plt

    grouped: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        grouped.setdefault(run.get("orth", ""), []).append(run)

    labels, values, colors = [], [], []
    for orth in ORTH_ORDER:
        orth_runs = grouped.get(orth, [])
        orth_values = []
        for run in orth_runs:
            v = _final_metric(run, "val/loss")
            if v is not None:
                orth_values.append(v)
        if not orth_values:
            continue
        labels.append(ORTH_LABEL[orth])
        values.append(mean(orth_values))
        colors.append(ORTH_COLOR[orth])
    if not labels:
        return
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(labels)), values, color=colors)
    plt.xticks(range(len(labels)), labels)
    plt.title("Final validation loss")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "final_val_loss.png", dpi=180)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/figures")
    parser.add_argument("--orths", nargs="*", default=None)
    args = parser.parse_args()

    runs_dir = (ROOT / "runs").resolve()
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    orths = set(args.orths) if args.orths else None

    runs = _load_runs(runs_dir, orths)
    if not runs:
        print("No run metrics found.")
        return 0

    plot_val_loss_vs_tokens(runs, out_dir)
    plot_val_loss_vs_wall_time(runs, out_dir)
    plot_final_val_loss_bars(runs, out_dir)
    print(f"Wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

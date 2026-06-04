#!/usr/bin/env python
"""Plot figures for the fixed 5x3 experiment comparison."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from src.utils import ROOT, read_jsonl

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


def load_runs(runs_dir: Path, orths: set[str] | None) -> list[dict]:
    runs = []
    for metrics_path in sorted(runs_dir.rglob("metrics.jsonl")):
        run_dir = metrics_path.parent
        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        orth = config.get("orthogonalizer_type")
        if orths and orth not in orths:
            continue
        name = config.get("run_name", run_dir.name)
        seed = None
        if "seed" in name:
            try:
                seed = int(name.rsplit("seed", 1)[1])
            except ValueError:
                seed = None
        runs.append({
            "dir": run_dir,
            "name": name,
            "seed": seed,
            "orth": orth,
            "schedule": config.get("orth_schedule_name", ""),
            "rows": read_jsonl(metrics_path),
        })
    return runs


def val_points(run: dict, x_key: str) -> list[tuple[float, float]]:
    return [
        (float(row[x_key]), float(row["val/loss"]))
        for row in run["rows"]
        if x_key in row and "val/loss" in row
    ]


def final_metric(run: dict, key: str) -> float | None:
    value = None
    for row in run["rows"]:
        if key in row:
            value = float(row[key])
    return value


def grouped_runs(runs: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        orth = run.get("orth")
        if orth:
            grouped[orth].append(run)
    return grouped


def plot_mean_val_curve(runs: list[dict], out_dir: Path, x_key: str, filename: str, xlabel: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    any_series = False
    for orth in ORTH_ORDER:
        orth_runs = grouped_runs(runs).get(orth, [])
        by_x: dict[float, list[float]] = defaultdict(list)
        for run in orth_runs:
            for x, y in val_points(run, x_key):
                by_x[x].append(y)
        if not by_x:
            continue
        xs = sorted(by_x)
        ys = [mean(by_x[x]) for x in xs]
        stds = [stdev(by_x[x]) if len(by_x[x]) > 1 else 0.0 for x in xs]
        color = ORTH_COLOR[orth]
        label = ORTH_LABEL[orth]
        any_series = True
        plt.plot(xs, ys, color=color, linewidth=2.2, label=label)
        lower = [y - s for y, s in zip(ys, stds)]
        upper = [y + s for y, s in zip(ys, stds)]
        plt.fill_between(xs, lower, upper, color=color, alpha=0.15)
    if not any_series:
        plt.close()
        return
    plt.xlabel(xlabel)
    plt.ylabel("val/loss")
    plt.title("Mean validation loss across seeds")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=180)
    plt.close()


def plot_seed_curves(runs: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    grouped = grouped_runs(runs)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax in axes[len(ORTH_ORDER):]:
        ax.axis("off")
    for idx, orth in enumerate(ORTH_ORDER):
        ax = axes[idx]
        orth_runs = sorted(grouped.get(orth, []), key=lambda run: (run["seed"] is None, run["seed"]))
        for run in orth_runs:
            points = val_points(run, "val/global_train_tokens")
            if not points:
                continue
            xs = [x for x, _ in points]
            ys = [y for _, y in points]
            label = f"seed {run['seed']}" if run["seed"] is not None else run["name"]
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.6, label=label)
        ax.set_title(ORTH_LABEL[orth])
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.supxlabel("train tokens")
    fig.supylabel("val/loss")
    fig.suptitle("Validation curves by seed")
    fig.tight_layout()
    fig.savefig(out_dir / "val_loss_by_seed_tokens.png", dpi=180)
    plt.close(fig)


def plot_scatter(runs: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for orth in ORTH_ORDER:
        orth_runs = grouped_runs(runs).get(orth, [])
        xs, ys = [], []
        for run in orth_runs:
            throughput = final_metric(run, "train/throughput_tokens_per_sec")
            final_val = final_metric(run, "val/loss")
            if throughput is None or final_val is None:
                continue
            xs.append(throughput)
            ys.append(final_val)
        if not xs:
            continue
        plt.scatter(xs, ys, s=48, color=ORTH_COLOR[orth], label=ORTH_LABEL[orth])
    plt.xlabel("train/throughput_tokens_per_sec")
    plt.ylabel("final val/loss")
    plt.title("Throughput vs final validation loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "throughput_vs_final_val_loss.png", dpi=180)
    plt.close()


def plot_group_bars(runs: list[dict], out_dir: Path, metric_key: str, filename: str, title: str) -> None:
    import matplotlib.pyplot as plt

    labels, means, errs, colors = [], [], [], []
    for orth in ORTH_ORDER:
        values = []
        for run in grouped_runs(runs).get(orth, []):
            value = final_metric(run, metric_key)
            if value is not None:
                values.append(value)
        if not values:
            continue
        labels.append(ORTH_LABEL[orth])
        means.append(mean(values))
        errs.append(stdev(values) if len(values) > 1 else 0.0)
        colors.append(ORTH_COLOR[orth])
    if not labels:
        return
    plt.figure(figsize=(9, 5.5))
    x = range(len(labels))
    plt.bar(x, means, yerr=errs, color=colors, capsize=5)
    plt.xticks(list(x), labels)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=180)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-dir", default="results/figures")
    parser.add_argument("--orths", nargs="*", default=None)
    args = parser.parse_args()

    runs_dir = (ROOT / args.runs_dir).resolve()
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    orths = set(args.orths) if args.orths else None

    runs = load_runs(runs_dir, orths)
    if not runs:
        print("No run metrics found.")
        return 0

    plot_mean_val_curve(runs, out_dir, "val/global_train_tokens", "val_loss_mean_vs_tokens.png", "train tokens")
    plot_mean_val_curve(runs, out_dir, "val/global_wall_time_s", "val_loss_mean_vs_wall_time.png", "wall time (s)")
    plot_seed_curves(runs, out_dir)
    plot_scatter(runs, out_dir)
    plot_group_bars(runs, out_dir, "val/loss", "final_val_loss_mean_std.png", "Final validation loss by orthogonalizer")
    plot_group_bars(runs, out_dir, "spec/update_orth_error", "orth_error_mean_std.png", "Update orthogonality error by orthogonalizer")
    plot_group_bars(runs, out_dir, "train/step_time_ms", "step_time_ms_mean_std.png", "Step time by orthogonalizer")
    print(f"Wrote figures under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

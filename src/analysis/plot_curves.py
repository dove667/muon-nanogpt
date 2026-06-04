#!/usr/bin/env python
"""Create local PNG curves from training metrics.

W&B receives the same metric keys. These plots are for local records and reports.
"""


import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from src.utils import ROOT, read_jsonl


def load_runs(runs_dir: Path, groups: set[str] | None) -> list[dict]:
    loaded = []
    for metrics_path in sorted(runs_dir.rglob("metrics.jsonl")):
        run_dir = metrics_path.parent
        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        group = config.get("wandb_group", run_dir.parent.name)
        if groups and group not in groups:
            continue
        loaded.append({
            "dir": run_dir,
            "group": group,
            "name": config.get("run_name", run_dir.name),
            "schedule": config.get("orth_schedule_name", run_dir.name),
            "config": config,
            "rows": read_jsonl(metrics_path),
        })
    return loaded


def final_value(run: dict, key: str):
    value = None
    for row in run["rows"]:
        if key in row:
            value = row[key]
    return value


def best_value(run: dict, key: str):
    values = [row[key] for row in run["rows"] if key in row]
    return min(values) if values else None


def val_auc_tokens(run: dict) -> float | None:
    points = sorted(
        (float(row["val/global_train_tokens"]), float(row["val/loss"]))
        for row in run["rows"]
        if "val/global_train_tokens" in row and "val/loss" in row
    )
    if len(points) < 2:
        return None
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        total += (x1 - x0) * (y0 + y1) / 2
    denom = points[-1][0] - points[0][0]
    return total / denom if denom > 0 else None


def p2_tf(run: dict) -> tuple[int, int] | None:
    match = re.match(r"p2_T(?P<T>\d+)_f(?P<f>\d+)_s(?P<s>\d+)", run["schedule"] or "")
    if not match:
        return None
    try:
        return int(match.group("T")), int(match.group("f"))
    except ValueError:
        return None


def pe_t_l(run: dict) -> tuple[int, float, str] | None:
    match = re.match(r"pe_T(?P<T>\d+)_l(?P<lb>.+)", run["schedule"] or "")
    if not match:
        return None
    lb_text = match.group("lb")
    try:
        return int(match.group("T")), float(lb_text), lb_text
    except ValueError:
        return None


def plot_val_loss(runs: list[dict], out_dir: Path, x_key: str, filename: str, xlabel: str) -> None:
    import matplotlib.pyplot as plt

    by_group = defaultdict(list)
    for run in runs:
        by_group[run["group"]].append(run)

    for group, group_runs in by_group.items():
        plt.figure(figsize=(10, 6))
        for run in group_runs:
            xs, ys = [], []
            for row in run["rows"]:
                if "val/loss" in row and x_key in row:
                    xs.append(row[x_key])
                    ys.append(row["val/loss"])
            if xs:
                plt.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=run["name"])
        plt.xlabel(xlabel)
        plt.ylabel("val/loss")
        plt.title(f"{group}: validation loss")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(out_dir / f"{group}_{filename}", dpi=180)
        plt.close()


def plot_pareto(runs: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    xs, ys, labels = [], [], []
    for run in runs:
        final_val = None
        tok_s = None
        for row in run["rows"]:
            if "val/loss" in row:
                final_val = row["val/loss"]
            if "train/throughput_tokens_per_sec" in row:
                tok_s = row["train/throughput_tokens_per_sec"]
        if final_val is not None and tok_s is not None:
            xs.append(tok_s)
            ys.append(final_val)
            labels.append(run["name"])
    if not xs:
        return
    plt.figure(figsize=(10, 6))
    plt.scatter(xs, ys, s=28)
    for x, y, label in zip(xs, ys, labels):
        plt.annotate(label, (x, y), fontsize=6, alpha=0.8)
    plt.xlabel("train/throughput_tokens_per_sec")
    plt.ylabel("final val/loss")
    plt.title("Wall-clock Pareto proxy")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "pareto_throughput_tokens_per_sec_vs_val_loss.png", dpi=180)
    plt.close()


def plot_phase2_heatmaps(runs: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    p2_runs = []
    for run in runs:
        tf = p2_tf(run)
        if tf is None:
            continue
        lr = float(run["config"].get("lr_mul", 1.0))
        if abs(lr - 1.0) > 1e-12:
            continue
        p2_runs.append((tf[0], tf[1], run))
    if not p2_runs:
        return

    metrics = [
        ("final_val_loss", "Phase2 final val/loss", lambda r: final_value(r, "val/loss")),
        ("val_auc_tokens", "Phase2 val-loss AUC over tokens", val_auc_tokens),
        ("throughput_tokens_per_sec", "Phase2 train throughput (tok/s)", lambda r: final_value(r, "train/throughput_tokens_per_sec")),
    ]
    Ts = sorted({T for T, _, _ in p2_runs})
    max_f = max(f for _, f, _ in p2_runs)
    for filename, title, metric_fn in metrics:
        grid = np.full((len(Ts), max_f + 1), np.nan, dtype=float)
        for T, f, run in p2_runs:
            value = metric_fn(run)
            if value is not None:
                grid[Ts.index(T), f] = float(value)
        if np.isnan(grid).all():
            continue
        plt.figure(figsize=(10, 5.5))
        image = plt.imshow(grid, aspect="auto", interpolation="nearest", cmap="viridis")
        plt.colorbar(image, label=title)
        plt.xticks(range(max_f + 1), range(max_f + 1))
        plt.yticks(range(len(Ts)), Ts)
        plt.xlabel("fast steps f")
        plt.ylabel("total steps T")
        plt.title(title)
        for y, T in enumerate(Ts):
            for x in range(max_f + 1):
                if x > T or math.isnan(grid[y, x]):
                    continue
                plt.text(x, y, f"{grid[y, x]:.3g}", ha="center", va="center", fontsize=7, color="white")
        plt.tight_layout()
        plt.savefig(out_dir / f"p2_all_T_heatmap_{filename}.png", dpi=180)
        plt.close()


def plot_phase2_orth_pareto(runs: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    xs, ys, labels = [], [], []
    for run in runs:
        if p2_tf(run) is None:
            continue
        if abs(float(run["config"].get("lr_mul", 1.0)) - 1.0) > 1e-12:
            continue
        orth_ms = final_value(run, "train/throughput_tokens_per_sec")
        final_val = final_value(run, "val/loss")
        if orth_ms is None or final_val is None:
            continue
        xs.append(float(orth_ms))
        ys.append(float(final_val))
        labels.append(run["schedule"])
    if not xs:
        return
    plt.figure(figsize=(10, 6))
    plt.scatter(xs, ys, s=34)
    for x, y, label in zip(xs, ys, labels):
        plt.annotate(label, (x, y), fontsize=6, alpha=0.8)
    plt.xlabel("train/throughput_tokens_per_sec")
    plt.ylabel("final val/loss")
    plt.title("Phase2 Pareto: throughput vs final val/loss")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "p2_pareto_throughput_vs_val_loss.png", dpi=180)
    plt.close()


def plot_pe_lower_bound(runs: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    series = defaultdict(list)
    for run in runs:
        parsed = pe_t_l(run)
        if parsed is None:
            continue
        T, lb, lb_text = parsed
        final_val = final_value(run, "val/loss")
        auc = val_auc_tokens(run)
        if final_val is None:
            continue
        lr = float(run["config"].get("lr_mul", 1.0))
        series[(T, lr)].append((lb, lb_text, float(final_val), auc))
    if not series:
        return
    for use_auc in (False, True):
        plt.figure(figsize=(10, 6))
        any_points = False
        for (T, lr), points in sorted(series.items()):
            points = sorted(points, key=lambda item: item[0])
            xs = [item[0] for item in points]
            ys = [item[3] if use_auc else item[2] for item in points]
            if any(y is None for y in ys):
                continue
            any_points = True
            plt.plot(xs, ys, marker="o", linewidth=1.5, label=f"T={T}, lr={lr:g}")
            for point in points:
                x = point[0]
                y = point[3] if use_auc else point[2]
                if y is not None:
                    plt.annotate(f"{x:g}", (x, y), fontsize=6, alpha=0.75)
        if not any_points:
            plt.close()
            continue
        plt.xscale("log")
        plt.xlabel("PE lower bound")
        plt.ylabel("val-loss AUC over tokens" if use_auc else "final val/loss")
        plt.title("Polar Express lower-bound sensitivity")
        plt.grid(True, alpha=0.25, which="both")
        plt.legend(fontsize=8)
        plt.tight_layout()
        suffix = "auc_tokens" if use_auc else "final_val"
        plt.savefig(out_dir / f"pe_lower_bound_{suffix}.png", dpi=180)
        plt.close()


def comparison_runs(runs: list[dict]) -> list[dict]:
    selected: list[dict] = []

    def add(run: dict | None) -> None:
        if run and all(existing["name"] != run["name"] for existing in selected):
            selected.append(run)

    add(next((r for r in runs if r["name"] == "old_fast5_lr1.0_seed0"), None))

    p2_t5 = [
        r for r in runs
        if p2_tf(r) and p2_tf(r)[0] == 5 and abs(float(r["config"].get("lr_mul", 1.0)) - 1.0) < 1e-12
    ]
    add(min(p2_t5, key=lambda r: final_value(r, "val/loss") or 999) if p2_t5 else None)

    p2_long = [
        r for r in runs
        if p2_tf(r) and p2_tf(r)[0] >= 6 and abs(float(r["config"].get("lr_mul", 1.0)) - 1.0) < 1e-12
    ]
    add(min(p2_long, key=lambda r: final_value(r, "val/loss") or 999) if p2_long else None)

    pe_runs = [
        r for r in runs
        if pe_t_l(r) and final_value(r, "val/loss") is not None
    ]
    add(min(pe_runs, key=lambda r: final_value(r, "val/loss") or 999) if pe_runs else None)
    return selected


def plot_comparison_curves(runs: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    selected = comparison_runs(runs)
    if len(selected) < 2:
        return
    for x_key, filename, xlabel in [
        ("val/global_train_tokens", "comparison_best_val_loss_vs_tokens.png", "train tokens"),
        ("val/global_wall_time_s", "comparison_best_val_loss_vs_wall_time.png", "wall time (s)"),
    ]:
        plt.figure(figsize=(10, 6))
        any_series = False
        for run in selected:
            xs, ys = [], []
            for row in run["rows"]:
                if "val/loss" in row and x_key in row:
                    xs.append(row[x_key])
                    ys.append(row["val/loss"])
            if xs:
                any_series = True
                plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, label=run["name"])
        if not any_series:
            plt.close()
            continue
        plt.xlabel(xlabel)
        plt.ylabel("val/loss")
        plt.title("old_fast5 vs best P2 vs best PE")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=180)
        plt.close()


def plot_comparison_spectral(runs: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    selected = comparison_runs(runs)
    if len(selected) < 2:
        return
    spectral = [
        ("spec/update_sv_std", "comparison_spec_update_sv_std.png", "update singular-value std"),
        ("spec/update_orth_error", "comparison_spec_update_orth_error.png", "update orthogonality error"),
        ("spec/update_stable_rank", "comparison_spec_update_stable_rank.png", "update stable rank"),
        ("spec/update_svd_entropy", "comparison_spec_update_svd_entropy.png", "update SVD entropy"),
        ("spec/momentum_sv_std", "comparison_spec_momentum_sv_std.png", "momentum singular-value std"),
        ("spec/momentum_orth_error", "comparison_spec_momentum_orth_error.png", "momentum orthogonality error"),
    ]
    for key, filename, ylabel in spectral:
        plt.figure(figsize=(10, 6))
        any_series = False
        for run in selected:
            xs, ys = [], []
            for row in run["rows"]:
                if key in row and "train/tokens" in row:
                    xs.append(row["train/tokens"])
                    ys.append(row[key])
            if xs:
                any_series = True
                plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, label=run["name"])
        if not any_series:
            plt.close()
            continue
        plt.xlabel("train tokens")
        plt.ylabel(ylabel)
        plt.title(f"Spectral comparison: {ylabel}")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=180)
        plt.close()


def plot_top_bar(runs: list[dict], out_dir: Path, top_n: int = 20) -> None:
    import matplotlib.pyplot as plt

    ranked = [
        (float(final_value(run, "val/loss")), run["name"])
        for run in runs
        if final_value(run, "val/loss") is not None
    ]
    ranked = sorted(ranked)[:top_n]
    if not ranked:
        return
    values = [item[0] for item in ranked]
    labels = [item[1] for item in ranked]
    plt.figure(figsize=(11, 7))
    y = range(len(ranked))
    plt.barh(y, values)
    plt.yticks(y, labels, fontsize=7)
    plt.gca().invert_yaxis()
    plt.xlabel("final val/loss")
    plt.title(f"Top {len(ranked)} runs by final validation loss")
    plt.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "top_runs_final_val_loss.png", dpi=180)
    plt.close()


def plot_metric_over_tokens(runs: list[dict], out_dir: Path, y_key: str, filename: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    by_group = defaultdict(list)
    for run in runs:
        by_group[run["group"]].append(run)

    for group, group_runs in by_group.items():
        plt.figure(figsize=(10, 6))
        any_series = False
        for run in group_runs:
            xs, ys = [], []
            for row in run["rows"]:
                if y_key in row and "train/tokens" in row:
                    xs.append(row["train/tokens"])
                    ys.append(row[y_key])
            if xs:
                any_series = True
                plt.plot(xs, ys, marker="o", linewidth=1.3, markersize=3, label=run["name"])
        if not any_series:
            plt.close()
            continue
        plt.xlabel("train tokens")
        plt.ylabel(ylabel)
        plt.title(f"{group}: {ylabel}")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(out_dir / f"{group}_{filename}", dpi=180)
        plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-dir", default="results/figures")
    parser.add_argument("--groups", nargs="*", default=None)
    args = parser.parse_args()

    runs_dir = (ROOT / args.runs_dir).resolve()
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = set(args.groups) if args.groups else None

    runs = load_runs(runs_dir, groups)
    if not runs:
        print("No run metrics found.")
        return 0

    plot_val_loss(runs, out_dir, "val/global_train_tokens", "val_loss_vs_tokens.png", "train tokens")
    plot_val_loss(runs, out_dir, "val/global_wall_time_s", "val_loss_vs_wall_time.png", "wall time (s)")
    plot_pareto(runs, out_dir)
    plot_phase2_heatmaps(runs, out_dir)
    plot_phase2_orth_pareto(runs, out_dir)
    plot_pe_lower_bound(runs, out_dir)
    plot_comparison_curves(runs, out_dir)
    plot_comparison_spectral(runs, out_dir)
    plot_top_bar(runs, out_dir)
    metric_plots = [
        ("train/throughput_tokens_per_sec", "train_throughput_tokens_per_sec.png", "train/throughput_tokens_per_sec"),
        ("train/grad_norm", "grad_norm.png", "train/grad_norm"),
        ("spec/update_orth_error", "spec_update_orth_error.png", "spec/update_orth_error"),
        ("spec/update_sv_std", "spec_update_sv_std.png", "spec/update_sv_std"),
        ("spec/update_stable_rank", "spec_update_stable_rank.png", "spec/update_stable_rank"),
        ("spec/momentum_sv_std", "spec_momentum_sv_std.png", "spec/momentum_sv_std"),
        ("spec/momentum_orth_error", "spec_momentum_orth_error.png", "spec/momentum_orth_error"),
    ]
    for y_key, filename, ylabel in metric_plots:
        plot_metric_over_tokens(runs, out_dir, y_key, filename, ylabel)
    print(f"Wrote figures under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from src.paths import ROOT, read_jsonl


def detect_mode(rows: list[dict]) -> str:
    has_benchmark = any("benchmark/wall_clock_s" in row for row in rows)
    has_spectral = any("spec/sample_count" in row for row in rows)
    if has_benchmark and has_spectral:
        raise ValueError("Mixed benchmark and spectral metrics in one run.")
    if has_benchmark:
        return "benchmark"
    if has_spectral:
        return "spectral"
    return "train"


def auc(points: list[tuple[float, float]]) -> float | None:
    points = sorted(points)
    if len(points) < 2:
        return None
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2
    span = points[-1][0] - points[0][0]
    return area / span if span > 0 else None


def last_metric(rows: list[dict], key: str):
    for row in reversed(rows):
        if key in row:
            return row[key]
    return None


def load_runs(runs_dir: Path) -> list[dict]:
    runs = []
    for metrics_path in sorted(runs_dir.rglob("metrics.jsonl")):
        run_dir = metrics_path.parent
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        rows = read_jsonl(metrics_path)
        if not rows:
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        vals = [row for row in rows if "val/loss" in row]
        token_points = [
            (float(row["val/global_train_tokens"]), float(row["val/loss"]))
            for row in vals
            if "val/global_train_tokens" in row
        ]
        wall_points = [
            (float(row["val/global_wall_time_s"]), float(row["val/loss"]))
            for row in vals
            if "val/global_wall_time_s" in row
        ]
        mode = detect_mode(rows)
        runs.append({
            "run_dir": str(run_dir.relative_to(ROOT)),
            "run_name": config.get("run_name"),
            "mode": mode,
            "orthogonalizer_type": config.get("orthogonalizer_type"),
            "orth_schedule_name": config.get("orth_schedule_name"),
            "seed": config.get("seed"),
            "lr_mul": config.get("lr_mul"),
            "T_ns": config.get("T_ns"),
            "fast_steps": config.get("fast_steps"),
            "stable_steps": config.get("stable_steps"),
            "pe_T": config.get("pe_T"),
            "pe_lower_bound": config.get("pe_lower_bound"),
            "train_token_budget": config.get("train_token_budget"),
            "final_val_loss": last_metric(rows, "val/loss"),
            "best_val_loss": min((float(row["val/loss"]) for row in vals), default=None),
            "val_auc_tokens": auc(token_points),
            "val_auc_wall": auc(wall_points),
            "final_wall_time_s": last_metric(rows, "val/global_wall_time_s"),
            "benchmark_wall_clock_s": last_metric(rows, "benchmark/wall_clock_s"),
            "g_post_semi_orth_error": last_metric(rows, "spec/g_post_semi_orth_error"),
            "g_pre_semi_orth_error": last_metric(rows, "spec/g_pre_semi_orth_error"),
            "buffer_post_semi_orth_error": last_metric(rows, "spec/buffer_post_semi_orth_error"),
            "peak_allocated_mb": last_metric(rows, "memory/peak_allocated_mb"),
            "status": last_metric(rows, "status") or "running_or_incomplete",
            "rows": rows,
        })
    return runs


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row if key != "rows"})
    prefix = [
        "run_name", "mode", "orthogonalizer_type", "orth_schedule_name",
        "seed", "lr_mul", "T_ns", "fast_steps", "stable_steps",
        "pe_T", "pe_lower_bound", "train_token_budget",
    ]
    ordered = [key for key in prefix if key in fieldnames] + [key for key in fieldnames if key not in prefix]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key != "rows"})


def write_val_points(path: Path, runs: list[dict]) -> None:
    fields = ["run_name", "mode", "orth_schedule_name", "seed", "lr_mul", "tokens", "wall_time_s", "val_loss"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            if run["mode"] != "train":
                continue
            for row in run["rows"]:
                if "val/loss" not in row:
                    continue
                writer.writerow({
                    "run_name": run["run_name"],
                    "mode": run["mode"],
                    "orth_schedule_name": run["orth_schedule_name"],
                    "seed": run["seed"],
                    "lr_mul": run["lr_mul"],
                    "tokens": row.get("val/global_train_tokens"),
                    "wall_time_s": row.get("val/global_wall_time_s"),
                    "val_loss": row.get("val/loss"),
                })


def load_spectral_detail_rows(runs_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(runs_dir.rglob("spectral_details.jsonl")):
        run_dir = path.parent
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.jsonl"
        if not config_path.exists() or not metrics_path.exists():
            continue
        if detect_mode(read_jsonl(metrics_path)) != "spectral":
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for row in read_jsonl(path):
            rows.append({
                "run_name": config.get("run_name"),
                "orth_schedule_name": config.get("orth_schedule_name"),
                "orthogonalizer_type": config.get("orthogonalizer_type"),
                "seed": config.get("seed"),
                "lr_mul": config.get("lr_mul"),
                **row,
            })
    return rows


def write_spectral_details_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    prefix = ["run_name", "orth_schedule_name", "orthogonalizer_type", "seed", "lr_mul", "train/tokens", "spec/label"]
    ordered = [key for key in prefix if key in fieldnames] + [key for key in fieldnames if key not in prefix]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def label_for(run: dict) -> str:
    schedule = run.get("orth_schedule_name") or run.get("orthogonalizer_type") or "unknown"
    seed = run.get("seed")
    lr_mul = run.get("lr_mul")
    suffix = []
    if seed is not None:
        suffix.append(f"s{seed}")
    if lr_mul is not None and abs(float(lr_mul) - 1.0) > 1e-12:
        suffix.append(f"lr{lr_mul}")
    return schedule if not suffix else f"{schedule} ({', '.join(suffix)})"


def plot_val_curves(runs: list[dict], out_dir: Path, *, x_key: str, out_name: str, xlabel: str) -> None:
    import matplotlib.pyplot as plt

    train_runs = [run for run in runs if run["mode"] == "train"]
    if not train_runs:
        return
    plt.figure(figsize=(12, 7))
    any_series = False
    for run in sorted(train_runs, key=lambda row: label_for(row)):
        points = []
        for row in run["rows"]:
            if "val/loss" in row and x_key in row:
                points.append((float(row[x_key]), float(row["val/loss"])))
        if not points:
            continue
        xs, ys = zip(*sorted(points))
        plt.plot(xs, ys, linewidth=1.8, label=label_for(run))
        any_series = True
    if not any_series:
        plt.close()
        return
    plt.xlabel(xlabel)
    plt.ylabel("validation loss")
    plt.title(f"Validation loss by {xlabel}")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / out_name, dpi=180)
    plt.close()


def plot_final_bar(runs: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    train_runs = [run for run in runs if run["mode"] == "train" and run["final_val_loss"] is not None]
    if not train_runs:
        return
    train_runs.sort(key=lambda row: float(row["final_val_loss"]))
    labels = [label_for(run) for run in train_runs]
    values = [float(run["final_val_loss"]) for run in train_runs]
    plt.figure(figsize=(max(10, len(labels) * 0.45), 6))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=8)
    plt.ylabel("final validation loss")
    plt.title("Final validation loss by schedule")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "final_val_loss_by_schedule.png", dpi=180)
    plt.close()


def plot_manual_depth(runs: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    train_runs = [
        run for run in runs
        if run["mode"] == "train"
        and run["orthogonalizer_type"] == "manual"
        and run["final_val_loss"] is not None
    ]
    if not train_runs:
        return
    train_runs.sort(key=lambda row: (int(row["T_ns"]), int(row["fast_steps"])))
    labels = [run["orth_schedule_name"] for run in train_runs]
    values = [float(run["final_val_loss"]) for run in train_runs]
    plt.figure(figsize=(max(9, len(labels) * 0.5), 5.5))
    plt.plot(range(len(labels)), values, marker="o")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("final validation loss")
    plt.title("Manual fast-to-stable schedules")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "manual_depth_final_loss.png", dpi=180)
    plt.close()


def plot_pe_lower_bound(runs: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    pe_runs = [
        run for run in runs
        if run["mode"] == "train"
        and run["orthogonalizer_type"] == "polar_express"
        and run["pe_T"] == 5
        and run["final_val_loss"] is not None
    ]
    if not pe_runs:
        return
    pe_runs.sort(key=lambda row: float(row["pe_lower_bound"]))
    xs = [float(run["pe_lower_bound"]) for run in pe_runs]
    ys = [float(run["final_val_loss"]) for run in pe_runs]
    labels = [run["orth_schedule_name"] for run in pe_runs]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o")
    for x, y, label in zip(xs, ys, labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(4, 5), fontsize=8)
    plt.xscale("log")
    plt.xlabel("PE lower bound")
    plt.ylabel("final validation loss")
    plt.title("Polar Express lower-bound sensitivity")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "pe_lower_bound_final_loss.png", dpi=180)
    plt.close()


def plot_spectral_objects(detail_rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not detail_rows:
        return
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    objects = ["buffer_post", "g_pre", "g_post"]
    for row in detail_rows:
        label = row.get("orth_schedule_name") or row.get("run_name")
        for obj in objects:
            key = f"spec/{obj}_semi_orth_error"
            if key in row:
                grouped[label][obj].append(float(row[key]))
    labels = sorted(grouped)
    if not labels:
        return
    x = np.arange(len(labels))
    width = 0.24
    plt.figure(figsize=(max(10, len(labels) * 0.65), 6))
    for idx, obj in enumerate(objects):
        values = [
            sum(grouped[label][obj]) / len(grouped[label][obj])
            if grouped[label][obj] else math.nan
            for label in labels
        ]
        plt.bar(x + (idx - 1) * width, values, width=width, label=obj)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("mean semi-orthogonality error")
    plt.title("Spectral objects before and after orthogonalization")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "spectral_object_error_by_schedule.png", dpi=180)
    plt.close()


def plot_attention_mlp(detail_rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in detail_rows:
        label = row.get("orth_schedule_name") or row.get("run_name")
        layer_label = str(row.get("spec/label", ""))
        if ".attn." in layer_label:
            group = "attention"
        elif ".mlp." in layer_label:
            group = "mlp"
        else:
            continue
        grouped[label][group].append(float(row["spec/g_post_semi_orth_error"]))
    labels = sorted(label for label, values in grouped.items() if values["attention"] and values["mlp"])
    if not labels:
        return
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(max(10, len(labels) * 0.65), 6))
    for idx, group in enumerate(["attention", "mlp"]):
        values = [sum(grouped[label][group]) / len(grouped[label][group]) for label in labels]
        plt.bar(x + (idx - 0.5) * width, values, width=width, label=group)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("mean g_post semi-orthogonality error")
    plt.title("Attention vs MLP post-orthogonality")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "attention_vs_mlp_gpost_error.png", dpi=180)
    plt.close()


def main() -> int:
    runs_dir = ROOT / "runs"
    out_dir = ROOT / "results" / "followup"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(runs_dir)
    write_csv(out_dir / "summary.csv", runs)
    write_val_points(out_dir / "val_points.csv", runs)
    detail_rows = load_spectral_detail_rows(runs_dir)
    write_spectral_details_csv(out_dir / "spectral_details.csv", detail_rows)

    plot_val_curves(
        runs, fig_dir,
        x_key="val/global_train_tokens",
        out_name="val_loss_vs_tokens.png",
        xlabel="train tokens",
    )
    plot_val_curves(
        runs, fig_dir,
        x_key="val/global_wall_time_s",
        out_name="val_loss_vs_wall_time.png",
        xlabel="wall time (s)",
    )
    plot_final_bar(runs, fig_dir)
    plot_manual_depth(runs, fig_dir)
    plot_pe_lower_bound(runs, fig_dir)
    plot_spectral_objects(detail_rows, fig_dir)
    plot_attention_mlp(detail_rows, fig_dir)

    print(f"Wrote follow-up summaries and figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

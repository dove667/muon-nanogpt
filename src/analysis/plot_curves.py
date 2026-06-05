#!/usr/bin/env python
import json
from pathlib import Path

from src.paths import ROOT, read_jsonl

ORTHOGONALIZER_ORDER = ["adamw", "vanilla", "manual", "fast", "polar_express"]
ORTHOGONALIZER_LABEL = {
    "adamw": "AdamW",
    "vanilla": "Vanilla",
    "manual": "Manual",
    "fast": "Fast",
    "polar_express": "Polar Express",
}
ORTHOGONALIZER_COLOR = {
    "adamw": "#4c78a8",
    "vanilla": "#72b7b2",
    "manual": "#f58518",
    "fast": "#e45756",
    "polar_express": "#54a24b",
}


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


def _load_runs(runs_dir: Path) -> list[dict]:
    runs = []
    for metrics_path in sorted(runs_dir.rglob("metrics.jsonl")):
        run_dir = metrics_path.parent
        config_path = run_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config file: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rows = read_jsonl(metrics_path)
        mode = _detect_mode(rows)
        runs.append({
            "name": config["run_name"],
            "orthogonalizer_type": config["orthogonalizer_type"],
            "mode": mode,
            "rows": rows,
        })
    return runs


def _val_points(run: dict, x_key: str) -> list[tuple[float, float]]:
    return [
        (float(row[x_key]), float(row["val/loss"]))
        for row in run["rows"]
        if x_key in row and "val/loss" in row
    ]


def _last_metric(run: dict, key: str) -> float | None:
    for row in reversed(run["rows"]):
        if key in row:
            return float(row[key])
    return None


def _index_runs(runs: list[dict]) -> dict[tuple[str, str], dict]:
    indexed: dict[tuple[str, str], dict] = {}
    for run in runs:
        key = (run["orthogonalizer_type"], run["mode"])
        if key in indexed:
            raise ValueError(
                "Duplicate "
                f"{run['mode']} run for orthogonalizer_type={run['orthogonalizer_type']}: "
                f"{indexed[key]['name']} and {run['name']}"
            )
        indexed[key] = run
    return indexed


def plot_val_loss_vs_tokens(runs: list[dict], out_dir: Path) -> None:
    """One train-mode val/loss curve per orth vs train tokens."""
    import matplotlib.pyplot as plt

    indexed = _index_runs(runs)

    plt.figure(figsize=(10, 6))
    any_series = False
    for orthogonalizer_type in ORTHOGONALIZER_ORDER:
        run = indexed.get((orthogonalizer_type, "train"))
        if run is None:
            continue
        points = sorted(_val_points(run, "val/global_train_tokens"))
        if not points:
            continue
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        any_series = True
        plt.plot(
            xs,
            ys,
            color=ORTHOGONALIZER_COLOR[orthogonalizer_type],
            linewidth=2.2,
            label=ORTHOGONALIZER_LABEL[orthogonalizer_type],
        )
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


def plot_benchmark_wall_clock(runs: list[dict], out_dir: Path) -> None:
    """Bar chart of end-to-end benchmark wall-clock time per orth."""
    import matplotlib.pyplot as plt

    indexed = _index_runs(runs)

    labels, values, colors = [], [], []
    for orthogonalizer_type in ORTHOGONALIZER_ORDER:
        run = indexed.get((orthogonalizer_type, "benchmark"))
        if run is None:
            continue
        wall_time = _last_metric(run, "benchmark/wall_clock_s")
        if wall_time is None:
            continue
        labels.append(ORTHOGONALIZER_LABEL[orthogonalizer_type])
        values.append(wall_time)
        colors.append(ORTHOGONALIZER_COLOR[orthogonalizer_type])
    if not labels:
        return
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(labels)), values, color=colors)
    plt.xticks(range(len(labels)), labels)
    plt.ylabel("wall time (s)")
    plt.title("End-to-end benchmark wall clock")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_wall_clock.png", dpi=180)
    plt.close()


def plot_final_val_loss_bars(runs: list[dict], out_dir: Path) -> None:
    """Bar chart of final train-mode val/loss per orth."""
    import matplotlib.pyplot as plt

    indexed = _index_runs(runs)

    labels, values, colors = [], [], []
    for orthogonalizer_type in ORTHOGONALIZER_ORDER:
        run = indexed.get((orthogonalizer_type, "train"))
        if run is None:
            continue
        value = _last_metric(run, "val/loss")
        if value is None:
            continue
        labels.append(ORTHOGONALIZER_LABEL[orthogonalizer_type])
        values.append(value)
        colors.append(ORTHOGONALIZER_COLOR[orthogonalizer_type])
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
    runs_dir = (ROOT / "runs").resolve()
    out_dir = (ROOT / "results" / "figures").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not runs_dir.exists():
        print(f"No runs directory found: {runs_dir}")
        return 0

    runs = _load_runs(runs_dir)
    if not runs:
        print(f"No run metrics found under {runs_dir}")
        return 0

    plot_val_loss_vs_tokens(runs, out_dir)
    plot_benchmark_wall_clock(runs, out_dir)
    plot_final_val_loss_bars(runs, out_dir)
    print(f"Wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

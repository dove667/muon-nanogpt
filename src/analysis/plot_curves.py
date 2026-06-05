#!/usr/bin/env python
import json
from collections import defaultdict
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

SPECTRAL_ORTH_ORDER = ["vanilla", "manual", "fast", "polar_express"]

OBJECT_KEYS = [
    ("buffer_post", "Momentum Buffer"),
    ("g_pre", "Pre-Orth Update"),
    ("g_post", "Post-Orth Update"),
]


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


def _load_spectral_details(runs_dir: Path) -> list[dict]:
    all_rows: list[dict] = []
    for detail_path in sorted(runs_dir.rglob("spectral_details.jsonl")):
        run_dir = detail_path.parent
        metric_rows = read_jsonl(run_dir / "metrics.jsonl")
        mode = _detect_mode(metric_rows)
        if mode != "spectral":
            continue
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        for row in read_jsonl(detail_path):
            all_rows.append({
                "orthogonalizer_type": config["orthogonalizer_type"],
                **row,
            })
    return all_rows


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


def plot_g_post_error_vs_tokens(rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        orth = row["orthogonalizer_type"]
        if orth not in SPECTRAL_ORTH_ORDER:
            continue
        grouped[orth][int(row["train/tokens"])].append(float(row["spec/g_post_semi_orth_error"]))

    plt.figure(figsize=(10, 6))
    any_series = False
    for orth in SPECTRAL_ORTH_ORDER:
        token_map = grouped.get(orth)
        if not token_map:
            continue
        xs = sorted(token_map)
        ys = [sum(token_map[x]) / len(token_map[x]) for x in xs]
        plt.plot(
            xs,
            ys,
            color=ORTHOGONALIZER_COLOR[orth],
            linewidth=2.2,
            label=ORTHOGONALIZER_LABEL[orth],
        )
        any_series = True
    if not any_series:
        plt.close()
        return
    plt.xlabel("train tokens")
    plt.ylabel("mean g_post semi-orth error")
    plt.title("Post-orth semi-orthogonality over training")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "g_post_semi_orth_error_vs_tokens.png", dpi=180)
    plt.close()


def plot_object_error_bars(rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    means: dict[str, dict[str, float]] = defaultdict(dict)
    for orth in SPECTRAL_ORTH_ORDER:
        orth_rows = [row for row in rows if row["orthogonalizer_type"] == orth]
        if not orth_rows:
            continue
        for object_key, _ in OBJECT_KEYS:
            metric = f"spec/{object_key}_semi_orth_error"
            values = [float(row[metric]) for row in orth_rows]
            means[orth][object_key] = sum(values) / len(values)

    if not means:
        return

    x = np.arange(len(SPECTRAL_ORTH_ORDER))
    width = 0.24
    plt.figure(figsize=(10, 6))
    for idx, (object_key, label) in enumerate(OBJECT_KEYS):
        values = [means[orth][object_key] for orth in SPECTRAL_ORTH_ORDER]
        plt.bar(x + (idx - 1) * width, values, width=width, label=label)
    plt.xticks(x, [ORTHOGONALIZER_LABEL[orth] for orth in SPECTRAL_ORTH_ORDER])
    plt.ylabel("mean semi-orth error")
    plt.title("Semi-orthogonality before and after orthogonalization")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "object_semi_orth_error.png", dpi=180)
    plt.close()


def plot_attn_mlp_breakdown(rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    values: dict[str, dict[str, float]] = defaultdict(dict)
    for orth in SPECTRAL_ORTH_ORDER:
        orth_rows = [row for row in rows if row["orthogonalizer_type"] == orth]
        if not orth_rows:
            continue
        attn_rows = [row for row in orth_rows if ".attn." in row["spec/label"]]
        mlp_rows = [row for row in orth_rows if ".mlp." in row["spec/label"]]
        if not attn_rows or not mlp_rows:
            continue
        values[orth]["attn"] = sum(float(row["spec/g_post_semi_orth_error"]) for row in attn_rows) / len(attn_rows)
        values[orth]["mlp"] = sum(float(row["spec/g_post_semi_orth_error"]) for row in mlp_rows) / len(mlp_rows)

    if not values:
        return

    x = np.arange(len(SPECTRAL_ORTH_ORDER))
    width = 0.35
    attn_vals = [values[orth]["attn"] for orth in SPECTRAL_ORTH_ORDER]
    mlp_vals = [values[orth]["mlp"] for orth in SPECTRAL_ORTH_ORDER]

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, attn_vals, width=width, label="Attention")
    plt.bar(x + width / 2, mlp_vals, width=width, label="MLP")
    plt.xticks(x, [ORTHOGONALIZER_LABEL[orth] for orth in SPECTRAL_ORTH_ORDER])
    plt.ylabel("mean g_post semi-orth error")
    plt.title("Post-orth semi-orthogonality by module type")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "g_post_semi_orth_error_attn_vs_mlp.png", dpi=180)
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

    spectral_rows = _load_spectral_details(runs_dir)
    if spectral_rows:
        plot_g_post_error_vs_tokens(spectral_rows, out_dir)
        plot_object_error_bars(spectral_rows, out_dir)
        plot_attn_mlp_breakdown(spectral_rows, out_dir)

    print(f"Wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

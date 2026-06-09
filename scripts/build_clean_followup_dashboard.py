#!/usr/bin/env python3
"""Build a standalone Chinese dashboard for the clean Muon experiments."""

from __future__ import annotations

import csv
import html
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TMP_REPO = ROOT
FOLLOW = ROOT / "results" / "followup_4090_20260608"
POLY = FOLLOW / "polynomial_maps"
READABLE = FOLLOW / "readable_figures"
OUT = ROOT / "docs" / "clean_followup_dashboard.html"

sys.path.insert(0, str(TMP_REPO))
from src.optim.orth import FAST_COEFF, STABLE_COEFF, polar_express_coefficients  # noqa: E402


COLORS = {
    "adamw": "#4C78A8",
    "stable": "#7F7F7F",
    "fast": "#1B9E77",
    "manual": "#D95F02",
    "pe": "#7570B3",
    "reference": "#333333",
}

MARKERS = {
    "adamw": "o",
    "stable": "s",
    "fast": "D",
    "manual": "^",
    "pe": "P",
}


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def rel(path: Path) -> str:
    return Path(os.path.relpath(path, OUT.parent)).as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: object, default: float = math.nan) -> float:
    try:
        text = str(value).strip()
        return float(text) if text else default
    except (TypeError, ValueError):
        return default


def fmt(value: object, digits: int = 4) -> str:
    value = to_float(value)
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def family(schedule: str) -> str:
    if schedule == "adamw":
        return "adamw"
    if schedule.startswith("stable") or schedule == "vanilla":
        return "stable"
    if schedule.startswith("fast"):
        return "fast"
    if schedule.startswith("manual"):
        return "manual"
    if schedule.startswith("pe"):
        return "pe"
    return "reference"


def short_label(schedule: str, seed: str | int | None = None, lr: str | float | None = None) -> str:
    label = schedule
    if schedule == "vanilla":
        label = "stable5"
    if lr is not None and str(lr) not in {"", "1", "1.0", "nan"}:
        label += f", lr={lr}"
    if seed is not None and str(seed) not in {"", "0", "0.0", "nan"}:
        label += f", s{int(float(seed))}"
    return label


def line_style(seed: str | int | None, lr: str | float | None, schedule: str) -> tuple[str | tuple[int, tuple[int, ...]], str]:
    lr_text = "" if lr is None else str(lr)
    seed_text = "" if seed is None else str(seed)
    if lr_text in {"0.5", "0.50"}:
        return "--", "v"
    if lr_text in {"2.0", "2"}:
        return "-.", "^"
    if seed_text in {"1", "1.0"}:
        return (0, (2, 2)), "o"
    if seed_text in {"2", "2.0"}:
        return (0, (1, 1)), "s"
    if schedule.startswith("pe_T10"):
        return (0, (6, 2)), "X"
    if schedule.startswith("pe_T9"):
        return (0, (3, 1, 1, 1)), "P"
    if schedule.startswith("pe_") and "l3e-3" in schedule:
        return "--", "o"
    if schedule.startswith("pe_") and "l3e-4" in schedule:
        return "-.", "^"
    if schedule.startswith("pe_") and "l3e-5" in schedule:
        return ":", "s"
    if "T10" in schedule:
        return (0, (6, 2)), "X"
    if "T9" in schedule:
        return (0, (4, 2)), "P"
    if "T8" in schedule:
        return (0, (2, 1, 1, 1)), "D"
    if "T7" in schedule:
        return "--", "o"
    return "-", MARKERS.get(family(schedule), "o")


def plot_curve(
    ax,
    val_rows: list[dict[str, str]],
    schedule: str,
    *,
    label: str | None = None,
    seed: str = "0",
    lr: str = "1.0",
) -> None:
    rows = [
        r for r in val_rows
        if r["orth_schedule_name"] == schedule and str(r["seed"]) == seed and str(r["lr_mul"]) == lr
    ]
    if not rows:
        return
    rows.sort(key=lambda r: to_float(r["tokens"]))
    first = rows[0]
    fam = family(schedule)
    linestyle, marker = line_style(first.get("seed"), first.get("lr_mul"), schedule)
    ax.plot(
        [to_float(r["tokens"]) / 1e6 for r in rows],
        [to_float(r["val_loss"]) for r in rows],
        label=label or short_label(schedule, first.get("seed"), first.get("lr_mul")),
        color=COLORS[fam],
        linestyle=linestyle,
        marker=marker,
        markevery=max(1, len(rows) // 6),
        linewidth=2.0,
        markersize=4,
        alpha=0.95,
    )


def apply_schedule(xs: np.ndarray, coeffs: list[tuple[float, float, float]]) -> np.ndarray:
    ys = xs.copy()
    for a, b, c in coeffs:
        ys = a * ys + b * ys**3 + c * ys**5
    return ys


def single_step_derivative(xs: np.ndarray, coeff: tuple[float, float, float]) -> np.ndarray:
    a, b, c = coeff
    return a + 3 * b * xs**2 + 5 * c * xs**4


def composed_derivative(xs: np.ndarray, coeffs: list[tuple[float, float, float]]) -> np.ndarray:
    ys = xs.copy()
    deriv = np.ones_like(xs)
    for coeff in coeffs:
        deriv *= single_step_derivative(ys, coeff)
        a, b, c = coeff
        ys = a * ys + b * ys**3 + c * ys**5
    return deriv


def schedule_coeffs() -> dict[str, list[tuple[float, float, float]]]:
    return {
        "stable5": [STABLE_COEFF] * 5,
        "fast5": [FAST_COEFF] * 5,
        "manual_T5_f3_s2": [FAST_COEFF] * 3 + [STABLE_COEFF] * 2,
        "manual_T7_f4_s3": [FAST_COEFF] * 4 + [STABLE_COEFF] * 3,
        "manual_T8_f5_s3": [FAST_COEFF] * 5 + [STABLE_COEFF] * 3,
        "manual_T9_f4_s5": [FAST_COEFF] * 4 + [STABLE_COEFF] * 5,
        "manual_T10_f5_s5": [FAST_COEFF] * 5 + [STABLE_COEFF] * 5,
        "pe_T5_l1e-3": polar_express_coefficients(1e-3, 5, 0.02, 0.02),
        "pe_T5_l3e-3": polar_express_coefficients(3e-3, 5, 0.02, 0.02),
        "pe_T5_l3e-4": polar_express_coefficients(3e-4, 5, 0.02, 0.02),
        "pe_T5_l3e-5": polar_express_coefficients(3e-5, 5, 0.02, 0.02),
        "pe_T9_l3e-5": polar_express_coefficients(3e-5, 9, 0.02, 0.02),
        "pe_T10_l3e-5": polar_express_coefficients(3e-5, 10, 0.02, 0.02),
    }


def setup_axis(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def savefig(name: str) -> Path:
    path = READABLE / name
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def zoom_ylim(ax, values: list[float], min_pad: float = 0.006) -> None:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return
    lo, hi = min(vals), max(vals)
    pad = max(min_pad, (hi - lo) * 0.22)
    ax.set_ylim(lo - pad, hi + pad)


def label_bars(ax, bars, values: list[float], digits: int = 4) -> None:
    for bar, value in zip(bars, values):
        if math.isnan(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.{digits}f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_readable_figures() -> None:
    READABLE.mkdir(parents=True, exist_ok=True)
    train_summary = read_csv(TMP_REPO / "results" / "train" / "summary.csv")
    benchmark_summary = read_csv(TMP_REPO / "results" / "benchmark" / "summary.csv")
    spectral_summary = read_csv(TMP_REPO / "results" / "spectral" / "summary.csv")
    follow_summary = read_csv(FOLLOW / "summary.csv")
    val_rows = read_csv(FOLLOW / "val_points.csv")

    # Basic clean T=5 summary: final loss, wall-clock, and geometry.
    basic_order = [
        ("AdamW", "adamw", "adamw"),
        ("stable5", "vanilla", "stable"),
        ("manual_T5", "manual", "manual"),
        ("fast5", "fast", "fast"),
        ("PE_T5", "polar_express", "pe"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    train_map = {r["orthogonalizer_type"]: r for r in train_summary}
    bench_map = {r["orthogonalizer_type"]: r for r in benchmark_summary}
    spec_map = {r["orthogonalizer_type"]: r for r in spectral_summary}
    xs = np.arange(len(basic_order))
    colors = [COLORS[fam] for _, _, fam in basic_order]
    axes[0].bar(xs, [to_float(train_map[k]["train_final_val_loss"]) for _, k, _ in basic_order], color=colors)
    axes[0].set_xticks(xs, [name for name, _, _ in basic_order], rotation=25, ha="right")
    setup_axis(axes[0], "Basic T=5: final validation loss", ylabel="lower is better")
    axes[1].bar(xs, [to_float(bench_map[k]["benchmark_wall_clock_s"]) for _, k, _ in basic_order], color=colors)
    axes[1].set_xticks(xs, [name for name, _, _ in basic_order], rotation=25, ha="right")
    setup_axis(axes[1], "Basic T=5: wall-clock", ylabel="seconds")
    axes[2].bar(xs[1:], [to_float(spec_map[k]["spectral_g_post_semi_orth_error"]) for _, k, _ in basic_order[1:]], color=colors[1:])
    axes[2].set_xticks(xs[1:], [name for name, _, _ in basic_order[1:]], rotation=25, ha="right")
    setup_axis(axes[2], "Basic T=5: g_post error", ylabel="lower is closer to semi-orthogonal")
    savefig("basic_t5_summary.png")

    # Core T=5 validation curves from follow-up.
    fig, ax = plt.subplots(figsize=(10, 6))
    core_schedules = [
        ("adamw", "AdamW"),
        ("stable5", "stable5"),
        ("manual_T5_f3_s2", "manual_T5_f3_s2"),
        ("fast5", "fast5"),
        ("pe_T5_l1e-3", "PE_T5_l1e-3"),
        ("pe_T5_l3e-3", "PE_T5_l3e-3 dashed"),
    ]
    for schedule, label in core_schedules:
        plot_curve(ax, val_rows, schedule, label=label)
    setup_axis(ax, "Core T=5 comparison", xlabel="train tokens (M)", ylabel="validation loss")
    ax.legend(fontsize=9, ncol=2, handlelength=3.2)
    savefig("val_loss_core_readable.png")

    # Four-panel follow-up curves with color + line-style encoding.
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    panels = [
        (axes[0, 0], "Core T=5", ["adamw", "stable5", "manual_T5_f3_s2", "fast5", "pe_T5_l1e-3", "pe_T5_l3e-3"]),
        (
            axes[0, 1],
            "LR sanity",
            ["fast5", "manual_T9_f4_s5", "pe_T9_l3e-5"],
        ),
        (
            axes[1, 0],
            "Manual depth",
            ["manual_T5_f3_s2", "manual_T7_f4_s3", "manual_T8_f5_s3", "manual_T9_f3_s6", "manual_T9_f4_s5", "manual_T10_f5_s5", "fast5"],
        ),
        (
            axes[1, 1],
            "PE lower bound / depth",
            ["pe_T5_l3e-3", "pe_T5_l1e-3", "pe_T5_l3e-4", "pe_T5_l3e-5", "pe_T9_l3e-5", "pe_T10_l3e-5", "fast5"],
        ),
    ]
    for ax, title, schedules in panels:
        if title == "LR sanity":
            for schedule in schedules:
                for lr in ["0.5", "1.0", "2.0"]:
                    rows = [
                        r for r in val_rows
                        if r["orth_schedule_name"] == schedule and str(r["lr_mul"]) == lr and str(r["seed"]) == "0"
                    ]
                    if not rows:
                        continue
                    rows.sort(key=lambda r: to_float(r["tokens"]))
                    fam = family(schedule)
                    linestyle, marker = line_style(0, lr, schedule)
                    ax.plot(
                        [to_float(r["tokens"]) / 1e6 for r in rows],
                        [to_float(r["val_loss"]) for r in rows],
                        color=COLORS[fam],
                        linestyle=linestyle,
                        marker=marker,
                        markevery=max(1, len(rows) // 6),
                        linewidth=1.9,
                        markersize=3.8,
                        label=f"{schedule}, lr={lr}",
                    )
        else:
            for schedule in schedules:
                rows_for_schedule = [
                    r for r in val_rows
                    if r["orth_schedule_name"] == schedule and str(r["seed"]) == "0" and str(r["lr_mul"]) == "1.0"
                ]
                if not rows_for_schedule:
                    continue
                plot_curve(ax, rows_for_schedule, schedule, label=schedule)
        setup_axis(ax, title, xlabel="train tokens (M)", ylabel="validation loss")
        ax.legend(fontsize=7, ncol=1)
    savefig("val_loss_followup_facets.png")

    # Multi-seed final loss.
    seed_targets = ["fast5", "manual_T9_f4_s5", "pe_T9_l3e-5"]
    values_by_schedule: dict[str, list[float]] = defaultdict(list)
    for row in follow_summary:
        if row["mode"] != "train" or row["orth_schedule_name"] not in seed_targets:
            continue
        if str(row["lr_mul"]) != "1.0":
            continue
        if str(row["seed"]) not in {"0", "1", "2"}:
            continue
        values_by_schedule[row["orth_schedule_name"]].append(to_float(row["final_val_loss"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [float(np.mean(values_by_schedule[s])) for s in seed_targets]
    stds = [float(np.std(values_by_schedule[s], ddof=1)) for s in seed_targets]
    bars = ax.bar(np.arange(len(seed_targets)), means, yerr=stds, capsize=5, color=[COLORS[family(s)] for s in seed_targets])
    ax.set_xticks(np.arange(len(seed_targets)), seed_targets, rotation=20, ha="right")
    setup_axis(ax, "Multi-seed final validation loss (zoomed y-axis)", ylabel="mean +/- std")
    zoom_ylim(ax, [m + s for m, s in zip(means, stds)] + [m - s for m, s in zip(means, stds)], min_pad=0.002)
    label_bars(ax, bars, means, digits=4)
    savefig("multi_seed_final_loss.png")

    # Final-loss control-variable bars.
    def bar_from_summary(filename: str, title: str, schedules: list[str]) -> None:
        rows_by_schedule = {r["orth_schedule_name"]: r for r in follow_summary if r["mode"] == "train" and r["status"] == "completed"}
        fig, ax = plt.subplots(figsize=(9, 5))
        vals = [to_float(rows_by_schedule[s]["final_val_loss"]) for s in schedules]
        bars = ax.bar(np.arange(len(schedules)), vals, color=[COLORS[family(s)] for s in schedules])
        ax.set_xticks(np.arange(len(schedules)), schedules, rotation=25, ha="right")
        setup_axis(ax, f"{title} (zoomed y-axis)", ylabel="final validation loss")
        zoom_ylim(ax, vals, min_pad=0.008)
        label_bars(ax, bars, vals, digits=4)
        savefig(filename)

    bar_from_summary(
        "manual_depth_final_loss_readable.png",
        "Manual family depth trend",
        ["manual_T5_f3_s2", "manual_T7_f4_s3", "manual_T8_f5_s3", "manual_T9_f3_s6", "manual_T9_f4_s5", "manual_T10_f5_s5", "fast5"],
    )
    bar_from_summary(
        "pe_lower_depth_final_loss_readable.png",
        "PE lower bound and depth",
        ["pe_T5_l3e-3", "pe_T5_l1e-3", "pe_T5_l3e-4", "pe_T5_l3e-5", "pe_T9_l3e-5", "pe_T10_l3e-5", "fast5"],
    )

    # Wall-clock bars.
    bench_rows = {r["orth_schedule_name"]: r for r in follow_summary if r["mode"] == "benchmark" and r["status"] == "completed"}
    bench_schedules = ["adamw", "stable5", "fast5", "manual_T5_f3_s2", "manual_T9_f4_s5", "pe_T5_l3e-5", "pe_T9_l3e-5"]
    fig, ax = plt.subplots(figsize=(10, 5))
    wall_vals = [to_float(bench_rows[s]["benchmark_wall_clock_s"]) for s in bench_schedules]
    bars = ax.bar(np.arange(len(bench_schedules)), wall_vals, color=[COLORS[family(s)] for s in bench_schedules])
    ax.set_xticks(np.arange(len(bench_schedules)), bench_schedules, rotation=25, ha="right")
    setup_axis(ax, "Benchmark wall-clock", ylabel="seconds")
    label_bars(ax, bars, wall_vals, digits=0)
    savefig("benchmark_wall_clock_readable.png")

    # AUC efficiency summary. AUC is a more stable speed/quality proxy than a local slope.
    auc_schedules = ["adamw", "stable5", "manual_T5_f3_s2", "fast5", "pe_T5_l1e-3", "manual_T9_f4_s5", "pe_T9_l3e-5"]
    train_rows_by_schedule = {
        r["orth_schedule_name"]: r
        for r in follow_summary
        if r["mode"] == "train" and r["status"] == "completed" and str(r["seed"]) == "0" and str(r["lr_mul"]) == "1.0"
    }
    auc_tokens = [to_float(train_rows_by_schedule[s]["val_auc_tokens"]) for s in auc_schedules]
    auc_wall = [to_float(train_rows_by_schedule[s]["val_auc_wall"]) for s in auc_schedules]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    idx_auc = np.arange(len(auc_schedules))
    bars0 = axes[0].bar(idx_auc, auc_tokens, color=[COLORS[family(s)] for s in auc_schedules])
    bars1 = axes[1].bar(idx_auc, auc_wall, color=[COLORS[family(s)] for s in auc_schedules])
    for ax, bars, vals, title in [
        (axes[0], bars0, auc_tokens, "Val-loss AUC over tokens"),
        (axes[1], bars1, auc_wall, "Val-loss AUC over wall-clock"),
    ]:
        ax.set_xticks(idx_auc, auc_schedules, rotation=25, ha="right")
        setup_axis(ax, f"{title} (lower is better)", ylabel="average validation loss along curve")
        zoom_ylim(ax, vals, min_pad=0.01)
        label_bars(ax, bars, vals, digits=3)
    savefig("auc_efficiency_readable.png")

    # Spectral object/module summary.
    spectral_rows = {r["orth_schedule_name"]: r for r in follow_summary if r["mode"] == "spectral" and r["status"] == "completed"}
    spec_schedules = ["stable5", "fast5", "manual_T5_f3_s2", "manual_T9_f4_s5", "pe_T5_l1e-3", "pe_T9_l3e-5"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    width = 0.24
    idx = np.arange(len(spec_schedules))
    for offset, key, label in [(-width, "buffer_post_semi_orth_error", "buffer_post"), (0, "g_pre_semi_orth_error", "g_pre"), (width, "g_post_semi_orth_error", "g_post")]:
        axes[0].bar(idx + offset, [to_float(spectral_rows[s][key]) for s in spec_schedules], width=width, label=label)
    axes[0].set_xticks(idx, spec_schedules, rotation=25, ha="right")
    setup_axis(axes[0], "Before / after orthogonalization", ylabel="semi-orthogonality error")
    axes[0].legend(fontsize=8)

    detail_rows = read_csv(FOLLOW / "spectral_details.csv")
    attn_mlp: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in detail_rows:
        schedule = row.get("orth_schedule_name", "")
        label = row.get("spec/label", "")
        module_type = "attention" if ".attn." in label else "mlp" if ".mlp." in label else ""
        if schedule in {"fast5", "manual_T9_f4_s5", "pe_T9_l3e-5"} and module_type:
            attn_mlp[(schedule, module_type)].append(to_float(row.get("spec/g_post_semi_orth_error")))
    schedules_module = ["fast5", "manual_T9_f4_s5", "pe_T9_l3e-5"]
    idx2 = np.arange(len(schedules_module))
    axes[1].bar(idx2 - 0.18, [float(np.mean(attn_mlp[(s, "attention")])) for s in schedules_module], width=0.36, label="attention")
    axes[1].bar(idx2 + 0.18, [float(np.mean(attn_mlp[(s, "mlp")])) for s in schedules_module], width=0.36, label="MLP")
    axes[1].set_xticks(idx2, schedules_module, rotation=25, ha="right")
    setup_axis(axes[1], "g_post error by module type", ylabel="semi-orthogonality error")
    axes[1].legend(fontsize=8)
    savefig("spectral_readable.png")

    # Polynomial maps: use representative schedules and split panels.
    coeffs_map = schedule_coeffs()
    xs = np.geomspace(1e-5, 1.0, 1000)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    left = ["stable5", "fast5", "manual_T5_f3_s2", "manual_T9_f4_s5"]
    right = ["pe_T5_l1e-3", "pe_T5_l3e-5", "pe_T9_l3e-5", "pe_T10_l3e-5"]
    for ax, schedules, title in [(axes[0], left, "Manual / fixed coefficients"), (axes[1], right, "Polar Express representatives")]:
        for schedule in schedules:
            fam = family(schedule)
            linestyle, marker = line_style(0, 1.0, schedule)
            ys = apply_schedule(xs, coeffs_map[schedule])
            ax.plot(xs, np.clip(ys, -0.1, 2.0), color=COLORS[fam], linestyle=linestyle, linewidth=2.0, label=schedule)
        ax.plot(xs, xs, color=COLORS["reference"], linestyle="--", linewidth=1.1, label="identity")
        ax.set_xscale("log")
        setup_axis(ax, title, xlabel="input singular value sigma", ylabel="mapped singular value")
        ax.set_ylim(-0.05, 2.02)
        ax.legend(fontsize=8)
    savefig("composed_maps_readable.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sigma = np.linspace(0.0, 1.0, 800)
    for name, coeff, color in [("stable", STABLE_COEFF, COLORS["stable"]), ("fast", FAST_COEFF, COLORS["fast"])]:
        a, b, c = coeff
        ys = a * sigma + b * sigma**3 + c * sigma**5
        axes[0].plot(sigma, ys, color=color, linewidth=2.2, label=f"{name}: p(sigma)")
        axes[1].plot(sigma, single_step_derivative(sigma, coeff), color=color, linestyle="--", linewidth=2.2, label=f"{name}: p'(sigma)")
    axes[0].plot(sigma, sigma, color=COLORS["reference"], linestyle=":", label="identity")
    axes[1].axhline(0, color=COLORS["reference"], linestyle=":", linewidth=1.1)
    setup_axis(axes[0], "Single-step map", xlabel="sigma", ylabel="p(sigma)")
    setup_axis(axes[1], "Single-step derivative", xlabel="sigma", ylabel="p'(sigma)")
    axes[0].legend(fontsize=9)
    axes[1].legend(fontsize=9)
    savefig("single_step_maps_derivatives_readable.png")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for schedule in ["stable5", "fast5", "manual_T9_f4_s5", "pe_T9_l3e-5"]:
        fam = family(schedule)
        linestyle, _ = line_style(0, 1.0, schedule)
        deriv = composed_derivative(xs, coeffs_map[schedule])
        ax.plot(xs, np.clip(deriv, -20, 20), color=COLORS[fam], linestyle=linestyle, linewidth=2.0, label=schedule)
    ax.axhline(0, color=COLORS["reference"], linestyle=":", linewidth=1.1)
    ax.set_xscale("log")
    setup_axis(ax, "Derivative of composed map, clipped to [-20, 20]", xlabel="input singular value sigma", ylabel="d p_T(sigma) / d sigma")
    ax.legend(fontsize=8)
    savefig("composed_derivatives_readable.png")


def image_card(filename: str, title: str, read: str, analysis: str, conclusion: str) -> str:
    path = READABLE / filename
    if not path.exists():
        return f"""
        <section class="figure-block missing">
          <h4>{esc(title)}</h4>
          <p>图像未生成。</p>
        </section>
        """
    return f"""
      <section class="figure-block searchable">
        <h4>{esc(title)}</h4>
        <img src="{esc(rel(path))}" alt="{esc(title)}" loading="lazy">
        <div class="fig-note">
          <p><strong>怎么看：</strong>{esc(read)}</p>
          <p><strong>分析：</strong>{esc(analysis)}</p>
          <p><strong>结论：</strong>{esc(conclusion)}</p>
        </div>
      </section>
    """


CSS = """
:root {
  --paper: #f7f2e8;
  --panel: #fffaf0;
  --ink: #14233a;
  --muted: #5f6f82;
  --line: #dfd3bc;
  --blue: #1e5aa8;
  --green: #0c7a55;
  --orange: #c5652f;
  --shadow: 0 14px 30px rgba(37, 31, 20, 0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Inter", "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
  color: var(--ink);
  background: var(--paper);
  line-height: 1.65;
}
code {
  font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
  background: #f0eadf;
  border: 1px solid #e2d6c2;
  border-radius: 5px;
  padding: 1px 4px;
}
header {
  padding: 30px clamp(18px, 4vw, 56px) 22px;
  background: linear-gradient(180deg, #fff8e8 0%, #f7f2e8 100%);
  border-bottom: 1px solid var(--line);
}
.eyebrow {
  color: var(--orange);
  font-weight: 700;
  letter-spacing: .02em;
  text-transform: uppercase;
  font-size: 13px;
}
h1 {
  margin: 8px 0 10px;
  font-size: clamp(30px, 4vw, 48px);
  line-height: 1.12;
  letter-spacing: 0;
}
.subtitle {
  max-width: 980px;
  color: var(--muted);
  font-size: 17px;
}
.toolbar {
  position: sticky;
  top: 0;
  z-index: 20;
  padding: 12px clamp(18px, 4vw, 56px);
  display: flex;
  gap: 10px;
  align-items: center;
  border-bottom: 1px solid var(--line);
  background: rgba(247, 242, 232, .94);
  backdrop-filter: blur(10px);
}
#keywordSearch {
  width: min(560px, 68vw);
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: #fffdf7;
  color: var(--ink);
}
.search-nav {
  border: 1px solid var(--line);
  background: #fffdf7;
  color: var(--ink);
  border-radius: 8px;
  padding: 9px 12px;
  cursor: pointer;
}
.search-status { color: var(--muted); font-size: 13px; }
.layout {
  display: grid;
  grid-template-columns: 250px minmax(0, 1fr);
  gap: 22px;
  padding: 22px clamp(18px, 4vw, 56px) 56px;
}
aside nav {
  position: sticky;
  top: 76px;
  display: grid;
  gap: 7px;
}
aside a {
  border: 1px solid var(--line);
  background: #fffdf7;
  color: var(--ink);
  border-radius: 8px;
  padding: 9px 10px;
  text-decoration: none;
}
main { display: grid; gap: 18px; }
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 18px;
  box-shadow: var(--shadow);
}
.panel h2 { margin: 0 0 12px; font-size: 24px; }
.panel h3 { margin: 20px 0 8px; font-size: 19px; }
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
}
.card {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fffdf7;
  padding: 13px;
}
.card strong { display: block; margin-bottom: 5px; }
.table-wrap {
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fffdf7;
}
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td {
  border-bottom: 1px solid var(--line);
  padding: 9px 10px;
  text-align: left;
  vertical-align: top;
}
th { background: #f1eadc; color: #31445d; }
tr:last-child td { border-bottom: 0; }
.callout {
  border-left: 4px solid var(--blue);
  background: #eef4ff;
  border-radius: 8px;
  padding: 12px 14px;
  margin: 12px 0;
}
.figure-block {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fffdf7;
  padding: 12px;
  margin-top: 12px;
}
.figure-block h4 {
  margin: 0 0 8px;
  font-size: 17px;
}
.figure-block img {
  display: block;
  width: 100%;
  max-height: 760px;
  object-fit: contain;
  border: 1px solid #eadfce;
  border-radius: 6px;
  background: white;
}
.fig-note {
  margin-top: 10px;
  color: #26384f;
}
.fig-note p { margin: 6px 0; }
.legend-note {
  color: var(--muted);
  font-size: 14px;
}
.search-highlight {
  background: #ffe38a;
  color: #1f2630;
  border-radius: 3px;
  padding: 0 2px;
}
.search-highlight.active {
  background: #ff9f43;
  outline: 2px solid #c5652f;
}
@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
  aside nav { position: static; }
  .toolbar { flex-wrap: wrap; }
}
"""


JS = """
const searchBox = document.getElementById("keywordSearch");
const prevSearch = document.getElementById("prevSearch");
const nextSearch = document.getElementById("nextSearch");
const searchStatus = document.getElementById("searchStatus");
const root = document.querySelector("main");
let hits = [];
let hitIndex = -1;

function clearHighlights() {
  root.querySelectorAll("span.search-highlight").forEach(span => {
    const text = document.createTextNode(span.textContent);
    span.replaceWith(text);
  });
  root.normalize();
  hits = [];
  hitIndex = -1;
}

function highlightText(node, query) {
  const text = node.nodeValue;
  const lower = text.toLowerCase();
  const q = query.toLowerCase();
  let pos = 0;
  let idx = lower.indexOf(q, pos);
  if (idx === -1) return;
  const frag = document.createDocumentFragment();
  while (idx !== -1) {
    if (idx > pos) frag.appendChild(document.createTextNode(text.slice(pos, idx)));
    const span = document.createElement("span");
    span.className = "search-highlight";
    span.textContent = text.slice(idx, idx + query.length);
    frag.appendChild(span);
    pos = idx + query.length;
    idx = lower.indexOf(q, pos);
  }
  if (pos < text.length) frag.appendChild(document.createTextNode(text.slice(pos)));
  node.replaceWith(frag);
}

function collectTextNodes(node, out = []) {
  if (node.nodeType === Node.TEXT_NODE) {
    if (node.nodeValue.trim()) out.push(node);
    return out;
  }
  if (node.nodeType !== Node.ELEMENT_NODE) return out;
  const tag = node.tagName.toLowerCase();
  if (["script", "style", "textarea", "input"].includes(tag)) return out;
  node.childNodes.forEach(child => collectTextNodes(child, out));
  return out;
}

function runSearch() {
  clearHighlights();
  const q = searchBox.value.trim();
  if (!q) {
    searchStatus.textContent = "";
    return;
  }
  collectTextNodes(root).forEach(node => highlightText(node, q));
  hits = [...root.querySelectorAll(".search-highlight")];
  if (!hits.length) {
    searchStatus.textContent = "0 个结果";
    return;
  }
  hitIndex = 0;
  activateHit();
}

function activateHit() {
  hits.forEach(h => h.classList.remove("active"));
  if (hitIndex < 0 || !hits.length) return;
  const hit = hits[hitIndex];
  hit.classList.add("active");
  hit.scrollIntoView({ behavior: "smooth", block: "center" });
  searchStatus.textContent = `${hitIndex + 1} / ${hits.length}`;
}

searchBox.addEventListener("input", runSearch);
nextSearch.addEventListener("click", () => {
  if (!hits.length) return;
  hitIndex = (hitIndex + 1) % hits.length;
  activateHit();
});
prevSearch.addEventListener("click", () => {
  if (!hits.length) return;
  hitIndex = (hitIndex - 1 + hits.length) % hits.length;
  activateHit();
});
"""


def build_html() -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Muon 系数序列实验解读</title>
  <style>{CSS}</style>
  <script>
    window.MathJax = {{ tex: {{ inlineMath: [['\\\\(', '\\\\)'], ['$', '$']] }}, svg: {{ fontCache: 'global' }} }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>
<body>
  <header>
    <div class="eyebrow">Muon / Polar Express experiments</div>
    <h1>Muon 系数序列实验解读</h1>
    <p class="subtitle">
      本页整理统一 clean training stack 下的基础控制变量实验与后续扩展实验。
      关注点是 Newton-Schulz / Polar Express 系数序列如何改变训练表现、运行成本和更新矩阵的谱几何。
    </p>
  </header>

  <section class="toolbar">
    <input id="keywordSearch" type="search" placeholder="搜索 fast5、gain、lower bound、attention、wall-clock">
    <button id="prevSearch" class="search-nav" type="button" aria-label="上一个搜索结果">&lt;</button>
    <button id="nextSearch" class="search-nav" type="button" aria-label="下一个搜索结果">&gt;</button>
    <span id="searchStatus" class="search-status"></span>
  </section>

  <div class="layout">
    <aside>
      <nav>
        <a href="#question">研究问题</a>
        <a href="#glossary">名词速查</a>
        <a href="#map">实验地图</a>
        <a href="#basic">基础控制变量</a>
        <a href="#followup">扩展实验</a>
        <a href="#spectral">谱几何</a>
        <a href="#polynomial">多项式映射</a>
        <a href="#takeaways">最终结论</a>
      </nav>
    </aside>

    <main>
      <section id="question" class="panel searchable">
        <h2>研究问题</h2>
        <div class="callout">
          Muon 的矩阵参数更新会经过一个多步五次多项式正交化过程。若奇异值为 \\(\\sigma\\)，单步可写作
          \\(p(\\sigma)=a\\sigma+b\\sigma^3+c\\sigma^5\\)。实验真正改变的是这些 \\((a,b,c)\\) 序列，
          也就是复合映射 \\(p_T(\\sigma)\\)。
        </div>
        <div class="cards">
          <div class="card"><strong>训练表现</strong>validation loss、val-loss AUC、multi-seed final loss。</div>
          <div class="card"><strong>运行成本</strong>端到端 wall-clock，重点看 T=5 与 T=9/10 的成本差别。</div>
          <div class="card"><strong>谱几何</strong><code>buffer_post</code>、<code>g_pre</code>、<code>g_post</code> 的半正交误差。</div>
          <div class="card"><strong>数学解释</strong>复合奇异值映射、gain 和导数解释为什么某些曲线更激进或更平滑。</div>
        </div>
      </section>

      <section id="glossary" class="panel searchable">
        <h2>名词速查</h2>
        <div class="table-wrap">
          <table>
            <thead><tr><th>术语 / 代号</th><th>含义</th></tr></thead>
            <tbody>
              <tr><td><code>AdamW</code></td><td>不做矩阵正交化的 optimizer baseline。</td></tr>
              <tr><td><code>stable5</code></td><td>连续 5 步 stable Newton-Schulz 系数，也对应代码里的 <code>vanilla</code> 调度。</td></tr>
              <tr><td><code>fast5</code></td><td>连续 5 步 fast Muon 系数，是当前实验中最强的 practical baseline。</td></tr>
              <tr><td><code>manual_T5_f3_s2</code></td><td>总 5 步：3 步 fast 后接 2 步 stable。</td></tr>
              <tr><td><code>manual_T9_f4_s5</code></td><td>总 9 步：4 步 fast 后接 5 步 stable。</td></tr>
              <tr><td><code>PE</code></td><td>Polar Express，根据 lower bound 逐步生成五次多项式系数。</td></tr>
              <tr><td><code>lower bound</code></td><td>PE 设计系数时假设的最小奇异值尺度，改变它会改变小奇异值区域的映射形状。</td></tr>
              <tr><td><code>buffer_post</code></td><td>momentum buffer 更新后的矩阵。</td></tr>
              <tr><td><code>g_pre</code></td><td>进入正交化前的 Nesterov mixed matrix。</td></tr>
              <tr><td><code>g_post</code></td><td>正交化后的 update matrix，是最终用于参数更新的矩阵。</td></tr>
              <tr><td><code>gain</code></td><td>\\(p_T(\\sigma)/\\sigma\\)，表示某个奇异值经过完整 schedule 后被放大多少倍。</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      <section id="map" class="panel searchable">
        <h2>实验地图</h2>
        <div class="table-wrap">
          <table>
            <thead><tr><th>实验组</th><th>控制变量</th><th>回答的问题</th></tr></thead>
            <tbody>
              <tr><td>基础 T=5 对照</td><td>固定 T=5，只换系数序列</td><td>stable / fast / manual / PE 的核心差异。</td></tr>
              <tr><td>Benchmark</td><td>同样训练预算，记录 wall-clock</td><td>正交化矩阵更新和更深迭代带来多少时间成本。</td></tr>
              <tr><td>Spectral</td><td>采样 update 矩阵并做 SVD</td><td>不同系数序列是否真的改变 update geometry。</td></tr>
              <tr><td>Multi-seed</td><td>代表配置 seeds 0/1/2</td><td>final loss 差距是否大于 seed 波动。</td></tr>
              <tr><td>LR sanity</td><td>代表配置 lr=0.5/1.0/2.0</td><td>主结论是否依赖某个配置偶然拿到更好 LR。</td></tr>
              <tr><td>Manual depth</td><td>比较 T=5/7/8/9/10 的 manual family</td><td>增加迭代深度是否持续带来收益。</td></tr>
              <tr><td>PE lower/depth</td><td>改变 PE lower bound 和 T</td><td>PE 的数学假设如何影响训练和谱映射。</td></tr>
              <tr><td>Polynomial maps</td><td>只用系数计算 \\(p_T(\\sigma)\\)</td><td>从数学上解释不同 schedule 的行为。</td></tr>
            </tbody>
          </table>
        </div>
        <p class="legend-note">图中的颜色表示方法家族：蓝色 AdamW，灰色 stable，绿色 fast，橙色 manual，紫色 PE；marker 表示具体配置，虚线/点线只在同一方法家族有多条曲线时用于区分 LR、lower bound 或 depth。</p>
      </section>

      <section id="basic" class="panel searchable">
        <h2>基础控制变量实验</h2>
        <p>这一组是最干净的核心对照：固定 T=5、模型、数据、训练预算、batch 和学习率调度，只改变正交化系数序列。</p>
        {image_card(
            "basic_t5_summary.png",
            "基础 T=5：loss、wall-clock 与 g_post error",
            "左图看最终 validation loss，中图看端到端 wall-clock，右图看正交化后的 g_post 半正交误差。三张图使用同一颜色编码。",
            "如果只看 loss，fast 与 PE 位于第一梯队，stable 明显落后；如果看 wall-clock，同样 T=5 的 Muon 变体几乎重合；如果看几何，PE 的 g_post error 最低。",
            "系数序列本身会影响训练和几何；同样五步迭代下，改变标量系数几乎不改变运行时间。"
        )}
        {image_card(
            "val_loss_core_readable.png",
            "核心 T=5：validation loss 曲线",
            "横轴是训练 token，纵轴是 validation loss；同一 token 下曲线越低，表示同样数据量学得越快。这里固定 seed=0、lr=1.0；三角点对应 manual，紫色 PE 中若有虚线只是用于区分不同 lower bound。",
            "AdamW 曲线整体更高，说明缺少矩阵正交化更新时当前设置明显较弱；stable5 虽然属于 Muon，但下降速度慢于 fast5 和 PE；manual_T5 位于 stable 与 fast 之间。fast5 与 PE 曲线非常接近，不能只凭肉眼说某一个绝对胜出。",
            "T=5 固定时，fast coefficients 和较好的 PE 系数能带来更好的 token efficiency。"
        )}
      </section>

      <section id="followup" class="panel searchable">
        <h2>扩展实验</h2>
        <p>扩展实验围绕代表性问题展开：seed 稳定性、LR 敏感性、manual depth、PE lower bound 和 PE depth。</p>
        {image_card(
            "val_loss_followup_facets.png",
            "Follow-up validation loss：分面曲线",
            "每个小图只比较一个问题：核心 T=5、LR sanity、manual depth、PE lower/depth。这里画的是代表性 seed=0 曲线；多 seed 稳定性看下一张柱状图。颜色表示方法家族，虚线/点线/marker 表示 LR、lower bound 或 depth。",
            "曲线轻微抖动来自有限 eval tokens、随机 minibatch 轨迹和较激进配置的局部波动；持续平滑下降说明优化稳定，平台期或下降变慢说明该配置在当前预算下学习效率下降。LR=2.0 的曲线更容易高位停滞，说明步长偏激进会压低稳定性。",
            "把所有线拆成分面后，核心规律更清楚：fast5 稳定强，manual 加深能追近，PE lower/depth 会改变曲线但没有稳定超过 fast5。"
        )}
        {image_card(
            "auc_efficiency_readable.png",
            "Val-loss AUC：学习效率与时间性价比",
            "左图是 validation loss 关于 token 的曲线面积，右图是关于 wall-clock 的曲线面积；两者都越低越好。",
            "局部斜率会受到起点、eval 噪声和选取区间影响，容易误导。AUC 把整条曲线压成一个指标：如果一个方法早期下降快、后期也保持较低 loss，它的 AUC 就会更低。token AUC 更适合比较样本效率，wall-clock AUC 更适合比较训练性价比。",
            "在当前设置里，fast5 和若干更深 manual/PE 的 AUC 很接近；这说明后续结论应保守表述为第一梯队接近，而不是只凭某个终点或局部斜率排序。"
        )}
        {image_card(
            "multi_seed_final_loss.png",
            "Multi-seed final loss",
            "柱子是 seeds 0/1/2 的平均 final loss，误差条是 seed 间标准差；这张图使用局部 y 轴，所以细小差距和误差条能看清。",
            "fast5、manual_T9_f4_s5、pe_T9_l3e-5 的差距都不大。fast5 的均值最低，但差距和 seed 波动处在同一个小量级，说明结论应强调稳定 baseline，而不是绝对胜出。",
            "当前 100M-token 设置下，fast5 是最稳的 practical baseline；更复杂的几何改进没有转化为更低 final loss。"
        )}
        {image_card(
            "manual_depth_final_loss_readable.png",
            "Manual depth trend",
            "横轴是 manual schedule，纵轴是 final validation loss；橙色是 manual family，绿色 fast5 是参考线。这里使用局部 y 轴，专门观察 0.01 量级差距。",
            "T=5 到 T=7/8/9/10 有明显改善，但 T=7 以后进入窄区间。继续加深会改变 update geometry，也会增加计算，但 loss 收益不再单调扩大。",
            "manual depth 有帮助，但不是越深越好；它更像是在接近 fast5 这一强 baseline。"
        )}
        {image_card(
            "pe_lower_depth_final_loss_readable.png",
            "PE lower bound and depth",
            "横轴比较不同 PE lower bound 与 T，绿色 fast5 作为参考；纵轴越低越好。这里也使用局部 y 轴来显示小差距。",
            "固定 T=5 时，lower bound 改变 final loss；固定 lower bound 后增加 T 可以改善部分 PE 配置，但仍未稳定低于 fast5。",
            "PE lower bound 是实质算法参数，它改变谱映射假设；PE depth 能改善某些配置，但额外迭代需要和成本一起看。"
        )}
        {image_card(
            "benchmark_wall_clock_readable.png",
            "Benchmark wall-clock",
            "柱子表示端到端训练时间；越低越快。",
            "T=5 的 Muon variants 接近，说明只换系数不会明显改变成本。T=9 manual/PE 更慢，主要来自多做几步矩阵正交化。AdamW 更快，因为没有正交化矩阵更新。",
            "最终比较不能只看 loss，也要看同 wall-clock 下是否划算。"
        )}
      </section>

      <section id="spectral" class="panel searchable">
        <h2>谱几何</h2>
        {image_card(
            "spectral_readable.png",
            "正交化前后与模块分解",
            "左图比较 buffer_post、g_pre、g_post；右图把 g_post error 拆成 attention 和 MLP。误差越低，update 越接近半正交目标。",
            "所有 Muon schedule 都会把 g_pre 推向更低误差的 g_post。更深 manual 和 PE 的 g_post error 低于 fast5，尤其在 attention projection matrices 上差距最大；MLP 上差距较小。",
            "系数序列确实改变了 update geometry；但 geometry 更接近半正交和 validation loss 更低之间不是简单单调关系。"
        )}
      </section>

      <section id="polynomial" class="panel searchable">
        <h2>多项式映射解释</h2>
        <p>这一部分不来自训练日志，而是直接由系数序列计算。它用于解释为什么不同 schedule 会产生不同几何行为。</p>
        {image_card(
            "composed_maps_readable.png",
            "Composed singular-value maps",
            "横轴是输入奇异值 \\(\\sigma\\)，纵轴是完整 schedule 后的 \\(p_T(\\sigma)\\)。虚线 identity 表示不改变奇异值。",
            "stable 曲线平滑且保守；fast 和更深 manual 对小奇异值更激进；PE 的曲线更容易出现振荡，是因为每一步多项式根据 lower-bound 区间构造，复合后在某些区间会发生 overshoot 和非单调折返。",
            "曲线越激进，通常 g_post error 越低；但过强的折返和振荡也可能让训练 loss 不再受益。"
        )}
        {image_card(
            "single_step_maps_derivatives_readable.png",
            "Single-step maps and derivatives",
            "左图是单步 \\(p(\\sigma)\\)，右图是单步导数 \\(p'(\\sigma)\\)。实线看映射值，虚线看局部斜率。",
            "fast 的导数在中高奇异值区域会跨过 0 并变成负数，说明该映射不是全区间单调。导数为 0 的点表示局部敏感度最低：附近不同 \\(\\sigma\\) 会被压到相近输出，而不是 update 变成 0。",
            "fast 的小奇异值放大更强，因此训练早期更有效；但非单调区间也解释了为什么更激进的映射需要通过训练结果检验。"
        )}
        {image_card(
            "composed_derivatives_readable.png",
            "Derivative of composed map",
            "这张图看 \\(d p_T(\\sigma)/d\\sigma\\)，为避免少数极大值压扁图像，纵轴裁剪到 [-20, 20]。",
            "复合导数会把每一步的局部斜率相乘，因此比单步导数更容易出现符号变化和大幅振荡。振荡更强的 schedule 对小奇异值更激进，但也更可能把相邻奇异值区间折叠到相近输出。",
            "这解释了几何和训练之间的张力：更强的正交化可以降低 g_post error，但未必给语言建模 loss 带来单调收益。"
        )}
      </section>

      <section id="takeaways" class="panel searchable">
        <h2>最终结论</h2>
        <div class="cards">
          <div class="card"><strong>结论 1</strong><code>fast5</code> 是当前 clean setup 下最强 practical baseline。</div>
          <div class="card"><strong>结论 2</strong><code>stable5</code> 偏保守，说明数学上更稳定的系数不一定带来最优 pretraining loss。</div>
          <div class="card"><strong>结论 3</strong>manual / PE 能显著降低 <code>g_post</code> error，尤其 attention matrices，但 loss 收益没有单调对应。</div>
          <div class="card"><strong>结论 4</strong>PE lower bound 和 iteration depth 通过 \\(p_T(\\sigma)\\) 改变谱映射，是有数学含义的算法参数。</div>
        </div>
        <div class="callout">
          最自然的汇报主线是：系数序列改变复合奇异值映射 \\(p_T(\\sigma)\\)，进而改变 update geometry；
          但更强的半正交化不会自动转化为更低 validation loss。训练曲线、wall-clock、谱诊断和多项式图应一起读。
        </div>
      </section>
    </main>
  </div>

  <script>{JS}</script>
</body>
</html>
"""


def main() -> int:
    plot_readable_figures()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_html(), encoding="utf-8", newline="\n")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

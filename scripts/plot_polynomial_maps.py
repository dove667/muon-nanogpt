#!/usr/bin/env python
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optim.orth import FAST_COEFF, STABLE_COEFF, polar_express_coefficients


OUT_DIR = ROOT / "results" / "polynomial_maps"


def apply_schedule(xs: np.ndarray, coeffs: list[tuple[float, float, float]]) -> np.ndarray:
    ys = xs.copy()
    for a, b, c in coeffs:
        ys = a * ys + b * ys**3 + c * ys**5
    return ys


def single_step_derivative(xs: np.ndarray, coeff: tuple[float, float, float]) -> np.ndarray:
    a, b, c = coeff
    return a + 3 * b * xs**2 + 5 * c * xs**4


def schedules() -> dict[str, list[tuple[float, float, float]]]:
    return {
        "stable5": [STABLE_COEFF] * 5,
        "fast5": [FAST_COEFF] * 5,
        "manual_T5_f3_s2": [FAST_COEFF] * 3 + [STABLE_COEFF] * 2,
        "manual_T7_f4_s3": [FAST_COEFF] * 4 + [STABLE_COEFF] * 3,
        "manual_T8_f5_s3": [FAST_COEFF] * 5 + [STABLE_COEFF] * 3,
        "manual_T9_f4_s5": [FAST_COEFF] * 4 + [STABLE_COEFF] * 5,
        "manual_T10_f5_s5": [FAST_COEFF] * 5 + [STABLE_COEFF] * 5,
        "pe_T5_l3e-3": polar_express_coefficients(3e-3, 5, 0.02, 0.02),
        "pe_T5_l1e-3": polar_express_coefficients(1e-3, 5, 0.02, 0.02),
        "pe_T5_l3e-4": polar_express_coefficients(3e-4, 5, 0.02, 0.02),
        "pe_T5_l3e-5": polar_express_coefficients(3e-5, 5, 0.02, 0.02),
        "pe_T9_l3e-5": polar_express_coefficients(3e-5, 9, 0.02, 0.02),
        "pe_T10_l3e-5": polar_express_coefficients(3e-5, 10, 0.02, 0.02),
    }


def plot_all_maps(xs: np.ndarray, schedule_map: dict[str, list[tuple[float, float, float]]]) -> None:
    plt.figure(figsize=(11, 7))
    for name, coeffs in schedule_map.items():
        ys = apply_schedule(xs, coeffs)
        plt.plot(xs, ys, label=name, linewidth=1.8)
    plt.plot(xs, xs, color="black", linestyle="--", linewidth=1.0, label="identity")
    plt.xlabel("input singular value sigma")
    plt.ylabel("after full coefficient schedule")
    plt.title("Composed singular-value maps")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "composed_maps.png", dpi=180)
    plt.close()


def plot_delta_maps(xs: np.ndarray, schedule_map: dict[str, list[tuple[float, float, float]]]) -> None:
    plt.figure(figsize=(11, 7))
    for name, coeffs in schedule_map.items():
        ys = apply_schedule(xs, coeffs)
        plt.plot(xs, ys - xs, label=name, linewidth=1.8)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("input singular value sigma")
    plt.ylabel("p_1:T(sigma) - sigma")
    plt.title("Net singular-value displacement")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "composed_map_delta.png", dpi=180)
    plt.close()


def plot_one_step_maps(xs: np.ndarray) -> None:
    plt.figure(figsize=(10, 6))
    for name, coeff in {"stable": STABLE_COEFF, "fast": FAST_COEFF}.items():
        a, b, c = coeff
        ys = a * xs + b * xs**3 + c * xs**5
        plt.plot(xs, ys, label=f"{name}: p(sigma)")
        plt.plot(xs, single_step_derivative(xs, coeff), linestyle="--", label=f"{name}: p'(sigma)")
    plt.plot(xs, xs, color="black", linewidth=1.0, alpha=0.7, label="identity")
    plt.xlabel("sigma")
    plt.title("Single-step Newton-Schulz maps and derivatives")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "single_step_maps_and_derivatives.png", dpi=180)
    plt.close()


def write_map_samples(xs: np.ndarray, schedule_map: dict[str, list[tuple[float, float, float]]]) -> None:
    sample_points = np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.6, 1.0])
    with (OUT_DIR / "map_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["schedule", "sigma", "mapped_sigma", "delta", "gain"])
        writer.writeheader()
        for name, coeffs in schedule_map.items():
            mapped = apply_schedule(sample_points, coeffs)
            for sigma, mapped_sigma in zip(sample_points, mapped):
                writer.writerow({
                    "schedule": name,
                    "sigma": float(sigma),
                    "mapped_sigma": float(mapped_sigma),
                    "delta": float(mapped_sigma - sigma),
                    "gain": float(mapped_sigma / max(float(sigma), 1e-30)),
                })


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xs = np.geomspace(1e-5, 1.0, 1200)
    schedule_map = schedules()
    plot_all_maps(xs, schedule_map)
    plot_delta_maps(xs, schedule_map)
    plot_one_step_maps(xs)
    write_map_samples(xs, schedule_map)
    print(f"Wrote polynomial-map figures to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

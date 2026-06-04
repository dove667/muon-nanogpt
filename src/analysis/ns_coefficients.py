#!/usr/bin/env python
"""Analyze the scalar singular-value map induced by Muon Newton-Schulz coefficients."""


import argparse
import csv
from pathlib import Path

from src.paths import ROOT
COEFF_SETS = {
    "simple_track3": (2.0, -1.5, 0.5, 12),
    "deepseek": (3.4445, -4.7750, 2.0315, 5),
    "mild_aggressive": (2.6, -2.8, 1.2, 8),
}


def step_singular_value(s: float, a: float, b: float, c: float) -> float:
    return (a + b * s * s + c * s**4) * s


def iterate(s: float, a: float, b: float, c: float, steps: int) -> float:
    for _ in range(steps):
        s = step_singular_value(s, a, b, c)
    return s


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="analysis")
    parser.add_argument("--grid-size", type=int, default=400)
    args = parser.parse_args()

    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = []
    for name, (a, b, c, steps) in COEFF_SETS.items():
        final_values = []
        unstable_count = 0
        for idx in range(args.grid_size + 1):
            s0 = max(1e-6, idx / args.grid_size)
            s_final = iterate(s0, a, b, c, steps)
            if abs(s_final) > 10 or s_final != s_final:
                unstable_count += 1
            final_values.append(s_final)
            rows.append({
                "name": name,
                "a": a,
                "b": b,
                "c": c,
                "steps": steps,
                "s0": s0,
                "s_final": s_final,
                "abs_error_to_1": abs(abs(s_final) - 1),
            })
        abs_errors = [abs(abs(v) - 1) for v in final_values if abs(v) <= 10]
        summary.append({
            "name": name,
            "a": a,
            "b": b,
            "c": c,
            "steps": steps,
            "mean_abs_error_to_1": sum(abs_errors) / len(abs_errors),
            "max_abs_error_to_1": max(abs_errors),
            "min_final": min(final_values),
            "max_final": max(final_values),
            "unstable_grid_points": unstable_count,
        })

    csv_path = out_dir / "ns_coefficients.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {csv_path}")
    print("name,steps,mean_abs_error_to_1,max_abs_error_to_1,unstable_grid_points")
    for row in summary:
        print(
            f"{row['name']},{row['steps']},{row['mean_abs_error_to_1']:.6f},"
            f"{row['max_abs_error_to_1']:.6f},{row['unstable_grid_points']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

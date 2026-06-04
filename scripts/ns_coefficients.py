#!/usr/bin/env python
"""Analyze the scalar singular-value map of Newton-Schulz coefficient sets.

For each (a,b,c) coefficient triple, applies T iterations to a grid of
singular values s0 ∈ [0, 1] and measures convergence error |s_final| − 1.
"""

import argparse
from pathlib import Path

COEFF_SETS = {
    "vanilla":  (2.0,    -1.5,    0.5,    12),   # stable (standard NS)
    "fast":     (3.4445, -4.7750, 2.0315,  5),   # aggressive (Keller–Jordan),
    "mild":     (2.6,    -2.8,    1.2,     8),   # intermediate
}


def _step(s: float, a: float, b: float, c: float) -> float:
    return (a + b * s * s + c * s ** 4) * s


def _iterate(s: float, a: float, b: float, c: float, steps: int) -> float:
    for _ in range(steps):
        s = _step(s, a, b, c)
    return s


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--grid-size", type=int, default=400)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("name,steps,mean_abs_error_to_1,max_abs_error_to_1,unstable_grid_points")

    for name, (a, b, c, steps) in COEFF_SETS.items():
        values = []
        unstable = 0
        for idx in range(args.grid_size + 1):
            s0 = max(1e-6, idx / args.grid_size)
            sf = _iterate(s0, a, b, c, steps)
            if abs(sf) > 10 or sf != sf:
                unstable += 1
            values.append(sf)

        abs_errors = [abs(abs(v) - 1) for v in values if abs(v) <= 10]
        mean_err = sum(abs_errors) / len(abs_errors)
        max_err = max(abs_errors)

        print(f"{name},{steps},{mean_err:.6f},{max_err:.6f},{unstable}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import subprocess
import sys
from dataclasses import dataclass

from src.paths import ROOT, RUNS_ROOT

SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class RunSpec:
    orth: str
    seed: int

    @property
    def name(self) -> str:
        if self.orth == "adamw":
            return f"adamw_seed{self.seed}"
        if self.orth == "vanilla":
            return f"vanilla_seed{self.seed}"
        if self.orth == "fast":
            return f"fast_seed{self.seed}"
        if self.orth == "manual":
            return f"manual_f3_s2_seed{self.seed}"
        if self.orth == "polar_express":
            return f"polar_express_l1e-3_seed{self.seed}"
        raise SystemExit(f"Unknown orth={self.orth}")


def run_completed(name: str) -> bool:
    metrics_path = RUNS_ROOT / name / "metrics.jsonl"
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        return False
    last_line = metrics_path.read_text(encoding="utf-8").splitlines()[-1]
    return '"status": "completed"' in last_line or '"status":"completed"' in last_line


def build_run_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for seed in SEEDS:
        specs.extend([
            RunSpec("adamw", seed),
            RunSpec("vanilla", seed),
            RunSpec("manual", seed),
            RunSpec("fast", seed),
            RunSpec("polar_express", seed),
        ])
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the fixed 15-run experiment plan.")
    parser.add_argument("--data-path", required=True,
                        help="Path to training data directory")
    parser.add_argument("--skip-completed-runs", action="store_true",
                        help="Skip runs that have already completed")
    return parser.parse_args()


def launch(spec: RunSpec, data_path: str, skip_completed: bool) -> None:
    if skip_completed and run_completed(spec.name):
        print("=" * 80)
        print(f"skip completed: {spec.name}")
        print("=" * 80)
        return

    command = [
        sys.executable, "-m", "src.training.train",
        "--orth", spec.orth,
        "--seed", str(spec.seed),
        "--data-path", data_path,
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    specs = build_run_specs()
    for spec in specs:
        launch(spec, args.data_path, args.skip_completed_runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

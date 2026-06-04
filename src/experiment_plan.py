import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from src.utils import ROOT, RUNS_ROOT


DEFAULT_TRAIN_TOKEN_BUDGET = 100_000_000
DEFAULT_EVAL_EVERY_TOKENS = 2_000_000
DEFAULT_EVAL_TOKENS = 524_288
DEFAULT_TRAIN_GRAD_ACCUM_STEPS = 16
DEFAULT_SPECTRAL_EVERY_TOKENS = 10_000_000
DEFAULT_SPECTRAL_MAX_MATRICES = 5
DEFAULT_SPECTRAL_MAX_DIM = 1024
SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class RunSpec:
    orth: str
    name: str
    lr_mul: float = 1.0
    seed: int = 0
    train_token_budget: int = DEFAULT_TRAIN_TOKEN_BUDGET
    eval_every_tokens: int = DEFAULT_EVAL_EVERY_TOKENS
    eval_tokens: int = DEFAULT_EVAL_TOKENS
    fast_steps: int | None = None
    stable_steps: int | None = None
    pe_lower_bound: str | None = None

    def to_cli_args(self) -> list[str]:
        args = [
            "--orth", self.orth,
            "--name", self.name,
            "--lr-mul", str(self.lr_mul),
            "--seed", str(self.seed),
            "--train-token-budget", str(self.train_token_budget),
            "--eval-every-tokens", str(self.eval_every_tokens),
            "--eval-tokens", str(self.eval_tokens),
        ]
        if self.orth == "manual":
            args.extend([
                "--fast-steps", str(self.fast_steps),
                "--stable-steps", str(self.stable_steps),
            ])
        elif self.orth == "polar_express":
            args.extend([
                "--pe-lower-bound", str(self.pe_lower_bound),
            ])
        return args


def run_completed(name: str, runs_root: str | None = None) -> bool:
    if runs_root is None:
        root = RUNS_ROOT
    else:
        root = Path(runs_root).expanduser()
        if not root.is_absolute():
            root = ROOT / root
    metrics_path = root / name / "metrics.jsonl"
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        return False
    last_line = metrics_path.read_text(encoding="utf-8").splitlines()[-1]
    return '"status": "completed"' in last_line or '"status":"completed"' in last_line


def build_run_specs(
    *,
    train_token_budget: int,
    eval_every_tokens: int,
    eval_tokens: int,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for seed in SEEDS:
        specs.extend([
            RunSpec("adamw", f"adamw_seed{seed}", seed=seed, train_token_budget=train_token_budget, eval_every_tokens=eval_every_tokens, eval_tokens=eval_tokens),
            RunSpec("vanilla", f"vanilla_seed{seed}", seed=seed, train_token_budget=train_token_budget, eval_every_tokens=eval_every_tokens, eval_tokens=eval_tokens),
            RunSpec("manual", f"manual_seed{seed}", seed=seed, train_token_budget=train_token_budget, eval_every_tokens=eval_every_tokens, eval_tokens=eval_tokens, fast_steps=3, stable_steps=2),
            RunSpec("fast", f"fast_seed{seed}", seed=seed, train_token_budget=train_token_budget, eval_every_tokens=eval_every_tokens, eval_tokens=eval_tokens),
            RunSpec("polar_express", f"polar_express_seed{seed}", seed=seed, train_token_budget=train_token_budget, eval_every_tokens=eval_every_tokens, eval_tokens=eval_tokens, pe_lower_bound="1e-3"),
        ])
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the fixed 15-run experiment plan.")
    parser.add_argument("--skip-completed-runs", action="store_true",
                        help="Skip runs that have already completed")

    parser.add_argument("--train-token-budget", type=int, default=DEFAULT_TRAIN_TOKEN_BUDGET,
                        help="Total token budget for training")
    parser.add_argument("--eval-every-tokens", type=int, default=DEFAULT_EVAL_EVERY_TOKENS,
                        help="Run evaluation every N tokens")
    parser.add_argument("--eval-tokens", type=int, default=DEFAULT_EVAL_TOKENS,
                        help="Number of tokens per evaluation step")
    parser.add_argument("--train-grad-accum-steps", type=int, default=DEFAULT_TRAIN_GRAD_ACCUM_STEPS,
                        help="Gradient accumulation steps for training")
    parser.add_argument("--eval-batch-size", type=int, default=None,
                        help="Batch size for evaluation")
    parser.add_argument("--eval-at-start", action="store_true",
                        help="Run evaluation at the start of training")
    parser.add_argument("--log-every-steps", type=int, default=20,
                        help="Log training metrics every N steps")

    parser.add_argument("--wandb", choices=["on", "off"], default="on",
                        help="Enable or disable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="muon-nanogpt",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb-entity",
                        help="Weights & Biases entity name")
    parser.add_argument("--wandb-mode",
                        help="Weights & Biases mode (e.g. offline, dryrun)")

    parser.add_argument("--runs-root",
                        help="Root directory for run outputs")
    parser.add_argument("--training-root",
                        help="Root directory for training artifacts")
    parser.add_argument("--trainer-py",
                        help="Path to custom trainer Python module")
    parser.add_argument("--nproc-per-node", type=int, default=1,
                        help="Number of processes per node")
    parser.add_argument("--data-path",
                        help="Path to training data directory")
    parser.add_argument("--model-max-seq-len", type=int, default=0,
                        help="Maximum sequence length for the model (0 for default)")
    
    parser.add_argument("--spectral-every-tokens", type=int, default=DEFAULT_SPECTRAL_EVERY_TOKENS,
                        help="Run spectral analysis every N tokens")
    parser.add_argument("--spectral-max-matrices", type=int, default=DEFAULT_SPECTRAL_MAX_MATRICES,
                        help="Maximum number of matrices for spectral analysis")
    parser.add_argument("--spectral-max-dim", type=int, default=DEFAULT_SPECTRAL_MAX_DIM,
                        help="Maximum dimension for spectral analysis")
    return parser.parse_args()


def launch(spec: RunSpec, args: argparse.Namespace) -> None:
    if args.skip_completed_runs and run_completed(spec.name, args.runs_root):
        print("=" * 80)
        print(f"skip completed: {spec.name}")
        print("=" * 80)
        return

    command = [sys.executable, "-m", "src.run_training", *spec.to_cli_args()]
    command.extend([
        "--train-grad-accum-steps", str(args.train_grad_accum_steps),
        "--log-every-steps", str(args.log_every_steps),
        "--wandb", args.wandb,
        "--wandb-project", args.wandb_project,
        "--nproc-per-node", str(args.nproc_per_node),
        "--spectral-every-tokens", str(args.spectral_every_tokens),
        "--spectral-max-matrices", str(args.spectral_max_matrices),
        "--spectral-max-dim", str(args.spectral_max_dim),
    ])
    if args.eval_batch_size is not None:
        command.extend(["--eval-batch-size", str(args.eval_batch_size)])
    if args.eval_at_start:
        command.append("--eval-at-start")
    if args.wandb_entity:
        command.extend(["--wandb-entity", args.wandb_entity])
    if args.wandb_mode:
        command.extend(["--wandb-mode", args.wandb_mode])
    if args.runs_root:
        command.extend(["--runs-root", args.runs_root])
    if args.training_root:
        command.extend(["--training-root", args.training_root])
    if args.trainer_py:
        command.extend(["--trainer-py", args.trainer_py])
    if args.data_path:
        command.extend(["--data-path", args.data_path])
    if args.model_max_seq_len > 0:
        command.extend(["--model-max-seq-len", str(args.model_max_seq_len)])
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    specs = build_run_specs(
        train_token_budget=args.train_token_budget,
        eval_every_tokens=args.eval_every_tokens,
        eval_tokens=args.eval_tokens,
    )
    for spec in specs:
        launch(spec, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

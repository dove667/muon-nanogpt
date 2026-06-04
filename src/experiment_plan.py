import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from src.utils import ROOT, RUNS_ROOT


DEFAULT_GROUP = "fixed_t5"
DEFAULT_TRAIN_TOKEN_BUDGET = 100_000_000
DEFAULT_EVAL_EVERY_TOKENS = 10_000_000
DEFAULT_EVAL_TOKENS = 2_097_152
DEFAULT_TRAIN_GRAD_ACCUM_STEPS = 32
DEFAULT_SPECTRAL_EVERY_TOKENS = 10_000_000
DEFAULT_SPECTRAL_MAX_MATRICES = 5
DEFAULT_SPECTRAL_MAX_DIM = 1024
SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class RunSpec:
    orth: str
    group: str
    name: str
    lr_mul: float = 1.0
    seed: int = 0
    train_token_budget: int = DEFAULT_TRAIN_TOKEN_BUDGET
    eval_every_tokens: int = DEFAULT_EVAL_EVERY_TOKENS
    eval_tokens: int = DEFAULT_EVAL_TOKENS
    ns_t: int | None = None
    fast_steps: int | None = None
    stable_steps: int | None = None
    pe_t: int | None = None
    pe_lower_bound: str | None = None

    def to_cli_args(self) -> list[str]:
        args = [
            "--orth", self.orth,
            "--group", self.group,
            "--name", self.name,
            "--lr-mul", str(self.lr_mul),
            "--seed", str(self.seed),
            "--train-token-budget", str(self.train_token_budget),
            "--eval-every-tokens", str(self.eval_every_tokens),
            "--eval-tokens", str(self.eval_tokens),
        ]
        if self.orth == "manual":
            args.extend([
                "--ns-t", str(self.ns_t),
                "--fast-steps", str(self.fast_steps),
                "--stable-steps", str(self.stable_steps),
            ])
        elif self.orth == "polar_express":
            args.extend([
                "--pe-t", str(self.pe_t),
                "--pe-lower-bound", str(self.pe_lower_bound),
            ])
        return args


def run_completed(group: str, name: str, runs_root: str | None = None) -> bool:
    if runs_root is None:
        root = RUNS_ROOT
    else:
        root = Path(runs_root).expanduser()
        if not root.is_absolute():
            root = ROOT / root
    metrics_path = root / group / name / "metrics.jsonl"
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        return False
    last_line = metrics_path.read_text(encoding="utf-8").splitlines()[-1]
    return '"status": "completed"' in last_line or '"status":"completed"' in last_line


def build_run_specs(
    *,
    train_token_budget: int,
    eval_every_tokens: int,
    eval_tokens: int,
    group: str,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for seed in SEEDS:
        specs.extend([
            RunSpec("adamw", group, f"adamw_seed{seed}", seed=seed, train_token_budget=train_token_budget, eval_every_tokens=eval_every_tokens, eval_tokens=eval_tokens),
            RunSpec("vanilla", group, f"vanilla_seed{seed}", seed=seed, train_token_budget=train_token_budget, eval_every_tokens=eval_every_tokens, eval_tokens=eval_tokens),
            RunSpec("manual", group, f"manual_seed{seed}", seed=seed, train_token_budget=train_token_budget, eval_every_tokens=eval_every_tokens, eval_tokens=eval_tokens, ns_t=5, fast_steps=3, stable_steps=2),
            RunSpec("fast", group, f"fast_seed{seed}", seed=seed, train_token_budget=train_token_budget, eval_every_tokens=eval_every_tokens, eval_tokens=eval_tokens),
            RunSpec("polar_express", group, f"polar_express_seed{seed}", seed=seed, train_token_budget=train_token_budget, eval_every_tokens=eval_every_tokens, eval_tokens=eval_tokens, pe_t=5, pe_lower_bound="1e-3"),
        ])
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the fixed 15-run experiment plan.")
    parser.add_argument("--group", default=DEFAULT_GROUP)
    parser.add_argument("--skip-completed-runs", action="store_true")
    parser.add_argument("--train-token-budget", type=int, default=DEFAULT_TRAIN_TOKEN_BUDGET)
    parser.add_argument("--eval-every-tokens", type=int, default=DEFAULT_EVAL_EVERY_TOKENS)
    parser.add_argument("--eval-tokens", type=int, default=DEFAULT_EVAL_TOKENS)
    parser.add_argument("--train-grad-accum-steps", type=int, default=DEFAULT_TRAIN_GRAD_ACCUM_STEPS)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-at-start", action="store_true")
    parser.add_argument("--log-every-steps", type=int, default=20)
    parser.add_argument("--extension-steps", type=int, default=0)
    parser.add_argument("--val-loss-every-steps", type=int, default=0)
    parser.add_argument("--wandb", choices=["on", "off"], default="on")
    parser.add_argument("--wandb-project", default="muon-nanogpt")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode")
    parser.add_argument("--runs-root")
    parser.add_argument("--training-root")
    parser.add_argument("--trainer-py")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--data-path")
    parser.add_argument("--train-files")
    parser.add_argument("--val-files")
    parser.add_argument("--model-max-seq-len", type=int, default=0)
    parser.add_argument("--spectral-every-tokens", type=int, default=DEFAULT_SPECTRAL_EVERY_TOKENS)
    parser.add_argument("--spectral-max-matrices", type=int, default=DEFAULT_SPECTRAL_MAX_MATRICES)
    parser.add_argument("--spectral-max-dim", type=int, default=DEFAULT_SPECTRAL_MAX_DIM)
    return parser.parse_args()


def launch(spec: RunSpec, args: argparse.Namespace) -> None:
    if args.skip_completed_runs and run_completed(spec.group, spec.name, args.runs_root):
        print("=" * 80)
        print(f"skip completed: {spec.group}/{spec.name}")
        print("=" * 80)
        return

    command = [sys.executable, "-m", "src.run_training", *spec.to_cli_args()]
    command.extend([
        "--train-grad-accum-steps", str(args.train_grad_accum_steps),
        "--log-every-steps", str(args.log_every_steps),
        "--extension-steps", str(args.extension_steps),
        "--val-loss-every-steps", str(args.val_loss_every_steps),
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
    if args.train_files:
        command.extend(["--train-files", args.train_files])
    if args.val_files:
        command.extend(["--val-files", args.val_files])
    if args.model_max_seq_len > 0:
        command.extend(["--model-max-seq-len", str(args.model_max_seq_len)])
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    specs = build_run_specs(
        train_token_budget=args.train_token_budget,
        eval_every_tokens=args.eval_every_tokens,
        eval_tokens=args.eval_tokens,
        group=args.group,
    )
    for spec in specs:
        launch(spec, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

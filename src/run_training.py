import argparse
import math
import os
import subprocess
from pathlib import Path

from src.utils import ROOT, TRAINING_ROOT, RUNS_ROOT
FIXED_BATCH_TOKENS = 8 * 2048 * 8
FIXED_SEQ_LEN = 2048
TRAINER_PY = (TRAINING_ROOT / "train.py").resolve()


def compute_train_steps(train_token_budget: int) -> int:
    return math.ceil(train_token_budget / FIXED_BATCH_TOKENS)


def default_run_name(args: argparse.Namespace) -> str:
    if args.orth == "adamw":
        return f"adamw_seed{args.seed}"
    if args.orth == "vanilla":
        return f"vanilla_seed{args.seed}"
    if args.orth == "fast":
        return f"fast_seed{args.seed}"
    if args.orth == "manual":
        return f"manual_f{args.fast_steps}_s{args.stable_steps}_seed{args.seed}"
    if args.orth == "polar_express":
        return f"polar_express_l{args.pe_lower_bound}_seed{args.seed}"
    raise SystemExit(f"Unknown orth={args.orth}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch one configured training run.")

    parser.add_argument("--orth", choices=["adamw", "vanilla", "fast", "manual", "polar_express"], default="fast",
                        help="Orthogonalization strategy: adamw / vanilla / fast / manual / polar_express")
    parser.add_argument("--lr-mul", type=float, default=1.0,
                        help="Learning rate multiplier")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--name", default=None,
                        help="Run name (auto-generated if not set)")
    
    parser.add_argument("--train-token-budget", type=int, default=100_000_000,
                        help="Total token budget for training")
    parser.add_argument("--eval-every-tokens", type=int, default=2_000_000,
                        help="Run evaluation every N tokens")
    parser.add_argument("--eval-tokens", type=int, default=524_288,
                        help="Number of tokens per evaluation step")
    parser.add_argument("--train-grad-accum-steps", type=int, default=16,
                        help="Gradient accumulation steps for training")
    parser.add_argument("--eval-batch-size", type=int, default=None,
                        help="Batch size for evaluation (auto-computed if not set)")
    parser.add_argument("--eval-at-start", action="store_true",
                        help="Run evaluation at the start of training")
    parser.add_argument("--log-every-steps", type=int, default=20,
                        help="Log training metrics every N steps")
    # wandn args
    parser.add_argument("--wandb", choices=["on", "off"], default="on",
                        help="Enable or disable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="muon-nanogpt",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", default=None,
                        help="Weights & Biases entity name")
    parser.add_argument("--wandb-mode", default=None,
                        help="Weights & Biases mode (e.g. offline, dryrun)")
    
    parser.add_argument("--nproc-per-node", type=int, default=1,
                        help="Number of processes per node (for torchrun)")
    parser.add_argument("--data-path", default=None,
                        help="Path to training data directory")
    # Manual mode specific args
    parser.add_argument("--fast-steps", type=int, default=None,
                        help="Number of fast Newton-Schulz iterations (manual mode, defaults to 5)")
    parser.add_argument("--stable-steps", type=int, default=None,
                        help="Number of stable Newton-Schulz iterations (manual mode, auto-computed)")
    
    # polar express specific args
    parser.add_argument("--pe-lower-bound", default="1e-3",
                        help="Polar Express singular value lower bound")
    parser.add_argument("--pe-cushion", type=float, default=2e-2,
                        help="Polar Express cushion parameter")
    parser.add_argument("--pe-safety-factor", type=float, default=2e-2,
                        help="Polar Express safety factor")
    
    parser.add_argument("--model-max-seq-len", type=int, default=0,
                        help="Maximum sequence length (0 uses default 2048)")
    
    # spectral analysis args
    parser.add_argument("--spectral-every-tokens", type=int, default=10_000_000,
                        help="Run spectral analysis every N tokens")
    parser.add_argument("--spectral-max-matrices", type=int, default=5,
                        help="Maximum number of matrices for spectral analysis")
    parser.add_argument("--spectral-max-dim", type=int, default=1024,
                        help="Maximum dimension for spectral analysis")
    
    args = parser.parse_args()
    if args.fast_steps is None:
        args.fast_steps = 5 if args.orth == "manual" else args.fast_steps
    if args.stable_steps is None:
        args.stable_steps = max(5 - args.fast_steps, 0) if args.orth == "manual" else args.stable_steps
    if args.name is None:
        args.name = default_run_name(args)
    if args.eval_batch_size is None:
        args.eval_batch_size = args.train_grad_accum_steps * 2048
    return args


def prepare_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    run_dir = (RUNS_ROOT / args.name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    env["ORTH"] = args.orth
    env["LR_MUL"] = str(args.lr_mul)
    env["BASE_LR"] = "0.008" if args.orth == "adamw" else "0.023"
    env["SEED"] = str(args.seed)
    env["TRAIN_TOKEN_BUDGET"] = str(args.train_token_budget)
    env["TRAIN_STEPS"] = str(compute_train_steps(args.train_token_budget))
    env["TRAIN_GRAD_ACCUM_STEPS"] = str(args.train_grad_accum_steps)
    env["TRAIN_SEQ_LEN"] = str(FIXED_SEQ_LEN)
    env["EVAL_EVERY_TOKENS"] = str(args.eval_every_tokens)
    env["EVAL_TOKENS"] = str(args.eval_tokens)
    env["EVAL_BATCH_SIZE"] = str(args.eval_batch_size)
    env["EVAL_AT_START"] = "1" if args.eval_at_start else "0"
    env["LOG_EVERY_STEPS"] = str(args.log_every_steps)
    env["EXTENSION_STEPS"] = "0"
    env["VAL_LOSS_EVERY_STEPS"] = "0"

    env["WANDB"] = "1" if args.wandb == "on" else "0"
    env["WANDB_PROJECT"] = args.wandb_project
    env["WANDB_NAME"] = args.name
    env.pop("WANDB_GROUP", None)

    env["RUNS_ROOT"] = str(RUNS_ROOT)
    env["RUN_DIR"] = str(run_dir)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".torchinductor"))
    env["PYTHONPATH"] = str(ROOT) + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")

    if args.wandb_entity:
        env["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_mode:
        env["WANDB_MODE"] = args.wandb_mode
    elif args.wandb == "on" and not env.get("WANDB_API_KEY") and not Path.home().joinpath(".netrc").exists():
        env["WANDB_MODE"] = "offline"

    if args.data_path:
        env["DATA_PATH"] = args.data_path

    if args.orth == "manual":
        env["FAST_STEPS"] = str(args.fast_steps)
        env["STABLE_STEPS"] = str(args.stable_steps)
    elif args.orth == "polar_express":
        env["PE_LOWER_BOUND"] = str(args.pe_lower_bound)
        env["PE_CUSHION"] = str(args.pe_cushion)
        env["PE_SAFETY_FACTOR"] = str(args.pe_safety_factor)

    if args.model_max_seq_len > 0:
        env["MODEL_MAX_SEQ_LEN"] = str(args.model_max_seq_len)
    env["SPECTRAL_EVERY_TOKENS"] = str(args.spectral_every_tokens)
    env["SPECTRAL_MAX_MATRICES"] = str(args.spectral_max_matrices)
    env["SPECTRAL_MAX_DIM"] = str(args.spectral_max_dim)
    return env


def run() -> int:
    args = parse_args()
    env = prepare_env(args)
    print("=" * 80)
    print(f"run: {args.name}")
    print(f"orth: {args.orth}")
    print(f"train_token_budget: {args.train_token_budget}")
    print(f"train_steps: {env['TRAIN_STEPS']}")
    print(f"grad_accum: {args.train_grad_accum_steps}")
    print(f"eval_every_tokens: {args.eval_every_tokens}")
    print(f"eval_tokens: {args.eval_tokens}")
    print("=" * 80)
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        str(TRAINER_PY),
    ]
    with open(Path(env["RUN_DIR"]) / "console.log", "a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=TRAINING_ROOT,
            env={**env, "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0")},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


if __name__ == "__main__":
    raise SystemExit(run())

import argparse
import math
import os
import sys
from pathlib import Path

import torch
from torch import nn

from model import GPT, setup_model_runtime
from optim import TrainingManager
from orthogonalization import build_orthogonalizer_config_from_env
from polar import make_polar_express
from run_support import (
    LoopConfig,
    RunLogger,
    nvidia_smi_output,
    run_training_loop,
    setup_device,
)
from schedule import Hyperparameters, TrainingSchedule, default_training_stages

ROOT = Path(__file__).resolve().parents[2]
FIXED_BATCH_TOKENS = 8 * 2048 * 8
FIXED_SEQ_LEN = 2048


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
    parser.add_argument("--wandb", choices=["on", "off"], default="on",
                        help="Enable or disable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="muon-nanogpt",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", default=None,
                        help="Weights & Biases entity name")
    parser.add_argument("--wandb-mode", default=None,
                        help="Weights & Biases mode (e.g. offline, dryrun)")
    parser.add_argument("--runs-root", default=None,
                        help="Root directory for run outputs")
    parser.add_argument("--data-path", default=None,
                        help="Path to training data directory")
    parser.add_argument("--fast-steps", type=int, default=None,
                        help="Number of fast Newton-Schulz iterations (manual mode, defaults to 5)")
    parser.add_argument("--stable-steps", type=int, default=None,
                        help="Number of stable Newton-Schulz iterations (manual mode, auto-computed)")
    parser.add_argument("--pe-lower-bound", default="1e-3",
                        help="Polar Express singular value lower bound")
    parser.add_argument("--pe-cushion", type=float, default=2e-2,
                        help="Polar Express cushion parameter")
    parser.add_argument("--pe-safety-factor", type=float, default=2e-2,
                        help="Polar Express safety factor")
    parser.add_argument("--model-max-seq-len", type=int, default=0,
                        help="Maximum sequence length (0 uses default 2048)")
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


def build_model(
    args: Hyperparameters,
    training_stages,
    device: torch.device,
) -> nn.Module:
    model_max_seq_len = int(
        os.environ.get(
            "MODEL_MAX_SEQ_LEN",
            max(
                args.val_batch_size,
                max(stage.train_max_seq_len for stage in training_stages),
            ),
        )
    )
    model = GPT(
        vocab_size=50257,
        num_layers=11,
        num_heads=6,
        head_dim=128,
        model_dim=768,
        max_seq_len=model_max_seq_len,
    ).to(device=device)
    for module in model.modules():
        if isinstance(module, (nn.Embedding, nn.Linear)):
            module.weight.data = module.weight.data.bfloat16()
    model.attn_gate_bank.data = model.attn_gate_bank.data.bfloat16()
    model.ve_gate_bank.data = model.ve_gate_bank.data.bfloat16()
    model.qk_bank.data = model.qk_bank.data.bfloat16()
    model.vo_bank.data = model.vo_bank.data.bfloat16()
    model.mlp_bank.data = model.mlp_bank.data.bfloat16()
    return model


def run_worker(args: argparse.Namespace) -> None:
    runs_root = Path(args.runs_root).expanduser() if args.runs_root else ROOT / "runs"
    if not runs_root.is_absolute():
        runs_root = ROOT / runs_root
    run_dir = (runs_root / args.name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    os.environ["ORTH"] = args.orth
    os.environ["LR_MUL"] = str(args.lr_mul)
    os.environ["BASE_LR"] = "0.008" if args.orth == "adamw" else "0.023"
    os.environ["SEED"] = str(args.seed)
    os.environ["TRAIN_TOKEN_BUDGET"] = str(args.train_token_budget)
    os.environ["TRAIN_STEPS"] = str(compute_train_steps(args.train_token_budget))
    os.environ["TRAIN_GRAD_ACCUM_STEPS"] = str(args.train_grad_accum_steps)
    os.environ["TRAIN_SEQ_LEN"] = str(FIXED_SEQ_LEN)
    os.environ["EVAL_EVERY_TOKENS"] = str(args.eval_every_tokens)
    os.environ["EVAL_TOKENS"] = str(args.eval_tokens)
    os.environ["EVAL_BATCH_SIZE"] = str(args.eval_batch_size)
    os.environ["EVAL_AT_START"] = "1" if args.eval_at_start else "0"
    os.environ["LOG_EVERY_STEPS"] = str(args.log_every_steps)
    os.environ["EXTENSION_STEPS"] = "0"
    os.environ["VAL_LOSS_EVERY_STEPS"] = "0"
    os.environ["WANDB"] = "1" if args.wandb == "on" else "0"
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_NAME"] = args.name
    os.environ.pop("WANDB_GROUP", None)
    os.environ["RUNS_ROOT"] = str(runs_root)
    os.environ["RUN_DIR"] = str(run_dir)
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".torchinductor"))
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
    elif args.wandb == "on" and not os.environ.get("WANDB_API_KEY") and not Path.home().joinpath(".netrc").exists():
        os.environ["WANDB_MODE"] = "offline"
    if args.data_path:
        os.environ["DATA_PATH"] = args.data_path
    if args.orth == "manual":
        os.environ["FAST_STEPS"] = str(args.fast_steps)
        os.environ["STABLE_STEPS"] = str(args.stable_steps)
    elif args.orth == "polar_express":
        os.environ["PE_LOWER_BOUND"] = str(args.pe_lower_bound)
        os.environ["PE_CUSHION"] = str(args.pe_cushion)
        os.environ["PE_SAFETY_FACTOR"] = str(args.pe_safety_factor)
    if args.model_max_seq_len > 0:
        os.environ["MODEL_MAX_SEQ_LEN"] = str(args.model_max_seq_len)
    os.environ["SPECTRAL_EVERY_TOKENS"] = str(args.spectral_every_tokens)
    os.environ["SPECTRAL_MAX_MATRICES"] = str(args.spectral_max_matrices)
    os.environ["SPECTRAL_MAX_DIM"] = str(args.spectral_max_dim)

    grad_accum_steps = args.train_grad_accum_steps
    train_steps = compute_train_steps(args.train_token_budget)

    print("=" * 80)
    print(f"run: {args.name}")
    print(f"orth: {args.orth}")
    print(f"train_token_budget: {args.train_token_budget}")
    print(f"train_steps: {train_steps}")
    print(f"grad_accum: {grad_accum_steps}")
    print(f"eval_every_tokens: {args.eval_every_tokens}")
    print(f"eval_tokens: {args.eval_tokens}")
    print(f"device: cuda:0")
    print("=" * 80)

    device, _ = setup_device(base_seed=args.seed)
    logger = None

    try:
        orth_state = build_orthogonalizer_config_from_env()
        polar_express = make_polar_express(
            coeff_schedule=orth_state.coeff_schedule,
            norm_factor=orth_state.norm_factor,
        )

        hparams = Hyperparameters()
        hparams.val_batch_size = args.eval_batch_size

        training_stages = default_training_stages()
        training_schedule = TrainingSchedule(
            training_stages,
            hparams.num_scheduled_iterations,
            hparams.num_extension_iterations,
            device=device,
        )

        setup_model_runtime(
            args_value=hparams,
            grad_accum_steps_value=grad_accum_steps,
            device_value=device,
        )

        logger = RunLogger(
            args=hparams,
            orth_config=orth_state.to_record(),
            orth_mode=orth_state.orth_mode,
            orth_schedule_name=orth_state.schedule_name,
            lr_mul=orth_state.lr_mul,
            train_token_budget=args.train_token_budget,
            eval_every_tokens=args.eval_every_tokens,
            grad_accum_steps=grad_accum_steps,
            device=device,
        )

        logger.print0("=" * 100)
        logger.print0(f"Running Python {sys.version}")
        logger.print0(
            f"Running PyTorch {torch.__version__} compiled for CUDA {torch.version.cuda}"
        )
        logger.print0(nvidia_smi_output())
        logger.print0("=" * 100)

        model = build_model(hparams, training_stages, device)

        training_manager = TrainingManager(
            model,
            device=device,
            args=hparams,
            training_schedule=training_schedule,
            lr_mul=orth_state.lr_mul,
            orth_mode=orth_state.orth_mode,
            polar_express=polar_express,
            grad_accum_steps=grad_accum_steps,
        )
        loop_config = LoopConfig.from_env(
            orth_mode=orth_state.orth_mode,
            orth_schedule_name=orth_state.schedule_name,
            polar_express_coeffs=orth_state.coeff_schedule,
            orth_norm_factor=orth_state.norm_factor,
        )

        run_training_loop(
            model=model,
            training_manager=training_manager,
            args=hparams,
            training_stages=training_stages,
            training_schedule=training_schedule,
            grad_accum_steps=grad_accum_steps,
            logger=logger,
            loop_config=loop_config,
        )
    finally:
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    run_worker(parse_args())

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from model import GPT
from optim import TrainingManager
from orthogonalization import OrthogonalizerConfig, build_orthogonalizer_config
from polar import make_polar_express
from run_support import (
    RunLogger,
    run_training_loop,
    setup_device,
)
from utils import resolve_data_files

FIXED_BATCH_TOKENS = 8 * 2048 * 8
FIXED_SEQ_LEN = 2048


@dataclass(slots=True)
class Hyperparameters:
    val_tokens: int = 524288
    val_batch_size: int = 2048
    num_scheduled_iterations: int = 1440
    run_id: str = ""
    save_checkpoint: bool = False
    bigram_vocab_size: int = 50304 * 5
    val_loss_every: int = 0
    train_files: str = ""
    val_files: str = ""


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


def dispatch_orth_config(args: argparse.Namespace) -> tuple[OrthogonalizerConfig, object, float]:
    orth_config = build_orthogonalizer_config(
        orth_mode=args.orth,
        fast_steps=args.fast_steps or 5,
        stable_steps=args.stable_steps or (max(5 - (args.fast_steps or 5), 0)),
        pe_lower_bound_raw=args.pe_lower_bound,
        pe_cushion=args.pe_cushion,
        pe_safety_factor=args.pe_safety_factor,
        lr_mul=args.lr_mul,
    )
    polar_express = make_polar_express(
        coeff_schedule=orth_config.coeff_schedule,
        norm_factor=orth_config.norm_factor,
    )
    base_lr = 0.008 if args.orth == "adamw" else 0.023
    return orth_config, polar_express, base_lr


def build_model(
    args: Hyperparameters,
    device: torch.device,
    model_max_seq_len: int,
) -> nn.Module:
    if model_max_seq_len <= 0:
        model_max_seq_len = max(args.val_batch_size, FIXED_SEQ_LEN)
    model = GPT(
        vocab_size=50257,
        num_layers=11,
        num_heads=6,
        head_dim=128,
        model_dim=768,
        max_seq_len=model_max_seq_len,
        device=device,
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


def main(args: argparse.Namespace) -> None:
    run_dir = (Path(__file__).resolve().parents[2] / "runs" / args.name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    grad_accum_steps = args.train_grad_accum_steps
    train_steps = math.ceil(args.train_token_budget / FIXED_BATCH_TOKENS)

    print("=" * 80)
    print(f"run: {args.name}")
    print(f"orth: {args.orth}")
    print("=" * 80)

    device = setup_device(base_seed=args.seed)
    orth_config, polar_express, base_lr = dispatch_orth_config(args)

    train_files, val_files = resolve_data_files(args.data_path)

    hparams = Hyperparameters(
        train_files=train_files,
        val_files=val_files,
        val_tokens=args.eval_tokens,
        val_batch_size=args.eval_batch_size,
        num_scheduled_iterations=train_steps,
    )

    logger = RunLogger(
        args=hparams,
        orth_config=orth_config,
        base_lr=base_lr,
        train_token_budget=args.train_token_budget,
        eval_every_tokens=args.eval_every_tokens,
        grad_accum_steps=grad_accum_steps,
        device=device,
        seed=args.seed,
        seq_len=FIXED_SEQ_LEN,
        run_dir=run_dir,
        run_name=args.name,
    )

    print("=" * 100)
    print(f"Running Python {sys.version}")
    print(f"Running PyTorch {torch.__version__} compiled for CUDA {torch.version.cuda}")
    print("=" * 100)

    model = build_model(hparams, device, args.model_max_seq_len)

    training_manager = TrainingManager(
        model,
        device=device,
        args=hparams,
        total_steps=train_steps,
        lr_mul=orth_config.lr_mul,
        orth_mode=orth_config.orth_mode,
        polar_express=polar_express,
        grad_accum_steps=grad_accum_steps,
    )

    run_training_loop(
        model=model,
        training_manager=training_manager,
        args=hparams,
        train_steps=train_steps,
        grad_accum_steps=grad_accum_steps,
        logger=logger,
        log_every_steps=args.log_every_steps,
        eval_every_tokens=args.eval_every_tokens,
        eval_at_start=args.eval_at_start,
        spectral_every_tokens=args.spectral_every_tokens,
        spectral_max_matrices=args.spectral_max_matrices,
        spectral_max_dim=args.spectral_max_dim,
        polar_express_coeffs=tuple(tuple(c) for c in orth_config.coeff_schedule),
        orth_norm_factor=orth_config.norm_factor,
    )


if __name__ == "__main__":
    main(parse_args())

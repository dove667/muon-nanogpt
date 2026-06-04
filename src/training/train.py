import argparse
import math
import sys
from pathlib import Path

import torch
from torch import nn

from model import GPT
from optim import TrainingManager
from orth import OrthogonalizerConfig, build_orthogonalizer_config, make_polar_express
from run_support import (
    Logger,
    run_training_loop,
    setup_device,
)
from utils import resolve_data_files
from config import TRAINING, MODEL, OPTIMIZER, get_orthogonalization


def default_run_name(orth: str, seed: int) -> str:
    orth_cfg = get_orthogonalization()
    if orth == "adamw":
        return f"adamw_seed{seed}"
    if orth == "vanilla":
        return f"vanilla_seed{seed}"
    if orth == "fast":
        return f"fast_seed{seed}"
    if orth == "manual":
        return f"manual_f{orth_cfg.fast_steps}_s{orth_cfg.stable_steps}_seed{seed}"
    if orth == "polar_express":
        return f"polar_express_l{orth_cfg.pe_lower_bound}_seed{seed}"
    raise SystemExit(f"Unknown orth={orth}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch one configured training run.")
    parser.add_argument("--orth", choices=["adamw", "vanilla", "fast", "manual", "polar_express"], default="fast",
                        help="Orthogonalization strategy")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--name", default=None,
                        help="Run name (auto-generated if not set)")
    parser.add_argument("--data-path", required=True,
                        help="Path to training data directory")
    args = parser.parse_args()
    if args.name is None:
        args.name = default_run_name(args.orth, args.seed)
    return args


def dispatch_orth_config(orth: str) -> tuple[OrthogonalizerConfig, object, float]:
    orth_cfg = get_orthogonalization()
    orth_config = build_orthogonalizer_config(
        orth_mode=orth,
        fast_steps=int(orth_cfg.fast_steps),
        stable_steps=int(orth_cfg.stable_steps),
        pe_lower_bound_raw=orth_cfg.pe_lower_bound,
        pe_cushion=float(orth_cfg.pe_cushion),
        pe_safety_factor=float(orth_cfg.pe_safety_factor),
        lr_mul=float(OPTIMIZER.lr_mul),
    )
    polar_express = make_polar_express(
        coeff_schedule=orth_config.coeff_schedule,
        norm_factor=orth_config.norm_factor,
    )
    base_lr = OPTIMIZER.base_lr_adamw if orth == "adamw" else OPTIMIZER.base_lr_muon
    return orth_config, polar_express, base_lr


def build_model(
    val_batch_size: int,
    device: torch.device,
    model_max_seq_len: int,
) -> nn.Module:
    if model_max_seq_len <= 0:
        model_max_seq_len = max(val_batch_size, TRAINING.seq_len)
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


def main() -> None:
    args = parse_args()

    run_dir = (Path(__file__).resolve().parents[2] / "runs" / args.name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    orth_config, polar_express, base_lr = dispatch_orth_config(args.orth)
    train_files, val_files = resolve_data_files(args.data_path)

    train_steps = math.ceil(TRAINING.train_token_budget / TRAINING.batch_tokens)

    logger = Logger(
        run_name=args.name,
        seed=args.seed,
        base_lr=base_lr,
        train_token_budget=TRAINING.train_token_budget,
        orth_config=orth_config,
        run_dir=run_dir,
    )

    val_batch_size = TRAINING.eval_batch_size
    if val_batch_size is None:
        val_batch_size = TRAINING.grad_accum_steps * TRAINING.seq_len

    print("=" * 80)
    print(f"run: {args.name}")
    print(f"orth: {args.orth}")
    print("=" * 80)

    device = setup_device(base_seed=args.seed)

    print("=" * 100)
    print(f"Running Python {sys.version}")
    print(f"Running PyTorch {torch.__version__} compiled for CUDA {torch.version.cuda}")
    print("=" * 100)

    model = build_model(val_batch_size, device, TRAINING.model_max_seq_len)

    training_manager = TrainingManager(
        model,
        device=device,
        total_steps=train_steps,
        lr_mul=orth_config.lr_mul,
        orth_mode=orth_config.orth_mode,
        polar_express=polar_express,
        grad_accum_steps=TRAINING.grad_accum_steps,
    )

    run_training_loop(
        model=model,
        training_manager=training_manager,
        train_files=train_files,
        val_files=val_files,
        val_tokens=TRAINING.eval_tokens,
        val_batch_size=val_batch_size,
        bigram_vocab_size=MODEL.bigram_vocab_size,
        train_steps=train_steps,
        grad_accum_steps=TRAINING.grad_accum_steps,
        logger=logger,
        log_every_steps=TRAINING.log_every_steps,
        eval_every_tokens=TRAINING.eval_every_tokens,
        eval_at_start=TRAINING.eval_at_start,
        val_loss_every=TRAINING.val_loss_every,
        spectral_every_tokens=TRAINING.spectral_every_tokens,
        spectral_max_matrices=TRAINING.spectral_max_matrices,
        spectral_max_dim=TRAINING.spectral_max_dim,
        polar_express_coeffs=tuple(tuple(c) for c in orth_config.coeff_schedule),
        orth_norm_factor=orth_config.norm_factor,
    )


if __name__ == "__main__":
    main()
import argparse
import math
import sys
from pathlib import Path

import torch
from torch import nn

from model import GPT
from optim import TrainingManager
from orth import OrthogonalizerConfig, build_orthogonalizer_config, make_polar_express
from run_support import Logger, run_training_loop, setup_device
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


def _resolve_data_files(data_path: str) -> tuple[str, str]:
    dp = Path(data_path)
    train_pattern = "fineweb_train_*.bin"
    val_pattern = "fineweb_val_*.bin"
    if not any(dp.glob(train_pattern)):
        raise FileNotFoundError(f"No training files matching {train_pattern} in {dp}")
    if not any(dp.glob(val_pattern)):
        raise FileNotFoundError(f"No validation files matching {val_pattern} in {dp}")
    return str(dp / train_pattern), str(dp / val_pattern)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch one configured training run.")
    parser.add_argument("--orth", choices=["adamw", "vanilla", "fast", "manual", "polar_express"], default="fast",
                        help="Orthogonalization strategy")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--data-path", required=True,
                        help="Path to training data directory")
    args = parser.parse_args()
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


def build_model(device: torch.device) -> nn.Module:
    model_max_seq_len = TRAINING.seq_len
    model = GPT(
        vocab_size=int(MODEL.vocab_size),
        num_layers=int(MODEL.num_layers),
        num_heads=int(MODEL.num_heads),
        head_dim=int(MODEL.head_dim),
        model_dim=int(MODEL.model_dim),
        max_seq_len=model_max_seq_len,
        mlp_ratio=int(MODEL.mlp_ratio),
    ).to(device=device, dtype=torch.bfloat16)
    return model


def main() -> None:
    args = parse_args()

    run_dir = (Path(__file__).resolve().parents[2] / "runs" / args.name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    orth_config, polar_express, base_lr = dispatch_orth_config(args.orth)
    train_files, val_files = _resolve_data_files(args.data_path)

    train_steps = math.ceil(TRAINING.train_token_budget / TRAINING.batch_tokens)

    logger = Logger(
        run_name=args.name,
        seed=args.seed,
        base_lr=base_lr,
        train_token_budget=TRAINING.train_token_budget,
        orth_config=orth_config,
        run_dir=run_dir,
    )

    print("=" * 80)
    print(f"run: {args.name}")
    print(f"orth: {args.orth}")
    print("=" * 80)

    device = setup_device(base_seed=args.seed)

    print("=" * 100)
    print(f"Running Python {sys.version}")
    print(f"Running PyTorch {torch.__version__} compiled for CUDA {torch.version.cuda}")
    print("=" * 100)

    model = build_model(device)

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
        train_steps=train_steps,
        grad_accum_steps=TRAINING.grad_accum_steps,
        logger=logger,
        log_every_steps=TRAINING.log_every_steps,
        eval_every_tokens=TRAINING.eval_every_tokens,
        spectral_every_tokens=TRAINING.spectral_every_tokens,
        spectral_max_matrices=TRAINING.spectral_max_matrices,
        spectral_max_dim=TRAINING.spectral_max_dim,
        polar_express_coeffs=tuple(tuple(c) for c in orth_config.coeff_schedule),
        orth_norm_factor=orth_config.norm_factor,
    )


if __name__ == "__main__":
    main()

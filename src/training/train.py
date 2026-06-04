import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn

from src.model.gpt import build_model
from src.optim import build_optimizer, step_optimizer
from src.optim import build_coeff_schedule, make_polar_express, orth_norm_factor, orth_record
from src.config import TRAINING, OPTIMIZER, get_orthogonalization
from src.training.utils import FIXED_SEED, default_run_name, resolve_data_path, setup_device, primary_lr
from src.training.logger import Logger
from src.data import data_generator
from src.training.metrics import collect_spectral_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch one configured training run.")
    parser.add_argument("--orth", choices=["adamw", "vanilla", "fast", "manual", "polar_express"],
                        default="fast", help="Orthogonalization strategy")
    parser.add_argument("--data-path", required=True, help="Path to training data directory")
    parser.add_argument("--benchmark", action="store_true",
                        help="Enable wall-clock timing measurements (adds cuda synchronize)")
    parser.add_argument("--spectral", action="store_true",
                        help="Enable spectral metric collection on optimizer states")
    args = parser.parse_args()
    args.name = default_run_name(args.orth)
    return args


def dispatch_orth_state(orth: str) -> tuple[list[tuple[float, float, float]], float, dict[str, object], float]:
    orth_cfg = get_orthogonalization()
    coeff_schedule = build_coeff_schedule(
        orth,
        fast_steps=int(orth_cfg.fast_steps),
        stable_steps=int(orth_cfg.stable_steps),
        pe_lower_bound_raw=orth_cfg.pe_lower_bound,
        pe_cushion=float(orth_cfg.pe_cushion),
        pe_safety_factor=float(orth_cfg.pe_safety_factor),
    )
    norm_factor = orth_norm_factor(orth, float(orth_cfg.pe_safety_factor))
    record = orth_record(
        orth,
        coeff_schedule,
        fast_steps=int(orth_cfg.fast_steps),
        stable_steps=int(orth_cfg.stable_steps),
        pe_lower_bound_raw=orth_cfg.pe_lower_bound,
        pe_cushion=float(orth_cfg.pe_cushion),
        pe_safety_factor=float(orth_cfg.pe_safety_factor),
        lr_mul=float(OPTIMIZER.lr_mul),
    )
    base_lr = float(OPTIMIZER.base_lr_adamw if orth == "adamw" else OPTIMIZER.base_lr_muon)
    return coeff_schedule, norm_factor, record, base_lr


def run_validation(
    *,
    model: nn.Module,
    device: torch.device,
    val_data_path: str,
    logger: Logger,
    step: int,
    train_steps: int,
    global_train_tokens: int,
) -> None:
    val_tokens = int(TRAINING.eval_tokens)
    model.eval()

    assert val_tokens % TRAINING.tokens_per_step == 0
    val_steps = TRAINING.grad_accum_steps * val_tokens // TRAINING.tokens_per_step
    val_loader = data_generator(
        val_data_path, TRAINING.tokens_per_step, TRAINING.seq_len, TRAINING.grad_accum_steps,
    )
    val_loss = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for _ in range(val_steps):
            inputs, targets = next(val_loader)
            val_loss += model(inputs, targets).mean()
    val_loss /= val_steps
    del val_loader

    record: dict = {
        "val/loss": float(val_loss),
        "val/ppl": float(math.exp(min(float(val_loss), TRAINING.val_ppl_max))),
        "val/global_train_tokens": int(global_train_tokens),
        "train/step": int(step),
    }

    print(f"step:{step}/{train_steps} val_loss:{val_loss:.4f}")
    logger.log_metric(record)
    model.train()


def run_training_loop(
    *,
    model: nn.Module,
    optimizer,
    device: torch.device,
    train_data_path: str,
    val_data_path: str,
    logger: Logger,
    polar_express_coeffs: tuple[tuple[float, float, float], ...],
    orth_norm_factor: float,
    benchmark: bool = False,
    spectral: bool = False,
) -> None:
    train_steps = math.ceil(TRAINING.train_token_budget / TRAINING.tokens_per_step)
    train_loader = data_generator(
        train_data_path, TRAINING.tokens_per_step, TRAINING.seq_len, TRAINING.grad_accum_steps,
    )

    global_train_tokens = 0
    next_eval_tokens = TRAINING.eval_interval_tokens
    next_spectral_tokens = TRAINING.spectral_interval_tokens

    if benchmark:
        torch.cuda.synchronize()
        t_wall = time.perf_counter()

    for step in range(train_steps + 1):
        last_step = step == train_steps

        should_eval = last_step or (
            TRAINING.eval_interval_tokens > 0 and global_train_tokens >= next_eval_tokens
        )
        if should_eval:
            run_validation(
                model=model, device=device, val_data_path=val_data_path,
                logger=logger, step=step, train_steps=train_steps,
                global_train_tokens=global_train_tokens,
            )
            if TRAINING.eval_interval_tokens > 0:
                while next_eval_tokens <= global_train_tokens:
                    next_eval_tokens += TRAINING.eval_interval_tokens

        if last_step:
            break

        should_log = TRAINING.log_every_steps > 0 and (
            (step + 1) % TRAINING.log_every_steps == 0 or step < 3
        )
        train_loss_accum = 0.0

        grad_scale = 1.0 / TRAINING.grad_accum_steps
        for _ in range(TRAINING.grad_accum_steps):
            inputs, targets = next(train_loader)
            loss = model(inputs, targets).sum() * grad_scale
            if should_log:
                train_loss_accum += float(loss.detach())
            loss.backward()

        step_optimizer(optimizer, step=step, total_steps=train_steps)
        global_train_tokens += TRAINING.tokens_per_step

        print(f"step:{step + 1}/{train_steps}")

        if should_log:
            record: dict = {
                "train/loss_raw": float(train_loss_accum),
                "train/lr": float(primary_lr(optimizer)),
                "train/tokens": int(global_train_tokens),
                "train/step": int(step + 1),
            }
            logger.log_metric(record)

        if spectral and TRAINING.spectral_interval_tokens > 0 and global_train_tokens >= next_spectral_tokens:
            spectral_summary, _ = collect_spectral_metrics(
                optimizer,
                global_train_tokens=global_train_tokens,
                master_process=True,
                num_matrices=TRAINING.spectral_num_matrices,
                svd_dim_cap=TRAINING.spectral_dim_cap,
                coeffs=polar_express_coeffs,
                norm_factor=orth_norm_factor,
            )
            if spectral_summary:
                logger.log_spectral(spectral_summary)
            while next_spectral_tokens <= global_train_tokens:
                next_spectral_tokens += TRAINING.spectral_interval_tokens

    if benchmark:
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - t_wall
        logger.log_metric({"benchmark/wall_clock_s": float(wall_s)})

    peak_mb = int(torch.cuda.max_memory_allocated() // 1024 // 1024)
    logger.log_metric({
        "memory/peak_allocated_mb": peak_mb,
        "train/final_tokens": int(global_train_tokens),
        "status": "completed",
    })

    print(f"peak memory allocated: {peak_mb} MiB")


def main() -> None:
    args = parse_args()

    run_dir = (Path(__file__).resolve().parents[2] / "runs" / args.name).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    coeff_schedule, norm_factor, orth_record_data, base_lr = dispatch_orth_state(args.orth)
    polar_express = make_polar_express(coeff_schedule=coeff_schedule, norm_factor=norm_factor)
    train_data_path, val_data_path = resolve_data_path(args.data_path)

    logger = Logger(
        run_name=args.name,
        seed=FIXED_SEED,
        base_lr=base_lr,
        orth_record=orth_record_data,
        run_dir=run_dir,
    )

    print("=" * 80)
    print(f"run:      {args.name}")
    print(f"orth:     {args.orth}")
    if args.benchmark:
        print(f"mode:     benchmark")
    elif args.spectral:
        print(f"mode:     spectral")
    print("=" * 80)

    device = setup_device(base_seed=FIXED_SEED)

    print(f"PyTorch {torch.__version__}  CUDA {torch.version.cuda}")
    print(f"Python  {sys.version}")

    model = build_model(device)
    optimizer = build_optimizer(model, orth_mode=args.orth, polar_express=polar_express)

    run_training_loop(
        model=model,
        optimizer=optimizer,
        device=device,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        logger=logger,
        polar_express_coeffs=tuple(tuple(c) for c in coeff_schedule),
        orth_norm_factor=norm_factor,
        benchmark=args.benchmark,
        spectral=args.spectral,
    )


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path
import time
import math
import time
from pathlib import Path

import torch
from torch import nn

from src.model.gpt import build_model
from src.optim import build_optimizer, step_optimizer
from src.optim import build_coeff_schedule, make_polar_express, orth_norm_factor, orth_record
from src.config import TRAINING, MODEL, OPTIMIZER, get_orthogonalization
from src.training.utils import default_run_name, resolve_data_path, setup_device
from src.training.logger import Logger
from src.data import data_generator
from src.training.metrics import collect_spectral_metrics, current_grad_norm



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
    val_data: str,
    logger: Logger,
    step: int,
    train_steps: int,
    training_time_ms: float,
    wall_start_time: float,
    global_train_tokens: int,
) -> None:
    val_batch_tokens = TRAINING.batch_tokens
    val_tokens = int(TRAINING.eval_tokens)
    eval_start_time = time.perf_counter()
    model.eval()
    assert val_tokens % val_batch_tokens == 0
    val_steps = TRAINING.grad_accum_steps * val_tokens // val_batch_tokens
    val_loader = data_generator(
        val_data, val_batch_tokens, TRAINING.seq_len, TRAINING.grad_accum_steps,
    )
    val_loss = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for _ in range(val_steps):
            inputs, targets = next(val_loader)
            val_loss += model(inputs, targets).mean()
    val_loss /= val_steps
    del val_loader
    eval_time_s = time.perf_counter() - eval_start_time
    print(
        f"step:{step}/{train_steps} val_loss:{val_loss:.4f} "
        f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms",
    )
    logger.log_metric(
        {
            "val/loss": float(val_loss),
            "val/ppl": float(math.exp(min(float(val_loss), TRAINING.val_ppl_clip))),
            "val/tokens": int(val_tokens),
            "val/time_s": float(eval_time_s),
            "val/tokens_per_sec": float(val_tokens / max(eval_time_s, 1e-9)),
            "val/global_train_tokens": int(global_train_tokens),
            "val/global_wall_time_s": float(time.perf_counter() - wall_start_time),
            "train/time_s": float(training_time_ms / 1000),
            "train/tokens": int(global_train_tokens),
            "train/step": int(step),
            "wall/elapsed_s": float(time.perf_counter() - wall_start_time),
        },
    )
    model.train()


def run_training_loop(
    *,
    model: nn.Module,
    optimizer,
    device: torch.device,
    train_data: str,
    val_data: str,
    logger: Logger,
    polar_express_coeffs: tuple[tuple[float, float, float], ...],
    orth_norm_factor: float,
) -> None:
    train_steps = math.ceil(TRAINING.train_token_budget / TRAINING.batch_tokens)
    train_loader = data_generator(
        train_data, TRAINING.batch_tokens, TRAINING.seq_len, TRAINING.grad_accum_steps,
    )

    training_time_ms = 0.0
    global_train_tokens = 0
    next_eval_tokens = TRAINING.eval_every_tokens
    next_spectral_tokens = TRAINING.spectral_every_tokens

    wall_start_time = time.perf_counter()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(train_steps + 1):
        last_step = step == train_steps

        should_eval_by_tokens = (
            TRAINING.eval_every_tokens > 0
            and global_train_tokens >= next_eval_tokens
        )
        if last_step or should_eval_by_tokens:
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            run_validation(
                model=model,
                device=device,
                val_data=val_data,
                logger=logger,
                step=step,
                train_steps=train_steps,
                training_time_ms=training_time_ms,
                wall_start_time=wall_start_time,
                global_train_tokens=global_train_tokens,
            )
            if TRAINING.eval_every_tokens > 0:
                while next_eval_tokens <= global_train_tokens:
                    next_eval_tokens += TRAINING.eval_every_tokens
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        should_log = TRAINING.log_every_steps > 0 and (
            (step + 1) % TRAINING.log_every_steps == 0 or step < 3
        )
        train_loss_accum = 0.0
        if should_log:
            torch.cuda.synchronize()
            step_wall_t0 = time.perf_counter()

        grad_scale = 1.0 / TRAINING.grad_accum_steps
        for _ in range(TRAINING.grad_accum_steps):
            inputs, targets = train_loader.send(None)
            loss = model(inputs, targets).sum() * grad_scale
            if should_log:
                train_loss_accum += float(loss.detach())
            loss.backward()

        if should_log:
            grad_norm_value = current_grad_norm(model)
            del loss
            torch.cuda.synchronize()

        step_optimizer(optimizer, step=step, total_steps=train_steps)
        global_train_tokens += TRAINING.batch_tokens

        if should_log:
            torch.cuda.synchronize()
            step_wall_ms = 1000 * (time.perf_counter() - step_wall_t0)
            step_wall_s = max(step_wall_ms / 1000.0, 1e-9)
        else:
            step_wall_ms = float("nan")
            step_wall_s = float("nan")

        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        print(
            f"step:{step + 1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms "
            f"step_avg:{approx_training_time_ms / (step + 1):.2f}ms",
        )

        if should_log:
            train_time_s = approx_training_time_ms / 1000
            logger.log_metric(
                {
                    "train/loss_raw": float(train_loss_accum),
                    "train/lr": float(_primary_train_lr_float(optimizer)),
                    "train/tokens": int(global_train_tokens),
                    "train/step": int(step + 1),
                    "train/throughput_tokens_per_sec": float(TRAINING.batch_tokens / step_wall_s),
                    "train/step_time_ms": float(step_wall_ms),
                    "train/grad_norm": float(grad_norm_value),
                    "train/total_time_s": float(train_time_s),
                    "wall/elapsed_s": float(time.perf_counter() - wall_start_time),
                },
            )

        if (
            TRAINING.spectral_every_tokens > 0
            and global_train_tokens >= next_spectral_tokens
        ):
            spectral_summary, spectral_details = collect_spectral_metrics(
                optimizer,
                global_train_tokens=global_train_tokens,
                master_process=True,
                spectral_max_matrices=TRAINING.spectral_max_matrices,
                spectral_max_dim=TRAINING.spectral_max_dim,
                coeffs=polar_express_coeffs,
                norm_factor=orth_norm_factor,
            )
            if spectral_summary:
                logger.log_spectral(spectral_summary)
            while next_spectral_tokens <= global_train_tokens:
                next_spectral_tokens += TRAINING.spectral_every_tokens

    logger.log_metric(
        {
            "memory/peak_allocated_mb": int(torch.cuda.max_memory_allocated() // 1024 // 1024),
            "memory/peak_reserved_mb": int(torch.cuda.max_memory_reserved() // 1024 // 1024),
            "train/final_tokens": int(global_train_tokens),
            "wall/final_elapsed_s": float(time.perf_counter() - wall_start_time),
            "status": "completed",
        },
    )
    print(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB",
    )


def _primary_train_lr_float(optimizer) -> float:
    for param_cfg in optimizer.param_cfgs.values():
        if param_cfg.optim == "normuon":
            return float(param_cfg.lr * param_cfg.lr_mul)
    return float("nan")


def main() -> None:
    args = parse_args()

    run_dir = (Path(__file__).resolve().parents[2] / "runs" / args.name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    coeff_schedule, norm_factor, orth_record_data, base_lr = dispatch_orth_state(args.orth)
    polar_express = make_polar_express(
        coeff_schedule=coeff_schedule,
        norm_factor=norm_factor,
    )
    train_data, val_data = resolve_data_path(args.data_path)

    logger = Logger(
        run_name=args.name,
        seed=args.seed,
        base_lr=base_lr,
        orth_record=orth_record_data,
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
    optimizer = build_optimizer(
        model,
        orth_mode=args.orth,
        polar_express=polar_express,
    )

    run_training_loop(
        model=model,
        optimizer=optimizer,
        device=device,
        train_data=train_data,
        val_data=val_data,
        logger=logger,
        polar_express_coeffs=tuple(tuple(c) for c in coeff_schedule),
        orth_norm_factor=norm_factor,
    )


if __name__ == "__main__":
    main()

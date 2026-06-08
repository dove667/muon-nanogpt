import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn

from src.model.gpt import build_model
from src.optim import build_optimizer, step_optimizer, build_coeff_schedule, make_orthogonalize_fn, orth_norm_factor, orth_record
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
    parser.add_argument("--name", default=None, help="Run directory name. Defaults to a timestamped name.")
    parser.add_argument("--seed", type=int, default=FIXED_SEED, help="Random seed")
    parser.add_argument("--train-token-budget", type=int, default=None,
                        help="Override training.train_token_budget")
    parser.add_argument("--eval-interval-tokens", type=int, default=None,
                        help="Override training.eval_interval_tokens")
    parser.add_argument("--eval-tokens", type=int, default=None,
                        help="Override training.eval_tokens")
    parser.add_argument("--log-every-steps", type=int, default=None,
                        help="Override training.log_every_steps")
    parser.add_argument("--spectral-interval-tokens", type=int, default=None,
                        help="Override training.spectral_interval_tokens")
    parser.add_argument("--spectral-num-matrices", type=int, default=None,
                        help="Override training.spectral_num_matrices")
    parser.add_argument("--spectral-dim-cap", type=int, default=None,
                        help="Override training.spectral_dim_cap")
    parser.add_argument("--lr-mul", type=float, default=None, help="Override optimizer.lr_mul")
    parser.add_argument("--ns-iterations", type=int, default=None,
                        help="Newton-Schulz iterations for vanilla/fast/manual modes")
    parser.add_argument("--fast-steps", type=int, default=None,
                        help="Manual schedule fast coefficient steps")
    parser.add_argument("--stable-steps", type=int, default=None,
                        help="Manual schedule stable coefficient steps")
    parser.add_argument("--pe-lower-bound", default=None, help="Polar Express lower bound")
    parser.add_argument("--pe-iterations", type=int, default=None,
                        help="Polar Express iteration count")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--benchmark", action="store_true",
                            help="Enable wall-clock timing measurements (adds cuda synchronize)")
    mode_group.add_argument("--spectral", action="store_true",
                            help="Enable spectral metric collection on Muon update objects")
    args = parser.parse_args()
    apply_cli_overrides(args)
    if args.name is None:
        args.name = default_run_name(args.orth)
    return args


def apply_cli_overrides(args: argparse.Namespace) -> None:
    training_overrides = {
        "train_token_budget": args.train_token_budget,
        "eval_interval_tokens": args.eval_interval_tokens,
        "eval_tokens": args.eval_tokens,
        "log_every_steps": args.log_every_steps,
        "spectral_interval_tokens": args.spectral_interval_tokens,
        "spectral_num_matrices": args.spectral_num_matrices,
        "spectral_dim_cap": args.spectral_dim_cap,
    }
    for key, value in training_overrides.items():
        if value is not None:
            TRAINING._data[key] = value

    if args.lr_mul is not None:
        OPTIMIZER._data["lr_mul"] = args.lr_mul

    orth_cfg = get_orthogonalization()
    if args.ns_iterations is not None:
        orth_cfg._data["ns_iterations"] = args.ns_iterations
        orth_cfg._data["default_iterations"] = args.ns_iterations
    if args.fast_steps is not None:
        orth_cfg._data["fast_steps"] = args.fast_steps
    if args.stable_steps is not None:
        orth_cfg._data["stable_steps"] = args.stable_steps
    if args.pe_lower_bound is not None:
        orth_cfg._data["pe_lower_bound"] = args.pe_lower_bound
    if args.pe_iterations is not None:
        orth_cfg._data["pe_iterations"] = args.pe_iterations


def dispatch_orth_state(orth: str) -> tuple[list[tuple[float, float, float]], float, dict[str, object], float]:
    orth_cfg = get_orthogonalization()
    ns_iterations = int(orth_cfg._data.get("ns_iterations", orth_cfg.default_iterations))
    pe_iterations = int(orth_cfg._data.get("pe_iterations", ns_iterations))
    coeff_schedule = build_coeff_schedule(
        orth,
        fast_steps=int(orth_cfg.fast_steps),
        stable_steps=int(orth_cfg.stable_steps),
        pe_lower_bound_raw=orth_cfg.pe_lower_bound,
        pe_cushion=float(orth_cfg.pe_cushion),
        pe_safety_factor=float(orth_cfg.pe_safety_factor),
        ns_iterations=ns_iterations,
        pe_iterations=pe_iterations,
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
        ns_iterations=ns_iterations,
        pe_iterations=pe_iterations,
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
    elapsed_s: float,
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
        "val/global_wall_time_s": float(elapsed_s),
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
    t_start = time.perf_counter()

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
                elapsed_s=time.perf_counter() - t_start,
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

        should_capture_spectral = (
            spectral
            and TRAINING.spectral_interval_tokens > 0
            and global_train_tokens + TRAINING.tokens_per_step >= next_spectral_tokens
        )
        captured_normuon_stats = step_optimizer(
            optimizer,
            step=step,
            total_steps=train_steps,
            capture_normuon_stats=should_capture_spectral,
        )
        global_train_tokens += TRAINING.tokens_per_step

        print(f"step:{step + 1}/{train_steps}")

        if should_log:
            record: dict = {
                "train/loss_raw": float(train_loss_accum),
                "train/lr": float(primary_lr(optimizer)),
                "train/tokens": int(global_train_tokens),
                "train/step": int(step + 1),
                "train/elapsed_s": float(time.perf_counter() - t_start),
            }
            logger.log_metric(record)

        if should_capture_spectral:
            spectral_summary, detail_records = collect_spectral_metrics(
                optimizer,
                global_train_tokens=global_train_tokens,
                master_process=True,
                num_matrices=TRAINING.spectral_num_matrices,
                svd_dim_cap=TRAINING.spectral_dim_cap,
                captured_normuon_stats=captured_normuon_stats,
            )
            if spectral_summary:
                logger.log_spectral(spectral_summary)
                logger.log_spectral_details(detail_records)
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
    orthogonalize_fn = make_orthogonalize_fn(coeff_schedule=coeff_schedule, norm_factor=norm_factor)
    train_data_path, val_data_path = resolve_data_path(args.data_path)

    logger = Logger(
        run_name=args.name,
        seed=args.seed,
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

    device = setup_device(base_seed=args.seed)

    print(f"PyTorch {torch.__version__}  CUDA {torch.version.cuda} on device {device}")
    print(f"Python  {sys.version}")

    model = build_model(device)
    optimizer = build_optimizer(model, orth_mode=args.orth, orthogonalize_fn=orthogonalize_fn)

    run_training_loop(
        model=model,
        optimizer=optimizer,
        device=device,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        logger=logger,
        benchmark=args.benchmark,
        spectral=args.spectral,
    )


if __name__ == "__main__":
    main()

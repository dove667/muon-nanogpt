import gc
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data import data_generator
from metrics import collect_spectral_metrics, current_grad_norm
from optim import step_optimizer
from config import TRAINING

def setup_device(*, base_seed: int) -> torch.device:
    assert torch.cuda.is_available()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed_all(base_seed)

    return device

class Logger:
    def __init__(
        self,
        *,
        run_name: str,
        seed: int,
        base_lr: float,
        train_token_budget: int,
        orth_record: dict[str, object],
        run_dir: Path,
    ) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.run_dir / "metrics.jsonl"
        self.config_file = self.run_dir / "config.json"

        run_config = {
            "run_name": run_name,
            "seed": seed,
            "base_lr": base_lr,
            "train_token_budget": train_token_budget,
            "batch_tokens": int(TRAINING.batch_tokens),
            "seq_len": int(TRAINING.seq_len),
            "grad_accum_steps": int(TRAINING.grad_accum_steps),
            **orth_record,
        }
        self.config_file.write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")

    def log_metric(self, record: dict) -> None:
        with self.metrics_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def log_spectral(self, summary_record: dict) -> None:
        if summary_record:
            self.log_metric(summary_record)


def run_validation(
    *,
    model: nn.Module,
    device: torch.device,
    val_files: str,
    val_tokens: int,
    grad_accum_steps: int,
    logger: Logger,
    step: int,
    train_steps: int,
    training_time_ms: float,
    wall_start_time: float,
    global_train_tokens: int,
) -> None:
    val_batch_tokens = TRAINING.batch_tokens
    eval_start_time = time.perf_counter()
    model.eval()
    assert val_tokens % val_batch_tokens == 0
    val_steps = grad_accum_steps * val_tokens // val_batch_tokens
    val_loader = data_generator(
        val_files, val_batch_tokens, TRAINING.seq_len, grad_accum_steps,
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
    train_files: str,
    val_files: str,
    val_tokens: int,
    train_steps: int,
    logger: Logger,
    log_every_steps: int,
    eval_every_tokens: int,
    spectral_every_tokens: int,
    spectral_max_matrices: int,
    spectral_max_dim: int,
    polar_express_coeffs: tuple[tuple[float, float, float], ...],
    orth_norm_factor: float,
) -> None:
    train_loader = data_generator(
        train_files, TRAINING.batch_tokens, TRAINING.seq_len, TRAINING.grad_accum_steps,
    )
    gc.collect()

    training_time_ms = 0.0
    global_train_tokens = 0
    next_eval_tokens = eval_every_tokens
    next_spectral_tokens = spectral_every_tokens

    wall_start_time = time.perf_counter()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(train_steps + 1):
        last_step = step == train_steps

        should_eval_by_tokens = (
            eval_every_tokens > 0
            and global_train_tokens >= next_eval_tokens
        )
        if last_step or should_eval_by_tokens:
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            run_validation(
                model=model,
                device=device,
                val_files=val_files,
                val_tokens=val_tokens,
                grad_accum_steps=TRAINING.grad_accum_steps,
                logger=logger,
                step=step,
                train_steps=train_steps,
                training_time_ms=training_time_ms,
                wall_start_time=wall_start_time,
                global_train_tokens=global_train_tokens,
            )
            if eval_every_tokens > 0:
                while next_eval_tokens <= global_train_tokens:
                    next_eval_tokens += eval_every_tokens
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        should_log = log_every_steps > 0 and (
            (step + 1) % log_every_steps == 0 or step < 3
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
            spectral_every_tokens > 0
            and global_train_tokens >= next_spectral_tokens
        ):
            spectral_summary, spectral_details = collect_spectral_metrics(
                optimizer,
                global_train_tokens=global_train_tokens,
                master_process=True,
                spectral_max_matrices=spectral_max_matrices,
                spectral_max_dim=spectral_max_dim,
                coeffs=polar_express_coeffs,
                norm_factor=orth_norm_factor,
            )
            if spectral_summary:
                logger.log_spectral(spectral_summary)
            while next_spectral_tokens <= global_train_tokens:
                next_spectral_tokens += spectral_every_tokens

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

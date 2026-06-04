import gc
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from data_pipeline import data_generator
from metrics import collect_spectral_metrics, current_grad_norm

FIXED_BATCH_TOKENS = 8 * 2048 * 8
FIXED_SEQ_LEN = 2048

if TYPE_CHECKING:
    from orthogonalization import OrthogonalizerConfig


def setup_device(*, base_seed: int) -> torch.device:
    assert torch.cuda.is_available()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed_all(base_seed)

    return device


def nvidia_smi_output() -> str:
    return subprocess.run(
        ["nvidia-smi"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout


class RunLogger:
    def __init__(
        self,
        *,
        args,
        orth_config: "OrthogonalizerConfig",
        base_lr: float,
        train_token_budget: int,
        eval_every_tokens: int,
        grad_accum_steps: int,
        device: torch.device,
        seed: int,
        seq_len: int,
        run_dir: Path,
        run_name: str,
    ) -> None:
        self.args = args
        self.orth_mode = orth_config.orth_mode
        self.orth_schedule_name = orth_config.schedule_name
        self.run_dir = run_dir
        self.run_name = run_name

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.run_dir / "metrics.jsonl"
        self.config_file = self.run_dir / "config.json"

        run_config = {
            **{
                key: getattr(args, key)
                for key in dir(args)
                if not key.startswith("_") and not callable(getattr(args, key))
            },
            **orth_config.to_record(),
            "run_name": run_name,
            "base_lr": base_lr,
            "actual_lr": base_lr * orth_config.lr_mul,
            "seed": seed,
            "model_size": "train_gpt_11L_768D",
            "seq_len": seq_len,
            "grad_accum": grad_accum_steps,
            "train_token_budget": train_token_budget,
            "eval_every_tokens": eval_every_tokens,
            "eval_tokens": args.val_tokens,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
        }
        self.config_file.write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")

    def _with_common_fields(self, record: dict, step_value: int | None = None) -> dict:
        payload = dict(record)
        payload.setdefault("run/name", self.run_name)
        payload.setdefault("orthogonalizer/type", self.orth_mode)
        payload.setdefault("orthogonalizer/schedule_name", self.orth_schedule_name)
        if step_value is not None:
            payload.setdefault("train/step", int(step_value))
        return payload

    def log_metric(self, record: dict, step_value: int | None = None) -> None:
        payload = self._with_common_fields(record, step_value=step_value)
        with self.metrics_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def log_spectral(self, summary_record: dict) -> None:
        if summary_record:
            self.log_metric(summary_record)


def run_validation(
    *,
    model: nn.Module,
    training_manager,
    args,
    grad_accum_steps: int,
    logger: RunLogger,
    step: int,
    train_steps: int,
    training_time_ms: float,
    wall_start_time: float,
) -> None:
    eval_start_time = time.perf_counter()
    model.eval()
    assert args.val_tokens % args.val_batch_size == 0
    val_steps = grad_accum_steps * args.val_tokens // args.val_batch_size
    val_loader = data_generator(
        args.val_files, args.val_batch_size, -1, grad_accum_steps,
        False, args.bigram_vocab_size,
    )
    val_loss = torch.tensor(0.0, device=training_manager.device)
    with torch.no_grad():
        for _ in range(val_steps):
            inputs, targets, cum_seqlens, bigram_inputs, _ = next(val_loader)
            val_loss += model(
                inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args(),
            ).mean()
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
            "val/ppl": float(math.exp(min(float(val_loss), 20.0))),
            "val/tokens": int(args.val_tokens),
            "val/time_s": float(eval_time_s),
            "val/tokens_per_sec": float(args.val_tokens / max(eval_time_s, 1e-9)),
            "val/global_train_tokens": int(training_manager.global_train_tokens),
            "val/global_wall_time_s": float(time.perf_counter() - wall_start_time),
            "train/time_s": float(training_time_ms / 1000),
            "train/tokens": int(training_manager.global_train_tokens),
            "train/step": int(step),
            "wall/elapsed_s": float(time.perf_counter() - wall_start_time),
        },
        step_value=step,
    )
    model.train()


def run_training_loop(
    *,
    model: nn.Module,
    training_manager,
    args,
    train_steps: int,
    grad_accum_steps: int,
    logger: RunLogger,
    log_every_steps: int,
    eval_every_tokens: int,
    eval_at_start: bool,
    spectral_every_tokens: int,
    spectral_max_matrices: int,
    spectral_max_dim: int,
    polar_express_coeffs: tuple[tuple[float, float, float], ...],
    orth_norm_factor: float,
) -> None:
    train_loader = data_generator(
        args.train_files, FIXED_BATCH_TOKENS, FIXED_SEQ_LEN,
        grad_accum_steps, True, args.bigram_vocab_size,
    )
    gc.collect()

    training_time_ms = 0.0
    training_manager.global_train_tokens = 0
    next_eval_tokens = 0 if eval_at_start else eval_every_tokens
    next_spectral_tokens = spectral_every_tokens

    wall_start_time = time.perf_counter()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(train_steps + 1):
        last_step = step == train_steps

        should_eval_by_tokens = (
            eval_every_tokens > 0
            and training_manager.global_train_tokens >= next_eval_tokens
        )
        should_eval_by_steps = args.val_loss_every > 0 and step % args.val_loss_every == 0
        if last_step or should_eval_by_tokens or should_eval_by_steps:
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            run_validation(
                model=model, training_manager=training_manager, args=args,
                grad_accum_steps=grad_accum_steps, logger=logger, step=step, train_steps=train_steps,
                training_time_ms=training_time_ms, wall_start_time=wall_start_time,
            )
            if eval_every_tokens > 0:
                while next_eval_tokens <= training_manager.global_train_tokens:
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

        grad_scale = 1.0 / grad_accum_steps
        for _ in range(grad_accum_steps):
            inputs, targets, cum_seqlens, bigram_inputs, bigram_cpu = train_loader.send(
                training_manager.train_loader_send_args
            )
            training_manager.sparse_index_update(step, bigram_cpu)
            loss = model(
                inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args(),
            ).sum() * grad_scale
            if should_log:
                train_loss_accum += float(loss.detach())
            training_manager.sparse_index_share(step)
            loss.backward()

        if should_log:
            grad_norm_value = current_grad_norm(model)
            del loss
            torch.cuda.synchronize()

        training_manager.step_optimizers(step)
        training_manager.global_train_tokens += FIXED_BATCH_TOKENS

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
                    "train/lr": float(_primary_train_lr_float(training_manager)),
                    "train/tokens": int(training_manager.global_train_tokens),
                    "train/step": int(step + 1),
                    "train/throughput_tokens_per_sec": float(FIXED_BATCH_TOKENS / step_wall_s),
                    "train/step_time_ms": float(step_wall_ms),
                    "train/grad_norm": float(grad_norm_value),
                    "train/total_time_s": float(train_time_s),
                    "wall/elapsed_s": float(time.perf_counter() - wall_start_time),
                },
                step_value=step + 1,
            )

        if (
            spectral_every_tokens > 0
            and training_manager.global_train_tokens >= next_spectral_tokens
        ):
            spectral_summary, spectral_details = collect_spectral_metrics(
                training_manager.optimizer,
                global_train_tokens=training_manager.global_train_tokens,
                master_process=True,
                spectral_max_matrices=spectral_max_matrices,
                spectral_max_dim=spectral_max_dim,
                coeffs=polar_express_coeffs,
                norm_factor=orth_norm_factor,
            )
            if spectral_summary:
                logger.log_spectral(spectral_summary)
            while next_spectral_tokens <= training_manager.global_train_tokens:
                next_spectral_tokens += spectral_every_tokens

    logger.log_metric(
        {
            "memory/peak_allocated_mb": int(torch.cuda.max_memory_allocated() // 1024 // 1024),
            "memory/peak_reserved_mb": int(torch.cuda.max_memory_reserved() // 1024 // 1024),
            "train/final_tokens": int(training_manager.global_train_tokens),
            "wall/final_elapsed_s": float(time.perf_counter() - wall_start_time),
            "status": "completed",
        },
        step_value=train_steps,
    )
    print(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB",
    )


def _primary_train_lr_float(training_manager) -> float:
    for param_cfg in training_manager.optimizer.param_cfgs.values():
        if param_cfg.label not in {"qk_bank", "vo_bank", "mlp_bank"}:
            continue
        return float(param_cfg.lr * param_cfg.lr_mul)
    return float("nan")

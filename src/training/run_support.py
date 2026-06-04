import gc
import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from data_pipeline import distributed_data_generator
from metrics import collect_spectral_metrics, current_grad_norm


@dataclass(slots=True)
class DistributedContext:
    rank: int
    world_size: int
    grad_accum_steps: int
    grad_scale: float
    device: torch.device
    master_process: bool
    base_seed: int


def build_code_snapshot(script_path: str) -> str:
    with open(script_path, "r", encoding="utf-8") as handle:
        return handle.read()


def setup_distributed_from_env() -> DistributedContext:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.empty(1, device=f"cuda:{local_rank}", requires_grad=True).backward()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert 8 % world_size == 0, "world_size must be a divisor of 8"

    grad_accum_steps = int(
        os.environ.get(
            "TRAIN_GRAD_ACCUM_STEPS",
            os.environ.get("SPEEDTEST_GRAD_ACCUM_STEPS", 8 // world_size),
        )
    )
    grad_scale = 1 / grad_accum_steps

    assert torch.cuda.is_available()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    dist.init_process_group(backend="cuda:nccl,cpu:gloo", device_id=device)
    dist.barrier()

    base_seed = int(os.environ.get("SEED", "0"))
    random.seed(base_seed + rank)
    np.random.seed(base_seed + rank)
    torch.manual_seed(base_seed + rank)
    torch.cuda.manual_seed_all(base_seed + rank)

    return DistributedContext(
        rank=rank,
        world_size=world_size,
        grad_accum_steps=grad_accum_steps,
        grad_scale=grad_scale,
        device=device,
        master_process=(rank == 0),
        base_seed=base_seed,
    )


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
        master_process: bool,
        args,
        orth_config: dict,
        orth_mode: str,
        orth_schedule_name: str,
        lr_mul: float,
        train_token_budget: int,
        eval_every_tokens: int,
        world_size: int,
        grad_accum_steps: int,
        device: torch.device,
    ) -> None:
        self.master_process = master_process
        self.args = args
        self.orth_mode = orth_mode
        self.orth_schedule_name = orth_schedule_name
        self.wandb_run = None
        self.spectral_details_file = None
        self.spectral_details_enabled = os.environ.get("SPECTRAL_LOG_DETAILS", "0").lower() in {
            "1", "true", "yes", "on",
        }

        self.run_name = os.environ.get("WANDB_NAME") or os.environ.get("RUN_NAME") or args.run_id
        self.wandb_project = os.environ.get("WANDB_PROJECT", "muon-nanogpt")
        self.wandb_entity = os.environ.get("WANDB_ENTITY") or None
        self.wandb_mode = os.environ.get("WANDB_MODE") or None
        self.wandb_enabled = os.environ.get("WANDB", "1").lower() not in {
            "0", "false", "no", "disabled",
        }

        runs_root = Path(os.environ.get("RUNS_ROOT", "runs"))
        self.run_dir = Path(os.environ.get("RUN_DIR", runs_root / self.run_name))
        self.metrics_file = None
        self.config_file = None

        if not self.master_process:
            return

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.run_dir / "metrics.jsonl"
        self.config_file = self.run_dir / "config.json"
        if self.spectral_details_enabled:
            self.spectral_details_file = self.run_dir / "spectral_metrics.jsonl"

        run_config = {
            **{
                key: getattr(args, key)
                for key in dir(args)
                if not key.startswith("_") and not callable(getattr(args, key))
            },
            **orth_config,
            "run_name": self.run_name,
            "wandb_project": self.wandb_project,
            "base_lr": float(os.environ.get("BASE_LR", "0.023")),
            "actual_lr": float(os.environ.get("BASE_LR", "0.023")) * lr_mul,
            "seed": int(os.environ.get("SEED", "0")),
            "model_size": "train_gpt_11L_768D",
            "seq_len": int(os.environ.get("TRAIN_SEQ_LEN", "2048")),
            "grad_accum": grad_accum_steps,
            "world_size": world_size,
            "train_token_budget": train_token_budget,
            "eval_every_tokens": eval_every_tokens,
            "eval_tokens": args.val_tokens,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
            "spectral_log_details": self.spectral_details_enabled,
        }
        self.config_file.write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")

        if self.wandb_enabled:
            try:
                import wandb
                wandb_kwargs = dict(
                    project=self.wandb_project,
                    name=self.run_name,
                    config=run_config,
                )
                if self.wandb_entity:
                    wandb_kwargs["entity"] = self.wandb_entity
                if self.wandb_mode:
                    wandb_kwargs["mode"] = self.wandb_mode
                self.wandb_run = wandb.init(**wandb_kwargs)
            except Exception as exc:
                print(f"W&B init failed, continuing with local logs only: {exc}", flush=True)
                self.wandb_run = None

    def _with_common_fields(self, record: dict, step_value: int | None = None) -> dict:
        payload = dict(record)
        payload.setdefault("run/name", self.run_name)
        payload.setdefault("orthogonalizer/type", self.orth_mode)
        payload.setdefault("orthogonalizer/schedule_name", self.orth_schedule_name)
        if step_value is not None:
            payload.setdefault("train/step", int(step_value))
        return payload

    def print0(self, message: str) -> None:
        if not self.master_process:
            return
        print(message)

    def log_metric(self, record: dict, step_value: int | None = None) -> None:
        if not self.master_process:
            return
        payload = self._with_common_fields(record, step_value=step_value)
        with self.metrics_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        if self.wandb_run is not None:
            self.wandb_run.log(payload, step=step_value)

    def log_spectral(self, summary_record: dict, detail_records: list[dict], step_value: int | None = None) -> None:
        if summary_record:
            self.log_metric(summary_record, step_value=step_value)
        if not self.master_process or not self.spectral_details_enabled or not detail_records:
            return
        with self.spectral_details_file.open("a", encoding="utf-8") as handle:
            for record in detail_records:
                payload = self._with_common_fields(record, step_value=step_value)
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def close(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()


@dataclass(slots=True)
class LoopConfig:
    log_every_steps: int
    eval_every_tokens: int
    eval_at_start: bool
    spectral_every_tokens: int
    spectral_max_matrices: int
    spectral_max_dim: int
    orth_mode: str
    orth_schedule_name: str
    polar_express_coeffs: tuple[tuple[float, float, float], ...]
    orth_norm_factor: float

    @classmethod
    def from_env(cls, *, orth_mode: str, orth_schedule_name: str, polar_express_coeffs, orth_norm_factor: float):
        return cls(
            log_every_steps=int(os.environ.get("LOG_EVERY_STEPS", "20")),
            eval_every_tokens=int(float(os.environ.get("EVAL_EVERY_TOKENS", "0"))),
            eval_at_start=os.environ.get("EVAL_AT_START", "0").lower() in {"1", "true", "yes"},
            spectral_every_tokens=int(float(os.environ.get("SPECTRAL_EVERY_TOKENS", "10000000"))),
            spectral_max_matrices=int(os.environ.get("SPECTRAL_MAX_MATRICES", "5")),
            spectral_max_dim=int(os.environ.get("SPECTRAL_MAX_DIM", "1024")),
            orth_mode=orth_mode,
            orth_schedule_name=orth_schedule_name,
            polar_express_coeffs=tuple(tuple(coeff) for coeff in polar_express_coeffs),
            orth_norm_factor=orth_norm_factor,
        )


def run_validation(
    *,
    model: nn.Module,
    training_manager,
    args,
    dist_ctx,
    logger,
    step: int,
    train_steps: int,
    training_time_ms: float,
    wall_start_time: float,
) -> None:
    eval_start_time = time.perf_counter()
    model.eval()
    assert args.val_tokens % args.val_batch_size == 0
    val_steps = dist_ctx.grad_accum_steps * args.val_tokens // args.val_batch_size
    val_loader = distributed_data_generator(
        args.val_files, args.val_batch_size, -1, dist_ctx.grad_accum_steps,
        False, dist_ctx.rank, dist_ctx.world_size, args.bigram_vocab_size,
    )
    val_loss = 0
    with torch.no_grad():
        for _ in range(val_steps):
            inputs, targets, cum_seqlens, bigram_inputs, _ = next(val_loader)
            val_loss += model(
                inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args(),
            ).mean()
    val_loss /= val_steps
    del val_loader
    dist.reduce(val_loss, 0, op=dist.ReduceOp.AVG)
    eval_time_s = time.perf_counter() - eval_start_time
    logger.print0(
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
    training_stages,
    training_schedule,
    dist_ctx,
    logger,
    loop_config: LoopConfig,
) -> None:
    train_loader = distributed_data_generator(
        args.train_files, training_stages[0].batch_size, training_stages[0].train_max_seq_len,
        dist_ctx.grad_accum_steps, True, dist_ctx.rank, dist_ctx.world_size, args.bigram_vocab_size,
    )
    gc.collect()

    training_time_ms = 0.0
    training_manager.global_train_tokens = 0
    next_eval_tokens = 0 if loop_config.eval_at_start else loop_config.eval_every_tokens
    next_spectral_tokens = loop_config.spectral_every_tokens

    wall_start_time = time.perf_counter()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    train_steps = training_schedule.total_steps
    for step in range(train_steps + 1):
        last_step = step == train_steps
        training_manager.advance_schedule(step)

        should_eval_by_tokens = (
            loop_config.eval_every_tokens > 0
            and training_manager.global_train_tokens >= next_eval_tokens
        )
        should_eval_by_steps = args.val_loss_every > 0 and step % args.val_loss_every == 0
        if last_step or should_eval_by_tokens or should_eval_by_steps:
            if last_step:
                training_manager.apply_final_ws_ext()
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            run_validation(
                model=model, training_manager=training_manager, args=args,
                dist_ctx=dist_ctx, logger=logger, step=step, train_steps=train_steps,
                training_time_ms=training_time_ms, wall_start_time=wall_start_time,
            )
            if loop_config.eval_every_tokens > 0:
                while next_eval_tokens <= training_manager.global_train_tokens:
                    next_eval_tokens += loop_config.eval_every_tokens
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        should_log = loop_config.log_every_steps > 0 and (
            (step + 1) % loop_config.log_every_steps == 0 or step < 3
        )
        train_loss_accum = 0.0
        if should_log:
            torch.cuda.synchronize()
            step_wall_t0 = time.perf_counter()

        for _ in range(dist_ctx.grad_accum_steps):
            inputs, targets, cum_seqlens, bigram_inputs, bigram_cpu = train_loader.send(
                training_manager.train_loader_send_args
            )
            training_manager.sparse_index_update(step, bigram_cpu)
            loss = model(
                inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args(),
            ).sum() * dist_ctx.grad_scale
            if should_log:
                train_loss_accum += float(loss.detach())
            training_manager.sparse_index_share(step)
            loss.backward()

        if should_log:
            grad_norm_value = current_grad_norm(model)
            del loss
            torch.cuda.synchronize()

        training_manager.step_optimizers(step)
        stage_for_tokens, _ = training_schedule.lookup(step)
        step_tokens = int(stage_for_tokens.batch_size)
        training_manager.global_train_tokens += step_tokens

        if should_log:
            torch.cuda.synchronize()
            step_wall_ms = 1000 * (time.perf_counter() - step_wall_t0)
            step_wall_s = max(step_wall_ms / 1000.0, 1e-9)
        else:
            step_wall_ms = float("nan")
            step_wall_s = float("nan")

        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        logger.print0(
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
                    "train/throughput_tokens_per_sec": float(step_tokens / step_wall_s),
                    "train/step_time_ms": float(step_wall_ms),
                    "train/grad_norm": float(grad_norm_value),
                    "train/total_time_s": float(train_time_s),
                    "wall/elapsed_s": float(time.perf_counter() - wall_start_time),
                },
                step_value=step + 1,
            )

        if (
            loop_config.spectral_every_tokens > 0
            and training_manager.global_train_tokens >= next_spectral_tokens
        ):
            spectral_summary, spectral_details = collect_spectral_metrics(
                training_manager.optimizer,
                global_train_tokens=training_manager.global_train_tokens,
                master_process=dist_ctx.master_process,
                spectral_max_matrices=loop_config.spectral_max_matrices,
                spectral_max_dim=loop_config.spectral_max_dim,
                coeffs=loop_config.polar_express_coeffs,
                norm_factor=loop_config.orth_norm_factor,
            )
            if spectral_summary:
                logger.log_spectral(spectral_summary, spectral_details, step_value=step + 1)
            while next_spectral_tokens <= training_manager.global_train_tokens:
                next_spectral_tokens += loop_config.spectral_every_tokens

    if dist_ctx.master_process:
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
    logger.print0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB",
    )


def _primary_train_lr_float(training_manager) -> float:
    for param_cfg in training_manager.optimizer.param_cfgs.values():
        if param_cfg.label not in {"qk_bank", "vo_bank", "mlp_bank"}:
            continue
        return float(param_cfg.lr * param_cfg.lr_mul)
    return float("nan")

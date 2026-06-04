import argparse
import io
import math
import os
import subprocess
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
    setup_distributed_from_env,
)
from schedule import Hyperparameters, TrainingSchedule, default_training_stages

ROOT = Path(__file__).resolve().parents[2]
FIXED_BATCH_TOKENS = 8 * 2048 * 8
FIXED_SEQ_LEN = 2048


class Tee(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


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
    parser.add_argument("--nproc-per-node", type=int, default=1,
                        help="Number of processes per node")
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


def args_to_cli(args: argparse.Namespace) -> list[str]:
    cli = [
        "--orth", args.orth,
        "--lr-mul", str(args.lr_mul),
        "--seed", str(args.seed),
        "--name", args.name,
        "--train-token-budget", str(args.train_token_budget),
        "--eval-every-tokens", str(args.eval_every_tokens),
        "--eval-tokens", str(args.eval_tokens),
        "--train-grad-accum-steps", str(args.train_grad_accum_steps),
        "--eval-batch-size", str(args.eval_batch_size),
        "--log-every-steps", str(args.log_every_steps),
        "--wandb", args.wandb,
        "--wandb-project", args.wandb_project,
        "--nproc-per-node", str(args.nproc_per_node),
        "--pe-lower-bound", str(args.pe_lower_bound),
        "--pe-cushion", str(args.pe_cushion),
        "--pe-safety-factor", str(args.pe_safety_factor),
        "--spectral-every-tokens", str(args.spectral_every_tokens),
        "--spectral-max-matrices", str(args.spectral_max_matrices),
        "--spectral-max-dim", str(args.spectral_max_dim),
    ]
    if args.eval_at_start:
        cli.append("--eval-at-start")
    if args.wandb_entity:
        cli.extend(["--wandb-entity", args.wandb_entity])
    if args.wandb_mode:
        cli.extend(["--wandb-mode", args.wandb_mode])
    if args.runs_root:
        cli.extend(["--runs-root", args.runs_root])
    if args.data_path:
        cli.extend(["--data-path", args.data_path])
    if args.fast_steps is not None:
        cli.extend(["--fast-steps", str(args.fast_steps)])
    if args.stable_steps is not None:
        cli.extend(["--stable-steps", str(args.stable_steps)])
    if args.model_max_seq_len > 0:
        cli.extend(["--model-max-seq-len", str(args.model_max_seq_len)])
    return cli


def prepare_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.runs_root:
        runs_root = Path(args.runs_root).expanduser()
    else:
        runs_root = Path(env.get("RUNS_ROOT", ROOT / "runs"))
    if not runs_root.is_absolute():
        runs_root = ROOT / runs_root
    run_dir = (runs_root / args.name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    env["ORTH"] = args.orth
    env["LR_MUL"] = str(args.lr_mul)
    env["BASE_LR"] = "0.008" if args.orth == "adamw" else "0.023"
    env["SEED"] = str(args.seed)
    env["TRAIN_TOKEN_BUDGET"] = str(args.train_token_budget)
    env["TRAIN_STEPS"] = str(compute_train_steps(args.train_token_budget))
    env["TRAIN_GRAD_ACCUM_STEPS"] = str(args.train_grad_accum_steps)
    env["TRAIN_SEQ_LEN"] = str(FIXED_SEQ_LEN)
    env["EVAL_EVERY_TOKENS"] = str(args.eval_every_tokens)
    env["EVAL_TOKENS"] = str(args.eval_tokens)
    env["EVAL_BATCH_SIZE"] = str(args.eval_batch_size)
    env["EVAL_AT_START"] = "1" if args.eval_at_start else "0"
    env["LOG_EVERY_STEPS"] = str(args.log_every_steps)
    env["EXTENSION_STEPS"] = "0"
    env["VAL_LOSS_EVERY_STEPS"] = "0"
    env["WANDB"] = "1" if args.wandb == "on" else "0"
    env["WANDB_PROJECT"] = args.wandb_project
    env["WANDB_NAME"] = args.name
    env.pop("WANDB_GROUP", None)
    env["RUNS_ROOT"] = str(runs_root)
    env["RUN_DIR"] = str(run_dir)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".torchinductor"))

    if args.wandb_entity:
        env["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_mode:
        env["WANDB_MODE"] = args.wandb_mode
    elif args.wandb == "on" and not env.get("WANDB_API_KEY") and not Path.home().joinpath(".netrc").exists():
        env["WANDB_MODE"] = "offline"
    if args.data_path:
        env["DATA_PATH"] = args.data_path
    if args.orth == "manual":
        env["FAST_STEPS"] = str(args.fast_steps)
        env["STABLE_STEPS"] = str(args.stable_steps)
    elif args.orth == "polar_express":
        env["PE_LOWER_BOUND"] = str(args.pe_lower_bound)
        env["PE_CUSHION"] = str(args.pe_cushion)
        env["PE_SAFETY_FACTOR"] = str(args.pe_safety_factor)
    if args.model_max_seq_len > 0:
        env["MODEL_MAX_SEQ_LEN"] = str(args.model_max_seq_len)
    env["SPECTRAL_EVERY_TOKENS"] = str(args.spectral_every_tokens)
    env["SPECTRAL_MAX_MATRICES"] = str(args.spectral_max_matrices)
    env["SPECTRAL_MAX_DIM"] = str(args.spectral_max_dim)
    return env


def launch_torchrun(args: argparse.Namespace) -> int:
    env = prepare_env(args)
    run_dir = Path(env["RUN_DIR"])
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        str(Path(__file__).resolve()),
        *args_to_cli(args),
    ]
    print("=" * 80)
    print(f"run: {args.name}")
    print(f"orth: {args.orth}")
    print(f"train_token_budget: {args.train_token_budget}")
    print(f"train_steps: {env['TRAIN_STEPS']}")
    print(f"grad_accum: {args.train_grad_accum_steps}")
    print(f"eval_every_tokens: {args.eval_every_tokens}")
    print(f"eval_tokens: {args.eval_tokens}")
    print(f"world_size: {args.nproc_per_node}")
    print("=" * 80)
    with (run_dir / "console.log").open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def build_model(
    args: Hyperparameters,
    training_stages,
    device: torch.device,
    world_batch_divisor: int,
) -> nn.Module:
    model_max_seq_len = int(
        os.environ.get(
            "MODEL_MAX_SEQ_LEN",
            max(
                args.val_batch_size // world_batch_divisor,
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


def broadcast_model(model: nn.Module) -> None:
    if not torch.distributed.is_initialized():
        return
    for param in model.parameters():
        torch.distributed.broadcast(param.detach(), 0)


def run_worker(args: argparse.Namespace) -> None:
    env = prepare_env(args)
    os.environ.update(env)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0 and "LOCAL_RANK" not in os.environ:
        log_path = Path(env["RUN_DIR"]) / "console.log"
        log_handle = log_path.open("a", encoding="utf-8")
        tee = Tee(sys.__stdout__, log_handle)
        sys.stdout = tee
        sys.stderr = tee
    if "LOCAL_RANK" not in os.environ:
        print("=" * 80)
        print(f"run: {args.name}")
        print(f"orth: {args.orth}")
        print(f"train_token_budget: {args.train_token_budget}")
        print(f"train_steps: {env['TRAIN_STEPS']}")
        print(f"grad_accum: {args.train_grad_accum_steps}")
        print(f"eval_every_tokens: {args.eval_every_tokens}")
        print(f"eval_tokens: {args.eval_tokens}")
        print(f"device: cuda:{local_rank}")
        print(f"world_size: {os.environ.get('WORLD_SIZE', '1')}")
        print("=" * 80)

    dist_ctx = setup_distributed_from_env()
    logger = None

    try:
        orth_state = build_orthogonalizer_config_from_env()
        polar_express = make_polar_express(
            coeff_schedule=orth_state.coeff_schedule,
            norm_factor=orth_state.norm_factor,
        )

        hparams = Hyperparameters()
        world_batch_divisor = dist_ctx.grad_accum_steps * dist_ctx.world_size
        hparams.val_batch_size = int(
            float(os.environ.get("EVAL_BATCH_SIZE", world_batch_divisor * 2048))
        )

        training_stages = default_training_stages()
        training_schedule = TrainingSchedule(
            training_stages,
            hparams.num_scheduled_iterations,
            hparams.num_extension_iterations,
            device=dist_ctx.device,
        )

        setup_model_runtime(
            args_value=hparams,
            world_size_value=dist_ctx.world_size,
            grad_accum_steps_value=dist_ctx.grad_accum_steps,
            grad_scale_value=dist_ctx.grad_scale,
            device_value=dist_ctx.device,
        )

        logger = RunLogger(
            master_process=dist_ctx.master_process,
            args=hparams,
            orth_config=orth_state.to_record(),
            orth_mode=orth_state.orth_mode,
            orth_schedule_name=orth_state.schedule_name,
            lr_mul=orth_state.lr_mul,
            train_token_budget=int(float(os.environ.get("TRAIN_TOKEN_BUDGET", "0"))),
            eval_every_tokens=int(float(os.environ.get("EVAL_EVERY_TOKENS", "0"))),
            world_size=dist_ctx.world_size,
            grad_accum_steps=dist_ctx.grad_accum_steps,
            device=dist_ctx.device,
        )

        logger.print0("=" * 100)
        logger.print0(f"Running Python {sys.version}")
        logger.print0(
            f"Running PyTorch {torch.__version__} compiled for CUDA {torch.version.cuda}"
        )
        logger.print0(nvidia_smi_output())
        logger.print0("=" * 100)

        model = build_model(hparams, training_stages, dist_ctx.device, world_batch_divisor)
        broadcast_model(model)

        training_manager = TrainingManager(
            model,
            rank=dist_ctx.rank,
            world_size=dist_ctx.world_size,
            grad_accum_steps=dist_ctx.grad_accum_steps,
            device=dist_ctx.device,
            args=hparams,
            training_schedule=training_schedule,
            lr_mul=orth_state.lr_mul,
            orth_mode=orth_state.orth_mode,
            polar_express=polar_express,
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
            dist_ctx=dist_ctx,
            logger=logger,
            loop_config=loop_config,
        )
    finally:
        if logger is not None:
            logger.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.nproc_per_node > 1 and "LOCAL_RANK" not in os.environ:
        raise SystemExit(launch_torchrun(parsed_args))
    run_worker(parsed_args)


import os
import sys

import torch
from torch import nn

from model import GPT, setup_model_runtime
from optim import TrainingManager
from orthogonalization import build_orthogonalizer_config_from_env
from polar import make_polar_express
from run_support import (
    LoopConfig,
    RunLogger,
    build_code_snapshot,
    nvidia_smi_output,
    run_training_loop,
    setup_distributed_from_env,
)
from schedule import Hyperparameters, TrainingSchedule, default_training_stages


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
                max(stage.batch_size for stage in training_stages) // world_batch_divisor,
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
    for param in model.parameters():
        torch.distributed.broadcast(param.detach(), 0)


def main() -> None:
    code_snapshot = build_code_snapshot(__file__)
    dist_ctx = setup_distributed_from_env()
    logger = None

    try:
        orth_state = build_orthogonalizer_config_from_env()
        polar_express = make_polar_express(
            coeff_schedule=orth_state.coeff_schedule,
            norm_factor=orth_state.norm_factor,
        )

        args = Hyperparameters()
        world_batch_divisor = dist_ctx.grad_accum_steps * dist_ctx.world_size
        args.val_batch_size = int(
            float(os.environ.get("EVAL_BATCH_SIZE", world_batch_divisor * 2048))
        )

        training_stages = default_training_stages()
        training_schedule = TrainingSchedule(
            training_stages,
            args.num_scheduled_iterations,
            args.num_extension_iterations,
            device=dist_ctx.device,
            cooldown_frac=0.60,
        )

        setup_model_runtime(
            args_value=args,
            world_size_value=dist_ctx.world_size,
            grad_accum_steps_value=dist_ctx.grad_accum_steps,
            grad_scale_value=dist_ctx.grad_scale,
            device_value=dist_ctx.device,
        )

        logger = RunLogger(
            master_process=dist_ctx.master_process,
            args=args,
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

        logger.print0(code_snapshot)
        logger.print0("=" * 100)
        logger.print0(f"Running Python {sys.version}")
        logger.print0(
            f"Running PyTorch {torch.__version__} compiled for CUDA {torch.version.cuda}"
        )
        logger.print0(nvidia_smi_output())
        logger.print0("=" * 100)

        model = build_model(args, training_stages, dist_ctx.device, world_batch_divisor)
        broadcast_model(model)

        training_manager = TrainingManager(
            model,
            rank=dist_ctx.rank,
            world_size=dist_ctx.world_size,
            grad_accum_steps=dist_ctx.grad_accum_steps,
            device=dist_ctx.device,
            args=args,
            training_schedule=training_schedule,
            lr_mul=orth_state.lr_mul,
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
            args=args,
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
    main()

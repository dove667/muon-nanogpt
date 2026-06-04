
import argparse
import math
import os
import subprocess
from pathlib import Path

from src.utils import ROOT, TRAINING_ROOT as DEFAULT_TRAINING_ROOT, RUNS_ROOT as DEFAULT_RUNS_ROOT

FIXED_BATCH_TOKENS = 16 * 2048 * 8
FIXED_SEQ_LEN = 2048


def compute_train_steps(train_token_budget: int) -> int:
    return math.ceil(train_token_budget / FIXED_BATCH_TOKENS)


def default_group(orth: str) -> str:
    del orth
    return "fixed_t5"


def default_run_name(args: argparse.Namespace) -> str:
    if args.orth == "adamw":
        return f"adamw_seed{args.seed}"
    if args.orth == "vanilla":
        return f"vanilla_seed{args.seed}"
    if args.orth == "fast":
        return f"fast_seed{args.seed}"
    if args.orth == "manual":
        return f"manual_T{args.ns_t}_f{args.fast_steps}_s{args.stable_steps}_seed{args.seed}"
    if args.orth == "polar_express":
        return f"polar_express_T{args.pe_t}_l{args.pe_lower_bound}_seed{args.seed}"
    raise SystemExit(f"Unknown orth={args.orth}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch one configured training run.")
    parser.add_argument("--orth", choices=["adamw", "vanilla", "fast", "manual", "polar_express"], default=os.environ.get("ORTH", "fast"))
    parser.add_argument("--lr-mul", type=float, default=float(os.environ.get("LR_MUL", "1.0")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    parser.add_argument("--group", default=os.environ.get("WANDB_GROUP"))
    parser.add_argument("--name", default=os.environ.get("WANDB_NAME") or os.environ.get("RUN_NAME"))
    parser.add_argument("--train-token-budget", type=int, default=int(float(os.environ.get("TRAIN_TOKEN_BUDGET", "30000000"))))
    parser.add_argument("--eval-every-tokens", type=int, default=int(float(os.environ.get("EVAL_EVERY_TOKENS", "5000000"))))
    parser.add_argument("--eval-tokens", type=int, default=int(float(os.environ.get("EVAL_TOKENS", "1048576"))))
    parser.add_argument("--train-grad-accum-steps", type=int, default=int(os.environ.get("TRAIN_GRAD_ACCUM_STEPS", os.environ.get("SPEEDTEST_GRAD_ACCUM_STEPS", "32"))))
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-at-start", action="store_true", default=os.environ.get("EVAL_AT_START", "0").lower() in {"1", "true", "yes"})
    parser.add_argument("--log-every-steps", type=int, default=int(os.environ.get("LOG_EVERY_STEPS", "20")))
    parser.add_argument("--extension-steps", type=int, default=int(float(os.environ.get("EXTENSION_STEPS", "0"))))
    parser.add_argument("--val-loss-every-steps", type=int, default=int(os.environ.get("VAL_LOSS_EVERY_STEPS", "0")))
    parser.add_argument("--wandb", choices=["on", "off"], default="off" if os.environ.get("WANDB", "1").lower() in {"0", "false", "no", "disabled"} else "on")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "muon-nanogpt"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE"))
    parser.add_argument("--runs-root", default=os.environ.get("RUNS_ROOT", str(DEFAULT_RUNS_ROOT)))
    parser.add_argument("--training-root", default=os.environ.get("TRAINING_ROOT", str(DEFAULT_TRAINING_ROOT)))
    parser.add_argument("--trainer-py", default=os.environ.get("TRAINER_PY"))
    parser.add_argument("--nproc-per-node", type=int, default=int(os.environ.get("NPROC_PER_NODE", "1")))
    parser.add_argument("--data-path", default=os.environ.get("DATA_PATH"))
    parser.add_argument("--train-files", default=os.environ.get("TRAIN_FILES"))
    parser.add_argument("--val-files", default=os.environ.get("VAL_FILES"))
    parser.add_argument("--ns-t", type=int, default=int(os.environ.get("NS_T", "5")))
    parser.add_argument("--fast-steps", type=int, default=None)
    parser.add_argument("--stable-steps", type=int, default=None)
    parser.add_argument("--pe-t", type=int, default=int(os.environ.get("PE_T", "5")))
    parser.add_argument("--pe-lower-bound", default=os.environ.get("PE_LOWER_BOUND", "1e-3"))
    parser.add_argument("--pe-cushion", type=float, default=float(os.environ.get("PE_CUSHION", "2e-2")))
    parser.add_argument("--pe-safety-factor", type=float, default=float(os.environ.get("PE_SAFETY_FACTOR", "2e-2")))
    parser.add_argument("--model-max-seq-len", type=int, default=int(os.environ.get("MODEL_MAX_SEQ_LEN", "0")))
    parser.add_argument("--spectral-every-tokens", type=int, default=int(float(os.environ.get("SPECTRAL_EVERY_TOKENS", "10000000"))))
    parser.add_argument("--spectral-max-matrices", type=int, default=int(os.environ.get("SPECTRAL_MAX_MATRICES", "5")))
    parser.add_argument("--spectral-max-dim", type=int, default=int(os.environ.get("SPECTRAL_MAX_DIM", "1024")))
    args = parser.parse_args()
    if args.fast_steps is None:
        args.fast_steps = args.ns_t if args.orth == "manual" else args.fast_steps
    if args.stable_steps is None:
        args.stable_steps = max(args.ns_t - args.fast_steps, 0) if args.orth == "manual" else args.stable_steps
    if args.group is None:
        args.group = default_group(args.orth)
    if args.name is None:
        args.name = default_run_name(args)
    if args.eval_batch_size is None:
        args.eval_batch_size = args.train_grad_accum_steps * 2048
    return args


def prepare_env(args: argparse.Namespace) -> tuple[dict[str, str], Path, Path]:
    env = os.environ.copy()
    training_root = Path(args.training_root).resolve()
    trainer_py = Path(args.trainer_py).resolve() if args.trainer_py else (training_root / "train.py").resolve()
    run_dir = Path(args.runs_root).resolve() / args.group / args.name
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
    env["EXTENSION_STEPS"] = str(args.extension_steps)
    env["VAL_LOSS_EVERY_STEPS"] = str(args.val_loss_every_steps)
    env["WANDB"] = "1" if args.wandb == "on" else "0"
    env["WANDB_PROJECT"] = args.wandb_project
    env["WANDB_GROUP"] = args.group
    env["WANDB_NAME"] = args.name
    env["RUNS_ROOT"] = str(Path(args.runs_root).resolve())
    env["RUN_DIR"] = str(run_dir)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".torchinductor"))
    env["PYTHONPATH"] = str(ROOT) + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")

    if args.wandb_entity:
        env["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_mode:
        env["WANDB_MODE"] = args.wandb_mode
    elif args.wandb == "on" and not env.get("WANDB_API_KEY") and not Path.home().joinpath(".netrc").exists():
        env["WANDB_MODE"] = "offline"

    if args.data_path:
        env["DATA_PATH"] = args.data_path
    if args.train_files:
        env["TRAIN_FILES"] = args.train_files
    if args.val_files:
        env["VAL_FILES"] = args.val_files

    if args.orth == "manual":
        env["NS_T"] = str(args.ns_t)
        env["FAST_STEPS"] = str(args.fast_steps)
        env["STABLE_STEPS"] = str(args.stable_steps)
    elif args.orth == "polar_express":
        env["PE_T"] = str(args.pe_t)
        env["PE_LOWER_BOUND"] = str(args.pe_lower_bound)
        env["PE_CUSHION"] = str(args.pe_cushion)
        env["PE_SAFETY_FACTOR"] = str(args.pe_safety_factor)

    if args.model_max_seq_len > 0:
        env["MODEL_MAX_SEQ_LEN"] = str(args.model_max_seq_len)
    env["SPECTRAL_EVERY_TOKENS"] = str(args.spectral_every_tokens)
    env["SPECTRAL_MAX_MATRICES"] = str(args.spectral_max_matrices)
    env["SPECTRAL_MAX_DIM"] = str(args.spectral_max_dim)
    return env, training_root, trainer_py


def run() -> int:
    args = parse_args()
    env, training_root, trainer_py = prepare_env(args)
    print("=" * 80)
    print(f"run: {args.group}/{args.name}")
    print(f"orth: {args.orth}")
    print(f"train_token_budget: {args.train_token_budget}")
    print(f"train_steps: {env['TRAIN_STEPS']}")
    print(f"grad_accum: {args.train_grad_accum_steps}")
    print(f"eval_every_tokens: {args.eval_every_tokens}")
    print(f"eval_tokens: {args.eval_tokens}")
    print(f"run_dir: {env['RUN_DIR']}")
    print(f"trainer: {trainer_py}")
    print(f"training_root: {training_root}")
    print("=" * 80)
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        str(trainer_py),
    ]
    with open(Path(env["RUN_DIR"]) / "console.log", "a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=training_root,
            env={**env, "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0")},
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


if __name__ == "__main__":
    raise SystemExit(run())

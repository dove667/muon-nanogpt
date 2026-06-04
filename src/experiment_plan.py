
import argparse
import subprocess
import sys

from src.utils import ROOT
from src.plan.selection import run_completed
from src.plan.stages import (
    DEFAULT_BUDGET,
    DEFAULT_EVAL_EVERY,
    DEFAULT_EVAL_TOKENS,
    DEFAULT_FINAL_BUDGET,
    DEFAULT_FINAL_EVAL_EVERY,
    DEFAULT_FINAL_EVAL_TOKENS,
    DEFAULT_MAIN_BUDGET,
    DEFAULT_MAIN_EVAL_EVERY,
    DEFAULT_MAIN_EVAL_TOKENS,
    DEFAULT_MAIN_TOP_N,
    iter_core75,
    iter_final,
    iter_main,
    iter_main_final,
    iter_minimal,
    iter_p2_t10,
    iter_p2_t5,
    iter_p2_t69,
    iter_p2_t78,
    iter_pe_expand,
    iter_pe_init,
    iter_pe_iter_expand,
    iter_pe_lower_expand,
    iter_pe_lr_expand,
    iter_vanilla,
)


STAGES = {
    "vanilla": iter_vanilla,
    "p2_t5": iter_p2_t5,
    "pe_init": iter_pe_init,
    "p2_t10": iter_p2_t10,
    "p2_t78": iter_p2_t78,
    "p2_t69": iter_p2_t69,
    "pe_lower_expand": iter_pe_lower_expand,
    "pe_iter_expand": iter_pe_iter_expand,
    "pe_lr_expand": iter_pe_lr_expand,
    "pe_expand": iter_pe_expand,
    "main": iter_main,
    "final": iter_final,
    "minimal": iter_minimal,
    "core75": iter_core75,
    "main_final": iter_main_final,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the experiment plan with explicit command-line parameters.")
    parser.add_argument("stage", nargs="?", default="minimal", choices=sorted(STAGES))
    parser.add_argument("--skip-completed-runs", action="store_true")
    parser.add_argument("--train-token-budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--eval-every-tokens", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--eval-tokens", type=int, default=DEFAULT_EVAL_TOKENS)
    parser.add_argument("--train-grad-accum-steps", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-at-start", action="store_true")
    parser.add_argument("--log-every-steps", type=int, default=20)
    parser.add_argument("--extension-steps", type=int, default=0)
    parser.add_argument("--val-loss-every-steps", type=int, default=0)
    parser.add_argument("--wandb", choices=["on", "off"], default="on")
    parser.add_argument("--wandb-project", default="muon-nanogpt")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode")
    parser.add_argument("--runs-root")
    parser.add_argument("--training-root")
    parser.add_argument("--trainer-py")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--data-path")
    parser.add_argument("--train-files")
    parser.add_argument("--val-files")
    parser.add_argument("--model-max-seq-len", type=int, default=0)
    parser.add_argument("--spectral-every-tokens", type=int, default=10_000_000)
    parser.add_argument("--spectral-max-matrices", type=int, default=5)
    parser.add_argument("--spectral-max-dim", type=int, default=1024)
    parser.add_argument("--main-token-budget", type=int, default=DEFAULT_MAIN_BUDGET)
    parser.add_argument("--main-eval-every-tokens", type=int, default=DEFAULT_MAIN_EVAL_EVERY)
    parser.add_argument("--main-eval-tokens", type=int, default=DEFAULT_MAIN_EVAL_TOKENS)
    parser.add_argument("--main-top-n", type=int, default=DEFAULT_MAIN_TOP_N)
    parser.add_argument("--final-token-budget", type=int, default=DEFAULT_FINAL_BUDGET)
    parser.add_argument("--final-eval-every-tokens", type=int, default=DEFAULT_FINAL_EVAL_EVERY)
    parser.add_argument("--final-eval-tokens", type=int, default=DEFAULT_FINAL_EVAL_TOKENS)
    return parser.parse_args()


def launch(spec, args: argparse.Namespace) -> None:
    if args.skip_completed_runs and run_completed(spec.group, spec.name):
        print("=" * 80)
        print(f"skip completed: {spec.group}/{spec.name}")
        print("=" * 80)
        return

    command = [sys.executable, "-m", "src.run_training", *spec.to_cli_args()]
    command.extend([
        "--train-grad-accum-steps", str(args.train_grad_accum_steps),
        "--log-every-steps", str(args.log_every_steps),
        "--extension-steps", str(args.extension_steps),
        "--val-loss-every-steps", str(args.val_loss_every_steps),
        "--wandb", args.wandb,
        "--wandb-project", args.wandb_project,
        "--nproc-per-node", str(args.nproc_per_node),
        "--spectral-every-tokens", str(args.spectral_every_tokens),
        "--spectral-max-matrices", str(args.spectral_max_matrices),
        "--spectral-max-dim", str(args.spectral_max_dim),
    ])
    if args.eval_batch_size is not None:
        command.extend(["--eval-batch-size", str(args.eval_batch_size)])
    if args.eval_at_start:
        command.append("--eval-at-start")
    if args.wandb_entity:
        command.extend(["--wandb-entity", args.wandb_entity])
    if args.wandb_mode:
        command.extend(["--wandb-mode", args.wandb_mode])
    if args.runs_root:
        command.extend(["--runs-root", args.runs_root])
    if args.training_root:
        command.extend(["--training-root", args.training_root])
    if args.trainer_py:
        command.extend(["--trainer-py", args.trainer_py])
    if args.data_path:
        command.extend(["--data-path", args.data_path])
    if args.train_files:
        command.extend(["--train-files", args.train_files])
    if args.val_files:
        command.extend(["--val-files", args.val_files])
    if args.model_max_seq_len > 0:
        command.extend(["--model-max-seq-len", str(args.model_max_seq_len)])
    subprocess.run(command, cwd=ROOT, check=True)


def stage_specs(args: argparse.Namespace):
    common = dict(
        budget=args.train_token_budget,
        eval_every=args.eval_every_tokens,
        eval_tokens=args.eval_tokens,
    )
    if args.stage == "main":
        return iter_main(
            budget=args.main_token_budget,
            eval_every=args.main_eval_every_tokens,
            eval_tokens=args.main_eval_tokens,
            top_n=args.main_top_n,
        )
    if args.stage == "final":
        return iter_final(
            budget=args.final_token_budget,
            eval_every=args.final_eval_every_tokens,
            eval_tokens=args.final_eval_tokens,
        )
    if args.stage == "main_final":
        return iter_main_final(
            **common,
            main_budget=args.main_token_budget,
            main_eval_every=args.main_eval_every_tokens,
            main_eval_tokens=args.main_eval_tokens,
            main_top_n=args.main_top_n,
            final_budget=args.final_token_budget,
            final_eval_every=args.final_eval_every_tokens,
            final_eval_tokens=args.final_eval_tokens,
        )
    return STAGES[args.stage](**common)


def main() -> int:
    args = parse_args()
    for spec in stage_specs(args):
        launch(spec, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


import math
import os
import uuid
from dataclasses import dataclass
from itertools import accumulate, pairwise
from pathlib import Path

import torch


_DEFAULT_FINEWEB_DIR = Path(__file__).resolve().parents[1] / "data" / "fineweb10B"


def _candidate_data_roots(raw_base: str | None) -> list[Path]:
    roots: list[Path] = []
    if raw_base:
        base = Path(raw_base).expanduser()
        roots.extend([
            base,
            base / "fineweb10B",
            base / "src" / "data" / "fineweb10B",
        ])
    roots.append(_DEFAULT_FINEWEB_DIR)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve(strict=False)
        if resolved not in seen:
            deduped.append(root)
            seen.add(resolved)
    return deduped


def _resolve_split_pattern(split: str) -> str:
    env_name = "TRAIN_FILES" if split == "train" else "VAL_FILES"
    override = os.environ.get(env_name)
    if override:
        return override

    roots = _candidate_data_roots(os.environ.get("DATA_PATH"))
    filename = f"fineweb_{split}_*.bin"
    for root in roots:
        if any(root.glob(filename)):
            return str(root / filename)
    return str(roots[0] / filename)


@dataclass(slots=True)
class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", ".")
    train_files: str = _resolve_split_pattern("train")
    val_files: str = _resolve_split_pattern("val")
    val_tokens: int = int(float(os.environ.get("EVAL_TOKENS", 10485760)))
    val_batch_size: int = int(float(os.environ.get("EVAL_BATCH_SIZE", 2048)))
    num_scheduled_iterations: int = int(float(os.environ.get("TRAIN_STEPS", 1440)))
    num_extension_iterations: int = int(float(os.environ.get("EXTENSION_STEPS", 0)))
    run_id: str = os.environ.get("WANDB_NAME") or os.environ.get("RUN_NAME") or f"{uuid.uuid4()}"
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY_STEPS", "0"))
    save_checkpoint: bool = False
    bigram_vocab_size: int = 50304 * 5


@dataclass(slots=True)
class TrainingStage:
    lr_mul: float
    batch_size: int
    window_sizes: tuple[int, int]
    mtp_weights_start: list[float]
    mtp_weights_end: list[float]
    train_max_seq_len: int
    duration: float = None


class TrainingSchedule:
    def __init__(
        self,
        stages: list[TrainingStage],
        scheduled_iterations: int,
        extension_iterations: int,
        device,
        warmup_frac: float = 0.10,
        min_lr_frac: float = 0.10,
        split_embed_stage: int | None = None,
        ws_post_yarn_ext: int = 7,
    ):
        self.stages = stages
        self.scheduled_iterations = scheduled_iterations
        self.warmup_frac = warmup_frac
        self.min_lr_frac = min_lr_frac
        self.ws_post_yarn_ext = ws_post_yarn_ext
        self.total_steps = self.scheduled_iterations + extension_iterations
        if len(stages) == 1:
            self.boundaries = [(0, self.total_steps)]
        else:
            ends = [0, *[round(c * scheduled_iterations) for c in accumulate(s.duration for s in stages[:-1])], self.total_steps]
            assert self.scheduled_iterations == ends[-2]
            self.boundaries = list(pairwise(ends))
        self.split_step = None if split_embed_stage is None else (self.boundaries[split_embed_stage][0] | 1)
        self.mtp_weights = []
        for step in range(self.total_steps + 1):
            stage, t = self.lookup(step)
            w = [a + (b - a) * t for a, b in zip(stage.mtp_weights_start, stage.mtp_weights_end)]
            self.mtp_weights.append(torch.tensor(w, device=device))

    def lookup(self, step: int) -> tuple[TrainingStage, float]:
        for i, (start, end) in enumerate(self.boundaries):
            if step < end:
                t = (step - start) / (end - start)
                return self.stages[i], t
        return self.stages[-1], 1.0

    def get_lr(self, step: int) -> float:
        stage, _ = self.lookup(step)
        if self.scheduled_iterations <= 1:
            return stage.lr_mul

        warmup_steps = max(1, round(self.scheduled_iterations * self.warmup_frac))
        if step < warmup_steps:
            return stage.lr_mul * ((step + 1) / warmup_steps)

        progress = min(1.0, (step - warmup_steps) / max(1, self.scheduled_iterations - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
        decay = self.min_lr_frac + (1.0 - self.min_lr_frac) * cosine
        return stage.lr_mul * decay


def default_training_stages() -> list[TrainingStage]:
    return [
        TrainingStage(
            duration=1.0,
            train_max_seq_len=2048,
            batch_size=16 * 2048 * 8,
            window_sizes=(3, 7),
            lr_mul=1.0,
            mtp_weights_start=[1.0, 0.0],
            mtp_weights_end=[1.0, 0.0],
        ),
    ]


def get_muon_momentum(step: int, total_steps: int, muon_warmup_steps=300, muon_cooldown_steps=50, momentum_min=0.85, momentum_max=0.95):
    del step, total_steps, muon_warmup_steps, muon_cooldown_steps, momentum_min
    return momentum_max

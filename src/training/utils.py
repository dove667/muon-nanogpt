import random
from pathlib import Path

import numpy as np
import torch

from src.config import get_orthogonalization


def setup_device(*, base_seed: int) -> torch.device:
    assert torch.cuda.is_available()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed_all(base_seed)

    return device


def default_run_name(orth: str, seed: int) -> str:
    orth_cfg = get_orthogonalization()
    if orth == "adamw":
        return f"adamw_seed{seed}"
    if orth == "vanilla":
        return f"vanilla_seed{seed}"
    if orth == "fast":
        return f"fast_seed{seed}"
    if orth == "manual":
        return f"manual_f{orth_cfg.fast_steps}_s{orth_cfg.stable_steps}_seed{seed}"
    if orth == "polar_express":
        return f"polar_express_l{orth_cfg.pe_lower_bound}_seed{seed}"
    raise SystemExit(f"Unknown orth={orth}")


def resolve_data_path(data_path: str) -> tuple[str, str]:
    dp = Path(data_path)
    train_pattern = "fineweb_train_*.bin"
    val_pattern = "fineweb_val_*.bin"
    if not any(dp.glob(train_pattern)):
        raise FileNotFoundError(f"No training files matching {train_pattern} in {dp}")
    if not any(dp.glob(val_pattern)):
        raise FileNotFoundError(f"No validation files matching {val_pattern} in {dp}")
    return str(dp / train_pattern), str(dp / val_pattern)


def primary_lr(optimizer) -> float:
    for param_cfg in optimizer.param_cfgs.values():
        if param_cfg.optim == "normuon":
            return float(param_cfg.lr * param_cfg.lr_mul)
    return float("nan")

import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.config import get_orthogonalization

FIXED_SEED = 0


def setup_device(*, base_seed: int = FIXED_SEED) -> torch.device:
    assert torch.cuda.is_available()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed_all(base_seed)

    return device


def default_run_name(orth: str) -> str:
    orth_cfg = get_orthogonalization()
    timestamp = datetime.now().strftime("%m%d_%H%M")
    ns_iterations = int(orth_cfg._data.get("ns_iterations", orth_cfg.default_iterations))
    pe_iterations = int(orth_cfg._data.get("pe_iterations", ns_iterations))
    if orth == "adamw":
        return f"adamw_{timestamp}"
    if orth == "vanilla":
        return f"stable{ns_iterations}_{timestamp}"
    if orth == "fast":
        return f"fast{ns_iterations}_{timestamp}"
    if orth == "manual":
        return f"manual_T{ns_iterations}_f{orth_cfg.fast_steps}_s{orth_cfg.stable_steps}_{timestamp}"
    if orth == "polar_express":
        return f"pe_T{pe_iterations}_l{orth_cfg.pe_lower_bound}_{timestamp}"
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
    fallback_lr = None
    for param_cfg in optimizer.param_cfgs.values():
        if param_cfg.optim == "normuon":
            return float(param_cfg.lr * param_cfg.lr_mul)
        if fallback_lr is None and param_cfg.optim == "adam":
            fallback_lr = float(param_cfg.lr * param_cfg.lr_mul)
    return float("nan") if fallback_lr is None else fallback_lr

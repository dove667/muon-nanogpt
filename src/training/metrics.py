
import math
import time

import torch
from torch import nn


def current_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad = param.grad.detach().float()
            total += float(grad.square().sum().item())
    return math.sqrt(total)


def downsample_matrix_for_svd(mat: torch.Tensor, spectral_max_dim: int) -> torch.Tensor:
    mat = mat.detach().float()
    if spectral_max_dim <= 0:
        return mat
    rows, cols = mat.shape[-2], mat.shape[-1]
    if rows > spectral_max_dim:
        row_idx = torch.linspace(0, rows - 1, spectral_max_dim, device=mat.device).round().long()
        mat = mat.index_select(-2, row_idx)
    if cols > spectral_max_dim:
        col_idx = torch.linspace(0, cols - 1, spectral_max_dim, device=mat.device).round().long()
        mat = mat.index_select(-1, col_idx)
    return mat


@torch.no_grad()
def orthogonalized_copy_for_stats(mat: torch.Tensor, coeffs: list[tuple[float, float, float]], norm_factor: float, spectral_max_dim: int) -> torch.Tensor:
    x = downsample_matrix_for_svd(mat, spectral_max_dim)
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * norm_factor + 1e-6)
    for a, b, c in coeffs:
        if x.size(-2) > x.size(-1):
            gram = x.mT @ x
            poly = b * gram + c * (gram @ gram)
            x = a * x + x @ poly
        else:
            gram = x @ x.mT
            poly = b * gram + c * (gram @ gram)
            x = a * x + poly @ x
    return x.float()


def svd_summary(mat: torch.Tensor, spectral_max_dim: int) -> dict[str, float]:
    mat_cpu = downsample_matrix_for_svd(mat, spectral_max_dim).detach().float().cpu()
    if mat_cpu.numel() == 0:
        return {}
    sv = torch.linalg.svdvals(mat_cpu)
    if sv.numel() == 0:
        return {}
    sv_max = float(sv.max())
    sv_min = float(sv.min())
    sv_std = float(sv.std(unbiased=False))
    denom = max(sv_max * sv_max, 1e-30)
    stable_rank = float((sv.square().sum() / denom).item()) if sv_max > 0 else 0.0
    probs = sv / sv.sum().clamp_min(1e-30)
    entropy = float(-(probs * probs.clamp_min(1e-30).log()).sum().item())
    gram = mat_cpu.T @ mat_cpu if mat_cpu.shape[-2] >= mat_cpu.shape[-1] else mat_cpu @ mat_cpu.T
    eye = torch.eye(gram.shape[0], dtype=gram.dtype)
    orth_error = float((gram - eye).norm().item() / max(gram.shape[0] ** 0.5, 1.0))
    return {
        "sv_min": sv_min,
        "sv_max": sv_max,
        "sv_std": sv_std,
        "orth_error": orth_error,
        "stable_rank": stable_rank,
        "svd_entropy": entropy,
    }


def collect_spectral_metrics(
    optimizer,
    global_train_tokens: int,
    master_process: bool,
    spectral_max_matrices: int,
    spectral_max_dim: int,
    coeffs: list[tuple[float, float, float]],
    norm_factor: float,
) -> tuple[dict, list[dict]]:
    if not master_process or spectral_max_matrices <= 0:
        return {}, []

    t_start = time.perf_counter()
    summary = {"train/tokens": int(global_train_tokens)}
    aggregates: dict[str, list[float]] = {}
    detail_records: list[dict] = []
    sample_count = 0

    def add_aggregate(name: str, value: float) -> None:
        aggregates.setdefault(name, []).append(float(value))

    for param, param_cfg in optimizer.param_cfgs.items():
        if param_cfg.optim != "normuon":
            continue
        momentum = optimizer.param_states[param]["momentum_buffer"].detach()
        matrices = momentum.reshape(-1, momentum.shape[-2], momentum.shape[-1])
        candidate_idxs = sorted({0, matrices.shape[0] // 2, matrices.shape[0] - 1})
        for mat_idx in candidate_idxs:
            if sample_count >= spectral_max_matrices:
                break
            mat = matrices[mat_idx]
            mom_stats = svd_summary(mat, spectral_max_dim)
            upd_stats = svd_summary(
                orthogonalized_copy_for_stats(mat, coeffs, norm_factor, spectral_max_dim),
                spectral_max_dim,
            )
            if not mom_stats or not upd_stats:
                continue
            detail = {
                "train/tokens": int(global_train_tokens),
                "spec/label": param_cfg.label,
                "spec/matrix_index": int(mat_idx),
            }
            for key, value in mom_stats.items():
                detail[f"spec/momentum_{key}"] = value
                add_aggregate(f"momentum_{key}", value)
            for key, value in upd_stats.items():
                detail[f"spec/update_{key}"] = value
                add_aggregate(f"update_{key}", value)
            detail_records.append(detail)
            sample_count += 1
        if sample_count >= spectral_max_matrices:
            break

    if sample_count == 0:
        return {}, []

    for key, values in aggregates.items():
        summary[f"spec/{key}"] = float(sum(values) / len(values))
    summary["spec/sample_count"] = int(sample_count)
    summary["spec/time_s"] = float(time.perf_counter() - t_start)
    return summary, detail_records

import math
import time

import torch
from torch import nn

SPECTRAL_OBJECTS = ("buffer_post", "g_pre", "g_post")


def current_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad = param.grad.detach().float()
            total += float(grad.square().sum().item())
    return math.sqrt(total)


def downsample_matrix_for_svd(mat: torch.Tensor, svd_dim_cap: int) -> torch.Tensor:
    mat = mat.detach().float()
    if svd_dim_cap <= 0:
        return mat
    rows, cols = mat.shape[-2], mat.shape[-1]
    if rows > svd_dim_cap:
        row_idx = torch.linspace(0, rows - 1, svd_dim_cap, device=mat.device).round().long()
        mat = mat.index_select(-2, row_idx)
    if cols > svd_dim_cap:
        col_idx = torch.linspace(0, cols - 1, svd_dim_cap, device=mat.device).round().long()
        mat = mat.index_select(-1, col_idx)
    return mat


def semi_orthogonality_side(mat: torch.Tensor) -> str:
    return "cols" if mat.shape[-2] >= mat.shape[-1] else "rows"


def svd_summary(mat: torch.Tensor, svd_dim_cap: int) -> dict[str, float]:
    mat_cpu = downsample_matrix_for_svd(mat, svd_dim_cap).detach().float().cpu()
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

    # Semi-orthogonality is always measured on the shorter side Gram matrix:
    # X^T X for tall matrices, XX^T for wide matrices.
    gram = mat_cpu.T @ mat_cpu if mat_cpu.shape[-2] >= mat_cpu.shape[-1] else mat_cpu @ mat_cpu.T
    eye = torch.eye(gram.shape[0], dtype=gram.dtype)
    semi_orth_error = float((gram - eye).norm().item() / max(gram.shape[0] ** 0.5, 1.0))
    return {
        "sv_min": sv_min,
        "sv_max": sv_max,
        "sv_std": sv_std,
        "semi_orth_error": semi_orth_error,
        "stable_rank": stable_rank,
        "svd_entropy": entropy,
    }


def _candidate_indices(num_matrices: int) -> list[int]:
    if num_matrices <= 0:
        return []
    if num_matrices <= 3:
        return list(range(num_matrices))
    return sorted({0, num_matrices // 2, num_matrices - 1})


def _sample_positions(total_candidates: int, target_samples: int) -> list[int]:
    if total_candidates <= 0 or target_samples <= 0:
        return []
    if target_samples >= total_candidates:
        return list(range(total_candidates))
    if target_samples == 1:
        return [total_candidates // 2]
    positions: list[int] = []
    used: set[int] = set()
    for i in range(target_samples):
        pos = round(i * (total_candidates - 1) / (target_samples - 1))
        while pos in used and pos + 1 < total_candidates:
            pos += 1
        while pos in used and pos - 1 >= 0:
            pos -= 1
        if pos not in used:
            positions.append(pos)
            used.add(pos)
    return sorted(positions)


@torch.no_grad()
def collect_spectral_metrics(
    optimizer,
    global_train_tokens: int,
    master_process: bool,
    num_matrices: int,
    svd_dim_cap: int,
    captured_normuon_stats: dict,
) -> tuple[dict, list[dict]]:
    if not master_process or num_matrices <= 0:
        return {}, []

    t_start = time.perf_counter()
    summary = {"train/tokens": int(global_train_tokens)}
    aggregates: dict[str, list[float]] = {}
    detail_records: list[dict] = []
    candidates: list[dict] = []

    def add_aggregate(name: str, value: float) -> None:
        aggregates.setdefault(name, []).append(float(value))

    for param, param_cfg in optimizer.param_cfgs.items():
        if param_cfg.optim != "normuon":
            continue
        captured = captured_normuon_stats.get(param)
        if not captured:
            continue
        first_object = captured[SPECTRAL_OBJECTS[0]]
        matrices = first_object.reshape(-1, first_object.shape[-2], first_object.shape[-1])
        for mat_idx in _candidate_indices(matrices.shape[0]):
            candidates.append(
                {
                    "label": param_cfg.label,
                    "matrix_index": int(mat_idx),
                    "shape": tuple(int(dim) for dim in matrices[mat_idx].shape),
                    "semi_orth_side": semi_orthogonality_side(matrices[mat_idx]),
                    "objects": {
                        object_name: captured[object_name].reshape(
                            -1,
                            captured[object_name].shape[-2],
                            captured[object_name].shape[-1],
                        )[mat_idx]
                        for object_name in SPECTRAL_OBJECTS
                    },
                }
            )

    selected_positions = _sample_positions(len(candidates), num_matrices)
    for sample_slot, candidate_pos in enumerate(selected_positions):
        candidate = candidates[candidate_pos]
        detail = {
            "train/tokens": int(global_train_tokens),
            "spec/label": candidate["label"],
            "spec/matrix_index": int(candidate["matrix_index"]),
            "spec/sample_slot": int(sample_slot),
            "spec/candidate_position": int(candidate_pos),
            "spec/candidate_count": int(len(candidates)),
            "spec/rows": int(candidate["shape"][0]),
            "spec/cols": int(candidate["shape"][1]),
            "spec/semi_orth_side": candidate["semi_orth_side"],
        }
        keep_candidate = True
        candidate_metrics: dict[str, float] = {}
        for object_name, mat in candidate["objects"].items():
            stats = svd_summary(mat, svd_dim_cap)
            if not stats:
                keep_candidate = False
                break
            for key, value in stats.items():
                metric_name = f"{object_name}_{key}"
                candidate_metrics[metric_name] = value
        if keep_candidate:
            for metric_name, value in candidate_metrics.items():
                detail[f"spec/{metric_name}"] = value
                add_aggregate(metric_name, value)
            detail_records.append(detail)

    if not detail_records:
        return {}, []

    for key, values in aggregates.items():
        summary[f"spec/{key}"] = float(sum(values) / len(values))
    summary["spec/sample_count"] = int(len(detail_records))
    summary["spec/candidate_count"] = int(len(candidates))
    summary["spec/time_s"] = float(time.perf_counter() - t_start)
    return summary, detail_records

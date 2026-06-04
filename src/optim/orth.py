import math

import numpy as np

FAST_COEFF = (3.4445, -4.7750, 2.0315)
STABLE_COEFF = (2.0, -1.5, 0.5)


def optimal_quintic(l: float, u: float) -> tuple[float, float, float]:
    assert 0 <= l <= u
    if 1 - 5e-6 <= l / u:
        return (15 / 8) / u, (-10 / 8) / (u ** 3), (3 / 8) / (u ** 5)
    q = (3 * l + u) / 4
    r = (l + 3 * u) / 4
    err = float("inf")
    old_err = None
    while old_err is None or abs(old_err - err) > 1e-15:
        old_err = err
        lhs = np.array([
            [l, l**3, l**5, 1],
            [q, q**3, q**5, -1],
            [r, r**3, r**5, 1],
            [u, u**3, u**5, -1],
        ])
        a, b, c, err = np.linalg.solve(lhs, np.ones(4))
        q, r = np.sqrt((-3 * b + np.array([-1, 1]) * math.sqrt(9 * b**2 - 20 * a * c)) / (10 * c))
    return float(a), float(b), float(c)


def polar_express_coefficients(lower_bound: float, num_iters: int, safety_factor_eps: float, cushion: float) -> list[tuple[float, float, float]]:
    u = 1.0
    l = lower_bound
    assert 0 <= l <= u
    safety_factor = 1.0 + safety_factor_eps
    coeffs = []
    for iter_idx in range(num_iters):
        a, b, c = optimal_quintic(max(l, cushion * u), u)
        if cushion * u > l:
            pl = a * l + b * l**3 + c * l**5
            pu = a * u + b * u**3 + c * u**5
            rescaler = 2 / (pl + pu)
            a *= rescaler
            b *= rescaler
            c *= rescaler
        if iter_idx < num_iters - 1:
            a /= safety_factor
            b /= safety_factor**3
            c /= safety_factor**5
        coeffs.append((a, b, c))
        l = a * l + b * l**3 + c * l**5
        u = 2 - l
    return coeffs


def build_coeff_schedule(
    orth_mode: str,
    fast_steps: int,
    stable_steps: int,
    pe_lower_bound_raw: str,
    pe_cushion: float,
    pe_safety_factor: float,
) -> list[tuple[float, float, float]]:
    if orth_mode == "adamw":
        coeffs = []
    elif orth_mode == "vanilla":
        coeffs = [STABLE_COEFF] * 5
    elif orth_mode == "fast":
        coeffs = [FAST_COEFF] * 5
    elif orth_mode == "manual":
        if fast_steps < 0 or stable_steps < 0:
            raise RuntimeError("manual schedule counts must be non-negative")
        if fast_steps + stable_steps != 5:
            raise RuntimeError(f"FAST_STEPS + STABLE_STEPS must equal 5, got {fast_steps}+{stable_steps}!={5}")
        coeffs = [FAST_COEFF] * fast_steps + [STABLE_COEFF] * stable_steps
    elif orth_mode == "polar_express":
        lower_bound = float(pe_lower_bound_raw)
        coeffs = polar_express_coefficients(lower_bound, 5, pe_safety_factor, pe_cushion)
    else:
        raise RuntimeError(f"unknown ORTH={orth_mode!r}; expected adamw, vanilla, fast, manual, or polar_express")
    return coeffs


def orth_schedule_name(
    orth_mode: str,
    pe_lower_bound_raw: str,
    *,
    fast_steps: int,
    stable_steps: int,
) -> str:
    if orth_mode in {"adamw", "vanilla", "fast"}:
        return orth_mode
    if orth_mode == "manual":
        return f"manual_f{fast_steps}_s{stable_steps}"
    if orth_mode == "polar_express":
        return f"pe_l{pe_lower_bound_raw}"
    raise RuntimeError(f"unknown ORTH={orth_mode!r}; expected adamw, vanilla, fast, manual, or polar_express")


def orth_norm_factor(orth_mode: str, pe_safety_factor: float) -> float:
    return 1.0 + (pe_safety_factor if orth_mode == "polar_express" else 0.0)


def orth_record(
    orth_mode: str,
    coeff_schedule: list[tuple[float, float, float]],
    *,
    fast_steps: int,
    stable_steps: int,
    pe_lower_bound_raw: str,
    pe_cushion: float,
    pe_safety_factor: float,
    lr_mul: float,
) -> dict[str, object]:
    schedule_name = orth_schedule_name(
        orth_mode,
        pe_lower_bound_raw,
        fast_steps=fast_steps,
        stable_steps=stable_steps,
    )
    if orth_mode == "manual":
        record_fast_steps = fast_steps
        record_stable_steps = stable_steps
    elif orth_mode == "fast":
        record_fast_steps = len(coeff_schedule)
        record_stable_steps = 0
    elif orth_mode == "vanilla":
        record_fast_steps = 0
        record_stable_steps = len(coeff_schedule)
    else:
        record_fast_steps = None
        record_stable_steps = None
    return {
        "orthogonalizer_type": orth_mode,
        "orth_schedule_name": schedule_name,
        "coeff_schedule": coeff_schedule,
        "T_ns": len(coeff_schedule),
        "fast_steps": record_fast_steps,
        "stable_steps": record_stable_steps,
        "pe_T": len(coeff_schedule) if orth_mode == "polar_express" else None,
        "pe_lower_bound": pe_lower_bound_raw if orth_mode == "polar_express" else None,
        "pe_cushion": pe_cushion if orth_mode == "polar_express" else None,
        "pe_safety_factor": pe_safety_factor if orth_mode == "polar_express" else None,
        "lr_mul": lr_mul,
    }

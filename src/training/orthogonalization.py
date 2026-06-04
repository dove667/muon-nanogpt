
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FAST_COEFF = (3.4445, -4.7750, 2.0315)
STABLE_COEFF = (2.0, -1.5, 0.5)


@dataclass(frozen=True, slots=True)
class OrthogonalizerConfig:
    orth_mode: str
    fast_steps: int
    stable_steps: int
    pe_lower_bound_raw: str
    pe_cushion: float
    pe_safety_factor: float
    lr_mul: float
    coeff_schedule: list[tuple[float, float, float]]
    norm_safety_eps: float
    schedule_name: str

    @property
    def norm_factor(self) -> float:
        return 1.0 + self.norm_safety_eps

    def to_record(self) -> dict[str, object]:
        if self.orth_mode == "manual":
            fast_steps = self.fast_steps
            stable_steps = self.stable_steps
        elif self.orth_mode == "fast":
            fast_steps = len(self.coeff_schedule)
            stable_steps = 0
        elif self.orth_mode == "vanilla":
            fast_steps = 0
            stable_steps = len(self.coeff_schedule)
        else:
            fast_steps = None
            stable_steps = None
        return {
            "orthogonalizer_type": self.orth_mode,
            "orth_schedule_name": self.schedule_name,
            "coeff_schedule": self.coeff_schedule,
            "T_ns": len(self.coeff_schedule),
            "fast_steps": fast_steps,
            "stable_steps": stable_steps,
            "pe_T": len(self.coeff_schedule) if self.orth_mode == "polar_express" else None,
            "pe_lower_bound": self.pe_lower_bound_raw if self.orth_mode == "polar_express" else None,
            "pe_cushion": self.pe_cushion if self.orth_mode == "polar_express" else None,
            "pe_safety_factor": self.pe_safety_factor if self.orth_mode == "polar_express" else None,
            "lr_mul": self.lr_mul,
        }


def _parse_lower_bound(value: str) -> float:
    if value == "lemp":
        empirical = os.environ.get("PE_EMP_LOWER_BOUND")
        if empirical is None:
            path = os.environ.get("PE_EMP_LOWER_BOUND_FILE")
            if path:
                empirical = Path(path).read_text(encoding="utf-8").strip()
        if empirical is None:
            raise RuntimeError("PE_LOWER_BOUND=lemp requires PE_EMP_LOWER_BOUND or PE_EMP_LOWER_BOUND_FILE")
        value = empirical
    return float(value)


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


def build_orthogonalizer_config_from_env() -> OrthogonalizerConfig:
    orth_mode = os.environ.get("ORTH", "fast").strip().lower()
    fast_steps = int(os.environ.get("FAST_STEPS", "5"))
    stable_steps = int(os.environ.get("STABLE_STEPS", str(max(5 - fast_steps, 0))))
    pe_lower_bound_raw = os.environ.get("PE_LOWER_BOUND", "1e-3").strip().lower()
    pe_cushion = float(os.environ.get("PE_CUSHION", "2e-2"))
    pe_safety_factor = float(os.environ.get("PE_SAFETY_FACTOR", "2e-2"))
    lr_mul = float(os.environ.get("LR_MUL", "1.0"))

    if orth_mode == "adamw":
        coeffs, eps, schedule = [], 0.0, "adamw"
    elif orth_mode == "vanilla":
        coeffs, eps, schedule = [STABLE_COEFF] * 5, 0.0, "vanilla"
    elif orth_mode == "fast":
        coeffs, eps, schedule = [FAST_COEFF] * 5, 0.0, "fast"
    elif orth_mode == "manual":
        if fast_steps < 0 or stable_steps < 0:
            raise RuntimeError("manual schedule counts must be non-negative")
        if fast_steps + stable_steps != 5:
            raise RuntimeError(f"FAST_STEPS + STABLE_STEPS must equal 5, got {fast_steps}+{stable_steps}!={5}")
        coeffs, eps, schedule = [FAST_COEFF] * fast_steps + [STABLE_COEFF] * stable_steps, 0.0, f"manual_f{fast_steps}_s{stable_steps}"
    elif orth_mode == "polar_express":
        lower_bound = _parse_lower_bound(pe_lower_bound_raw)
        coeffs = polar_express_coefficients(lower_bound, 5, pe_safety_factor, pe_cushion)
        eps, schedule = pe_safety_factor, f"pe_l{pe_lower_bound_raw}"
    else:
        raise RuntimeError(f"unknown ORTH={orth_mode!r}; expected adamw, vanilla, fast, manual, or polar_express")

    return OrthogonalizerConfig(
        orth_mode=orth_mode,
        fast_steps=fast_steps,
        stable_steps=stable_steps,
        pe_lower_bound_raw=pe_lower_bound_raw,
        pe_cushion=pe_cushion,
        pe_safety_factor=pe_safety_factor,
        lr_mul=lr_mul,
        coeff_schedule=coeffs,
        norm_safety_eps=eps,
        schedule_name=schedule,
    )

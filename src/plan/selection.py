
import json
import os
import re
from pathlib import Path

from src.utils import read_jsonl, RUNS_ROOT as DEFAULT_RUNS_ROOT

RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", DEFAULT_RUNS_ROOT))


def is_completed_metrics(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    rows = read_jsonl(path)
    return bool(rows) and rows[-1].get("status") == "completed"


def run_completed(group: str, name: str) -> bool:
    return is_completed_metrics(RUNS_ROOT / group / name / "metrics.jsonl")


def run_completed_any_group(name: str) -> bool:
    for metrics_path in RUNS_ROOT.glob(f"*/{name}/metrics.jsonl"):
        if is_completed_metrics(metrics_path):
            return True
    return False


def completed_records(excluded_groups=None, included_groups=None) -> list[dict]:
    excluded_groups = excluded_groups or set()
    records = []
    for metrics_path in RUNS_ROOT.glob("*/*/metrics.jsonl"):
        group = metrics_path.parent.parent.name
        if included_groups is not None and group not in included_groups:
            continue
        if group in excluded_groups:
            continue
        config_path = metrics_path.parent / "config.json"
        if not config_path.exists() or not is_completed_metrics(metrics_path):
            continue
        config = json.loads(config_path.read_text())
        vals = [row["val/loss"] for row in read_jsonl(metrics_path) if "val/loss" in row]
        if not vals:
            continue
        records.append({
            "val": vals[-1],
            "config": config,
            "group": group,
            "name": config.get("run_name") or metrics_path.parent.name,
        })
    return records


def best_record(records, predicate):
    pool = [record for record in records if predicate(record)]
    return min(pool, key=lambda record: record["val"]) if pool else None


def record_key(record: dict) -> tuple:
    config = record["config"]
    orth = config.get("orthogonalizer_type")
    lr = float(config.get("lr_mul", 1.0))
    if orth == "vanilla":
        return ("manual", 5, 5, 0, lr)
    if orth == "manual":
        return ("manual", int(config.get("T_ns")), int(config.get("fast_steps")), int(config.get("stable_steps")), lr)
    if orth == "polar_express":
        return ("pe", int(config.get("pe_T")), str(config.get("pe_lower_bound")), lr)
    return (orth, config.get("orth_schedule_name"), lr)


def best_pe_t5_lower_bound() -> str:
    candidates = []
    pattern = re.compile(r"pe_T5_l(?P<lb>.+)_lr1\.0_seed0$")
    for metrics_path in RUNS_ROOT.glob("*/pe_T5_l*_lr1.0_seed0/metrics.jsonl"):
        config_path = metrics_path.parent / "config.json"
        if not config_path.exists() or not is_completed_metrics(metrics_path):
            continue
        config = json.loads(config_path.read_text())
        if config.get("orthogonalizer_type") != "polar_express":
            continue
        vals = [row["val/loss"] for row in read_jsonl(metrics_path) if "val/loss" in row]
        if not vals:
            continue
        match = pattern.match(config.get("run_name") or metrics_path.parent.name)
        if not match:
            continue
        if str(config.get("pe_T")) == "5" and abs(float(config.get("lr_mul", 1.0)) - 1.0) < 1e-12:
            candidates.append((vals[-1], match.group("lb")))
    return min(candidates, key=lambda item: item[0])[1] if candidates else "1e-4"


def top_pe_lr_expand_specs() -> list[tuple[int, str]]:
    candidates = []
    pattern = re.compile(r"pe_T(?P<T>\d+)_l(?P<lb>.+)_lr1\.0_seed0$")
    for metrics_path in RUNS_ROOT.glob("*/pe_T*_l*_lr1.0_seed0/metrics.jsonl"):
        config_path = metrics_path.parent / "config.json"
        if not config_path.exists() or not is_completed_metrics(metrics_path):
            continue
        config = json.loads(config_path.read_text())
        if config.get("orthogonalizer_type") != "polar_express":
            continue
        try:
            lr_mul = float(config.get("lr_mul", 1.0))
        except Exception:
            continue
        if abs(lr_mul - 1.0) > 1e-12:
            continue
        vals = [row["val/loss"] for row in read_jsonl(metrics_path) if "val/loss" in row]
        if not vals:
            continue
        match = pattern.match(config.get("run_name") or metrics_path.parent.name)
        if not match:
            continue
        candidates.append((vals[-1], int(match.group("T")), match.group("lb")))
    seen = set()
    specs = []
    for _, pe_t, lower_bound in sorted(candidates):
        key = (pe_t, lower_bound)
        if key in seen:
            continue
        seen.add(key)
        specs.append(key)
        if len(specs) >= 3:
            break
    return specs

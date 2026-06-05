import csv
import json
from pathlib import Path

from src.paths import ROOT, read_jsonl


def _detect_mode(rows: list[dict]) -> str:
    has_benchmark = any("benchmark/wall_clock_s" in row for row in rows)
    has_spectral = any("spec/sample_count" in row for row in rows)
    if has_benchmark and has_spectral:
        raise ValueError("Mixed benchmark+spectrum runs are not supported.")
    if has_benchmark:
        return "benchmark"
    if has_spectral:
        return "spectral"
    return "train"


def main() -> int:
    runs_dir = (ROOT / "runs").resolve()
    out_dir = (ROOT / "results").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not runs_dir.exists():
        print(f"No runs directory found: {runs_dir}")
        return 0

    all_rows: list[dict] = []
    fieldnames: set[str] = set()

    for detail_path in sorted(runs_dir.rglob("spectral_details.jsonl")):
        run_dir = detail_path.parent
        metrics_path = run_dir / "metrics.jsonl"
        config_path = run_dir / "config.json"
        if not metrics_path.exists() or not config_path.exists():
            raise FileNotFoundError(f"Missing metrics/config next to {detail_path}")

        metrics_rows = read_jsonl(metrics_path)
        mode = _detect_mode(metrics_rows)
        if mode != "spectral":
            continue

        config = json.loads(config_path.read_text(encoding="utf-8"))
        for row in read_jsonl(detail_path):
            enriched = {
                "run": str(run_dir.relative_to(ROOT)),
                "run_name": config["run_name"],
                "orthogonalizer_type": config["orthogonalizer_type"],
                "orth_schedule_name": config.get("orth_schedule_name"),
                **row,
            }
            all_rows.append(enriched)
            fieldnames.update(enriched.keys())

    if not all_rows:
        print(f"No spectral detail logs found under {runs_dir}")
        return 0

    ordered_prefix = [
        "run",
        "run_name",
        "orthogonalizer_type",
        "orth_schedule_name",
        "train/tokens",
        "spec/label",
        "spec/matrix_index",
        "spec/sample_slot",
        "spec/candidate_position",
        "spec/candidate_count",
        "spec/rows",
        "spec/cols",
        "spec/semi_orth_side",
    ]
    remaining = sorted(name for name in fieldnames if name not in ordered_prefix)
    out_path = out_dir / "spectral_details.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*ordered_prefix, *remaining])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

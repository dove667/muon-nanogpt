import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "runs"
TRAINING_ROOT = ROOT / "src" / "training"
RESULTS_ROOT = ROOT / "results"
ARCHIVE_RESULTS_ROOT = ROOT / "5090_results"
DATA_ROOT = ROOT / "src" / "data"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

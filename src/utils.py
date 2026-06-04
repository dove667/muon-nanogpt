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


def resolve_data_files(data_path: str | Path | None = None) -> tuple[str, str]:
    if data_path is None:
        raise ValueError("data_path is required")
    data_path = Path(data_path)
    train_pattern = "fineweb_train_*.bin"
    val_pattern = "fineweb_val_*.bin"
    if not any(data_path.glob(train_pattern)):
        raise FileNotFoundError(f"No training files matching {train_pattern} in {data_path}")
    if not any(data_path.glob(val_pattern)):
        raise FileNotFoundError(f"No validation files matching {val_pattern} in {data_path}")
    return str(data_path / train_pattern), str(data_path / val_pattern)

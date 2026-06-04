import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "runs"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def _resolve_data_files(data_path: str) -> tuple[str, str]:
    dp = Path(data_path)
    train_pattern = "fineweb_train_*.bin"
    val_pattern = "fineweb_val_*.bin"
    if not any(dp.glob(train_pattern)):
        raise FileNotFoundError(f"No training files matching {train_pattern} in {dp}")
    if not any(dp.glob(val_pattern)):
        raise FileNotFoundError(f"No validation files matching {val_pattern} in {dp}")
    return str(dp / train_pattern), str(dp / val_pattern)
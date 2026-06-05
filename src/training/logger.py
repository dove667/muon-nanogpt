from pathlib import Path
import json
    
from src.config import TRAINING

class Logger:
    def __init__(
        self,
        *,
        run_name: str,
        seed: int,
        base_lr: float,
        orth_record: dict[str, object],
        run_dir: Path,
    ) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.run_dir / "metrics.jsonl"
        self.spectral_details_file = self.run_dir / "spectral_details.jsonl"
        self.config_file = self.run_dir / "config.json"

        run_config = {
            "run_name": run_name,
            "seed": seed,
            "base_lr": base_lr,
            "train_token_budget": int(TRAINING.train_token_budget),
            "tokens_per_step": int(TRAINING.tokens_per_step),
            "seq_len": int(TRAINING.seq_len),
            "grad_accum_steps": int(TRAINING.grad_accum_steps),
            **orth_record,
        }
        self.config_file.write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")

    def log_metric(self, record: dict) -> None:
        with self.metrics_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def log_spectral(self, summary_record: dict) -> None:
        if summary_record:
            self.log_metric(summary_record)

    def log_spectral_details(self, detail_records: list[dict]) -> None:
        if not detail_records:
            return
        with self.spectral_details_file.open("a", encoding="utf-8") as handle:
            for record in detail_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

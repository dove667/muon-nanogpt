import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


@dataclass(slots=True, frozen=True)
class _Loader:
    _data: dict[str, Any]

    @staticmethod
    def load(path: Path = _CONFIG_PATH) -> "_Loader":
        with open(path) as f:
            return _Loader(yaml.safe_load(f))

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        value = self._data[name]
        if isinstance(value, dict):
            return _Loader(value)
        return value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def items(self):
        return self._data.items()


_cfg = _Loader.load()


def get_training() -> _Loader:
    return _cfg.training

def get_model() -> _Loader:
    return _cfg.model

def get_optimizer() -> _Loader:
    return _cfg.optimizer

def get_orthogonalization() -> _Loader:
    return _cfg.orthogonalization


TRAINING = get_training()
MODEL = get_model()
OPTIMIZER = get_optimizer()

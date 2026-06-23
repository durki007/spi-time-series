from __future__ import annotations

from pathlib import Path

import yaml
from sklearn.base import BaseEstimator

from spi_time_series.config.schema import (
    ESTIMATOR_ALLOWLIST,
    ModelConfig,
    RunConfig,
)


def load_config(path: Path | str) -> RunConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RunConfig.model_validate(raw)


def save_config(config: RunConfig, path: Path | str) -> None:
    config.to_yaml(Path(path))


def build_estimator(model_cfg: ModelConfig) -> BaseEstimator:
    cls = ESTIMATOR_ALLOWLIST[model_cfg.model_type]
    return cls(**model_cfg.params)

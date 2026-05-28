from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.base import BaseEstimator

from spi_time_series.data.schemas import FeatureSet


@dataclass
class PipelineState:
    """Serializable snapshot of a Pipeline's fitted state.

    Stores per-model results so that individual models can be reused
    without retraining when their inputs are unchanged.
    """

    features: FeatureSet | None = None

    # Per-model storage (keyed by model name)
    optimized_models: dict[str, BaseEstimator] = field(default_factory=dict)
    trained_models: dict[str, Any] = field(default_factory=dict)

    # Cache keys
    extract_key: str | None = None
    search_keys: dict[str, str] = field(default_factory=dict)
    train_keys: dict[str, str] = field(default_factory=dict)

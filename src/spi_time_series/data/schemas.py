from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline as SklearnPipeline

type EventLog = pd.DataFrame
type FeatureMatrix = pd.DataFrame
type LabelSeries = pd.Series


@dataclass(frozen=True)
class RawData:
    """XES event log loaded verbatim from disk."""

    event_log: EventLog


@dataclass(frozen=True)
class TraceSample:
    case_id: str
    data: np.ndarray
    prefix_indexes: Iterable[tuple[int, int]]


@dataclass(frozen=True)
class PreprocessedData:
    """Cleaned event log split into train and test case sets."""

    train_log: Iterable[TraceSample]
    num_train_cases: int
    test_log: Iterable[TraceSample]
    num_test_cases: int
    col_idx: dict[str, int]  # mapping from column name to its index
    cleaned_log: EventLog = None  # type: ignore[assignment]


@dataclass(frozen=True)
class FeatureSet:
    """Feature matrices and labels for train and test splits."""

    X_train: FeatureMatrix
    X_test: FeatureMatrix
    y_train: LabelSeries
    y_test: LabelSeries
    feature_names: list[str]


@dataclass(frozen=True)
class ModelArtifact:
    """Trained scikit-learn pipelines keyed by model name."""

    models: dict[str, SklearnPipeline]
    feature_names: list[str]
    target_col: str


@dataclass
class EvaluationReport:
    """Evaluation metrics per model name and prefix length."""

    metrics: dict[str, dict[int, dict[str, float]]]
    model_names: list[str]
    prefix_lengths: list[int]


class WindowGenerator(Protocol):
    def __call__(self, trace: np.ndarray) -> Iterable[tuple[int, int]]: ...


class TargetGenerator(Protocol):
    def __call__(
        self,
        trace: np.ndarray,
        start_idx: int,
        end_idx: int,
        col_idx_mapping: dict[str, int],
    ) -> float | str: ...


class PrefixFeature(Protocol):
    def __call__(
        self, prefix: np.ndarray, col_idx_mapping: dict[str, int]
    ) -> pd.Series: ...

    def fit(
        self,
        event_log: Iterable[TraceSample],
        col_idx_mapping: dict[str, int],
        **config_kwargs,
    ): ...

    def name(self) -> str: ...

    feature_names: list[str]

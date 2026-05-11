from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

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
class PrefixSample:
    case_id: str
    prefix: EventLog
    target: float | str


@dataclass(frozen=True)
class PreprocessedData:
    """Cleaned event log split into train and test case sets."""

    train_log: Iterator[PrefixSample]
    test_log: Iterator[PrefixSample]
    activity_col: str
    timestamp_col: str


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
    def __call__(self, trace: pd.DataFrame) -> Iterator[pd.DataFrame]: ...


class TargetGenerator(Protocol):
    def __call__(
        self, trace: pd.DataFrame, prefix: pd.DataFrame
    ) -> float | str: ...

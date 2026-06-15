from collections.abc import Iterable
from dataclasses import dataclass, field
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
    prefix_indexes: np.ndarray


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
    """Evaluation metrics per model name and prefix length.

    Attributes:
        prefix_metrics: Per-model, per-prefix-length metric dictionaries
            (``{model_name: {prefix_length: {metric: value}}}``).
        model_metrics: Per-model metric dictionaries (``{model_name: {metric: value}}``).
        model_names: Ordered list of evaluated model names.
        prefix_lengths: Sorted list of prefix lengths present in the test set.
        prefix_counts: Number of test samples per prefix length
            (``{prefix_length: count}``).  Used to compute sample-weighted
            aggregate scores in :func:`~spi_time_series.evaluation.metrics.compare_models`.
    """

    # metrics per model per prefix
    prefix_metrics: dict[str, dict[int, dict[str, float]]] = field(
        default_factory=dict
    )
    # metrics per model
    model_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    model_names: list[str] = field(default_factory=list)
    prefix_lengths: list[int] = field(default_factory=list)
    prefix_counts: dict[int, int] = field(default_factory=dict)


@dataclass
class BestPrefixInfo:
    """Identifies the prefix length where predictive performance plateaus
    for a single model.

    Attributes:
        model_name: Name of the model.
        plateau_prefix: Smallest prefix length where the relative improvement
            (delta) of *metric* falls below *plateau_threshold*.
        metric: Name of the metric used to determine the plateau (e.g. ``rmse``).
        value: Metric value at the plateau prefix.
        plateau_threshold: Fractional threshold for relative improvement.
    """

    model_name: str
    plateau_prefix: int
    metric: str
    value: float
    plateau_threshold: float


@dataclass
class ModelRankEntry:
    """Aggregated performance for a single model across all prefix lengths.

    Attributes:
        model_name: Name of the model.
        aggregate_score: Mean of *metric* across all evaluated prefix lengths.
        metric: Name of the aggregation metric (e.g. ``rmse``).
        rank: 1-based rank within the comparison (1 = best).
    """

    model_name: str
    aggregate_score: float
    metric: str
    rank: int


@dataclass
class ModelComparisonResult:
    """Structured comparison of trained models and their optimal prefix lengths.

    Produced by :func:`spi_time_series.evaluation.metrics.compare_models`.

    Attributes:
        task: The task type (``regression`` or ``classification``).
        best_model: Model name with the best aggregate score.
        best_model_score: Aggregate score of the best model.
        ranking_metric: Metric used for ranking and plateau detection.
        model_rankings: Sorted list of per-model aggregate scores.
        best_prefixes: Per-model plateau information keyed by model name.
    """

    task: str
    best_model: str
    best_model_score: float
    ranking_metric: str
    model_rankings: list[ModelRankEntry] = field(default_factory=list)
    best_prefixes: dict[str, BestPrefixInfo] = field(default_factory=dict)


class WindowGenerator(Protocol):
    def __call__(self, trace: np.ndarray) -> np.ndarray: ...


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

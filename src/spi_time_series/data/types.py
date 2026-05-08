"""Callable type aliases for the Pipeline strategy interfaces.

Each alias describes the contract a pipeline stage must satisfy.
Pass plain functions or callable objects — no subclassing required.
"""

from collections.abc import Callable
from pathlib import Path

from spi_time_series.data.schemas import (
    EvaluationReport,
    EventLog,
    FeatureSet,
    ModelArtifact,
    PreprocessedData,
    RawData,
)

type Preprocessor = Callable[[RawData], EventLog]
type Splitter = Callable[[EventLog], PreprocessedData]
type FeatureExtractor = Callable[[PreprocessedData], FeatureSet]
type Evaluator = Callable[[ModelArtifact, FeatureSet], EvaluationReport]
type Reporter = Callable[[ModelArtifact, EvaluationReport, Path | None], None]

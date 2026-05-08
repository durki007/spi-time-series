from __future__ import annotations

from typing import Self

from sklearn.base import BaseEstimator

from spi_time_series.data import Dataset
from spi_time_series.data.types import (
    Evaluator,
    FeatureExtractor,
    Preprocessor,
    Reporter,
    Splitter,
)
from spi_time_series.pipeline.pipeline import Pipeline


class PipelineBuilder:
    """Fluent builder for constructing a Pipeline.

    Call ``with_*`` / ``add_*`` methods in any order, then call ``build()``
    to validate and return the assembled Pipeline.

    Required before ``build()``: ``with_dataset``, ``with_preprocessor``,
    ``with_splitter``, ``with_feature_extractor``.
    """

    def __init__(self) -> None:
        self._dataset: Dataset | None = None
        self._preprocessor: Preprocessor | None = None
        self._splitter: Splitter | None = None
        self._feature_extractor: FeatureExtractor | None = None
        self._models: dict[str, BaseEstimator] = {}
        self._evaluators: list[Evaluator] = []
        self._reporters: list[Reporter] = []

    def with_dataset(self, dataset: Dataset) -> Self:
        """Set the dataset that supplies the raw event log."""
        self._dataset = dataset
        return self

    def with_preprocessor(self, fn: Preprocessor) -> Self:
        """Set the preprocessing strategy: ``(RawData) -> EventLog``."""
        self._preprocessor = fn
        return self

    def with_splitter(self, fn: Splitter) -> Self:
        """Set the splitting strategy: ``(EventLog) -> PreprocessedData``."""
        self._splitter = fn
        return self

    def with_feature_extractor(self, fn: FeatureExtractor) -> Self:
        """Set the feature extraction strategy: ``(PreprocessedData) -> FeatureSet``."""
        self._feature_extractor = fn
        return self

    def add_model(self, name: str, model: BaseEstimator) -> Self:
        """Register a named scikit-learn estimator to be trained and evaluated."""
        self._models[name] = model
        return self

    def add_evaluator(self, fn: Evaluator) -> Self:
        """Add an evaluation strategy: ``(ModelArtifact, FeatureSet) -> EvaluationReport``."""
        self._evaluators.append(fn)
        return self

    def add_reporter(self, fn: Reporter) -> Self:
        """Add a reporting strategy: ``(ModelArtifact, EvaluationReport, Path | None) -> None``."""
        self._reporters.append(fn)
        return self

    def build(self) -> Pipeline:
        """Validate required components and return the assembled Pipeline.

        Raises:
            ValueError: If any required component has not been set.
        """
        missing = [
            name
            for name, val in [
                ("dataset", self._dataset),
                ("preprocessor", self._preprocessor),
                ("splitter", self._splitter),
                ("feature_extractor", self._feature_extractor),
            ]
            if val is None
        ]
        if missing:
            raise ValueError(
                f"Pipeline is missing required components: {', '.join(missing)}"
            )

        return Pipeline(
            dataset=self._dataset,  # type: ignore[arg-type]
            preprocessor=self._preprocessor,  # type: ignore[arg-type]
            splitter=self._splitter,  # type: ignore[arg-type]
            feature_extractor=self._feature_extractor,  # type: ignore[arg-type]
            models=self._models,
            evaluators=self._evaluators,
            reporters=self._reporters,
        )

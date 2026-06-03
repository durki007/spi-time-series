from __future__ import annotations

from typing import TYPE_CHECKING, Self, get_args

if TYPE_CHECKING:
    from spi_time_series.config.schema import RunConfig

from sklearn.base import BaseEstimator

from spi_time_series.config import TaskType
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
        self._task: TaskType = "regression"
        self._dataset: Dataset | None = None
        self._preprocessor: Preprocessor | None = None
        self._splitter: Splitter | None = None
        self._feature_extractor: FeatureExtractor | None = None
        self._pca_keep_percentage: float | None = None
        self._models: dict[str, BaseEstimator] = {}
        self._param_grids: dict[str, dict[str, list]] = {}
        self._evaluators: list[Evaluator] = []
        self._reporters: list[Reporter] = []

    def with_task(self, task: TaskType):
        valid_tasks = get_args(TaskType)
        if task not in valid_tasks:
            raise ValueError(f"Invalid task: {task}. Must be in {valid_tasks}")
        self._task = task
        return self

    def with_dataset(self, dataset: Dataset) -> Self:
        """Set the dataset that supplies the raw event log."""
        self._dataset = dataset
        return self

    def with_pca_keep_percentage(self, keep_percentage: float | None) -> Self:
        self._pca_keep_percentage = keep_percentage
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

    def add_hyperparams(
        self, model_name: str, param_grid: dict[str, list]
    ) -> Self:
        """Add a hyperparameter grid for a model to be used by the search stage."""
        if model_name not in self._models:
            raise ValueError(
                f"Cannot add hyperparameters for unknown model '{model_name}'."
            )
        self._param_grids[model_name] = param_grid
        return self

    @classmethod
    def from_config(cls, config: RunConfig) -> PipelineBuilder:
        """Construct a PipelineBuilder pre-populated from a RunConfig.

        Wires dataset, preprocessor, splitter, and models from the config.
        Does NOT wire with_feature_extractor — the target_generator encodes
        domain logic (e.g. remaining time vs. loan outcome) that cannot be
        expressed as a config scalar. Call builder.with_feature_extractor(...)
        after from_config() before build().
        """
        from spi_time_series.config.loader import build_estimator
        from spi_time_series.config.schema import RunConfig  # noqa: F401
        from spi_time_series.data.schemas import PreprocessedData, RawData
        from spi_time_series.preprocessing.preprocess import (
            _build_trace_samples,
            clean_event_log,
            filter_dev_cases,
            sliding_window_factory,
            split_data,
        )

        builder = cls()

        builder.with_dataset(Dataset())
        builder.with_task(config.task)
        builder.with_pca_keep_percentage(
            None
            if not config.pca_config.active
            else config.pca_config.keep_variability
        )

        valid_ends = config.data.valid_end_activities or None
        top_k = config.data.top_k_variants

        def _preprocessor(raw: RawData):
            cleaned_log = clean_event_log(
                raw.event_log,
                filter_valid_outcomes=config.task == "classification",
                valid_end_activities=valid_ends,
                top_k_variants=top_k,
            )
            return cleaned_log

        builder.with_preprocessor(_preprocessor)

        split_quantile = config.data.split_quantile
        prefix_gen = sliding_window_factory(
            min_length=config.prefix.min_length,
            max_length=config.prefix.max_length,
        )

        def _splitter(log) -> PreprocessedData:
            if config.data.dev_mode:
                log = filter_dev_cases(log)
            train_df, test_df = split_data(log, split_quantile=split_quantile)
            col_idx = {c: i for i, c in enumerate(train_df.columns)}
            return PreprocessedData(
                train_log=_build_trace_samples(train_df, prefix_gen),
                num_train_cases=len(train_df["case:concept:name"].unique()),
                test_log=_build_trace_samples(test_df, prefix_gen),
                num_test_cases=len(test_df["case:concept:name"].unique()),
                col_idx=col_idx,
                cleaned_log=log,
            )

        builder.with_splitter(_splitter)

        for name, model_cfg in config.models.items():
            builder.add_model(name, build_estimator(model_cfg))
            if model_cfg.param_grid:
                builder._param_grids[name] = dict(model_cfg.param_grid)

        return builder

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
            task=self._task,
            dataset=self._dataset,  # type: ignore[arg-type]
            preprocessor=self._preprocessor,  # type: ignore[arg-type]
            splitter=self._splitter,  # type: ignore[arg-type]
            feature_extractor=self._feature_extractor,  # type: ignore[arg-type]
            pca_keep_percentage=self._pca_keep_percentage,
            models=self._models,
            param_grids=self._param_grids,
            evaluators=self._evaluators,
            reporters=self._reporters,
        )

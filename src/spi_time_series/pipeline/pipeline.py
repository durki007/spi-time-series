from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import joblib
from sklearn.base import BaseEstimator

from spi_time_series.data import Dataset
from spi_time_series.data.schemas import (
    EvaluationReport,
    FeatureSet,
    ModelArtifact,
    RawData,
)
from spi_time_series.data.types import (
    Evaluator,
    FeatureExtractor,
    Preprocessor,
    Reporter,
    Splitter,
)
from spi_time_series.models import search_hyperparams, train
from spi_time_series.pipeline.state import PipelineState

if TYPE_CHECKING:
    from spi_time_series.config.schema import SearchConfig

logger = logging.getLogger(__name__)


def _merge_evaluations(reports: list[EvaluationReport]) -> EvaluationReport:
    """Deep-merge a list of evaluation reports into one."""
    if not reports:
        raise ValueError("No evaluators produced a report.")
    merged = EvaluationReport(metrics={}, model_names=[], prefix_lengths=[])
    for report in reports:
        for model, by_prefix in report.metrics.items():
            merged.metrics.setdefault(model, {})
            for prefix, metrics in by_prefix.items():
                merged.metrics[model].setdefault(prefix, {}).update(metrics)
        for name in report.model_names:
            if name not in merged.model_names:
                merged.model_names.append(name)
        for pl in report.prefix_lengths:
            if pl not in merged.prefix_lengths:
                merged.prefix_lengths.append(pl)
    return merged


@dataclass
class Pipeline:
    """Predictive process monitoring pipeline.

    Build with PipelineBuilder, then call fit() followed by evaluate().

    Attributes:
        dataset: Source of the raw event log.
        preprocessor: Cleans the raw event log → EventLog.
        splitter: Partitions the cleaned log into train/test → PreprocessedData.
        feature_extractor: Builds labelled feature matrices → FeatureSet.
        models: Named scikit-learn estimators to train and compare.
        param_grids: Per-model hyperparameter grids used by the search stage.
        evaluators: Compute evaluation metrics; their results are merged.
        reporters: Produce output artefacts (plots, CSVs, MLflow runs, …).
    """

    dataset: Dataset
    preprocessor: Preprocessor
    splitter: Splitter
    feature_extractor: FeatureExtractor
    models: dict[str, BaseEstimator]
    param_grids: dict[str, dict[str, list]] = field(default_factory=dict)
    evaluators: list[Evaluator] = field(default_factory=list)
    reporters: list[Reporter] = field(default_factory=list)

    # Internal state populated by fit()
    _features: FeatureSet | None = field(default=None, init=False, repr=False)
    _artifact: ModelArtifact | None = field(
        default=None, init=False, repr=False
    )

    # Per-model cached results
    _optimized_models: dict[str, BaseEstimator] = field(
        default_factory=dict, init=False, repr=False
    )
    _trained_models: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False
    )

    # Per-stage / per-model cache keys
    _extract_key: str | None = field(default=None, init=False, repr=False)
    _search_keys: dict[str, str] = field(
        default_factory=dict, init=False, repr=False
    )
    _train_keys: dict[str, str] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def is_fitted(self) -> bool:
        """True after fit() has completed at least the train stage."""
        return self._features is not None and self._artifact is not None

    def restore_state(self, state: PipelineState) -> None:
        """Load a previously saved PipelineState into this pipeline.

        Call before fit() so that fit() can skip stages whose inputs are
        unchanged.  If all current models are already present in the state,
        the pipeline is immediately ready for evaluate().
        """
        self._features = state.features
        self._optimized_models = dict(state.optimized_models)
        self._trained_models = dict(state.trained_models)
        self._extract_key = state.extract_key
        self._search_keys = dict(state.search_keys)
        self._train_keys = dict(state.train_keys)

        # Reconstruct artifact if all current model names are cached
        if self._features is not None and set(self.models).issubset(
            set(self._trained_models)
        ):
            self._artifact = ModelArtifact(
                models={k: self._trained_models[k] for k in self.models},
                feature_names=self._features.feature_names,
                target_col=self._features.y_train.name or "target",
            )

    def extract_state(self) -> PipelineState:
        """Capture the current fitted state as a serialisable PipelineState."""
        return PipelineState(
            features=self._features,
            optimized_models=dict(self._optimized_models),
            trained_models=dict(self._trained_models),
            extract_key=self._extract_key,
            search_keys=dict(self._search_keys),
            train_keys=dict(self._train_keys),
        )

    def fit(
        self,
        *,
        extract_key: str | None = None,
        force: bool = False,
        n_jobs: int = 1,
        search_config: SearchConfig | None = None,
    ) -> Pipeline:
        """Run feature extraction, hyperparameter search, and model training.

        Stages are skipped when their config-derived key is unchanged from the
        last run (loaded via restore_state).  If a stage re-runs, all
        downstream per-model results are invalidated automatically.

        Args:
            extract_key: Hash of the extract-relevant config sections.  When
                this matches the stored key and features are present, the
                extract stage is skipped.
            force: When True, re-run all stages regardless of cached keys.
            n_jobs: Number of parallel jobs forwarded to RandomizedSearchCV.
            search_config: Hyperparameter search settings.  Required when any
                model has a non-empty param_grid and is not cached.

        Returns:
            self, for method chaining.
        """
        # ---- extract (global) -------------------------------------------
        extract_changed = (
            force
            or extract_key is None
            or extract_key != self._extract_key
            or self._features is None
        )
        if extract_changed:
            logger.info("Preprocessing and splitting data…")
            raw = RawData(event_log=self.dataset.log)
            cleaned = self.preprocessor(raw)
            preprocessed = self.splitter(cleaned)

            logger.info("Extracting features…")
            self._features = self.feature_extractor(preprocessed)
            self._extract_key = extract_key

            # Cascade: all per-model results are now stale
            self._optimized_models.clear()
            self._trained_models.clear()
            self._search_keys.clear()
            self._train_keys.clear()
            self._artifact = None
        else:
            logger.info("Skipping extract (key: %s)", self._extract_key)

        # ---- search (per-model) -----------------------------------------
        models_needing_search: dict[str, BaseEstimator] = {}
        new_search_keys: dict[str, str] = {}
        for name, model in self.models.items():
            param_grid = self.param_grids.get(name, {})
            if not param_grid:
                continue
            key = joblib.hash(
                {
                    "type": type(model).__name__,
                    "params": model.get_params(),
                    "param_grid": param_grid,
                    "search_config": (
                        search_config.model_dump() if search_config else None
                    ),
                }
            )[:8]
            new_search_keys[name] = key
            if (
                not force
                and not extract_changed
                and name in self._optimized_models
                and self._search_keys.get(name) == key
            ):
                logger.info("Skipping search for '%s' (key: %s)", name, key)
            else:
                models_needing_search[name] = model

        if models_needing_search:
            if search_config is None:
                raise ValueError(
                    "search_config must be provided when any model has a param_grid."
                )
            assert self._features is not None
            logger.info(
                "Searching hyperparameters for %d model(s)…",
                len(models_needing_search),
            )
            new_opts = search_hyperparams(
                self._features,
                models_needing_search,
                {n: self.param_grids[n] for n in models_needing_search},
                search_config,
                n_jobs=n_jobs,
            )
            for name in models_needing_search:
                self._optimized_models[name] = new_opts[name]
                self._search_keys[name] = new_search_keys[name]
                # Cascade: invalidate train for this model
                self._trained_models.pop(name, None)
                self._train_keys.pop(name, None)

        # ---- train (per-model) ------------------------------------------
        # Use optimized model if available, else fall back to original
        effective_models = {
            name: self._optimized_models.get(name, model)
            for name, model in self.models.items()
        }
        models_needing_train: dict[str, BaseEstimator] = {}
        new_train_keys: dict[str, str] = {}
        for name, est in effective_models.items():
            key = joblib.hash(
                {
                    "type": type(est).__name__,
                    "params": est.get_params(),
                    "extract_key": self._extract_key,
                    "search_key": self._search_keys.get(name),
                }
            )[:8]
            new_train_keys[name] = key
            if (
                not force
                and not extract_changed
                and name in self._trained_models
                and self._train_keys.get(name) == key
            ):
                logger.info("Skipping train for '%s' (key: %s)", name, key)
            else:
                models_needing_train[name] = est

        assert self._features is not None
        if models_needing_train:
            logger.info("Training %d model(s)…", len(models_needing_train))
            new_artifact = train(self._features, models_needing_train)
            for name in models_needing_train:
                self._trained_models[name] = new_artifact.models[name]
                self._train_keys[name] = new_train_keys[name]

        # Rebuild artifact from the full set of (cached + newly trained) models
        self._artifact = ModelArtifact(
            models={k: self._trained_models[k] for k in self.models},
            feature_names=self._features.feature_names,
            target_col=self._features.y_train.name or "target",
        )

        logger.info("Fit complete.")
        return self

    def evaluate(self, output_dir=None) -> EvaluationReport:
        """Evaluate trained models and invoke reporters.

        Must be called after fit() has completed at least the train stage.

        Args:
            output_dir: Directory passed to each reporter for saving artefacts.

        Returns:
            Merged EvaluationReport across all evaluators.

        Raises:
            RuntimeError: If fit() has not been called or training was not run.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Pipeline has not been fully fitted. "
                "Call fit() before evaluate()."
            )

        assert self._artifact is not None
        assert self._features is not None
        logger.info("Evaluating %d model(s)…", len(self._artifact.models))
        reports = [e(self._artifact, self._features) for e in self.evaluators]
        evaluation = _merge_evaluations(reports)

        for reporter in self.reporters:
            reporter(self._artifact, evaluation, output_dir)

        logger.info("Evaluation complete.")
        return evaluation

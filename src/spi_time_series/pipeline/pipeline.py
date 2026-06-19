from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import joblib
from sklearn.base import BaseEstimator

from spi_time_series.config import TaskType
from spi_time_series.config.schema import SearchConfig
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

logger = logging.getLogger(__name__)


def _merge_evaluations(reports: list[EvaluationReport]) -> EvaluationReport:
    """Deep-merge a list of evaluation reports into one."""
    if not reports:
        raise ValueError("No evaluators produced a report.")
    merged = EvaluationReport(
        feature_set=None,
        model_predictions={},
        prefix_metrics={},
        model_metrics={},
        model_names=[],
        prefix_lengths=[],
    )
    for report in reports:
        if merged.feature_set is None and report.feature_set is not None:
            merged.feature_set = report.feature_set  # We assume if multiple reports have a feature set they all are the same.

        for model, pred in report.model_predictions.items():
            merged.model_predictions[model] = pred

        for model, by_prefix in report.prefix_metrics.items():
            merged.prefix_metrics.setdefault(model, {})
            for prefix, metrics in by_prefix.items():
                merged.prefix_metrics[model].setdefault(prefix, {}).update(
                    metrics
                )
        for model, by_prefix in report.feature_importance.items():
            merged.feature_importance.setdefault(model, {})
            for prefix, metrics in by_prefix.items():
                merged.feature_importance[model].setdefault(prefix, {}).update(
                    metrics
                )
        for model, by_metric in report.model_metrics.items():
            merged.model_metrics.setdefault(model, {})
            for metric, val in by_metric.items():
                merged.model_metrics[model][metric] = val
        for name in report.model_names:
            if name not in merged.model_names:
                merged.model_names.append(name)
        for pl in report.prefix_lengths:
            if pl not in merged.prefix_lengths:
                merged.prefix_lengths.append(pl)
        for pl, count in report.prefix_counts.items():
            if pl not in merged.prefix_counts:
                merged.prefix_counts[pl] = count
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

    task: TaskType
    dataset: Dataset
    preprocessor: Preprocessor
    splitter: Splitter
    feature_extractor: FeatureExtractor
    pca_keep_percentage: float | None
    models: dict[str, BaseEstimator]
    param_grids: dict[str, dict[str, list]] = field(default_factory=dict)
    evaluators: list[Evaluator] = field(default_factory=list)
    reporters: list[Reporter] = field(default_factory=list)

    # Internal state populated by fit()
    _features: FeatureSet | None = field(default=None, init=False, repr=False)
    _artifact: ModelArtifact | None = field(
        default=None, init=False, repr=False
    )

    # Fitted feature objects + column mapping (for prototype predictions)
    _fitted_features: list[Any] | None = field(
        default=None, init=False, repr=False
    )
    _fitted_col_idx_mapping: dict[str, int] | None = field(
        default=None, init=False, repr=False
    )

    # Per-model cached results
    _optimized_models: dict[str, BaseEstimator] = field(
        default_factory=dict, init=False, repr=False
    )
    _trained_models: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _best_params: dict[str, dict] = field(
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

    @property
    def best_params(self) -> dict[str, dict]:
        """Best hyperparameters found per model during the last search stage."""
        return dict(self._best_params)

    def restore_state(self, state: PipelineState) -> None:
        """Load a previously saved PipelineState into this pipeline.

        Call before fit() so that fit() can skip stages whose inputs are
        unchanged.  If all current models are already present in the state,
        the pipeline is immediately ready for evaluate().
        """
        self._features = state.features
        self._fitted_features = state.fitted_features
        self._fitted_col_idx_mapping = state.fitted_col_idx_mapping
        self._optimized_models = dict(state.optimized_models)
        self._trained_models = dict(state.trained_models)
        self._best_params = dict(state.best_params)
        self._extract_key = state.extract_key
        self._search_keys = dict(state.search_keys)
        self._train_keys = dict(state.train_keys)

        f = state.features
        feature_summary = (
            f"{len(f.X_train)} train rows × {len(f.feature_names)} features, "
            f"{len(f.X_test)} test rows"
            if f is not None
            else "no features"
        )
        opt_names = list(state.optimized_models) or ["none"]
        trained_names = list(state.trained_models) or ["none"]
        logger.info(
            "Checkpoint loaded: extract_key=%s | features: %s | "
            "optimized models [%d]: %s | trained models [%d]: %s",
            state.extract_key or "none",
            feature_summary,
            len(state.optimized_models),
            ", ".join(opt_names),
            len(state.trained_models),
            ", ".join(trained_names),
        )

        # Reconstruct artifact if all current model names are cached
        if self._features is not None and set(self.models).issubset(
            set(self._trained_models)
        ):
            self._artifact = ModelArtifact(
                models={k: self._trained_models[k] for k in self.models},
                feature_names=self._features.feature_names,
                target_col=self._features.y_train.name or "target",
            )
            logger.info(
                "All %d model(s) present in checkpoint — pipeline is fully fitted, "
                "evaluate() can be called without re-running fit stages.",
                len(self.models),
            )

    def extract_state(self) -> PipelineState:
        """Capture the current fitted state as a serialisable PipelineState."""
        return PipelineState(
            features=self._features,
            fitted_features=self._fitted_features,
            fitted_col_idx_mapping=self._fitted_col_idx_mapping,
            optimized_models=dict(self._optimized_models),
            trained_models=dict(self._trained_models),
            best_params=dict(self._best_params),
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
            logger.info(
                "Features extracted (key: %s): %d train rows × %d features, %d test rows",
                self._extract_key,
                len(self._features.X_train),
                len(self._features.feature_names),
                len(self._features.X_test),
            )

            if (
                hasattr(self.feature_extractor, "features")
                and self.feature_extractor.features
            ):
                self._fitted_features = self.feature_extractor.features
            self._fitted_col_idx_mapping = preprocessed.col_idx

            # Cascade: all per-model results are now stale
            self._optimized_models.clear()
            self._trained_models.clear()
            self._search_keys.clear()
            self._train_keys.clear()
            self._artifact = None
        else:
            f = self._features
            logger.info(
                "Skipping extract (key: %s) — reusing %d train rows × %d features, %d test rows",
                self._extract_key,
                len(f.X_train) if f is not None else 0,
                len(f.feature_names) if f is not None else 0,
                len(f.X_test) if f is not None else 0,
            )

        # ---- search (per-model) -----------------------------------------
        models_needing_search: dict[str, BaseEstimator] = {}
        new_search_keys: dict[str, str] = {}
        models_with_grid = [n for n in self.models if self.param_grids.get(n)]
        if not models_with_grid:
            logger.info(
                "Search stage: no models have a param_grid — skipping entirely."
            )
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
                cached_params = self._best_params.get(name, {})
                logger.info(
                    "Skipping search for '%s' (key: %s) — reusing cached params: %s",
                    name,
                    key,
                    cached_params if cached_params else "<not recorded>",
                )
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
            new_opts, new_best = search_hyperparams(
                self._features,
                models_needing_search,
                {n: self.param_grids[n] for n in models_needing_search},
                search_config,
                pca_keep_percentage=self.pca_keep_percentage,
                n_jobs=n_jobs,
            )
            for name in models_needing_search:
                self._optimized_models[name] = new_opts[name]
                self._search_keys[name] = new_search_keys[name]
                if name in new_best:
                    self._best_params[name] = new_best[name]
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
                logger.info(
                    "Skipping train for '%s' (key: %s) — reusing cached %s",
                    name,
                    key,
                    type(est).__name__,
                )
            else:
                models_needing_train[name] = est

        assert self._features is not None
        if not models_needing_train:
            logger.info(
                "Train stage: all %d model(s) loaded from cache — %s",
                len(effective_models),
                ", ".join(effective_models),
            )
        else:
            logger.info("Training %d model(s)…", len(models_needing_train))
            new_artifact = train(
                self._features, models_needing_train, self.pca_keep_percentage
            )
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
        reports = [
            e(self._artifact, self._features, self.task)
            for e in self.evaluators
        ]
        evaluation = _merge_evaluations(reports)

        for reporter in self.reporters:
            reporter(self._artifact, evaluation, output_dir)

        logger.info("Evaluation complete.")
        return evaluation

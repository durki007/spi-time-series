from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from spi_time_series.data.schemas import FeatureSet, ModelArtifact

if TYPE_CHECKING:
    from spi_time_series.config.schema import SearchConfig

logger = logging.getLogger(__name__)


def _build_numeric_preprocessor() -> ColumnTransformer:
    """Select all numeric columns, impute with median, scale, drop string columns."""
    return ColumnTransformer(
        [
            (
                "num",
                SklearnPipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                make_column_selector(dtype_include=np.number),
            )
        ],
        remainder="drop",
    )


def search_hyperparams(
    features: FeatureSet,
    models: dict[str, BaseEstimator],
    param_grids: dict[str, dict[str, list]],
    search_config: SearchConfig,
    *,
    n_jobs: int = 1,
) -> dict[str, BaseEstimator]:
    """Run RandomizedSearchCV for each model that has a param_grid.

    Returns estimators with best hyperparameters set but not yet fitted on the
    full training data — train() does the actual full fit.
    """
    X_train = features.X_train
    y_train = features.y_train

    if (
        search_config.search_sample_size is not None
        and search_config.search_sample_size < len(X_train)
    ):
        X_train = X_train.sample(
            n=search_config.search_sample_size,
            random_state=search_config.random_state,
        )
        y_train = y_train.loc[X_train.index]

    search_pre = _build_numeric_preprocessor()
    X_num = search_pre.fit_transform(X_train)

    optimized: dict[str, BaseEstimator] = {}
    total = len(models)
    for idx, (name, estimator) in enumerate(
        tqdm(models.items(), desc="Searching hyperparams", unit="model")
    ):
        remaining = total - idx - 1
        logger.info(
            "Searching '%s' (%d/%d, %d remaining)…",
            name,
            idx + 1,
            total,
            remaining,
        )
        grid = param_grids.get(name, {})
        if not grid:
            logger.info("'%s': no param_grid, using default params.", name)
            optimized[name] = estimator
            continue

        cv = RandomizedSearchCV(
            estimator,
            grid,
            n_iter=search_config.n_iter,
            cv=search_config.cv_folds,
            random_state=search_config.random_state,
            n_jobs=n_jobs,
            refit=False,
            verbose=2,
        )
        cv.fit(X_num, y_train)

        best_params = cv.best_params_
        logger.info("'%s' best params: %s", name, best_params)
        optimized[name] = clone(estimator).set_params(**best_params)

    return optimized


def train(
    features: FeatureSet, models: dict[str, BaseEstimator]
) -> ModelArtifact:
    """Fit baseline and extended models on training feature vectors.

    Each estimator is wrapped in a two-step sklearn Pipeline:
      preprocessor -> model

    The preprocessor selects all numeric columns, imputes NaN with the column
    median, and drops object-dtype columns (last_activity, last_transition)
    produced when one_hot_encode_categorical=False in BasicControlFlowFeatures.
    """
    fitted: dict[str, SklearnPipeline] = {}

    for name, estimator in tqdm(
        models.items(), desc="Training models", unit="model"
    ):
        logger.info("Fitting '%s'…", name)
        if "verbose" in estimator.get_params():
            estimator = clone(estimator).set_params(verbose=1)
        pipe = SklearnPipeline(
            steps=[
                ("preprocessor", _build_numeric_preprocessor()),
                ("model", estimator),
            ]
        )
        pipe.fit(features.X_train, features.y_train)
        fitted[name] = pipe
        logger.info("'%s' trained.", name)

    return ModelArtifact(
        models=fitted,
        feature_names=features.feature_names,
        target_col=features.y_train.name or "target",
    )

import logging

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline

from spi_time_series.data.schemas import FeatureSet, ModelArtifact

logger = logging.getLogger(__name__)


def _build_numeric_preprocessor() -> ColumnTransformer:
    """Select all numeric columns, impute with median, drop string columns."""
    return ColumnTransformer(
        [
            (
                "num",
                SimpleImputer(strategy="median"),
                make_column_selector(dtype_include=np.number),
            )
        ],
        remainder="drop",
    )


def train(
    features: FeatureSet, models: dict[str, BaseEstimator]
) -> ModelArtifact:
    """Fit baseline and extended models on training feature vectors.

    Each estimator is wrapped in a two-step sklearn Pipeline:
      preprocessor -> model

    The preprocessor selects all numeric columns, imputes NaN with the column
    median, and drops object-dtype columns (last_activity, last_transition)
    produced when one_hot_encode_categorical=False in BasicControlFlowFeatures.

    Pass a GridSearchCV or RandomizedSearchCV as the estimator to trigger
    hyperparameter search during fit.
    """
    fitted: dict[str, SklearnPipeline] = {}

    for name, estimator in models.items():
        logger.info("Training model: %s", name)
        pipe = SklearnPipeline(
            steps=[
                ("preprocessor", _build_numeric_preprocessor()),
                ("model", estimator),
            ]
        )
        pipe.fit(features.X_train, features.y_train)
        fitted[name] = pipe
        logger.info("Model %s trained.", name)

    return ModelArtifact(
        models=fitted,
        feature_names=features.feature_names,
        target_col=features.y_train.name or "target",
    )

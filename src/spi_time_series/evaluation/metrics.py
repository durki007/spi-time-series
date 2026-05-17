import logging
import warnings

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from spi_time_series.data.schemas import (
    EvaluationReport,
    FeatureSet,
    ModelArtifact,
)

logger = logging.getLogger(__name__)

_PREFIX_LENGTH_COL = "BasicControlFlowFeatures__prefix_length"


def evaluate(artifact: ModelArtifact, features: FeatureSet) -> EvaluationReport:
    """Compute per-model, per-prefix-length regression metrics on the test set.

    Metrics: MAE, RMSE, R².
    R² is nan for single-sample prefix groups (undefined); no exception is raised.

    Requires BasicControlFlowFeatures__prefix_length in features.X_test.
    """
    X_test = features.X_test
    y_test = features.y_test

    if _PREFIX_LENGTH_COL not in X_test.columns:
        raise ValueError(
            f"Column '{_PREFIX_LENGTH_COL}' not found in X_test. "
            "BasicControlFlowFeatures must be included in the feature pipeline."
        )

    groups: dict = X_test.groupby(_PREFIX_LENGTH_COL).groups
    prefix_lengths: list[int] = sorted(int(pl) for pl in groups)
    model_names: list[str] = list(artifact.models)
    all_metrics: dict[str, dict[int, dict[str, float]]] = {}

    for model_name, pipeline in artifact.models.items():
        logger.info("Evaluating model: %s", model_name)
        y_pred = pd.Series(pipeline.predict(X_test), index=X_test.index)

        model_metrics: dict[int, dict[str, float]] = {}
        for pl_val, group_idx in groups.items():
            y_true_g = y_test.loc[group_idx]
            y_pred_g = y_pred.loc[group_idx]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_metrics[int(pl_val)] = {
                    "mae": float(mean_absolute_error(y_true_g, y_pred_g)),
                    "rmse": float(root_mean_squared_error(y_true_g, y_pred_g)),
                    "r2": float(r2_score(y_true_g, y_pred_g)),
                }

        all_metrics[model_name] = model_metrics
        logger.info(
            "Model %s evaluated across %d prefix lengths.",
            model_name,
            len(prefix_lengths),
        )

    return EvaluationReport(
        metrics=all_metrics,
        model_names=model_names,
        prefix_lengths=prefix_lengths,
    )

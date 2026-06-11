import logging
import warnings

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from spi_time_series.config import TaskType
from spi_time_series.data.schemas import (
    BestPrefixInfo,
    EvaluationReport,
    FeatureSet,
    ModelArtifact,
    ModelComparisonResult,
    ModelRankEntry,
)

logger = logging.getLogger(__name__)

_PREFIX_LENGTH_COL = "BasicControlFlowFeatures__prefix_length"


def evaluate(
    artifact: ModelArtifact, features: FeatureSet, target_type: TaskType
) -> EvaluationReport:
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

                if target_type == "regression":
                    model_metrics[int(pl_val)] = {
                        "mae": float(mean_absolute_error(y_true_g, y_pred_g)),
                        "rmse": float(
                            root_mean_squared_error(y_true_g, y_pred_g)
                        ),
                        "r2": float(r2_score(y_true_g, y_pred_g)),
                    }

                elif target_type == "classification":
                    model_metrics[int(pl_val)] = {
                        "accuracy": float(accuracy_score(y_true_g, y_pred_g)),
                        "f1_macro": float(
                            f1_score(
                                y_true_g,
                                y_pred_g,
                                average="macro",
                                zero_division=0,
                            )
                        ),
                        "f1_weighted": float(
                            f1_score(
                                y_true_g,
                                y_pred_g,
                                average="weighted",
                                zero_division=0,
                            )
                        ),
                    }

                else:
                    raise ValueError(f"Unknown target_type: {target_type}")

        all_metrics[model_name] = model_metrics
        logger.info(
            "Model %s evaluated across %d prefix lengths.",
            model_name,
            len(prefix_lengths),
        )

    return EvaluationReport(
        prefix_metrics=all_metrics,
        model_names=model_names,
        prefix_lengths=prefix_lengths,
    )


# ---------------------------------------------------------------------------
# Model comparison & best-prefix detection
# ---------------------------------------------------------------------------


def _select_ranking_metric(task: TaskType) -> str:
    """Return the canonical metric name for ranking models given a task type.

    ==============  =============
    Task             Metric
    ==============  =============
    ``regression``   ``rmse``
    ``classification`` ``f1_weighted``
    ==============  =============
    """
    if task == "regression":
        return "rmse"
    if task == "classification":
        return "f1_weighted"
    raise ValueError(f"Unknown task type: {task}")


def _lower_is_better(metric: str) -> bool:
    """Return ``True`` when a smaller value indicates better performance."""
    return metric in ("mae", "rmse")


def _detect_plateau_prefix(
    model_name: str,
    prefix_metrics: dict[int, dict[str, float]],
    metric: str,
    *,
    plateau_threshold: float = 0.05,
) -> BestPrefixInfo:
    """Walk prefix lengths in ascending order and return the first length
    where the relative improvement of *metric* drops below *plateau_threshold*.

    Parameters
    ----------
    model_name:
        Model identifier used in the result label.
    prefix_metrics:
        Mapping ``prefix_length → {metric_name: value, …}`` for a single model.
    metric:
        Name of the metric to track (must be present in every entry).
    plateau_threshold:
        Fractional improvement below which the curve is considered to have
        plateaued (default ``0.05`` = 5 %).

    Returns
    -------
    BestPrefixInfo
        Plateau information.  When fewer than two prefix lengths are available
        the *plateau_prefix* is set to the sole available length.
    """
    if not prefix_metrics:
        return BestPrefixInfo(
            model_name=model_name,
            plateau_prefix=-1,
            metric=metric,
            value=float("nan"),
            plateau_threshold=plateau_threshold,
        )

    ordered: list[tuple[int, float]] = sorted(
        (pl, m[metric]) for pl, m in prefix_metrics.items() if metric in m
    )

    if len(ordered) < 2:
        pl, val = ordered[0]
        return BestPrefixInfo(
            model_name=model_name,
            plateau_prefix=pl,
            metric=metric,
            value=val,
            plateau_threshold=plateau_threshold,
        )

    better_down: bool = _lower_is_better(metric)

    for i in range(1, len(ordered)):
        prev_pl, prev_val = ordered[i - 1]
        curr_pl, curr_val = ordered[i]

        delta: float = (
            prev_val - curr_val if better_down else curr_val - prev_val
        )
        denom: float = abs(prev_val) if abs(prev_val) > 1e-12 else 1.0
        rel_improvement: float = delta / denom

        if rel_improvement < plateau_threshold:
            return BestPrefixInfo(
                model_name=model_name,
                plateau_prefix=prev_pl,
                metric=metric,
                value=prev_val,
                plateau_threshold=plateau_threshold,
            )

    # No plateau detected — return the last (longest) prefix
    last_pl, last_val = ordered[-1]
    return BestPrefixInfo(
        model_name=model_name,
        plateau_prefix=last_pl,
        metric=metric,
        value=last_val,
        plateau_threshold=plateau_threshold,
    )


def compare_models(
    report: EvaluationReport,
    task: TaskType,
    *,
    plateau_threshold: float = 0.05,
) -> ModelComparisonResult | None:
    """Aggregate per-prefix evaluation metrics into a structured model
    comparison, identifying the best model and the optimal prefix length
    (plateau) for every model.

    Parameters
    ----------
    report:
        Evaluation report produced by :func:`evaluate`.  Must contain
        ``prefix_metrics`` keyed by model name and prefix length.
    task:
        ``"regression"`` or ``"classification"``.
    plateau_threshold:
        Fractional improvement threshold for plateau detection
        (default ``0.05`` = 5 %).

    Returns
    -------
    ModelComparisonResult | None
        Structured comparison, or ``None`` when ``report.prefix_metrics`` is
        empty.
    """
    if not report.prefix_metrics:
        logger.warning("No prefix metrics available for model comparison.")
        return None

    metric: str = _select_ranking_metric(task)

    # ---- 1. Compute per-model aggregate scores & rank ---------------------
    aggregates: list[tuple[str, float]] = []
    for model_name, pm in report.prefix_metrics.items():
        values: list[float] = [m[metric] for m in pm.values() if metric in m]
        if not values:
            logger.warning(
                "Model '%s' has no '%s' values; skipping.", model_name, metric
            )
            continue
        aggregates.append((model_name, sum(values) / len(values)))

    if not aggregates:
        logger.warning("No models with metric '%s' available.", metric)
        return None

    # Sort: ascending for lower-is-better metrics, descending otherwise
    reverse: bool = not _lower_is_better(metric)
    aggregates.sort(key=lambda item: item[1], reverse=reverse)

    rankings: list[ModelRankEntry] = [
        ModelRankEntry(
            model_name=name,
            aggregate_score=score,
            metric=metric,
            rank=idx + 1,
        )
        for idx, (name, score) in enumerate(aggregates)
    ]

    best_name, best_score = aggregates[0]

    # ---- 2. Detect plateau prefix per model --------------------------------
    best_prefixes: dict[str, BestPrefixInfo] = {}
    for model_name, pm in report.prefix_metrics.items():
        best_prefixes[model_name] = _detect_plateau_prefix(
            model_name=model_name,
            prefix_metrics=pm,
            metric=metric,
            plateau_threshold=plateau_threshold,
        )
        logger.info(
            "Model '%s' plateau at prefix_length=%d (%s=%.4f)",
            model_name,
            best_prefixes[model_name].plateau_prefix,
            metric,
            best_prefixes[model_name].value,
        )

    logger.info(
        "Best model: '%s' (aggregate %s=%.4f)", best_name, metric, best_score
    )

    return ModelComparisonResult(
        task=task,
        best_model=best_name,
        best_model_score=best_score,
        ranking_metric=metric,
        model_rankings=rankings,
        best_prefixes=best_prefixes,
    )

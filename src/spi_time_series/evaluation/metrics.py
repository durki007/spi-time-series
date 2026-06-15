import logging
import warnings
from pathlib import Path

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
from spi_time_series.data.types import Reporter

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

    prefix_counts: dict[int, int] = {
        int(pl): len(idx) for pl, idx in groups.items()
    }

    return EvaluationReport(
        prefix_metrics=all_metrics,
        model_names=model_names,
        prefix_lengths=prefix_lengths,
        prefix_counts=prefix_counts,
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
    # Weight each prefix-length score by its sample count so that
    # metrics computed on many test instances carry more weight than
    # those computed on only a handful of instances.
    aggregates: list[tuple[str, float]] = []
    for model_name, pm in report.prefix_metrics.items():
        total_weight: float = 0.0
        weighted_sum: float = 0.0
        for pl, m in pm.items():
            if metric in m:
                count: int = report.prefix_counts.get(pl, 1)
                weighted_sum += m[metric] * count
                total_weight += count
        if total_weight == 0:
            logger.warning(
                "Model '%s' has no '%s' values; skipping.", model_name, metric
            )
            continue
        aggregates.append((model_name, weighted_sum / total_weight))

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


def _make_model_comparison_reporter(
    task: str,
) -> Reporter:
    """Factory: create a reporter that runs model comparison analysis.

    The returned reporter calls :func:`compare_models` on the merged evaluation
    report and persists two artefacts under ``output_dir / "reports"``:

    * ``model_comparison.csv`` — per-model aggregate scores and ranks.
    * ``best_prefixes.csv`` — per-model plateau prefix-length information.

    Parameters
    ----------
    task:
        ``"regression"`` or ``"classification"`` — forwarded to
        :func:`compare_models`.

    Returns
    -------
    Reporter
        A callable conforming to :data:`spi_time_series.data.types.Reporter`.
    """

    def _reporter(
        artifact: ModelArtifact,
        report: EvaluationReport,
        output_dir: Path | None,
    ) -> None:
        if output_dir is None:
            return

        comparison: ModelComparisonResult | None = compare_models(
            report,
            task,  # type: ignore[arg-type]
        )
        if comparison is None:
            logger.warning(
                "Model comparison could not be produced — "
                "no prefix metrics available."
            )
            return

        reports_dir: Path = output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # --- rankings CSV ---
        ranking_rows: list[dict[str, object]] = [
            {
                "rank": r.rank,
                "model": r.model_name,
                "aggregate_score": round(r.aggregate_score, 4),
                "metric": r.metric,
            }
            for r in comparison.model_rankings
        ]
        pd.DataFrame(ranking_rows).to_csv(
            reports_dir / "model_comparison.csv", index=False
        )
        logger.info(
            "Model comparison saved to %s",
            reports_dir / "model_comparison.csv",
        )

        # --- best prefixes CSV ---
        prefix_rows: list[dict[str, object]] = [
            {
                "model": info.model_name,
                "plateau_prefix_length": info.plateau_prefix,
                "metric": info.metric,
                "metric_value": round(info.value, 4),
            }
            for info in comparison.best_prefixes.values()
        ]
        pd.DataFrame(prefix_rows).to_csv(
            reports_dir / "best_prefixes.csv", index=False
        )
        logger.info(
            "Best prefix info saved to %s",
            reports_dir / "best_prefixes.csv",
        )

    return _reporter

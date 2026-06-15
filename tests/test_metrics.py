import numpy as np
import pandas as pd
import pytest

from spi_time_series.config.schema import TaskType
from spi_time_series.data.schemas import (
    EvaluationReport,
    FeatureSet,
    ModelArtifact,
)
from spi_time_series.evaluation.metrics import compare_models, evaluate

_PREFIX_COL = "BasicControlFlowFeatures__prefix_length"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ConstantPredictor:
    """Duck-type predict stub: always returns a fixed constant."""

    def __init__(self, constant: float):
        self._c = constant

    def predict(self, X):
        return np.full(len(X), self._c)


def _make_feature_set(
    n_per_prefix: int = 10,
    prefix_lengths: list[int] | None = None,
    task: TaskType = "regression",
) -> FeatureSet:
    if prefix_lengths is None:
        prefix_lengths = [1, 2, 3]
    rows = []
    targets: list[int | float] = []
    for pl in prefix_lengths:
        for _ in range(n_per_prefix):
            rows.append(
                {
                    _PREFIX_COL: pl,
                    "BasicControlFlowFeatures__elapsed_time_hours": float(pl),
                }
            )
            if task == "classification":
                targets.append(1)
            else:
                targets.append(float(pl * 10))

    X = pd.DataFrame(rows).reset_index(drop=True)
    if task == "classification":
        y = pd.Series(targets, name="outcome")
    else:
        y = pd.Series(targets, name="remaining_time_hours")

    return FeatureSet(
        X_train=X,
        X_test=X,
        y_train=y,
        y_test=y,
        feature_names=list(X.columns),
    )


def _make_artifact(model_dict: dict) -> ModelArtifact:
    return ModelArtifact(
        models=model_dict,
        feature_names=[_PREFIX_COL],
        target_col="remaining_time_hours",
    )


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_evaluate_returns_evaluation_report():
    fs = _make_feature_set()
    artifact = _make_artifact({"perfect": _ConstantPredictor(10.0)})
    report = evaluate(artifact, fs, "regression")
    assert isinstance(report, EvaluationReport)


def test_report_model_names_match_artifact():
    fs = _make_feature_set()
    artifact = _make_artifact(
        {"m1": _ConstantPredictor(10.0), "m2": _ConstantPredictor(20.0)}
    )
    report = evaluate(artifact, fs, "regression")
    assert set(report.model_names) == {"m1", "m2"}


def test_report_prefix_lengths_match_data():
    fs = _make_feature_set(prefix_lengths=[1, 3, 5])
    artifact = _make_artifact({"m": _ConstantPredictor(15.0)})
    report = evaluate(artifact, fs, "regression")
    assert sorted(report.prefix_lengths) == [1, 3, 5]


def test_regression_report_metrics_contains_mae_rmse_r2():
    fs = _make_feature_set(prefix_lengths=[2])
    artifact = _make_artifact({"m": _ConstantPredictor(20.0)})
    report = evaluate(artifact, fs, "regression")
    assert set(report.prefix_metrics["m"][2].keys()) == {"mae", "rmse", "r2"}


def test_classification_report_metrics_contains_accuracy_f1macro_f1weighted():
    fs = _make_feature_set(prefix_lengths=[2])
    artifact = _make_artifact({"m": _ConstantPredictor(20.0)})
    report = evaluate(artifact, fs, "classification")
    assert set(report.prefix_metrics["m"][2].keys()) == {
        "accuracy",
        "f1_macro",
        "f1_weighted",
    }


# ---------------------------------------------------------------------------
# Metric correctness
# ---------------------------------------------------------------------------


def test_mae_is_zero_for_perfect_predictions():
    fs = _make_feature_set(n_per_prefix=5, prefix_lengths=[2])
    artifact = _make_artifact({"perfect": _ConstantPredictor(20.0)})
    report = evaluate(artifact, fs, "regression")
    assert report.prefix_metrics["perfect"][2]["mae"] == pytest.approx(0.0)


def test_rmse_is_zero_for_perfect_predictions():
    fs = _make_feature_set(n_per_prefix=5, prefix_lengths=[3])
    artifact = _make_artifact({"perfect": _ConstantPredictor(30.0)})
    report = evaluate(artifact, fs, "regression")
    assert report.prefix_metrics["perfect"][3]["rmse"] == pytest.approx(0.0)


def test_r2_is_one_for_perfect_predictions():
    fs = _make_feature_set(n_per_prefix=5, prefix_lengths=[1])
    artifact = _make_artifact({"perfect": _ConstantPredictor(10.0)})
    report = evaluate(artifact, fs, "regression")
    assert report.prefix_metrics["perfect"][1]["r2"] == pytest.approx(1.0)


def test_mae_is_correct_for_constant_offset():
    fs = _make_feature_set(n_per_prefix=8, prefix_lengths=[2])
    artifact = _make_artifact({"biased": _ConstantPredictor(25.0)})  # y_true=20
    report = evaluate(artifact, fs, "regression")
    assert report.prefix_metrics["biased"][2]["mae"] == pytest.approx(5.0)


def test_rmse_is_correct_for_constant_offset():
    fs = _make_feature_set(n_per_prefix=8, prefix_lengths=[3])
    artifact = _make_artifact({"biased": _ConstantPredictor(40.0)})  # y_true=30
    report = evaluate(artifact, fs, "regression")
    assert report.prefix_metrics["biased"][3]["rmse"] == pytest.approx(10.0)


def test_accuracy_is_one_for_perfect_predictions():
    fs = _make_feature_set(
        n_per_prefix=5, prefix_lengths=[2], task="classification"
    )
    artifact = _make_artifact({"perfect": _ConstantPredictor(1)})
    report = evaluate(artifact, fs, "classification")
    assert report.prefix_metrics["perfect"][2]["accuracy"] == pytest.approx(1.0)


def test_f1_macro_is_one_for_perfect_predictions():
    fs = _make_feature_set(
        n_per_prefix=6, prefix_lengths=[3], task="classification"
    )
    artifact = _make_artifact({"perfect": _ConstantPredictor(1)})
    report = evaluate(artifact, fs, "classification")
    assert report.prefix_metrics["perfect"][3]["f1_macro"] == pytest.approx(1.0)


def test_f1_weighted_is_one_for_perfect_predictions():
    fs = _make_feature_set(
        n_per_prefix=7, prefix_lengths=[1], task="classification"
    )
    artifact = _make_artifact({"perfect": _ConstantPredictor(1)})
    report = evaluate(artifact, fs, "classification")
    assert report.prefix_metrics["perfect"][1]["f1_weighted"] == pytest.approx(
        1.0
    )


def test_accuracy_is_zero_for_wrong_constant_predictions():
    fs = _make_feature_set(
        n_per_prefix=10, prefix_lengths=[2], task="classification"
    )
    # assume ground truth is always 1, predictor always returns 0
    artifact = _make_artifact({"bad": _ConstantPredictor(0)})
    report = evaluate(artifact, fs, "classification")
    assert report.prefix_metrics["bad"][2]["accuracy"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_r2_is_nan_for_single_sample_prefix():
    fs = _make_feature_set(n_per_prefix=1, prefix_lengths=[1])
    artifact = _make_artifact({"m": _ConstantPredictor(10.0)})
    report = evaluate(artifact, fs, "regression")
    assert np.isnan(report.prefix_metrics["m"][1]["r2"])


def test_missing_prefix_length_column_raises():
    X = pd.DataFrame(
        {"BasicControlFlowFeatures__elapsed_time_hours": [1.0, 2.0]}
    )
    y = pd.Series([10.0, 20.0])
    fs = FeatureSet(
        X_train=X,
        X_test=X,
        y_train=y,
        y_test=y,
        feature_names=list(X.columns),
    )
    artifact = _make_artifact({"m": _ConstantPredictor(15.0)})
    with pytest.raises(
        ValueError, match="BasicControlFlowFeatures__prefix_length"
    ):
        evaluate(artifact, fs, "regression")


def test_all_prefix_lengths_have_all_metrics():
    fs = _make_feature_set(n_per_prefix=5, prefix_lengths=[1, 2, 3])
    artifact = _make_artifact({"m": _ConstantPredictor(20.0)})
    report = evaluate(artifact, fs, "regression")
    for pl in [1, 2, 3]:
        assert set(report.prefix_metrics["m"][pl].keys()) == {
            "mae",
            "rmse",
            "r2",
        }


def test_multiple_models_all_evaluated():
    fs = _make_feature_set(n_per_prefix=5, prefix_lengths=[2])
    artifact = _make_artifact(
        {"fast": _ConstantPredictor(20.0), "slow": _ConstantPredictor(25.0)}
    )
    report = evaluate(artifact, fs, "regression")
    assert "fast" in report.prefix_metrics
    assert "slow" in report.prefix_metrics


# ---------------------------------------------------------------------------
# prefix_counts
# ---------------------------------------------------------------------------


def test_prefix_counts_populated():
    """evaluate() stores the number of test samples per prefix length."""
    fs = _make_feature_set(n_per_prefix=7, prefix_lengths=[1, 3, 5])
    artifact = _make_artifact({"m": _ConstantPredictor(15.0)})
    report = evaluate(artifact, fs, "regression")
    assert report.prefix_counts == {1: 7, 3: 7, 5: 7}


# ---------------------------------------------------------------------------
# compare_models – weighted averaging
# ---------------------------------------------------------------------------


def test_compare_models_weighted_avg_equal_counts():
    """When all prefix lengths have equal sample counts, the weighted
    average equals the simple average."""
    fs = _make_feature_set(n_per_prefix=10, prefix_lengths=[1, 2, 3])
    artifact = _make_artifact({"m": _ConstantPredictor(20.0)})
    report = evaluate(artifact, fs, "regression")
    result = compare_models(report, "regression")
    assert result is not None

    # The simple mean should equal the weighted mean (all counts = 10)
    pm = report.prefix_metrics["m"]
    simple_mean = sum(m["rmse"] for m in pm.values()) / len(pm)
    assert result.best_model_score == pytest.approx(simple_mean)


def test_compare_models_weighted_avg_differs_when_unequal():
    """When prefix lengths have unequal sample counts, a prefix with many
    samples influences the aggregate more than a prefix with few samples."""
    # Build a custom FeatureSet: prefix 1 has 1000 samples, prefix 2 has 2.
    n_big, n_small = 1000, 2
    pl_big, pl_small = 1, 99
    rows: list[dict] = []
    targets: list[float] = []

    for _ in range(n_big):
        rows.append({_PREFIX_COL: pl_big, "f": 0.0})
        targets.append(10.0)  # y_true = 10, perfect_pred → rmse=0

    for _ in range(n_small):
        rows.append({_PREFIX_COL: pl_small, "f": 0.0})
        targets.append(100.0)  # y_true = 100 → rmse = 10 (offset by 10)

    X = pd.DataFrame(rows).reset_index(drop=True)
    y = pd.Series(targets, name="remaining_time_hours")
    fs = FeatureSet(
        X_train=X,
        X_test=X,
        y_train=y,
        y_test=y,
        feature_names=list(X.columns),
    )

    artifact = _make_artifact({"m": _ConstantPredictor(10.0)})
    report = evaluate(artifact, fs, "regression")

    # Verify the underlying metrics
    assert report.prefix_metrics["m"][pl_big]["rmse"] == pytest.approx(0.0)
    assert report.prefix_metrics["m"][pl_small]["rmse"] == pytest.approx(10.0)

    # Simple (unweighted) mean
    simple_mean = (0.0 + 10.0) / 2.0  # = 5.0
    # Weighted mean: (0*1000 + 10*2) / 1002 ≈ 0.01996
    weighted_mean = (0.0 * n_big + 10.0 * n_small) / (n_big + n_small)

    result = compare_models(report, "regression")
    assert result is not None
    # Weighted mean should be very close to 0, NOT the simple mean of 5.0
    assert result.best_model_score == pytest.approx(weighted_mean)
    assert abs(result.best_model_score - simple_mean) > 1.0  # clearly different


def test_compare_models_returns_none_for_empty_report():
    """compare_models returns None when prefix_metrics is empty."""
    empty = EvaluationReport()
    assert compare_models(empty, "regression") is None


def test_compare_models_f1_weighted_for_classification():
    """Classification task uses f1_weighted for ranking."""
    fs = _make_feature_set(
        n_per_prefix=5, prefix_lengths=[1, 2], task="classification"
    )
    artifact = _make_artifact({"perfect": _ConstantPredictor(1)})
    report = evaluate(artifact, fs, "classification")
    result = compare_models(report, "classification")
    assert result is not None
    assert result.ranking_metric == "f1_weighted"
    assert result.best_model_score == pytest.approx(1.0)


def test_compare_models_regression_uses_rmse():
    """Regression task uses rmse for ranking."""
    fs = _make_feature_set(n_per_prefix=5, prefix_lengths=[1])
    artifact = _make_artifact({"m": _ConstantPredictor(10.0)})
    report = evaluate(artifact, fs, "regression")
    result = compare_models(report, "regression")
    assert result is not None
    assert result.ranking_metric == "rmse"

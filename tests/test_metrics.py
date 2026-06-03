import numpy as np
import pandas as pd
import pytest

from spi_time_series.config.schema import TaskType
from spi_time_series.data.schemas import (
    EvaluationReport,
    FeatureSet,
    ModelArtifact,
)
from spi_time_series.evaluation.metrics import evaluate

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

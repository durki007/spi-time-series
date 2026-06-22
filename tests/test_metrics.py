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
from tests.conftest import ConstantPredictor

_PREFIX_COL = "BasicControlFlowFeatures__prefix_length"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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

    trace_ids = pd.Series(["A"] * len(y))
    pl_series = X[_PREFIX_COL].astype(int).reset_index(drop=True)

    return FeatureSet(
        X_train=X,
        X_test=X,
        y_train=y,
        y_test=y,
        feature_names=list(X.columns),
        trace_ids_train=trace_ids,
        trace_ids_test=trace_ids,
        prefix_lengths_train=pl_series,
        prefix_lengths_test=pl_series,
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
    artifact = _make_artifact({"perfect": ConstantPredictor(10.0)})
    report = evaluate(artifact, fs, "regression")
    assert isinstance(report, EvaluationReport)


def test_report_model_names_match_artifact():
    fs = _make_feature_set()
    artifact = _make_artifact(
        {"m1": ConstantPredictor(10.0), "m2": ConstantPredictor(20.0)}
    )
    report = evaluate(artifact, fs, "regression")
    assert set(report.model_names) == {"m1", "m2"}


def test_report_prefix_lengths_match_data():
    fs = _make_feature_set(prefix_lengths=[1, 3, 5])
    artifact = _make_artifact({"m": ConstantPredictor(15.0)})
    report = evaluate(artifact, fs, "regression")
    assert sorted(report.prefix_lengths) == [1, 3, 5]


def test_regression_report_metrics_contains_mae_rmse_r2_median_ae():
    fs = _make_feature_set(prefix_lengths=[2])
    artifact = _make_artifact({"m": ConstantPredictor(20.0)})
    report = evaluate(artifact, fs, "regression")
    assert set(report.prefix_metrics["m"][2].keys()) == {
        "mae",
        "rmse",
        "r2",
        "median_ae",
    }


def test_classification_report_metrics_contains_all_keys():
    fs = _make_feature_set(prefix_lengths=[2])
    artifact = _make_artifact({"m": ConstantPredictor(20.0)})
    report = evaluate(artifact, fs, "classification")
    assert set(report.prefix_metrics["m"][2].keys()) == {
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
        "roc_auc",
        "pr_auc",
    }


# ---------------------------------------------------------------------------
# Metric correctness
# ---------------------------------------------------------------------------


def test_mae_is_zero_for_perfect_predictions():
    fs = _make_feature_set(n_per_prefix=5, prefix_lengths=[2])
    artifact = _make_artifact({"perfect": ConstantPredictor(20.0)})
    report = evaluate(artifact, fs, "regression")
    assert report.prefix_metrics["perfect"][2]["mae"] == pytest.approx(0.0)


def test_rmse_is_zero_for_perfect_predictions():
    fs = _make_feature_set(n_per_prefix=5, prefix_lengths=[3])
    artifact = _make_artifact({"perfect": ConstantPredictor(30.0)})
    report = evaluate(artifact, fs, "regression")
    assert report.prefix_metrics["perfect"][3]["rmse"] == pytest.approx(0.0)


def test_r2_is_one_for_perfect_predictions():
    fs = _make_feature_set(n_per_prefix=5, prefix_lengths=[1])
    artifact = _make_artifact({"perfect": ConstantPredictor(10.0)})
    report = evaluate(artifact, fs, "regression")
    assert report.prefix_metrics["perfect"][1]["r2"] == pytest.approx(1.0)


def test_mae_is_correct_for_constant_offset():
    fs = _make_feature_set(n_per_prefix=8, prefix_lengths=[2])
    artifact = _make_artifact({"biased": ConstantPredictor(25.0)})  # y_true=20
    report = evaluate(artifact, fs, "regression")
    assert report.prefix_metrics["biased"][2]["mae"] == pytest.approx(5.0)


def test_rmse_is_correct_for_constant_offset():
    fs = _make_feature_set(n_per_prefix=8, prefix_lengths=[3])
    artifact = _make_artifact({"biased": ConstantPredictor(40.0)})  # y_true=30
    report = evaluate(artifact, fs, "regression")
    assert report.prefix_metrics["biased"][3]["rmse"] == pytest.approx(10.0)


def test_accuracy_is_one_for_perfect_predictions():
    fs = _make_feature_set(
        n_per_prefix=5, prefix_lengths=[2], task="classification"
    )
    artifact = _make_artifact({"perfect": ConstantPredictor(1)})
    report = evaluate(artifact, fs, "classification")
    assert report.prefix_metrics["perfect"][2]["accuracy"] == pytest.approx(1.0)


def test_f1_macro_is_one_for_perfect_predictions():
    fs = _make_feature_set(
        n_per_prefix=6, prefix_lengths=[3], task="classification"
    )
    artifact = _make_artifact({"perfect": ConstantPredictor(1)})
    report = evaluate(artifact, fs, "classification")
    assert report.prefix_metrics["perfect"][3]["f1_macro"] == pytest.approx(1.0)


def test_f1_weighted_is_one_for_perfect_predictions():
    fs = _make_feature_set(
        n_per_prefix=7, prefix_lengths=[1], task="classification"
    )
    artifact = _make_artifact({"perfect": ConstantPredictor(1)})
    report = evaluate(artifact, fs, "classification")
    assert report.prefix_metrics["perfect"][1]["f1_weighted"] == pytest.approx(
        1.0
    )


def test_accuracy_is_zero_for_wrong_constant_predictions():
    fs = _make_feature_set(
        n_per_prefix=10, prefix_lengths=[2], task="classification"
    )
    # assume ground truth is always 1, predictor always returns 0
    artifact = _make_artifact({"bad": ConstantPredictor(0)})
    report = evaluate(artifact, fs, "classification")
    assert report.prefix_metrics["bad"][2]["accuracy"] == pytest.approx(0.0)


def test_median_ae_is_zero_for_perfect_predictions():
    fs = _make_feature_set(n_per_prefix=5, prefix_lengths=[2])
    artifact = _make_artifact({"perfect": ConstantPredictor(20.0)})
    report = evaluate(artifact, fs, "regression")
    assert report.prefix_metrics["perfect"][2]["median_ae"] == pytest.approx(
        0.0
    )


def test_median_ae_is_correct_for_constant_offset():
    fs = _make_feature_set(n_per_prefix=8, prefix_lengths=[2])
    artifact = _make_artifact({"biased": ConstantPredictor(25.0)})
    report = evaluate(artifact, fs, "regression")
    assert report.prefix_metrics["biased"][2]["median_ae"] == pytest.approx(5.0)


def test_precision_macro_is_one_for_perfect_predictions():
    fs = _make_feature_set(
        n_per_prefix=5, prefix_lengths=[2], task="classification"
    )
    artifact = _make_artifact({"perfect": ConstantPredictor(1)})
    report = evaluate(artifact, fs, "classification")
    assert report.prefix_metrics["perfect"][2][
        "precision_macro"
    ] == pytest.approx(1.0)


def test_recall_macro_is_one_for_perfect_predictions():
    fs = _make_feature_set(
        n_per_prefix=6, prefix_lengths=[3], task="classification"
    )
    artifact = _make_artifact({"perfect": ConstantPredictor(1)})
    report = evaluate(artifact, fs, "classification")
    assert report.prefix_metrics["perfect"][3]["recall_macro"] == pytest.approx(
        1.0
    )


def test_roc_auc_is_nan_for_single_class_group():
    fs = _make_feature_set(
        n_per_prefix=5, prefix_lengths=[2], task="classification"
    )
    artifact = _make_artifact({"m": ConstantPredictor(1)})
    report = evaluate(artifact, fs, "classification")
    assert np.isnan(report.prefix_metrics["m"][2]["roc_auc"])


def test_pr_auc_is_one_for_single_class_perfect_predictions():
    fs = _make_feature_set(
        n_per_prefix=5, prefix_lengths=[2], task="classification"
    )
    artifact = _make_artifact({"m": ConstantPredictor(1)})
    report = evaluate(artifact, fs, "classification")
    assert report.prefix_metrics["m"][2]["pr_auc"] == pytest.approx(1.0)


def test_roc_auc_is_05_for_constant_predictor_multi_class():
    rows = []
    targets = []
    for i in range(10):
        rows.append({_PREFIX_COL: 2, "f": float(i)})
        targets.append(i % 2)
    X = pd.DataFrame(rows).reset_index(drop=True)
    y = pd.Series(targets, name="outcome")
    pl_series = pd.Series([2] * len(y), dtype=int)
    fs = FeatureSet(
        X_train=X,
        X_test=X,
        y_train=y,
        y_test=y,
        feature_names=list(X.columns),
        trace_ids_train=pd.Series(),
        trace_ids_test=pd.Series(),
        prefix_lengths_train=pl_series,
        prefix_lengths_test=pl_series,
    )
    artifact = _make_artifact({"constant": ConstantPredictor(0)})
    report = evaluate(artifact, fs, "classification")
    assert report.prefix_metrics["constant"][2]["roc_auc"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_r2_is_nan_for_single_sample_prefix():
    fs = _make_feature_set(n_per_prefix=1, prefix_lengths=[1])
    artifact = _make_artifact({"m": ConstantPredictor(10.0)})
    report = evaluate(artifact, fs, "regression")
    assert np.isnan(report.prefix_metrics["m"][1]["r2"])


def test_missing_prefix_length_column_does_not_raise():
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
        trace_ids_train=pd.Series(),
        trace_ids_test=pd.Series(),
        prefix_lengths_train=pd.Series(dtype=int),
        prefix_lengths_test=pd.Series(dtype=int),
    )
    artifact = _make_artifact({"m": ConstantPredictor(15.0)})
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = evaluate(artifact, fs, "regression")
    assert report is not None
    assert report.prefix_metrics["m"] != {}


def test_all_prefix_lengths_have_all_metrics():
    fs = _make_feature_set(n_per_prefix=5, prefix_lengths=[1, 2, 3])
    artifact = _make_artifact({"m": ConstantPredictor(20.0)})
    report = evaluate(artifact, fs, "regression")
    for pl in [1, 2, 3]:
        assert set(report.prefix_metrics["m"][pl].keys()) == {
            "mae",
            "rmse",
            "r2",
            "median_ae",
        }


def test_multiple_models_all_evaluated():
    fs = _make_feature_set(n_per_prefix=5, prefix_lengths=[2])
    artifact = _make_artifact(
        {"fast": ConstantPredictor(20.0), "slow": ConstantPredictor(25.0)}
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
    artifact = _make_artifact({"m": ConstantPredictor(15.0)})
    report = evaluate(artifact, fs, "regression")
    assert report.prefix_counts == {1: 7, 3: 7, 5: 7}


# ---------------------------------------------------------------------------
# compare_models – weighted averaging
# ---------------------------------------------------------------------------


def test_compare_models_weighted_avg_equal_counts():
    """When all prefix lengths have equal sample counts, the weighted
    average equals the simple average."""
    fs = _make_feature_set(n_per_prefix=10, prefix_lengths=[1, 2, 3])
    artifact = _make_artifact({"m": ConstantPredictor(20.0)})
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
        targets.append(100.0)  # y_true=100, pred=10 → rmse=90

    X = pd.DataFrame(rows).reset_index(drop=True)
    y = pd.Series(targets, name="remaining_time_hours")
    pl_series = X[_PREFIX_COL].astype(int).reset_index(drop=True)
    fs = FeatureSet(
        X_train=X,
        X_test=X,
        y_train=y,
        y_test=y,
        feature_names=list(X.columns),
        trace_ids_train=pd.Series(),
        trace_ids_test=pd.Series(),
        prefix_lengths_train=pl_series,
        prefix_lengths_test=pl_series,
    )

    artifact = _make_artifact({"m": ConstantPredictor(10.0)})
    report = evaluate(artifact, fs, "regression")

    # Verify the underlying metrics
    assert report.prefix_metrics["m"][pl_big]["rmse"] == pytest.approx(0.0)
    assert report.prefix_metrics["m"][pl_small]["rmse"] == pytest.approx(90.0)

    # Simple (unweighted) mean
    simple_mean = (0.0 + 90.0) / 2.0  # = 45.0
    # Weighted mean: (0*1000 + 90*2) / 1002 ≈ 0.1796
    weighted_mean = (0.0 * n_big + 90.0 * n_small) / (n_big + n_small)

    result = compare_models(report, "regression")
    assert result is not None
    # Weighted mean should be very close to 0, NOT the simple mean of 45.0
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
    artifact = _make_artifact({"perfect": ConstantPredictor(1)})
    report = evaluate(artifact, fs, "classification")
    result = compare_models(report, "classification")
    assert result is not None
    assert result.ranking_metric == "f1_weighted"
    assert result.best_model_score == pytest.approx(1.0)


def test_compare_models_regression_uses_rmse():
    """Regression task uses rmse for ranking."""
    fs = _make_feature_set(n_per_prefix=5, prefix_lengths=[1])
    artifact = _make_artifact({"m": ConstantPredictor(10.0)})
    report = evaluate(artifact, fs, "regression")
    result = compare_models(report, "regression")
    assert result is not None
    assert result.ranking_metric == "rmse"


# ---------------------------------------------------------------------------
# compare_models – plateau detection (windowed)
# ---------------------------------------------------------------------------


def _build_rmse_report(
    model_name: str, prefix_rmse: dict[int, float]
) -> EvaluationReport:
    """Build a minimal EvaluationReport with the given per-prefix RMSE values.

    Each prefix gets an equal sample count of 100 so weighted/unweighted
    averages coincide (weights don't matter for plateau tests).
    """
    prefix_metrics: dict[str, dict[int, dict[str, float]]] = {
        model_name: {pl: {"rmse": v} for pl, v in prefix_rmse.items()}
    }
    return EvaluationReport(
        prefix_metrics=prefix_metrics,
        model_names=[model_name],
        prefix_lengths=sorted(prefix_rmse),
        prefix_counts={pl: 100 for pl in prefix_rmse},
    )


def test_plateau_window_1_recovers_pairwise_behavior():
    """With window=1, the plateau detection is identical to the old
    consecutive-pair behavior."""
    # RMSE curve that plateaus early (small improvement between 1→2)
    report = _build_rmse_report("m", {1: 5.0, 2: 4.9, 3: 3.0})
    # Pair 1→2: improvement = (5.0 - 4.9) / 5.0 = 2% → below 5% threshold
    result = compare_models(report, "regression", plateau_window=1)
    assert result is not None
    bp = result.best_prefixes["m"]
    assert bp.plateau_prefix == 1  # plateau at the start of the tiny step

    # RMSE curve that keeps improving
    report2 = _build_rmse_report("m", {1: 5.0, 2: 4.0, 3: 3.0})
    result2 = compare_models(report2, "regression", plateau_window=1)
    assert result2 is not None
    bp2 = result2.best_prefixes["m"]
    assert bp2.plateau_prefix == 3  # never plateaued → last prefix


def test_plateau_window_ignores_isolated_noise():
    """A single noisy prefix length (one small improvement step) should not
    trigger a false plateau when the window size is > 1."""
    # RMSE curve: steady improvement, one flat step, then improvement again
    # 1→2: 5.0→4.0 = 20%  (big improvement)
    # 2→3: 4.0→3.95 = 1.25% (flat step — would trigger with window=1)
    # 3→4: 3.95→3.0 = 24% (big improvement again)
    # 4→5: 3.0→2.95 = 1.67% (flat)
    # 5→6: 2.95→2.9 = 1.69% (flat)
    # With window=3, the average of [20, 1.25, 24] = 15% — above threshold
    # Only at the end where [24, 1.67, 1.69] avg ≈ 9% — still above 5%
    report = _build_rmse_report(
        "m", {1: 5.0, 2: 4.0, 3: 3.95, 4: 3.0, 5: 2.95, 6: 2.9}
    )
    result = compare_models(report, "regression", plateau_window=3)
    assert result is not None
    bp = result.best_prefixes["m"]
    # Should NOT plateau at prefix 2 (where the flat step is)
    assert bp.plateau_prefix != 2
    # With window=3 and the given curve, should reach the end
    assert bp.plateau_prefix == 6


def test_plateau_window_detects_genuine_plateau():
    """When several consecutive steps show minimal improvement, a genuine
    plateau is detected."""
    # RMSE curve with a clear plateau at the end
    # 1→2: 10→8 = 20%
    # 2→3: 8→7.9 = 1.25%
    # 3→4: 7.9→7.81 = 1.14%
    # 4→5: 7.81→7.73 = 1.02%
    # Window=3 average of [1.25, 1.14, 1.02] = 1.14% → below 5%
    report = _build_rmse_report(
        "m", {1: 10.0, 2: 8.0, 3: 7.9, 4: 7.81, 5: 7.73}
    )
    result = compare_models(report, "regression", plateau_window=3)
    assert result is not None
    bp = result.best_prefixes["m"]
    # Plateau at prefix 2 (start of the flat window)
    assert bp.plateau_prefix == 2


def test_plateau_window_larger_than_data_falls_back():
    """When the window is larger than the number of improvements, the
    effective window shrinks to fit."""
    # Only 3 points → 2 improvements; window=5 becomes effectively window=2
    report = _build_rmse_report("m", {1: 10.0, 2: 9.9, 3: 9.85})
    result = compare_models(report, "regression", plateau_window=5)
    assert result is not None
    bp = result.best_prefixes["m"]
    # (10-9.9)/10 = 1%, (9.9-9.85)/9.9 ≈ 0.5% → avg ≈ 0.75%, below 5%
    assert bp.plateau_prefix == 1

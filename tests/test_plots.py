from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spi_time_series.evaluation.metrics import (
    detect_task,
    select_primary_metric,
)
from spi_time_series.evaluation.plots import (
    _plot_error_distribution,
    _plot_metric_vs_prefix,
    _plot_predicted_vs_actual,
    _plot_roc_pr_curves,
)


class _DummyModel:
    """Minimal model stub with predict and predict_proba."""

    def __init__(self, constant: float = 50.0, n_classes: int = 3):
        self._c = constant
        self._n_classes = n_classes

    def predict(self, X):
        return np.full(len(X), self._c)

    def predict_proba(self, X):
        n = len(X)
        proba = np.zeros((n, self._n_classes))
        proba[:, int(self._c) % self._n_classes] = 1.0
        return proba


def _make_regression_data(n: int = 60) -> tuple[dict, pd.DataFrame, pd.Series]:
    models = {
        "dummy": _DummyModel(constant=60.0, n_classes=1),
    }
    rng = np.random.default_rng(42)
    a = n // 3
    b = n // 3
    c = n - a - b
    prefix_lengths = [1] * a + [5] * b + [10] * c
    X = pd.DataFrame(
        {
            "BasicControlFlowFeatures__prefix_length": prefix_lengths,
            "feat_a": rng.normal(size=n),
        }
    ).reset_index(drop=True)
    y = pd.Series(rng.uniform(20, 120, n), name="remaining_time_hours")
    return models, X, y


def _make_classification_data(
    n: int = 60,
) -> tuple[dict, pd.DataFrame, pd.Series]:
    models = {
        "dummy": _DummyModel(constant=1.0, n_classes=3),
    }
    rng = np.random.default_rng(42)
    a = n // 3
    b = n // 3
    c = n - a - b
    prefix_lengths = [1] * a + [5] * b + [10] * c
    X = pd.DataFrame(
        {
            "BasicControlFlowFeatures__prefix_length": prefix_lengths,
            "feat_a": rng.normal(size=n),
        }
    ).reset_index(drop=True)
    y = pd.Series(rng.integers(0, 3, n).astype(int), name="outcome")
    return models, X, y


# ---------------------------------------------------------------------------
# Task detection
# ---------------------------------------------------------------------------


def test_detect_task_regression():
    assert detect_task({"mae", "rmse", "r2", "median_ae"}) == "regression"


def test_detect_task_classification():
    assert detect_task({"accuracy", "f1_weighted"}) == "classification"


def test_detect_task_raises_on_unknown():
    with pytest.raises(ValueError, match="Cannot detect task"):
        detect_task({"unknown_col"})


def test_select_primary_metric_regression():
    assert select_primary_metric("regression", {"rmse", "mae"}) == "rmse"


def test_select_primary_metric_fallback():
    assert select_primary_metric("regression", {"median_ae"}) == "median_ae"


def test_select_primary_metric_classification():
    assert (
        select_primary_metric("classification", {"f1_weighted"})
        == "f1_weighted"
    )


# ---------------------------------------------------------------------------
# Plot smoke tests
# ---------------------------------------------------------------------------


def test_plot_metric_vs_prefix(tmp_path: Path):
    csv = tmp_path / "report.csv"
    pd.DataFrame(
        {
            "model": ["m1", "m1", "m2", "m2"],
            "prefix_length": [1, 2, 1, 2],
            "rmse": [10.0, 9.0, 20.0, 18.0],
            "mae": [8.0, 7.0, 15.0, 14.0],
        }
    ).to_csv(csv, index=False)

    out = tmp_path / "metric_vs_prefix.png"
    _plot_metric_vs_prefix(csv, "rmse", out)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_error_distribution(tmp_path: Path):
    models, X, y = _make_regression_data()
    out = tmp_path / "error_dist.png"
    _plot_error_distribution(
        models,
        X,
        y,
        "BasicControlFlowFeatures__prefix_length",
        out,
    )
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_predicted_vs_actual(tmp_path: Path):
    models, X, y = _make_regression_data()
    out = tmp_path / "pred_vs_act.png"
    _plot_predicted_vs_actual(
        models,
        X,
        y,
        "BasicControlFlowFeatures__prefix_length",
        out,
    )
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_roc_pr_curves(tmp_path: Path):
    models, X, y = _make_classification_data()
    out = tmp_path / "roc_pr.png"
    _plot_roc_pr_curves(
        models,
        X,
        y,
        "BasicControlFlowFeatures__prefix_length",
        out,
    )
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_roc_pr_curves_small_data(tmp_path: Path):
    models, X, y = _make_classification_data(n=5)
    out = tmp_path / "roc_pr_small.png"
    _plot_roc_pr_curves(
        models,
        X,
        y,
        "BasicControlFlowFeatures__prefix_length",
        out,
    )
    assert out.exists()

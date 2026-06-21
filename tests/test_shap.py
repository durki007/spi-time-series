from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

from spi_time_series.data.schemas import (
    EvaluationReport,
    FeatureSet,
    ModelArtifact,
)
from spi_time_series.evaluation.shap_explainability import report_shap

_pl_col = "BasicControlFlowFeatures__prefix_length"


def _make_classification_data():
    rng = np.random.default_rng(42)
    n = 100
    n_features = 6
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    X[_pl_col] = rng.integers(1, 5, size=n)
    y = pd.Series(rng.integers(0, 2, size=n), name="outcome")

    model = Pipeline(
        [("rf", RandomForestClassifier(n_estimators=10, random_state=42))]
    )
    model.fit(X, y)

    trace_ids = pd.Series([f"c{i}" for i in range(n)])

    fs = FeatureSet(
        X_train=X,
        X_test=X,
        y_train=y,
        y_test=y,
        trace_ids_train=trace_ids,
        trace_ids_test=trace_ids,
        feature_names=list(X.columns),
    )

    artifact = ModelArtifact(
        models={"rf": model},
        feature_names=list(X.columns),
        target_col="outcome",
    )

    preds = model.predict(X)

    report = EvaluationReport(
        feature_set=fs,
        model_predictions={"rf": pd.Series(preds, index=X.index)},
    )

    return artifact, report


def _make_regression_data():
    rng = np.random.default_rng(42)
    n = 100
    n_features = 6
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    X[_pl_col] = rng.integers(1, 5, size=n)
    y = pd.Series(rng.normal(50, 10, size=n), name="remaining_time_hours")

    model = Pipeline(
        [("rf", RandomForestRegressor(n_estimators=10, random_state=42))]
    )
    model.fit(X, y)

    trace_ids = pd.Series([f"c{i}" for i in range(n)])

    fs = FeatureSet(
        X_train=X,
        X_test=X,
        y_train=y,
        y_test=y,
        trace_ids_train=trace_ids,
        trace_ids_test=trace_ids,
        feature_names=list(X.columns),
    )

    artifact = ModelArtifact(
        models={"rf": model},
        feature_names=list(X.columns),
        target_col="remaining_time_hours",
    )

    preds = model.predict(X)

    report = EvaluationReport(
        feature_set=fs,
        model_predictions={"rf": pd.Series(preds, index=X.index)},
    )

    return artifact, report


def test_shap_classification_generates_plots(tmp_path: Path):
    artifact, report = _make_classification_data()
    report_shap(artifact, report, tmp_path)

    shap_dir = tmp_path / "shap"
    assert shap_dir.is_dir()

    files = sorted(p.name for p in shap_dir.iterdir())
    assert "rf_shap_summary_bar.png" in files
    assert "rf_shap_summary_dot.png" in files
    assert "rf_shap_waterfall_cls0.png" in files
    assert "rf_shap_waterfall_cls1.png" in files

    for f in shap_dir.iterdir():
        assert f.stat().st_size > 1000, f"Empty file: {f.name}"


def test_shap_regression_generates_plots(tmp_path: Path):
    artifact, report = _make_regression_data()
    report_shap(artifact, report, tmp_path)

    shap_dir = tmp_path / "shap"
    assert shap_dir.is_dir()

    files = sorted(p.name for p in shap_dir.iterdir())
    assert "rf_shap_summary_bar.png" in files
    assert "rf_shap_summary_dot.png" in files
    assert "rf_shap_waterfall.png" in files

    for f in shap_dir.iterdir():
        assert f.stat().st_size > 1000, f"Empty file: {f.name}"


def test_shap_skips_when_no_output_dir():
    artifact, report = _make_classification_data()
    report_shap(artifact, report, None)  # should not crash


def test_shap_skips_when_no_feature_set():
    artifact, _ = _make_classification_data()
    report = EvaluationReport()
    report_shap(artifact, report, Path("ignored"))  # should not crash


def test_shap_non_tree_model_skips_gracefully(tmp_path: Path):
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(10, 3)), columns=["a", "b", "c"])
    X[_pl_col] = 1
    y = pd.Series(rng.integers(0, 2, size=10), name="outcome")
    trace_ids = pd.Series(["c0"] * 10)

    model = Pipeline([("lr", LogisticRegression())])
    model.fit(X, y)

    fs = FeatureSet(
        X_train=X,
        X_test=X,
        y_train=y,
        y_test=y,
        trace_ids_train=trace_ids,
        trace_ids_test=trace_ids,
        feature_names=list(X.columns),
    )

    artifact = ModelArtifact(
        models={"lr": model},
        feature_names=list(X.columns),
        target_col="outcome",
    )

    report = EvaluationReport(feature_set=fs)
    report_shap(artifact, report, tmp_path)  # should log warning, not crash

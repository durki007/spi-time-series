from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
from pydantic import ValidationError

from spi_time_series.config.schema import (
    RunConfig,
)
from spi_time_series.data.schemas import EvaluationReport
from spi_time_series.evaluation.ablation import (
    ComboResult,
    _aggregate_metric,
    _build_extractor,
    _lower_is_better,
    _write_ablation_report,
    _write_per_prefix_metrics,
)

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_ablation_config_classification_default_metric() -> None:
    """roc_auc is valid for classification task."""
    raw = {
        "task": "classification",
        "models": {
            "rf": {
                "model_type": "RandomForestClassifier",
                "params": {"random_state": 42},
            }
        },
        "ablation": {
            "metric": "roc_auc",
            "log": {
                "combinations": [
                    {"name": "dummy", "features": []},
                    {"name": "A", "features": ["ActivityCountFeatures"]},
                ]
            },
        },
    }
    config = RunConfig.model_validate(raw)
    assert config.ablation is not None
    assert config.ablation.metric == "roc_auc"


def test_ablation_config_classification_f1_weighted() -> None:
    raw = {
        "task": "classification",
        "models": {
            "rf": {
                "model_type": "RandomForestClassifier",
                "params": {"random_state": 42},
            }
        },
        "ablation": {
            "metric": "f1_weighted",
            "log": {
                "combinations": [
                    {"name": "dummy", "features": []},
                ]
            },
        },
    }
    config = RunConfig.model_validate(raw)
    assert config.ablation is not None
    assert config.ablation.metric == "f1_weighted"


def test_ablation_config_regression_rmse() -> None:
    raw = {
        "task": "regression",
        "models": {
            "rf": {
                "model_type": "RandomForestRegressor",
                "params": {"random_state": 42},
            }
        },
        "ablation": {
            "metric": "rmse",
            "log": {
                "combinations": [
                    {"name": "dummy", "features": []},
                ]
            },
        },
    }
    config = RunConfig.model_validate(raw)
    assert config.ablation is not None


def test_ablation_config_invalid_metric_for_task() -> None:
    with pytest.raises(
        ValidationError, match="ablation.metric.*rmse.*classification"
    ):
        RunConfig.model_validate(
            {
                "task": "classification",
                "models": {
                    "rf": {
                        "model_type": "RandomForestClassifier",
                        "params": {"random_state": 42},
                    }
                },
                "ablation": {
                    "metric": "rmse",
                    "log": {
                        "combinations": [
                            {"name": "dummy", "features": []},
                        ]
                    },
                },
            }
        )


def test_ablation_config_unknown_feature() -> None:
    with pytest.raises(ValidationError, match="not a recognized feature"):
        RunConfig.model_validate(
            {
                "task": "classification",
                "models": {
                    "rf": {
                        "model_type": "RandomForestClassifier",
                        "params": {"random_state": 42},
                    }
                },
                "ablation": {
                    "log": {
                        "combinations": [
                            {"name": "bad", "features": ["NonExistentFeature"]},
                        ]
                    },
                },
            }
        )


def test_ablation_config_ts_base_validated() -> None:
    with pytest.raises(ValidationError, match="not a recognized feature"):
        RunConfig.model_validate(
            {
                "task": "classification",
                "models": {
                    "rf": {
                        "model_type": "RandomForestClassifier",
                        "params": {"random_state": 42},
                    }
                },
                "ablation": {
                    "log": {
                        "combinations": [
                            {"name": "dummy", "features": []},
                        ]
                    },
                    "ts": {
                        "base": ["BadFeature"],
                        "combinations": [
                            {
                                "name": "t1",
                                "features": ["ActiveCaseCountFeature"],
                            },
                        ],
                    },
                },
            }
        )


def test_ablation_config_ts_optional() -> None:
    """TsAblation can be None — log-only ablation."""
    raw = {
        "task": "classification",
        "models": {
            "rf": {
                "model_type": "RandomForestClassifier",
                "params": {"random_state": 42},
            }
        },
        "ablation": {
            "log": {
                "combinations": [
                    {"name": "dummy", "features": []},
                    {"name": "A", "features": ["ActivityCountFeatures"]},
                ]
            },
        },
    }
    config = RunConfig.model_validate(raw)
    assert config.ablation is not None
    assert config.ablation.ts is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_lower_is_better() -> None:
    assert _lower_is_better("rmse") is True
    assert _lower_is_better("mae") is True
    assert _lower_is_better("median_ae") is True
    assert _lower_is_better("roc_auc") is False
    assert _lower_is_better("f1_weighted") is False
    assert _lower_is_better("r2") is False


def test_aggregate_metric() -> None:
    from spi_time_series.data.schemas import EvaluationReport

    report = EvaluationReport(
        model_predictions={},
        prefix_metrics={
            "rf": {
                3: {"roc_auc": 0.6, "f1_weighted": 0.5},
                4: {"roc_auc": 0.7, "f1_weighted": 0.6},
            }
        },
        model_names=["rf"],
        prefix_lengths=[3, 4],
        prefix_counts={3: 10, 4: 20},
    )
    val = _aggregate_metric(report, "roc_auc", "rf")
    expected = (0.6 * 10 + 0.7 * 20) / 30
    assert abs(val - expected) < 1e-6


def test_aggregate_metric_missing_model() -> None:
    from spi_time_series.data.schemas import EvaluationReport

    report = EvaluationReport(
        model_predictions={},
        prefix_metrics={"rf": {3: {"roc_auc": 0.6}}},
        model_names=["rf"],
        prefix_lengths=[3],
        prefix_counts={3: 10},
    )
    val = _aggregate_metric(report, "roc_auc", "hgb")
    assert val != val  # nan check


def test_aggregate_metric_missing_metric() -> None:
    from spi_time_series.data.schemas import EvaluationReport

    report = EvaluationReport(
        model_predictions={},
        prefix_metrics={"rf": {3: {"roc_auc": 0.6}}},
        model_names=["rf"],
        prefix_lengths=[3],
        prefix_counts={3: 10},
    )
    val = _aggregate_metric(report, "f1_weighted", "rf")
    assert val != val  # nan


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _make_dummy_report(
    model_name: str = "rf",
    metrics: dict[int, dict[str, float]] | None = None,
) -> EvaluationReport:
    if metrics is None:
        metrics = {3: {"roc_auc": 0.6, "f1_weighted": 0.5}}
    return EvaluationReport(
        model_predictions={},
        prefix_metrics={model_name: metrics},
        model_names=[model_name],
        prefix_lengths=sorted(metrics),
        prefix_counts={pl: 10 for pl in metrics},
    )


def test_write_ablation_report(tmp_path: Path) -> None:
    results = [
        ComboResult(
            name="A",
            group="log",
            report=_make_dummy_report(metrics={3: {"roc_auc": 0.7}}),
        ),
        ComboResult(
            name="B",
            group="log",
            report=_make_dummy_report(metrics={3: {"roc_auc": 0.6}}),
        ),
    ]
    path = _write_ablation_report(results, "roc_auc", tmp_path)
    assert path.exists()
    df = pd.read_csv(path)
    assert list(df.columns) == [
        "combo",
        "group",
        "model",
        "prefix_length",
        "metric",
        "metric_value",
        "rank",
    ]
    assert len(df) == 2
    # roc_auc: higher-is-better → A (0.7) rank 1, B (0.6) rank 2
    a_row = df[df["combo"] == "A"].iloc[0]
    b_row = df[df["combo"] == "B"].iloc[0]
    assert a_row["rank"] == 1
    assert b_row["rank"] == 2


def test_write_per_prefix_metrics(tmp_path: Path) -> None:
    results = [
        ComboResult(
            name="A",
            group="log",
            report=_make_dummy_report(
                metrics={3: {"roc_auc": 0.7}, 4: {"roc_auc": 0.8}}
            ),
        ),
    ]
    path = _write_per_prefix_metrics(results, "roc_auc", tmp_path)
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) == 2
    assert set(df["prefix_length"]) == {3, 4}


# ---------------------------------------------------------------------------
# _build_extractor
# ---------------------------------------------------------------------------


def test_build_extractor_dummy() -> None:
    """Empty feature list builds a valid extractor (dummy run)."""
    raw = {
        "task": "classification",
        "models": {
            "rf": {
                "model_type": "RandomForestClassifier",
                "params": {"random_state": 42},
            }
        },
        "ablation": {
            "log": {
                "combinations": [
                    {"name": "dummy", "features": []},
                ]
            },
        },
    }
    config = RunConfig.model_validate(raw)
    extractor = _build_extractor([], config)
    assert callable(extractor)


def test_build_extractor_with_features() -> None:
    """Known feature names produce a callable extractor."""
    raw = {
        "task": "classification",
        "models": {
            "rf": {
                "model_type": "RandomForestClassifier",
                "params": {"random_state": 42},
            }
        },
        "ablation": {
            "log": {
                "combinations": [
                    {
                        "name": "AW",
                        "features": [
                            "ActivityCountFeatures",
                            "WaitingStateFeatures",
                        ],
                    },
                ]
            },
        },
    }
    config = RunConfig.model_validate(raw)
    extractor = _build_extractor(
        ["ActivityCountFeatures", "WaitingStateFeatures"], config
    )
    assert callable(extractor)


# ---------------------------------------------------------------------------
# Integration: runner with mocked Pipeline
# ---------------------------------------------------------------------------


def test_ablation_runner_with_mock(monkeypatch) -> None:
    """Verify runner loops over all combos and produces output files."""
    from spi_time_series.data.schemas import EvaluationReport
    from spi_time_series.evaluation.ablation import AblationRunner

    call_log: list[str] = []

    class MockPipeline:
        def __init__(self, **kwargs):
            pass

        def fit(self, **kwargs):
            call_log.append("fit")

        def evaluate(self, output_dir=None):
            call_log.append("evaluate")
            return EvaluationReport(
                model_predictions={},
                prefix_metrics={
                    "rf": {3: {"roc_auc": 0.6}, 4: {"roc_auc": 0.7}}
                },
                model_names=["rf"],
                prefix_lengths=[3, 4],
                prefix_counts={3: 10, 4: 20},
            )

    class MockBuilder:
        @classmethod
        def from_config(cls, config):
            return cls()

        def with_feature_extractor(self, fn):
            return self

        def add_evaluator(self, fn):
            return self

        def build(self):
            return MockPipeline()

    monkeypatch.setattr(
        "spi_time_series.evaluation.ablation.PipelineBuilder", MockBuilder
    )

    raw = {
        "task": "classification",
        "models": {
            "rf": {
                "model_type": "RandomForestClassifier",
                "params": {"random_state": 42},
            }
        },
        "ablation": {
            "metric": "roc_auc",
            "log": {
                "combinations": [
                    {"name": "dummy", "features": []},
                    {"name": "A", "features": ["ActivityCountFeatures"]},
                ]
            },
            "ts": {
                "base": ["ActivityCountFeatures"],
                "combinations": [
                    {"name": "AC", "features": ["ActiveCaseCountFeature"]},
                ],
            },
        },
    }
    config = RunConfig.model_validate(raw)
    runner = AblationRunner(config)

    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        runner.run(out, n_jobs=1, search_config=config.search)

        assert len(call_log) == 6  # 3 combos × (fit + evaluate)
        assert (out / "ablation_report.csv").exists()
        assert (out / "per_prefix_metrics.csv").exists()
        assert (out / "figures" / "metric_vs_prefix.png").exists()


def test_ablation_runner_log_only(monkeypatch) -> None:
    """Runner works without TS phase."""
    from spi_time_series.data.schemas import EvaluationReport
    from spi_time_series.evaluation.ablation import AblationRunner

    call_log: list[str] = []

    class MockPipeline:
        def __init__(self, **kwargs):
            pass

        def fit(self, **kwargs):
            call_log.append("fit")

        def evaluate(self, output_dir=None):
            call_log.append("evaluate")
            return EvaluationReport(
                model_predictions={},
                prefix_metrics={"rf": {3: {"roc_auc": 0.6}}},
                model_names=["rf"],
                prefix_lengths=[3],
                prefix_counts={3: 10},
            )

    class MockBuilder:
        @classmethod
        def from_config(cls, config):
            return cls()

        def with_feature_extractor(self, fn):
            return self

        def add_evaluator(self, fn):
            return self

        def build(self):
            return MockPipeline()

    monkeypatch.setattr(
        "spi_time_series.evaluation.ablation.PipelineBuilder", MockBuilder
    )

    raw = {
        "task": "classification",
        "models": {
            "rf": {
                "model_type": "RandomForestClassifier",
                "params": {"random_state": 42},
            }
        },
        "ablation": {
            "metric": "roc_auc",
            "log": {
                "combinations": [
                    {"name": "dummy", "features": []},
                    {"name": "A", "features": ["ActivityCountFeatures"]},
                ]
            },
        },
    }
    config = RunConfig.model_validate(raw)
    runner = AblationRunner(config)

    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        runner.run(out, n_jobs=1, search_config=config.search)

    assert len(call_log) == 4  # 2 combos × (fit + evaluate)

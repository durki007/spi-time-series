"""Integration tests for the CLI entry point."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml

from spi_time_series.data.constants import EVENT_NAMES
from spi_time_series.main import main
from spi_time_series.pipeline.state import PipelineState


def _make_synthetic_log(
    n_cases: int = 50, events_per_case: int = 5
) -> pd.DataFrame:
    """Build a minimal event log DataFrame compatible with the pipeline."""
    base_time = datetime(2023, 1, 1, tzinfo=UTC)
    rows = []
    for case_idx in range(n_cases):
        case_id = f"case_{case_idx:03d}"
        case_start = base_time + timedelta(days=case_idx * (365 / n_cases))
        for event_idx in range(events_per_case):
            rows.append(
                {
                    "case:concept:name": case_id,
                    "concept:name": EVENT_NAMES[
                        (case_idx + event_idx) % len(EVENT_NAMES)
                    ],
                    "time:timestamp": case_start
                    + timedelta(minutes=event_idx * 10),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_log() -> pd.DataFrame:
    return _make_synthetic_log()


@pytest.fixture()
def regression_config(tmp_path: Path) -> Path:
    cfg = {
        "task": "regression",
        "data": {
            "valid_end_activities": [],
            "top_k_variants": None,
            "split_quantile": 0.7,
        },
        "prefix": {"min_length": 1, "max_length": 3},
        "features": {"one_hot_encode_categorical": False},
        "search": {
            "n_iter": 1,
            "cv_folds": 2,
            "random_state": 0,
            "search_sample_size": None,
        },
        "models": {
            "ridge": {
                "model_type": "Ridge",
                "params": {},
                "param_grid": {"alpha": [0.1]},
            }
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return config_path


def _patch_dataset(monkeypatch, tmp_path, log):
    from spi_time_series.data import dataset as dataset_mod

    def _mock_init(self, data_dir=None):
        self.data_dir = tmp_path
        self.log = log

    monkeypatch.setattr(dataset_mod.Dataset, "__init__", _mock_init)


def _report_path(output_dir: Path) -> Path:
    return output_dir / "reports" / "evaluation_report.csv"


@pytest.mark.integration
def test_main_regression_produces_report(
    tmp_path, regression_config, synthetic_log, monkeypatch
):
    """Full pipeline run produces a report JSON and a saved config."""
    _patch_dataset(monkeypatch, tmp_path, synthetic_log)

    output_dir = tmp_path / "output"
    main([str(regression_config), "--output-dir", str(output_dir)])

    report_file = _report_path(output_dir)
    assert report_file.exists()
    df = pd.read_csv(report_file)
    assert list(df["model"].unique()) == ["ridge"]
    assert len(df) > 0, "Expected at least one row per prefix length"
    assert (output_dir / "run_config.yaml").exists()


@pytest.mark.integration
def test_dry_run_prints_config_no_files(tmp_path, regression_config, capsys):
    """--dry-run prints YAML config and produces no output files."""
    output_dir = tmp_path / "output"
    main([str(regression_config), "--dry-run", "--output-dir", str(output_dir)])

    out = capsys.readouterr().out
    assert "task: regression" in out
    assert "ridge" in out
    assert not _report_path(output_dir).exists()


@pytest.mark.integration
def test_override_applies_and_revalidates(
    tmp_path, regression_config, synthetic_log, monkeypatch
):
    """--override updates a nested config value before validation."""
    _patch_dataset(monkeypatch, tmp_path, synthetic_log)

    output_dir = tmp_path / "output"
    main(
        [
            str(regression_config),
            "--override",
            "search.random_state=99",
            "--output-dir",
            str(output_dir),
        ]
    )

    saved_cfg = yaml.safe_load((output_dir / "run_config.yaml").read_text())
    assert saved_cfg["search"]["random_state"] == 99
    assert _report_path(output_dir).exists()


@pytest.mark.integration
def test_fit_reuses_stages_when_config_unchanged(
    tmp_path, regression_config, synthetic_log, monkeypatch
):
    """Second run with identical config has the same cache keys (all stages skipped)."""
    _patch_dataset(monkeypatch, tmp_path, synthetic_log)
    output_dir = tmp_path / "output"

    main([str(regression_config), "--output-dir", str(output_dir)])
    state1: PipelineState = joblib.load(output_dir / "checkpoint.joblib")

    main([str(regression_config), "--output-dir", str(output_dir)])
    state2: PipelineState = joblib.load(output_dir / "checkpoint.joblib")

    assert state1.extract_key == state2.extract_key
    assert state1.train_keys == state2.train_keys
    assert _report_path(output_dir).exists()


@pytest.mark.integration
def test_fit_invalidates_extract_on_data_config_change(
    tmp_path, regression_config, synthetic_log, monkeypatch
):
    """Changing data.split_quantile causes extract (and all models) to re-run."""
    _patch_dataset(monkeypatch, tmp_path, synthetic_log)
    output_dir = tmp_path / "output"

    main([str(regression_config), "--output-dir", str(output_dir)])
    state1: PipelineState = joblib.load(output_dir / "checkpoint.joblib")

    main(
        [
            str(regression_config),
            "--override",
            "data.split_quantile=0.6",
            "--output-dir",
            str(output_dir),
        ]
    )
    state2: PipelineState = joblib.load(output_dir / "checkpoint.joblib")

    assert state1.extract_key != state2.extract_key
    assert _report_path(output_dir).exists()


@pytest.mark.integration
def test_force_reruns_all_stages(
    tmp_path, regression_config, synthetic_log, monkeypatch
):
    """--force recomputes all stages even when config is unchanged."""
    _patch_dataset(monkeypatch, tmp_path, synthetic_log)
    output_dir = tmp_path / "output"

    main([str(regression_config), "--output-dir", str(output_dir)])
    main([str(regression_config), "--force", "--output-dir", str(output_dir)])
    assert _report_path(output_dir).exists()

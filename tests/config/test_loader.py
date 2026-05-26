import pytest
import yaml
from sklearn.linear_model import LogisticRegression, Ridge

from spi_time_series.config.loader import (
    build_estimator,
    load_config,
    save_config,
)
from spi_time_series.config.schema import ModelConfig, RunConfig


def test_load_config_from_file(tmp_path, regression_raw):
    config_file = tmp_path / "run.yaml"
    config_file.write_text(yaml.safe_dump(regression_raw), encoding="utf-8")
    config = load_config(config_file)
    assert config.task == "regression"
    assert "ridge" in config.models


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.yaml")


def test_save_load_round_trip(tmp_path, regression_raw):
    original = RunConfig.model_validate(regression_raw)
    out_path = tmp_path / "saved.yaml"
    save_config(original, out_path)
    reloaded = load_config(out_path)
    assert original.task == reloaded.task
    assert original.models == reloaded.models
    assert original.prefix == reloaded.prefix


def test_build_estimator_correct_class():
    cfg = ModelConfig(type="Ridge", params={}, param_grid={"alpha": [0.1]})
    estimator = build_estimator(cfg)
    assert isinstance(estimator, Ridge)


def test_build_estimator_applies_params():
    cfg = ModelConfig(
        type="Ridge", params={"alpha": 5.0}, param_grid={"alpha": [5.0]}
    )
    estimator = build_estimator(cfg)
    assert estimator.alpha == 5.0


def test_build_estimator_logistic_regression():
    cfg = ModelConfig(
        type="LogisticRegression",
        params={"max_iter": 200, "solver": "saga"},
        param_grid={"C": [0.1]},
    )
    estimator = build_estimator(cfg)
    assert isinstance(estimator, LogisticRegression)
    assert estimator.max_iter == 200

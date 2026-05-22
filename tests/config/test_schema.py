import joblib
import pytest
import yaml
from pydantic import ValidationError

from spi_time_series.config.schema import RunConfig


def test_valid_regression_config_parses(regression_raw):
    config = RunConfig.model_validate(regression_raw)
    assert config.task == "regression"
    assert config.prefix.min_length == 3
    assert config.prefix.max_length == 10


def test_valid_classification_config_parses(classification_raw):
    config = RunConfig.model_validate(classification_raw)
    assert config.task == "classification"
    assert config.search.search_sample_size == 1000


def test_invalid_task_raises(regression_raw):
    regression_raw["task"] = "forecasting"
    with pytest.raises(ValidationError, match="task"):
        RunConfig.model_validate(regression_raw)


def test_unknown_model_type_raises(regression_raw):
    regression_raw["models"]["ridge"]["type"] = "SVM"
    with pytest.raises(ValidationError, match="SVM"):
        RunConfig.model_validate(regression_raw)


def test_task_model_mismatch_raises(regression_raw):
    regression_raw["models"]["lr"] = {
        "type": "LogisticRegression",
        "params": {},
        "param_grid": {"C": [0.1]},
    }
    with pytest.raises(ValidationError, match="classifier"):
        RunConfig.model_validate(regression_raw)


def test_prefix_min_exceeds_max_raises(regression_raw):
    regression_raw["prefix"]["min_length"] = 10
    regression_raw["prefix"]["max_length"] = 5
    with pytest.raises(ValidationError, match="min_length"):
        RunConfig.model_validate(regression_raw)


def test_empty_param_grid_list_raises(regression_raw):
    regression_raw["models"]["ridge"]["param_grid"]["alpha"] = []
    with pytest.raises(ValidationError, match="param_grid"):
        RunConfig.model_validate(regression_raw)


def test_empty_models_dict_raises(regression_raw):
    regression_raw["models"] = {}
    with pytest.raises(ValidationError, match="models"):
        RunConfig.model_validate(regression_raw)


def test_top_k_zero_raises(regression_raw):
    regression_raw["data"]["top_k_variants"] = 0
    with pytest.raises(ValidationError, match="top_k_variants"):
        RunConfig.model_validate(regression_raw)


def test_split_quantile_boundary_raises(regression_raw):
    regression_raw["data"]["split_quantile"] = 1.0
    with pytest.raises(ValidationError, match="split_quantile"):
        RunConfig.model_validate(regression_raw)


def test_null_max_length_allowed(regression_raw):
    regression_raw["prefix"]["max_length"] = None
    config = RunConfig.model_validate(regression_raw)
    assert config.prefix.max_length is None


def test_null_top_k_allowed(regression_raw):
    config = RunConfig.model_validate(regression_raw)
    assert config.data.top_k_variants is None


def test_null_in_param_grid_allowed(regression_raw):
    regression_raw["models"]["rf"] = {
        "type": "RandomForestRegressor",
        "params": {},
        "param_grid": {"max_depth": [None, 5, 10]},
    }
    config = RunConfig.model_validate(regression_raw)
    assert None in config.models["rf"].param_grid["max_depth"]


def test_to_yaml_round_trip(regression_raw):
    config = RunConfig.model_validate(regression_raw)
    serialized = config.to_yaml()
    reloaded = RunConfig.model_validate(yaml.safe_load(serialized))
    assert config.task == reloaded.task
    assert config.prefix == reloaded.prefix
    assert config.search == reloaded.search
    assert config.models == reloaded.models


def test_checkpoint_params_excludes_checkpoint_dir(regression_raw):
    config = RunConfig.model_validate(regression_raw)
    params = config.checkpoint_params()
    assert "checkpoint_dir" not in params


def test_checkpoint_params_is_joblib_hashable(regression_raw):
    config = RunConfig.model_validate(regression_raw)
    h = joblib.hash(config.checkpoint_params())
    assert isinstance(h, str) and len(h) > 0


def test_checkpoint_params_differs_for_different_configs(
    regression_raw, classification_raw
):
    r = RunConfig.model_validate(regression_raw)
    c = RunConfig.model_validate(classification_raw)
    assert joblib.hash(r.checkpoint_params()) != joblib.hash(
        c.checkpoint_params()
    )

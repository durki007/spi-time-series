import copy

import pytest


@pytest.fixture
def regression_raw():
    return {
        "task": "regression",
        "data": {
            "valid_end_activities": ["A_Denied"],
            "top_k_variants": None,
            "split_quantile": 0.8,
        },
        "prefix": {"min_length": 3, "max_length": 10},
        "features": {"one_hot_encode_categorical": False},
        "search": {
            "n_iter": 5,
            "cv_folds": 2,
            "random_state": 0,
            "search_sample_size": None,
        },
        "models": {
            "ridge": {
                "model_type": "Ridge",
                "params": {},
                "param_grid": {"alpha": [0.1, 1.0]},
            }
        },
    }


@pytest.fixture
def classification_raw(regression_raw):
    raw = copy.deepcopy(regression_raw)
    raw["task"] = "classification"
    raw["search"]["search_sample_size"] = 1000
    raw["models"] = {
        "lr": {
            "model_type": "LogisticRegression",
            "params": {"max_iter": 100},
            "param_grid": {"C": [0.1, 1.0]},
        }
    }
    return raw

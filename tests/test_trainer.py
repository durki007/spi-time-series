import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline as SklearnPipeline

from spi_time_series.data.schemas import FeatureSet, ModelArtifact
from spi_time_series.models.trainer import train

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def numeric_feature_set():
    rng = np.random.default_rng(0)
    N_TRAIN, N_TEST, N_FEATURES = 60, 20, 5
    feature_names = [
        f"BasicControlFlowFeatures__feat_{i}" for i in range(N_FEATURES)
    ]

    X_train = pd.DataFrame(
        rng.random((N_TRAIN, N_FEATURES)), columns=feature_names
    )
    X_test = pd.DataFrame(
        rng.random((N_TEST, N_FEATURES)), columns=feature_names
    )
    X_train.iloc[0, 0] = float("nan")
    X_test.iloc[3, 2] = float("nan")

    y_train = pd.Series(rng.random(N_TRAIN) * 100, name="remaining_time_hours")
    y_test = pd.Series(rng.random(N_TEST) * 100, name="remaining_time_hours")

    return FeatureSet(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
    )


@pytest.fixture
def simple_models():
    return {"ridge": Ridge(alpha=1.0)}


@pytest.fixture
def search_models():
    return {
        "ridge_search": RandomizedSearchCV(
            Ridge(),
            param_distributions={"alpha": [0.1, 1.0, 10.0]},
            n_iter=2,
            cv=2,
            random_state=42,
        )
    }


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_train_returns_model_artifact(numeric_feature_set, simple_models):
    artifact = train(numeric_feature_set, simple_models)
    assert isinstance(artifact, ModelArtifact)


def test_artifact_has_correct_model_keys(numeric_feature_set, simple_models):
    artifact = train(numeric_feature_set, simple_models)
    assert set(artifact.models.keys()) == set(simple_models.keys())


def test_artifact_pipelines_are_sklearn_pipelines(
    numeric_feature_set, simple_models
):
    artifact = train(numeric_feature_set, simple_models)
    for name, pipe in artifact.models.items():
        assert isinstance(pipe, SklearnPipeline), (
            f"{name} is not an SklearnPipeline"
        )


def test_artifact_pipeline_has_preprocessor_and_model_steps(
    numeric_feature_set, simple_models
):
    artifact = train(numeric_feature_set, simple_models)
    step_names = [s[0] for s in artifact.models["ridge"].steps]
    assert "preprocessor" in step_names
    assert "model" in step_names


def test_artifact_feature_names_match_input(numeric_feature_set, simple_models):
    artifact = train(numeric_feature_set, simple_models)
    assert artifact.feature_names == numeric_feature_set.feature_names


def test_artifact_target_col_from_series_name(
    numeric_feature_set, simple_models
):
    artifact = train(numeric_feature_set, simple_models)
    assert artifact.target_col == "remaining_time_hours"


def test_artifact_target_col_fallback_when_name_is_none(
    numeric_feature_set, simple_models
):
    unnamed_fs = FeatureSet(
        X_train=numeric_feature_set.X_train,
        X_test=numeric_feature_set.X_test,
        y_train=pd.Series(numeric_feature_set.y_train.values),
        y_test=numeric_feature_set.y_test,
        feature_names=numeric_feature_set.feature_names,
    )
    artifact = train(unnamed_fs, simple_models)
    assert artifact.target_col == "target"


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


def test_fitted_pipeline_can_predict(numeric_feature_set, simple_models):
    artifact = train(numeric_feature_set, simple_models)
    preds = artifact.models["ridge"].predict(numeric_feature_set.X_test)
    assert preds.shape == (len(numeric_feature_set.X_test),)


def test_fitted_pipeline_predicts_finite_values(
    numeric_feature_set, simple_models
):
    artifact = train(numeric_feature_set, simple_models)
    preds = artifact.models["ridge"].predict(numeric_feature_set.X_test)
    assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# Multiple models
# ---------------------------------------------------------------------------


def test_multiple_models_are_all_trained(numeric_feature_set):
    models = {"ridge_a": Ridge(alpha=0.1), "ridge_b": Ridge(alpha=10.0)}
    artifact = train(numeric_feature_set, models)
    assert len(artifact.models) == 2
    for name in models:
        preds = artifact.models[name].predict(numeric_feature_set.X_test)
        assert preds.shape == (len(numeric_feature_set.X_test),)


def test_models_have_independent_preprocessors(numeric_feature_set):
    models = {"ridge_a": Ridge(alpha=0.1), "ridge_b": Ridge(alpha=10.0)}
    artifact = train(numeric_feature_set, models)
    pre_a = artifact.models["ridge_a"].named_steps["preprocessor"]
    pre_b = artifact.models["ridge_b"].named_steps["preprocessor"]
    assert pre_a is not pre_b


# ---------------------------------------------------------------------------
# GridSearchCV / RandomizedSearchCV pass-through
# ---------------------------------------------------------------------------


def test_train_with_randomized_search(numeric_feature_set, search_models):
    artifact = train(numeric_feature_set, search_models)
    pipe = artifact.models["ridge_search"]
    assert isinstance(pipe, SklearnPipeline)
    search = pipe.named_steps["model"]
    assert hasattr(search, "best_params_")


def test_search_model_can_predict(numeric_feature_set, search_models):
    artifact = train(numeric_feature_set, search_models)
    preds = artifact.models["ridge_search"].predict(numeric_feature_set.X_test)
    assert preds.shape == (len(numeric_feature_set.X_test),)


# ---------------------------------------------------------------------------
# Mixed-dtype columns (string categoricals dropped silently)
# ---------------------------------------------------------------------------


def test_train_with_mixed_columns_does_not_crash():
    rng = np.random.default_rng(1)
    N = 40
    X = pd.DataFrame(
        {
            "BasicControlFlowFeatures__elapsed_time_hours": rng.random(N),
            "BasicControlFlowFeatures__prefix_length": rng.integers(
                1, 10, N
            ).astype(float),
            "BasicControlFlowFeatures__last_activity": rng.choice(
                ["A", "B", None], N
            ),
            "BasicControlFlowFeatures__last_transition": rng.choice(
                ["A->B", None], N
            ),
        }
    )
    y = pd.Series(rng.random(N) * 100, name="remaining_time_hours")
    fs = FeatureSet(
        X_train=X,
        X_test=X,
        y_train=y,
        y_test=y,
        feature_names=list(X.columns),
    )
    artifact = train(fs, {"ridge": Ridge()})
    preds = artifact.models["ridge"].predict(X)
    assert preds.shape == (N,)
    assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


def test_train_adds_pca_step_when_requested(numeric_feature_set, simple_models):
    artifact = train(
        numeric_feature_set,
        simple_models,
        pca_keep_percentage=0.95,
    )

    pipe = artifact.models["ridge"]

    assert "pca" in pipe.named_steps
    assert isinstance(pipe.named_steps["pca"], PCA)


def test_train_does_not_add_pca_step_by_default(
    numeric_feature_set, simple_models
):
    artifact = train(numeric_feature_set, simple_models)

    pipe = artifact.models["ridge"]

    assert "pca" not in pipe.named_steps


def test_train_passes_keep_percentage_to_pca(
    numeric_feature_set, simple_models
):
    artifact = train(
        numeric_feature_set,
        simple_models,
        pca_keep_percentage=0.90,
    )

    pca = artifact.models["ridge"].named_steps["pca"]

    assert pca.n_components == 0.90


def test_pca_reduces_dimensionality():
    rng = np.random.default_rng(42)

    x1 = rng.normal(size=100)

    X = pd.DataFrame(
        {
            "a": x1,
            "b": x1 * 1.01,
            "c": x1 * 0.99,
            "d": rng.normal(size=100),
        }
    )

    y = pd.Series(rng.normal(size=100), name="target")

    fs = FeatureSet(
        X_train=X,
        X_test=X,
        y_train=y,
        y_test=y,
        feature_names=list(X.columns),
    )

    artifact = train(
        fs,
        {"ridge": Ridge()},
        pca_keep_percentage=0.95,
    )

    pca = artifact.models["ridge"].named_steps["pca"]

    assert pca.n_components_ < X.shape[1]


def test_pca_pipeline_can_predict(numeric_feature_set, simple_models):
    artifact = train(
        numeric_feature_set,
        simple_models,
        pca_keep_percentage=0.95,
    )

    preds = artifact.models["ridge"].predict(numeric_feature_set.X_test)

    assert preds.shape == (len(numeric_feature_set.X_test),)
    assert np.all(np.isfinite(preds))

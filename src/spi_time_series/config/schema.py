from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge

TaskType = Literal["regression", "classification"]
OutcomeFraming = Literal["3class", "2class"]

ESTIMATOR_ALLOWLIST: dict[str, type] = {
    "Ridge": Ridge,
    "LogisticRegression": LogisticRegression,
    "RandomForestRegressor": RandomForestRegressor,
    "RandomForestClassifier": RandomForestClassifier,
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
    "DummyRegressor": DummyRegressor,
    "DummyClassifier": DummyClassifier,
}

_REGRESSION_TYPES = {
    "Ridge",
    "RandomForestRegressor",
    "HistGradientBoostingRegressor",
    "DummyRegressor",
}
_CLASSIFICATION_TYPES = {
    "LogisticRegression",
    "RandomForestClassifier",
    "HistGradientBoostingClassifier",
    "DummyClassifier",
}

_FEATURES = {
    "BasicControlFlowFeatures",
    "OfferFeatures",
    "InteractionFeatures",
    "WaitingStateFeatures",
    "ActiveCaseCountFeature",
}


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    model_type: str
    params: dict[str, int | float | str | bool | None] = Field(
        default_factory=dict
    )
    param_grid: dict[str, list[int | float | str | None]] = Field(
        default_factory=dict
    )

    @field_validator("model_type")
    @classmethod
    def type_in_allowlist(cls, v: str) -> str:
        if v not in ESTIMATOR_ALLOWLIST:
            allowed = sorted(ESTIMATOR_ALLOWLIST)
            raise ValueError(
                f"'{v}' is not a recognized estimator. Valid choices: {allowed}"
            )
        return v

    @field_validator("param_grid")
    @classmethod
    def param_grid_lists_nonempty(cls, v: dict[str, list]) -> dict[str, list]:
        for key, lst in v.items():
            if len(lst) == 0:
                raise ValueError(
                    f"param_grid['{key}'] must contain at least one candidate value."
                )
        return v


class DataConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    valid_end_activities: list[str] = Field(default_factory=list)
    top_k_variants: int | None = None
    split_quantile: float = 0.8
    dev_mode: bool = False
    outcome_class: OutcomeFraming = "2class"

    @field_validator("top_k_variants")
    @classmethod
    def top_k_positive(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError(
                "top_k_variants must be a positive integer or null."
            )
        return v

    @field_validator("split_quantile")
    @classmethod
    def split_quantile_in_range(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError(
                "split_quantile must be strictly between 0.0 and 1.0."
            )
        return v


class PrefixConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_length: int = 1
    max_length: int | None = None

    @model_validator(mode="after")
    def min_not_exceeds_max(self) -> PrefixConfig:
        if self.max_length is not None and self.min_length > self.max_length:
            raise ValueError(
                f"prefix.min_length ({self.min_length}) must be <= "
                f"prefix.max_length ({self.max_length})."
            )
        return self


class FeaturesConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    one_hot_encode_categorical: bool = False
    enabled_features: list[str] = Field(
        default_factory=lambda: ["BasicControlFlowFeatures"]
    )
    exclude_features: list[str] = Field(
        default_factory=list,
        description=(
            "Column names to drop from the extracted feature matrices "
            "(X_train and X_test) before training.  Use this to prune "
            "features that showed low or negative permutation importance "
            "during prior evaluation runs.  Missing columns are logged "
            "as a warning and skipped silently."
        ),
    )

    @field_validator("enabled_features")
    @classmethod
    def features_in_allowlist(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError(
                "enabled_features must contain at least one feature."
            )
        for name in v:
            if name not in _FEATURES:
                allowed = sorted(_FEATURES)
                raise ValueError(
                    f"'{name}' is not a recognized feature. Valid choices: {allowed}"
                )
        return v


class SearchConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_iter: int = 10
    cv_folds: int = 3
    random_state: int = 42
    search_sample_size: int | None = None
    scoring: str | None = None

    @field_validator("n_iter", "cv_folds")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Must be a positive integer.")
        return v


class PCAConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    active: bool = False
    keep_variability: float = 0.95


class RunConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    task: TaskType
    data: DataConfig = Field(default_factory=DataConfig)
    prefix: PrefixConfig = Field(default_factory=PrefixConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    pca_config: PCAConfig = Field(default_factory=PCAConfig)
    models: dict[str, ModelConfig]

    @field_validator("models")
    @classmethod
    def models_nonempty(cls, v: dict) -> dict:
        if not v:
            raise ValueError("models must contain at least one entry.")
        return v

    @model_validator(mode="after")
    def task_model_types_align(self) -> RunConfig:
        for name, m in self.models.items():
            if (
                self.task == "regression"
                and m.model_type in _CLASSIFICATION_TYPES
            ):
                raise ValueError(
                    f"models.{name}.model_type '{m.model_type}' is a classifier "
                    f"but task is 'regression'."
                )
            if (
                self.task == "classification"
                and m.model_type in _REGRESSION_TYPES
            ):
                raise ValueError(
                    f"models.{name}.model_type '{m.model_type}' is a regressor "
                    f"but task is 'classification'."
                )
        return self

    def to_yaml(self, path: Path | None = None) -> str:
        raw = self.model_dump(mode="json")
        text = yaml.safe_dump(raw, sort_keys=True, allow_unicode=True)
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_yaml(cls, path: Path) -> RunConfig:
        raw: dict = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)

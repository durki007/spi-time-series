# SPI Time Series — Architecture

## Overview

Predictive process monitoring on the BPIC 2017 loan application log. The
pipeline supports **regression** (remaining time) and **classification**
(approved vs. not-approved outcome). A four-stage pipeline (`preprocess`
→ `extract` → `train` → `evaluate`) with joblib checkpointing skips
unchanged stages on re-runs.

---

## Directory Layout

```
spi-time-series/
├── configs/                             # YAML experiment definitions
├── src/spi_time_series/
│   ├── main.py                          # CLI entry point
│   ├── config/                          # YAML → RunConfig, estimator factory
│   │   ├── schema.py                    # Pydantic models (RunConfig, ModelConfig, …)
│   │   └── loader.py                    # load_config, build_estimator
│   ├── data/                            # BPIC 2017 dataset, datatypes
│   │   ├── dataset.py                   # Dataset (downloads & caches)
│   │   ├── schemas.py                   # RawData, TraceSample, FeatureSet, …
│   │   ├── types.py                     # Callable type aliases
│   │   └── constants.py                 # EVENT_NAMES, OUTCOME_EVENTS
│   ├── preprocessing/                   # Cleaning, splitting, windowing
│   │   ├── preprocess.py                # clean_event_log, split_data
│   │   └── window_generators.py         # sliding/outcome window factories
│   ├── features/                        # PrefixFeature implementations
│   │   ├── extraction.py                # extract_features_builder, generate_feature_matrix
│   │   ├── log_based_features.py        # BasicControlFlowFeatures, OfferFeatures, …
│   │   ├── time_series_features.py      # ActiveCaseCountFeature
│   │   └── targets.py                   # remaining_time/outcome/binary targets
│   ├── models/                          # Training & hyperparameter search
│   │   └── trainer.py                   # search_hyperparams (RandomizedSearchCV), train
│   ├── evaluation/                      # Metrics, plots, SHAP, comparison
│   │   ├── metrics.py                   # evaluate, compare_models, detect_task
│   │   ├── plots.py                     # metric_vs_prefix, error_dist, roc/pr curves
│   │   ├── feature_importance.py        # Permutation importance
│   │   ├── shap_explainability.py       # SHAP summary + waterfall plots
│   │   ├── evaluate_feature_impact.py   # Log-only vs. log+ts comparison CLI
│   │   └── prototype.py                 # Prediction demo (checkpoint → predict)
│   └── pipeline/                        # Orchestrator
│       ├── pipeline.py                  # Pipeline dataclass, fit/evaluate
│       ├── builder.py                   # PipelineBuilder (fluent API)
│       └── state.py                     # PipelineState (checkpoint schema)
├── tests/                               # Pytest (unit + integration)
├── notebooks/                           # EDA and analysis (not required for pipeline)
└── results/                             # Experiment outputs (gitignored)
```

---

## Data Flow

```
  Dataset().log
       │
       ▼
  RawData (pd.DataFrame)
       │  preprocessor = clean_event_log(…)
       ▼
  EventLog (cleaned pd.DataFrame)
       │  splitter = split_data(…)
       ▼
  PreprocessedData
     ├── train_log: list[TraceSample]
     ├── test_log:  list[TraceSample]
     ├── col_idx:   dict[column_name → int]
     └── cleaned_log: EventLog
       │  feature_extractor = extract_features_builder(…)
       ▼
  FeatureSet
     ├── X_train / X_test : DataFrames
     ├── y_train / y_test : Series
     ├── feature_names    : list[str]
     └── trace_ids_*    : Series
       │  search (optional) + train
       ▼
  ModelArtifact
     ├── models:        dict[str, SklearnPipeline]
     ├── feature_names: list[str]
     └── target_col:    str
       │  evaluators → reporters
       ▼
  EvaluationReport + disk artefacts (metrics CSV, plots, SHAP)
```

---

## Key Datatypes

| Type | Purpose | Defined in |
|------|---------|-----------|
| `RawData` | Raw event log loaded from disk | `data/schemas.py:15` |
| `TraceSample` | One case as numpy array + prefix windows | `data/schemas.py:22` |
| `PreprocessedData` | Train/test splits with column mapping | `data/schemas.py:29` |
| `FeatureSet` | Feature matrices and labels | `data/schemas.py:41` |
| `ModelArtifact` | Trained sklearn Pipelines | `data/schemas.py:54` |
| `EvaluationReport` | Per-model, per-prefix metrics | `data/schemas.py:63` |
| `PipelineState` | Serialisable checkpoint (joblib) | `pipeline/state.py:12` |
| `PrefixFeature` | Protocol: feature(prefix, col_idx) → vec | `data/schemas.py:175` |
| `TargetGenerator` | Protocol: label from trace window | `data/schemas.py:165` |

---

## CLI Usage

```bash
# Regression
python -m spi_time_series.main configs/regression.yaml --output-dir results/

# Classification
python -m spi_time_series.main configs/classification.yaml --output-dir results/

# Fast dev mode (few cases)
python -m spi_time_series.main configs/classification_dev.yaml --output-dir results/

# With overrides
python -m spi_time_series.main configs/classification.yaml \
    --override search.n_iter=5 --override data.split_quantile=0.7

# Force re-run
python -m spi_time_series.main configs/classification.yaml --force

# Prediction demo (load checkpoint → predict)
python -m spi_time_series.evaluation.prototype \
    --config configs/classification.yaml \
    --checkpoint results/classification/checkpoint.joblib \
    --prefix-csv new_prefix.csv
```

---

## Config Structure

Top-level keys in `configs/*.yaml`:

| Key | Role |
|-----|------|
| `task` | `"regression"` or `"classification"` |
| `data` | Dataset parameters: `split_quantile`, `dev_mode`, `outcome_class`, `valid_end_activities` |
| `prefix` | Window config: `min_length`, `max_length` |
| `features` | `enabled_features`, `exclude_features`, `one_hot_encode_categorical` |
| `search` | HPO settings: `n_iter`, `cv_folds`, `random_state`, `search_sample_size` |
| `pca_config` | PCA: `active`, `keep_variability` |
| `models` | Named estimators: `model_type`, `params`, `param_grid` |

---

## How to Add a Feature

1. Implement `PrefixFeature` protocol (`name()`, `fit()`, `__call__`, `feature_names`) —
   see `log_based_features.py` or `time_series_features.py` for examples
2. Add the feature name to `_FEATURES` allowlist in `config/schema.py:50-56`
3. Wire the instantiation in `_build_default_feature_extractor()` in `main.py`
4. Optionally add `feature_kwargs` support in the extract stage

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Temporal split** by case start/completion time | Prevents leakage — all prefixes of one case stay together |
| **SHAP as Reporter** (not Evaluator) | Avoids serializing large SHAP arrays into checkpoint; plots regenerated on re-eval |
| **TreeExplainer on raw estimator** | `shap.TreeExplainer` needs the unwrapped RF, not the sklearn Pipeline wrapper |
| **class_weight="balanced"** | Handles class imbalance during training without needing balanced_accuracy in HPO scoring |
| **PipelineState stores fitted features** | Enables the prototype to extract features from genuinely new prefix data |
| **Joblib checkpointing** | Cache keys hash config sections; re-running unchanged stages skips them |

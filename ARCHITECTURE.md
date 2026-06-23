# SPI Time Series — Architecture

## Overview

Predictive process monitoring on the BPI Challenge 2017 loan application log.
The pipeline supports **regression** (remaining time in hours) and
**classification** (approved vs. not-approved outcome). A four-stage pipeline
(`preprocess` → `extract` → `train` → `evaluate`) with joblib checkpointing
skips unchanged stages on re-runs.

---

## Directory Layout

```
spi-time-series/
├── experiments/                        # YAML experiment definitions
│   ├── classification/
│   │   ├── classification_compare.yaml
│   │   ├── classification_hgb_log.yaml
│   │   └── classification_hgb_time_series.yaml
│   └── regression/
│       ├── regression_compare.yaml
│       ├── regression_compare_full.yaml
│       ├── regression_rf_log.yaml
│       └── regression_rf_time_series.yaml
├── src/spi_time_series/
│   ├── main.py                         # CLI entry point
│   ├── config/                         # YAML → RunConfig, estimator factory
│   │   ├── schema.py                   # Pydantic models (RunConfig, ModelConfig, …)
│   │   └── loader.py                   # load_config, build_estimator
│   ├── data/                           # BPIC 2017 dataset, datatypes
│   │   ├── dataset.py                  # Dataset (downloads & caches)
│   │   ├── schemas.py                  # RawData, TraceSample, FeatureSet, …
│   │   ├── types.py                    # Callable type aliases
│   │   └── constants.py                # EVENT_NAMES, OUTCOME_EVENTS
│   ├── preprocessing/                  # Cleaning, splitting, windowing
│   │   ├── preprocess.py               # clean_event_log, split_data
│   │   └── window_generators.py        # sliding / outcome window factories
│   ├── features/                       # PrefixFeature implementations
│   │   ├── extraction.py               # extract_features_builder, generate_feature_matrix
│   │   ├── log_based_features.py       # ActivityCount, Temporal, WaitingState, Financial
│   │   ├── time_series_features.py     # ActiveCaseCount, FinancialVolume, DecisionRate
│   │   └── targets.py                  # remaining_time / outcome / binary targets
│   ├── models/                         # Training & hyperparameter search
│   │   └── trainer.py                  # search_hyperparams (RandomizedSearchCV), train
│   ├── evaluation/                     # Metrics, plots, SHAP, comparison
│   │   ├── metrics.py                  # evaluate, compare_models, detect_task
│   │   ├── plots.py                    # metric_vs_prefix, error_dist, roc/pr curves
│   │   ├── feature_importance.py       # Permutation importance
│   │   ├── shap_explainability.py      # SHAP summary + waterfall plots
│   │   ├── evaluate_feature_impact.py  # Log-only vs. log+ts comparison CLI
│   │   └── prototype.py                # Prediction demo (checkpoint → predict)
│   └── pipeline/                       # Orchestrator
│       ├── pipeline.py                 # Pipeline dataclass, fit/evaluate
│       ├── builder.py                  # PipelineBuilder (fluent API)
│       └── state.py                    # PipelineState (checkpoint schema)
├── tests/                              # Pytest (133 unit + integration tests)
├── doc/                                # Sprint and final reports
├── notebooks/                          # EDA and analysis (not required for pipeline)
└── results/                            # Experiment outputs (gitignored)
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
| `TargetGenerator` | Protocol: label from trace window | `data/schemas.py:165` |

---

## Feature Architecture

**34 log-based features:** ActivityCountFeatures (26), TemporalFeatures (2),
WaitingStateFeatures (4), FinancialFeatures (2).

**24 time-series features** (3 per base column: raw + `window_mean_8h` +
`trend_8h`): ActiveCaseCountFeature (1 base), FinancialVolumeFeature
(6 base: mean/total withdrawal, offer, monthly cost),
DecisionRateFeature (1 base: accept ratio).

Time-series features are precomputed once during `fit()` and aligned to each
prefix via backward-looking timestamp lookup (`merge_asof`, `direction="backward"`).

---

## CLI Usage

```bash
# Classification model comparison (4 models, log+TS, 58 features)
python -m spi_time_series.main experiments/classification/classification_compare.yaml \
    -o results/classification/classification_compare/

# Regression RF log-only (34 features)
python -m spi_time_series.main experiments/regression/regression_rf_log.yaml \
    -o results/regression/regression_rf_log/

# Force re-run (clear cache)
python -m spi_time_series.main experiments/classification/classification_compare.yaml \
    -o results/classification/classification_compare/ --force

# Prediction demo (load checkpoint → predict)
python -m spi_time_series.evaluation.prototype \
    --config experiments/classification/classification_compare.yaml \
    --checkpoint results/classification/classification_compare/checkpoint.joblib \
    --prefix-csv new_prefix.csv
```

---

## Config Structure

Top-level keys in `experiments/{task}/*.yaml`:

| Key | Role |
|-----|------|
| `task` | `"regression"` or `"classification"` |
| `data` | Dataset parameters: `split_quantile`, `dev_mode`, `outcome_class`, `valid_end_activities` |
| `prefix` | Window config: `min_length`, `max_length` |
| `features` | `enabled_features`, `exclude_features`, `one_hot_encode_categorical` |
| `search` | HPO settings: `n_iter`, `cv_folds`, `random_state`, `search_sample_size` |
| `pca_config` | PCA: `active`, `keep_variability` |
| `models` | Named estimators: `model_type`, `params`, `param_grid` |

Model types validated against an allowlist in `config/schema.py:24-31`.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Temporal split** by case start time | Prevents leakage — all prefixes of one case stay together |
| **SHAP as Reporter** (not Evaluator) | Avoids serializing large SHAP arrays into checkpoint; plots regenerated on re-eval |
| **TreeExplainer on raw estimator** | `shap.TreeExplainer` needs the unwrapped RF/HGB, not the sklearn Pipeline wrapper |
| **class_weight="balanced"** | Handles class imbalance during training |
| **Stage-level joblib checkpointing** | Cache keys hash config sections; re-running unchanged stages skips them |
| **PipelineState stores fitted features** | Enables the prototype to extract features from genuinely new prefix data |
| **Time-series alignment via merge_asof** | Leakage-safe backward lookup; precomputed once, aligned per prefix |

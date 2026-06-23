# SPI Time Series — Predictive Process Monitoring

Binary outcome prediction (approved vs. not-approved) and remaining-time
regression for the BPI Challenge 2017 loan application log, comparing
log-only and time-series-enhanced feature sets.

---

## Quick Start

```bash
# 1. Install dependencies
poetry install

# 2. Classification model comparison (4 models, log+TS)
poetry run python -m spi_time_series.main \
    experiments/classification/classification_compare.yaml \
    -o results/classification/classification_compare/

# 3. Regression RF log-only (34 features)
poetry run python -m spi_time_series.main \
    experiments/regression/regression_rf_log.yaml \
    -o results/regression/regression_rf_log/
```

The first run downloads the BPI Challenge 2017 dataset (~200 MB) automatically.

---

## Experiments

### Classification

| Config | Features | Models | Purpose |
|--------|----------|--------|---------|
| `classification_compare.yaml` | 58 (log+TS) | Dummy, LR, RF, HGB | Full comparison |
| `classification_hgb_log.yaml` | 34 (log-only) | HGB | TS impact (log-only baseline) |
| `classification_hgb_time_series.yaml` | 58 (log+TS) | HGB | TS impact (log+TS variant) |

### Regression

| Config | Features | Models | Purpose |
|--------|----------|--------|---------|
| `regression_compare.yaml` | 58 (log+TS), max_length=100 | Dummy, Ridge, RF, HGB | Full comparison |
| `regression_compare_full.yaml` | 58 (log+TS), max_length=null | Dummy, Ridge, RF, HGB | Full-length prefixes |
| `regression_rf_log.yaml` | 34 (log-only) | RF | TS impact (log-only baseline) |
| `regression_rf_time_series.yaml` | 58 (log+TS) | RF | TS impact (log+TS variant) |

**Feature groups** — log-based (34): activity counts (26), temporal (2),
financial (2), waiting states (4). Time-series (24): active-case count,
financial volume, decision rate — each with raw, 8-hour rolling mean, and
8-hour trend per base signal.

---

## Results

```
results/<experiment>/
├── figures/
│   ├── f1_weighted_vs_prefix.png       # Performance by prefix length
│   ├── rmse_vs_prefix.png              # RMSE by prefix length
│   ├── error_distribution.png          # Prediction error histogram
│   ├── predicted_vs_actual.png         # Scatter plot
│   ├── roc_pr_curves.png               # ROC and PR curves (classification)
│   ├── feature_importance.png          # Permutation importance
│   ├── shap/                           # SHAP summary bar/dot + waterfall
│   └── train_vs_test.png              # Overfit comparison
├── metrics/
│   ├── classification_report.csv       # Per-model, per-prefix metrics
│   ├── feature_drift_stats.csv         # KS-test drift between train/test
│   └── feature_importance.csv          # Permutation importance scores
├── checkpoint.joblib                   # Trained pipeline state
└── run_config.yaml                     # Resolved config
```

### Key Findings

- **HGB best for classification** (0.6601 F1-weighted, log+TS, 58 features)
- **RF best for regression** (233.14 RMSE, log-only, 34 features)
- **Time-series features add no value** for classification (+0.1%) and
  **harm regression** (+1.45 RMSE) — likely due to concept drift across the
  chronological train/test split
- **`count_W_Validate application`** dominates feature importance for both tasks
- Performance improves with longer prefixes but plateaus after ~25 events

---

## Prediction Demo

```bash
# Replay a random test-set prefix
poetry run python -m spi_time_series.evaluation.prototype \
    --config experiments/classification/classification_compare.yaml \
    --checkpoint results/classification/classification_compare/checkpoint.joblib

# Predict on a new prefix (CSV with concept:name, time:timestamp)
poetry run python -m spi_time_series.evaluation.prototype \
    --config experiments/classification/classification_compare.yaml \
    --checkpoint results/classification/classification_compare/checkpoint.joblib \
    --prefix-csv new_prefix.csv
```

Prints prediction + probabilities + top-5 SHAP features. Saves a waterfall
plot showing how each feature pushed the prediction.

---

## Architecture

Four-stage cacheable pipeline:

```
RawData → clean_event_log → PreprocessedData → extract_features
    → FeatureSet → train → ModelArtifact → evaluate → EvaluationReport
```

- **Temporal split** by case start time — no prefix from the same case
  appears in both train and test
- **joblib checkpointing** — re-running unchanged stages skips them
- **SHAP explainability** — local feature contributions per prediction
See [ARCHITECTURE.md](ARCHITECTURE.md) for the full module map and design
decisions.

---

## Development

```bash
poetry install --with dev,test
poetry run pre-commit install           # ruff (lint+fmt), mypy, cspell on commit
poetry run pre-commit run --all-files   # lint + typecheck + spell
poetry run pytest -m "not integration"  # unit-only (fast)
poetry run pytest                       # all tests (133 tests, 11 modules)
```

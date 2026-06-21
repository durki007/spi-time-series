# SPI Time Series — Predictive Process Monitoring

Binary outcome prediction and remaining-time regression for the BPIC 2017
loan application log, with optional time-series features.

## Quick Start

```bash
# 1. Install dependencies (one command, no Poetry needed)
pip install -r requirements.txt

# 2. Run classification (log-only features)
python -m spi_time_series.main configs/classification.yaml --output-dir results/classification/

# 3. Run classification with time-series features
python -m spi_time_series.main configs/classification_with_active_cases.yaml --output-dir results/classification_ts/

# 4. Compare results
ls results/classification/figures/
ls results/classification_ts/figures/
```

The first run downloads the BPIC 2017 dataset (~200 MB) automatically.

## Experiments

| Config | Task | Features | Model | Purpose |
|---|---|---|---|---|
| `classification.yaml` | Approved vs. denied | Log-based | RF (balanced) | Log-only baseline |
| `classification_with_active_cases.yaml` | Same | Log + time-series | RF (balanced) | Main approach |
| `classification_dummy.yaml` | Same | Log-based | Majority class | Baseline check |
| `regression.yaml` | Remaining time (hours) | Log-based | Ridge, RF, HGB | Log-only baseline |
| `regression_with_active_cases.yaml` | Same | Log + time-series | Ridge, RF, HGB | Main approach |

Append `_dev` to any config name (e.g. `classification_dev.yaml`) to run on a
small subset of cases — fast iteration.

**Feature groups** — log-based: control-flow, offers, interactions, waiting
states. Time-series: hourly active-case count with 1/6/12/24 h rolling windows.

## Results

```
results/<experiment>/
├── figures/
│   ├── metric_vs_prefix.png         # Performance by prefix length
│   ├── error_distribution.png       # Prediction error histogram
│   ├── predicted_vs_actual.png      # Scatter plot
│   ├── roc_pr_curves.png            # ROC and PR curves (classification)
│   ├── feature_importance.png       # Permutation importance
│   └── shap/                        # SHAP summary bar/dot + waterfall
├── reports/
│   └── evaluation_report.csv        # Metrics per prefix length
├── train.csv / test.csv             # Feature matrices + predictions
├── checkpoint.joblib                # Trained model + pipeline state
└── run_config.yaml                  # Resolved config
```

## Prediction Demo

```bash
# Replay a random test-set prefix
python -m spi_time_series.evaluation.prototype \
    --config configs/classification.yaml \
    --checkpoint results/classification/checkpoint.joblib

# Predict on a new prefix (CSV with concept:name, time:timestamp)
python -m spi_time_series.evaluation.prototype \
    --config configs/classification.yaml \
    --checkpoint results/classification/checkpoint.joblib \
    --prefix-csv new_prefix.csv
```

Prints prediction + probabilities + top-5 SHAP features. Saves a waterfall
plot showing how each feature pushed the prediction.

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

### Development (contributors)

```bash
poetry install --with dev,test
poetry run pre-commit install
poetry run pytest -m "not integration"
```

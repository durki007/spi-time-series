# spi-time-series — Agent Guide

## Setup
```bash
poetry install --with dev,test
poetry run pre-commit install          # ruff (lint+fmt), mypy, cspell on commit
```

## Run pipeline
```bash
# Regression
python -m spi_time_series.main configs/regression.yaml --output-dir results/
python -m spi_time_series.main configs/regression.yaml --override search.n_iter=5 --override data.split_quantile=0.7

# Classification (2class binary outcome: approved vs not-approved)
python -m spi_time_series.main configs/classification.yaml --output-dir results/            # RF log-only
python -m spi_time_series.main configs/classification_dummy.yaml --output-dir results/      # majority baseline
python -m spi_time_series.main configs/classification_with_active_cases.yaml --output-dir results/  # RF log+ts
python -m spi_time_series.main configs/classification_dev.yaml --output-dir results/        # dev variant (fast)
```
Four stages: `preprocess` → `extract` → `train` → `evaluate`. Stages are cached via `joblib` hash keys; `--force` flushes the cache. Model: RandomForestClassifier (chosen over LR/HGB via model comparison).

## Checks
```bash
poetry run pre-commit run --all-files   # lint + typecheck + spell (mirrors CI)
poetry run pytest                       # all tests
poetry run pytest -m "not integration"  # unit-only (fast)
```
Integration tests patch `Dataset.__init__` to avoid downloading 4TU data.

## Config
`configs/*.yaml` → Pydantic `RunConfig`. `_dev.yaml` configs enable `data.dev_mode: true` (filters to few cases). Model types validated against allowlist in `src/spi_time_series/config/schema.py:24-31`.

## Key packages
| Path | Role |
|---|---|
| `src/spi_time_series/main.py` | CLI entry point |
| `src/spi_time_series/pipeline/` | `Pipeline`, `PipelineBuilder`, `PipelineState` |
| `src/spi_time_series/config/` | YAML loading, Pydantic schema, estimator factory |
| `src/spi_time_series/features/` | `log_based_features`, `time_series_features`, `targets`, `extraction` |
| `src/spi_time_series/data/` | `Dataset` (BPIC 2017), schemas, callable type aliases |
| `src/spi_time_series/models/` | Training & hyperparameter search |
| `src/spi_time_series/evaluation/` | Metrics, feature importance |
| `src/spi_time_series/preprocessing/` | Event log cleaning, sliding window |

## Style & linting
- Ruff: `line-length = 80`, `quote-style = "double"`, `indent-style = "space"`
- Ruff lint: E, F, I, B, UP (E501 ignored)
- Mypy: `disallow_untyped_defs = false`, `ignore_missing_imports = true`
- Pre-commit hooks run ruff check --fix, ruff format, mypy, cspell

## Dataset
Default: `data/raw/` — on first use downloads BPI Challenge 2017 from 4TU (~200MB). Tests mock this. `data/raw/` is gitignored (except `.gitkeep`).

## Gotchas
- Notebooks may not auto-detect the Poetry venv; run `poetry config virtualenvs.in-project true` before install
- `checkpoint_dir` in YAML defaults to `../data/checkpoints` (relative to CWD, not the YAML)
- `prefix.max_length: null` = no upper bound (uses full trace)
- `search.search_sample_size` limits prefix count for faster HPO (used in `classification.yaml`)
- Add new features by implementing `PrefixFeature` protocol, adding name to `_FEATURES` allowlist in `schema.py`, and wiring in `main._build_default_feature_extractor`
- Results / checkpoints are gitignored; everyone runs their own
- `pyproject.toml` has `packages = [{ include = "src" }]` — `src` is the package root

## Assessment (Milestone 6)

### Core comparison
**Log-only model vs. log + aligned time-series features.** Every experiment must include both to justify the time-series addition.

### Leakage-safe split
- Split by **case start/completion time** or keep all prefixes of one case together. Never random-prefix-split.
- Train/val/test: fit → tune → once-only final eval. State exact split dates/percentages/counts.

### Metrics
- **Regression** (remaining time/duration): MAE (interpretable unit, e.g. hours), RMSE, median absolute error, R². State the target unit.
- **Classification**: Precision, Recall, F1, ROC AUC, PR AUC, accuracy. For imbalanced tasks prefer PR AUC and calibration over accuracy alone.
- Report metrics **by prefix length groups** (early: first 1–2 events, middle: ~5 or 25–50%, late: 10 or 75%+). Explain how groups are defined for this log.

### Baselines
- Always compare against a simple baseline: training-set mean/median for regression, majority class for classification, logistic regression for explainability.
- For this project: the **log-only model** is the baseline; the time-series-augmented model is the main approach.

### Time-series assessment specifics
- Document: resampling frequency, missing-value handling, lagged windows, sync with prefix timestamps.
- Analyze *when* time-series helps: high workload periods, specific activities, early vs. late prefixes, different horizons.
- Report computational overhead of time-series extraction.

### Plugs required
- Performance vs. prefix length (line plot)
- Prediction error distribution (histogram/boxplot/residuals)
- Predicted vs. actual (scatter for regression)
- Runtime vs. cases/events/prefixes/features
- ROC/PR curves (classification)
- Feature importance / SHAP summary

### Testing evidence
- Unit tests: label construction, prefix generation, split (no case overlap, correct time order), feature matrix shape/sanity, metric computations, model smoke test on toy data.
- Integration tests: pipeline end-to-end on tiny synthetic data.
- Test data must be small enough to inspect by hand.

### Prototype
- CLI script or notebook with clean demo: load trained artifact → features for new prefix → prediction → (if relevant) explanation.
- Must work on a fresh machine (no hardcoded paths, missing deps, hidden state).

### Reproducibility
- One command to run the full experiment (`python -m spi_time_series.main ...` already exists).
- Saved metrics + figures from scripts (not manual notebook runs).
- Fixed random seeds where meaningful.

### Critical evaluation — required sections
1. **Strong points** — what works well and why.
2. **Limitations** — where performance degrades (early prefixes, rare variants, etc.).
3. **Threats to validity** — data validity (representative?), label validity (business meaning?), evaluation validity (leakage?), external validity (generalize?), implementation validity (bugs?), interpretation validity (overinterpretation?).
4. **Next steps** — what would be tried next.

### Assessment document structure (recommended)
1. Project goal & prediction task
2. Data & labels (event log, prefixes, target, time-series attributes)
3. Implementation summary (pipeline, modules, config)
4. Experimental design (split, baselines, models, params, metrics)
5. Results (tables, plots, qualitative examples, runtime)
6. Testing evidence (unit/integration/manual, edge cases)
7. Critical evaluation (interpretation, limitations, threats, improvements)
8. Project retrospective (what worked, what didn't, lessons learned)

### Common mistakes to avoid
- No baseline, or only comparing complex models with each other
- Random row split that leaks prefixes of same case across train/test
- Reporting validation results as final test results
- Tables/plots without interpretation
- Unclear target definition or changing thresholds without explanation
- Evaluation that works only on one laptop
- Tests that don't check important logic
- Only reporting strong points, hiding limitations

## Implementation status (audit against assessment)

### ✅ Done / well-implemented
| Requirement | Details |
|---|---|
| **Baseline model** | `DummyRegressor(strategy='mean')` for regression, `DummyClassifier(strategy='most_frequent')` for classification; dedicated YAML configs (`*_dummy.yaml`, `*_dummy_dev.yaml`) |
| **Core comparison** | Configs for log-only (`regression.yaml`) and log+ts (`*_with_active_cases.yaml`); dedicated comparison CLI at `evaluate_feature_impact.py` |
| **Time-series feature** | `ActiveCaseCountFeature` with hourly resampling, rolling windows (1/6/12/24h), lag features, trend, time indicators |
| **Leakage-safe split** | Temporal split by case start/completion time (`split_quantile=0.8`); all prefixes of same case stay together; overlapping cases discarded |
| **Evaluation by prefix length** | `evaluate()` groups by `prefix_length`; plateau detection; per-length feature importance heatmaps/trajectories |
| **Regression metrics** | MAE, RMSE, R², median absolute error — implemented and tested; target unit = hours (`targets.py:16`) |
| **Classification metrics** | Accuracy, precision/recall (macro), F1 (macro/weighted), ROC AUC, PR AUC — implemented and tested |
| **Feature importance** | Permutation importance (overall + per prefix length); bar/heatmap/trajectory plots |
| **Evaluation plots** | `plots.py`: metric_vs_prefix (line), error distribution (histogram + boxplot), predicted-vs-actual (scatter), ROC/PR curves — all saved under `results/<exp>/figures/` |
| **Saved results** | `results/` with 6 experiment output trees, each containing `figures/`, `reports/` (CSVs), and `checkpoint/` |
| **Unit tests** | ~123 tests across 11 test files: label construction, prefix generation, metrics, model smoke tests, feature matrix shape |
| **Integration tests** | 7 tests in `test_main.py`; pipeline end-to-end on synthetic 50-case log |
| **Test data size** | All test data is tiny and hand-inspectable |
| **CLI prototype** | `python -m spi_time_series.main ...` — one command, dry-run, --override, --stages, --force all work |
| **Reproducibility** | `pyproject.toml` with deps; pipeline cached via joblib; one command to run full experiment |
| **Random seeds** | Fixed: `random.seed(42)`, `np.random.seed(42)` in `main()`, `random_state=42` in all model configs and search |
| **Model comparison** | LR vs RF vs HGB on dev data; RF won (f1_weighted=0.597 vs 0.514/0.507); confirmed on full data (0.644) |
| **Time-series assessment specifics** | Resampling: hourly. Windows: 1/6/12/24h rolling. Sync: prefix timestamp matched to nearest hourly bin. Analysis: TS helps early prefixes (len 3-10, +30% f1), fades mid-length, neutral late |

### ⚠️ Partial / needs work
| Requirement | Detail |
|---|---|
| **Prefix length groups** | Reports every length individually; no early/middle/late bucketing (assessment requires: early=1–2 events, middle=~5 or 25–50%, late=10+ or 75%+) |
| **Split test: case overlap assertion** | `split_data` exercised but no explicit assertion that `set(train.case_ids) & set(test.case_ids)` is empty |
| **Prototype: prediction demo** | No "load trained artifact → predict on new prefix" demo script/notebook; notebooks 02 (0/3 cells) and 03 (0/6 cells) unexecuted; referenced notebooks 04/05 don't exist |
| **Feature importance tests** | `feature_importance.py` has zero test coverage |
| **Runtime/overhead measurement** | Timing done (search ~80-190s, train ~60-130s, extraction ~5s) but no runtime vs. cases/events/prefixes plot |

### ❌ Missing entirely
| Requirement | Detail |
|---|---|
| **Assessment document** | No 8-section assessment report exists (required sections: project goal, data & labels, implementation summary, experimental design, results, testing evidence, critical evaluation, retrospective) |
| **Critical evaluation** | Not written — strong points, limitations, threats to validity (data/construction/evaluation/external/implementation/interpretation), next steps |
| **Project retrospective** | Not written — what worked, what didn't, lessons learned |
| **SHAP / model-specific explainability** | Only permutation importance; no SHAP or LIME |
| **Deployment sanity check** | No evidence of running on a different machine/environment; no Docker or documented env setup beyond `poetry install` |
| **Expert evaluation / debug info** | No documented inspection of: example prefixes & labels, feature vectors for small cases, predictions for easy synthetic examples, similar/different case output comparisons |
| **Runtime vs. cases/events/prefixes plot** | Only missing plot from the required set (error histogram, predicted-vs-actual, ROC/PR curves are done) |

# Supervised Process Intelligence (Lab) - SoSe2026
# MSC Project 1 (Time Series)

The goal of this assignment is to design, implement, and evaluate a time-series-enhanced
predictive process monitoring pipeline. You will work with at least one real-life (or realistically
simulated) event log and one or more external or derived time series (for example, workload traces,
resource utilization, system performance indicators, sensor measures, or demand forecasts). From a
process-centric perspective, you will first build “classical” supervised process monitoring models
using only event-log-based features (control-flow, temporal, case attributes). You will then extend
these models by integrating time-series information that is aligned with case prefixes, and
systematically investigate whether the additional data improves predictive performance and
business usefulness. The concrete prediction task must be selected and justified by your group (e.g.,
outcome prediction, remaining time prediction, risk of SLA violation, prediction of long waiting
times, prediction of resource overload). Emphasis is placed on sound preprocessing and
synchronization between event data and time series, modular and testable software design, and
rigorous empirical evaluation following best practices in supervised process mining.

## Development

### Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation)
- Node.js 22+ (required by the cspell spell-checker hook)

### Setup

**1. Install dependencies**

```bash
poetry install --with dev,test
```

This installs all runtime, development (`pre-commit`, linters), and test (`pytest`, `pytest-cov`) dependencies into a single virtual environment.

> **Note for Jupyter notebooks:** Poetry may create the virtual environment outside the project directory (e.g. in a global cache). In that case notebooks cannot auto-detect the kernel. To force Poetry to create the environment inside the project, run:
> ```bash
> poetry config virtualenvs.in-project true
> ```
> Then re-run `poetry install`. The environment will be created at `.venv/` in the project root and Jupyter/VS Code will pick it up automatically.

**2. Install the git hook**

```bash
poetry run pre-commit install
```

After this, ruff (lint + format), mypy, and cspell run automatically on every `git commit`. If any check fails the commit is blocked until the issue is fixed.

### VS Code setup

Install the following extensions:

| Extension | ID | Purpose |
|---|---|---|
| Python | `ms-python.python` | Python language support and Poetry env management |
| Ruff | `charliermarsh.ruff` | Linting and formatting (replaces flake8, isort, black) |
| Mypy Type Checker | `ms-python.mypy-type-checker` | Inline type error reporting |

Create `.vscode/settings.json` in the project root with the following content:

```json
{
    "python-envs.defaultEnvManager": "ms-python.python:poetry",
    "python-envs.defaultPackageManager": "ms-python.python:poetry",
    "python.analysis.typeCheckingMode": "off",
    "mypy-type-checker.preferDaemon": true,
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "explicit",
            "source.organizeImports.ruff": "explicit"
        }
    }
}
```

This configures:

- **Formatter** — Ruff runs on save and fixes all auto-fixable lint issues
- **Import sorting** — Ruff organises imports on save
- **Type checking** — Mypy runs in daemon mode (`dmypy`) for fast feedback; Pylance's built-in checker is disabled to avoid duplicate errors

### Running the pipeline

**Full run**

```bash
python -m spi_time_series.main configs/regression.yaml --output-dir results/
```

This runs all four stages (preprocess → extract → train → evaluate), saves stage checkpoints under `checkpoint_dir/cli/`, writes results to `results/`, and saves the resolved config to `results/run_config.yaml`.

**Dry-run** — print the resolved config and exit without touching the pipeline:

```bash
python -m spi_time_series.main configs/regression.yaml --dry-run
```

**Config overrides** — apply dot-notation key=value pairs after loading the YAML:

```bash
python -m spi_time_series.main configs/regression.yaml \
    --override search.n_iter=5 \
    --override prefix.max_length=10
```

**Staged execution** — run only selected stages; skipped stages load from their last checkpoint (raises a clear error if none exists):

```bash
# First run: preprocess and build feature matrices only
python -m spi_time_series.main configs/regression.yaml --stages preprocess,extract

# Later: train and evaluate using the cached feature matrices
python -m spi_time_series.main configs/regression.yaml --stages train,evaluate --output-dir results/
```

**Force recomputation** — ignore existing checkpoints and recompute every selected stage:

```bash
python -m spi_time_series.main configs/regression.yaml --no-cache --output-dir results/
```

All options together:

```
usage: python -m spi_time_series.main [-h] [--override KEY=VALUE]
                                       [--dry-run]
                                       [--stages STAGE[,STAGE...]]
                                       [--no-cache]
                                       [--output-dir PATH]
                                       config

positional arguments:
  config                Path to RunConfig YAML file

options:
  --override KEY=VALUE  Dot-notation config override (repeatable)
  --dry-run             Print resolved config and exit
  --stages              Comma-separated subset of: preprocess, extract, train, evaluate
  --no-cache            Skip reading existing checkpoints; always recompute
  --output-dir PATH     Directory for results and saved config (default: .)
```

### Running checks manually

Run all checks across the entire codebase (mirrors what CI does):

```bash
poetry run pre-commit run --all-files
```

Run only the test suite:

```bash
poetry run pytest
```
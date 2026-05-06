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

The project ships a [.vscode/settings.json](.vscode/settings.json) that configures everything automatically once the extensions are installed:

- **Formatter** — Ruff runs on save and fixes all auto-fixable lint issues
- **Import sorting** — Ruff organises imports on save
- **Type checking** — Mypy runs in daemon mode (`dmypy`) for fast feedback; Pylance's built-in checker is disabled to avoid duplicate errors

No manual configuration is needed beyond installing the extensions.

### Running checks manually

Run all checks across the entire codebase (mirrors what CI does):

```bash
poetry run pre-commit run --all-files
```

Run only the test suite:

```bash
poetry run pytest
```
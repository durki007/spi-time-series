"""Feature-impact comparison CLI: compare two pipeline evaluation reports side by side.

Usage::

    python -m spi_time_series.evaluation.evaluate_feature_impact \\
        --baseline results/regression_baseline/ \\
        --advanced  results/regression_advanced/

Produces ``results/feature_impact_comparison.png`` — a faceted line plot
comparing the primary performance metric of every model at each prefix length.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from spi_time_series.evaluation.metrics import (
    detect_task,
    select_primary_metric,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_model_comparison(
    df: pd.DataFrame,
    metric: str,
    baseline_label: str,
    advanced_label: str,
    output_path: Path,
    *,
    dpi: int = 300,
) -> Path:
    """Faceted line plot: one subplot per model, comparing baseline vs advanced.

    Parameters
    ----------
    df:
        Long-format dataframe with columns ``model``, ``prefix_length``,
        ``metric`` (the value), and ``source`` (``baseline`` / ``advanced``).
    metric:
        Name of the metric displayed on the y-axis.
    baseline_label:
        Legend label for the baseline run.
    advanced_label:
        Legend label for the advanced run.
    output_path:
        Where to save the PNG.
    dpi:
        Output resolution (default ``300``).

    Returns
    -------
    Path
        The path to the saved image.
    """
    models: list[str] = sorted(df["model"].unique())
    n_models: int = len(models)
    n_cols: int = min(n_models, 3)
    n_rows: int = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 5 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for idx, model_name in enumerate(models):
        row: int = idx // n_cols
        col: int = idx % n_cols
        ax = axes[row][col]

        model_df: pd.DataFrame = df[df["model"] == model_name]

        sns.lineplot(
            data=model_df,
            x="prefix_length",
            y="metric_value",
            hue="source",
            style="source",
            markers=True,
            linewidth=2.5,
            ax=ax,
        )

        ax.set_title(model_name, fontweight="bold")
        ax.set_xlabel("Prefix Length")
        ax.set_ylabel(metric.upper())
        ax.legend(title="", loc="best")

    # Hide unused subplots
    for idx in range(n_models, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"Feature-Impact Comparison — {metric.upper()} by Prefix Length",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spi_time_series.evaluation.evaluate_feature_impact",
        description="Compare two evaluation_report.csv files side by side.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory containing the baseline evaluation_report.csv.",
    )
    parser.add_argument(
        "--advanced",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory containing the advanced evaluation_report.csv.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("results/feature_impact_comparison.png"),
        metavar="PATH",
        help="Output path for the comparison plot (default: results/feature_impact_comparison.png).",
    )
    parser.add_argument(
        "--baseline-label",
        type=str,
        default="Baseline (Log-only)",
        help="Legend label for the baseline run (default: 'Baseline (Log-only)').",
    )
    parser.add_argument(
        "--advanced-label",
        type=str,
        default="Advanced (With Time-Series)",
        help="Legend label for the advanced run (default: 'Advanced (With Time-Series)').",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output resolution (default: 300).",
    )
    return parser.parse_args(argv)


def _load_report(dir_path: Path, source_label: str) -> pd.DataFrame:
    """Load ``evaluation_report.csv`` from *dir_path* and tag with *source_label*.

    Parameters
    ----------
    dir_path:
        Directory expected to contain ``reports/evaluation_report.csv``
        (the standard pipeline output layout) **or** the CSV file directly.
    source_label:
        Tag value written into a new ``source`` column (e.g. ``"baseline"``).

    Returns
    -------
    pd.DataFrame
        Loaded dataframe with an additional ``source`` column.

    Raises
    ------
    FileNotFoundError
        When neither ``reports/evaluation_report.csv`` nor ``evaluation_report.csv``
        can be found under *dir_path*.
    """
    # Try the standard pipeline output layout first, then the flat layout.
    candidates: list[Path] = [
        dir_path / "reports" / "evaluation_report.csv",
        dir_path / "evaluation_report.csv",
    ]
    for path in candidates:
        if path.is_file():
            logger.info("Loading %s from %s", source_label, path)
            df: pd.DataFrame = pd.read_csv(path)
            df["source"] = source_label
            return df

    searched: str = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"No evaluation_report.csv found for '{source_label}'. Searched:\n  {searched}"
    )


def main(argv: list[str] | None = None) -> None:
    """Run the feature-impact comparison from the command line."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    args: argparse.Namespace = _parse_args(argv)

    # --- load both reports ------------------------------------------------
    baseline_path: Path = args.baseline
    advanced_path: Path = args.advanced

    if not baseline_path.exists():
        logger.error("Baseline directory not found: %s", baseline_path)
        sys.exit(1)
    if not advanced_path.exists():
        logger.error("Advanced directory not found: %s", advanced_path)
        sys.exit(1)

    df_baseline: pd.DataFrame = _load_report(baseline_path, args.baseline_label)
    df_advanced: pd.DataFrame = _load_report(advanced_path, args.advanced_label)

    # --- detect task & metric ---------------------------------------------
    task = detect_task(set(df_baseline.columns))
    metric = select_primary_metric(task, set(df_baseline.columns))
    logger.info("Detected task='%s', primary metric='%s'", task, metric)

    # Verify the advanced report has the same metric
    if metric not in df_advanced.columns:
        logger.error(
            "Metric '%s' not found in advanced report. Columns: %s",
            metric,
            sorted(df_advanced.columns),
        )
        sys.exit(1)

    # --- merge into long format -------------------------------------------
    id_cols: list[str] = ["model", "prefix_length", "source"]
    df_combined: pd.DataFrame = pd.concat(
        [
            df_baseline[id_cols + [metric]].rename(
                columns={metric: "metric_value"}
            ),
            df_advanced[id_cols + [metric]].rename(
                columns={metric: "metric_value"}
            ),
        ],
        ignore_index=True,
    )

    # --- plot & save ------------------------------------------------------
    output_path: Path = _plot_model_comparison(
        df=df_combined,
        metric=metric,
        baseline_label=args.baseline_label,
        advanced_label=args.advanced_label,
        output_path=args.output,
        dpi=args.dpi,
    )
    logger.info("Feature-impact comparison saved to %s", output_path)


if __name__ == "__main__":
    main()

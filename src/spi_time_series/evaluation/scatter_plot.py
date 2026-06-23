"""Scatter-plot reporter: predicted vs actual remaining time (regression)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from spi_time_series.data.schemas import (
    EvaluationReport,
    FeatureSet,
    ModelArtifact,
)

logger = logging.getLogger(__name__)


def _build_scatter_df(
    feature_set: FeatureSet, y_pred: pd.Series
) -> pd.DataFrame:
    y_true = feature_set.y_test
    prefix_lengths = feature_set.prefix_lengths_test

    data: dict[str, object] = {
        "actual": y_true.values,
        "predicted": y_pred.loc[y_true.index].values,
    }
    if prefix_lengths is not None and len(prefix_lengths):
        data["prefix_length"] = prefix_lengths.loc[y_true.index].values

    return pd.DataFrame(data)


def _plot_model_scatter(
    df: pd.DataFrame, model_name: str, ax: plt.Axes
) -> None:
    hue_col = "prefix_length" if "prefix_length" in df.columns else None
    sns.scatterplot(
        data=df,
        x="actual",
        y="predicted",
        hue=hue_col,
        palette="viridis" if hue_col else None,
        alpha=0.5,
        s=20,
        ax=ax,
        legend="auto" if hue_col else False,
    )

    lims = [
        min(df["actual"].min(), df["predicted"].min()),
        max(df["actual"].max(), df["predicted"].max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")

    ax.set_xlabel("Actual remaining time")
    ax.set_ylabel("Predicted remaining time")
    ax.set_title(model_name)

    if hue_col:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles,
            labels=labels,
            title="Prefix length",
            fontsize=7,
            title_fontsize=8,
        )


def report_predicted_vs_actual(
    artifact: ModelArtifact,
    report: EvaluationReport,
    output_dir: Path | None,
    *,
    dpi: int = 150,
) -> None:
    """Reporter: scatter plot of predicted vs actual remaining time.

    Outputs ``output_dir/reports/predicted_vs_actual.png``.
    Skipped for non-regression tasks (non-float target) or when
    ``feature_set`` / ``model_predictions`` are absent.
    """
    if output_dir is None:
        logger.warning(
            "No output directory; skipping predicted vs actual plot."
        )
        return

    if report.feature_set is None or not report.model_predictions:
        logger.info(
            "No feature_set or model_predictions in report; "
            "skipping predicted vs actual plot."
        )
        return

    if not pd.api.types.is_float_dtype(report.feature_set.y_test):
        logger.info(
            "Target dtype is %s (not float); skipping predicted vs actual plot.",
            report.feature_set.y_test.dtype,
        )
        return

    model_names = report.model_names or list(report.model_predictions)
    n_models = len(model_names)
    n_cols = min(n_models, 3)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), squeeze=False
    )

    for idx, model_name in enumerate(model_names):
        if model_name not in report.model_predictions:
            continue
        row, col = divmod(idx, n_cols)
        df = _build_scatter_df(
            report.feature_set, report.model_predictions[model_name]
        )
        _plot_model_scatter(df, model_name, axes[row][col])

    for idx in range(n_models, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "Predicted vs Actual Remaining Time", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()

    out_dir = output_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predicted_vs_actual.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved predicted vs actual scatter plot: %s", out_path)

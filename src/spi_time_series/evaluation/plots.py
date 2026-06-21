"""Generate required assessment plots from a pipeline run.

Usage::

    python -m spi_time_series.evaluation.plots \\
        --checkpoint results/regression_dummy/checkpoint.joblib \\
        --output results/regression_dummy/figures/

Produces four figure types (task-dependent):
    - metric_vs_prefix.png       — line plot of primary metric vs prefix length
    - error_distribution.png     — histogram + boxplot of residuals (regression)
    - predicted_vs_actual.png    — scatter of y_pred vs y_true (regression)
    - roc_pr_curves.png          — ROC + PR curves (classification)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from spi_time_series.data.schemas import EvaluationReport, ModelArtifact
from spi_time_series.evaluation.metrics import (
    _PREFIX_LENGTH_COL,
    detect_task,
    select_primary_metric,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def _plot_metric_vs_prefix(
    csv_path: Path,
    metric: str,
    output_path: Path,
    *,
    dpi: int = 300,
) -> None:
    """Line plot: primary metric vs prefix length, one line per model."""
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="prefix_length",
        y=metric,
        hue="model",
        style="model",
        markers=True,
        linewidth=2,
        ax=ax,
    )
    ax.set_title(f"{metric.upper()} vs Prefix Length")
    ax.set_xlabel("Prefix Length")
    ax.set_ylabel(metric.upper())
    ax.legend(title="Model", loc="best")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved metric vs prefix plot: %s", output_path)


def _plot_error_distribution(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    prefix_length_col: str,
    output_path: Path,
    *,
    dpi: int = 300,
) -> None:
    """Histogram + boxplot of residuals (y_pred - y_true) per model."""
    rows: list[dict] = []
    for model_name, pipeline in models.items():
        y_pred = pipeline.predict(X_test)
        residuals = y_pred - y_test
        for i, res in enumerate(residuals):
            rows.append(
                {
                    "model": model_name,
                    "residual": res,
                    "prefix_length": X_test.iloc[i][prefix_length_col],
                }
            )

    residuals_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    sns.histplot(
        data=residuals_df,
        x="residual",
        hue="model",
        kde=True,
        alpha=0.4,
        ax=axes[0],
    )
    axes[0].set_title("Prediction Error Distribution (Residuals)")
    axes[0].set_xlabel("Residual (y_pred - y_true) [hours]")
    axes[0].axvline(0, color="red", linestyle="--", alpha=0.5)

    max_pl = int(residuals_df["prefix_length"].max())
    bin_size = max(1, max_pl // 10)
    residuals_df["prefix_group"] = (
        residuals_df["prefix_length"] // bin_size * bin_size
    ).astype(int)

    sns.boxplot(
        data=residuals_df,
        x="prefix_group",
        y="residual",
        ax=axes[1],
    )
    axes[1].set_title("Residual Distribution by Prefix Length Group")
    axes[1].set_xlabel("Prefix Length Group")
    axes[1].set_ylabel("Residual [hours]")
    axes[1].axhline(0, color="red", linestyle="--", alpha=0.5)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved error distribution plot: %s", output_path)


def _plot_predicted_vs_actual(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    prefix_length_col: str,
    output_path: Path,
    *,
    dpi: int = 300,
) -> None:
    """Scatter plot: y_pred vs y_true, one subplot per model."""
    n_models = len(models)
    n_cols = min(n_models, 3)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 6 * n_rows),
        squeeze=False,
    )

    for idx, (model_name, pipeline) in enumerate(models.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        y_pred = pipeline.predict(X_test)
        prefix_vals = X_test[prefix_length_col].values

        scatter = ax.scatter(
            y_test,
            y_pred,
            alpha=0.4,
            s=8,
            c=prefix_vals,
            cmap="viridis",
        )

        all_vals = np.concatenate([y_test.values, y_pred])
        lims = [all_vals.min(), all_vals.max()]
        ax.plot(lims, lims, "r--", alpha=0.7, label="Perfect prediction")

        ax.set_xlabel("Actual (hours)")
        ax.set_ylabel("Predicted (hours)")
        ax.set_title(model_name)
        ax.legend(loc="upper left")
        plt.colorbar(scatter, ax=ax, label="Prefix length")

    for idx in range(n_models, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].set_visible(False)

    fig.suptitle("Predicted vs Actual", fontsize=14, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved predicted vs actual plot: %s", output_path)


def _plot_roc_pr_curves(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    prefix_length_col: str,
    output_path: Path,
    *,
    dpi: int = 300,
) -> None:
    """ROC and PR curves per model, showing early/mid/late prefix groups."""
    n_models = len(models)
    n_cols = 2
    n_rows = n_models

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(14, 5 * n_rows),
        squeeze=False,
    )

    prefix_lengths = sorted(X_test[prefix_length_col].unique())
    n_pl = len(prefix_lengths)
    show_prefixes = [prefix_lengths[0]]
    if n_pl > 2:
        show_prefixes.append(prefix_lengths[n_pl // 2])
    if n_pl > 1:
        show_prefixes.append(prefix_lengths[-1])

    for idx, (model_name, pipeline) in enumerate(models.items()):
        if not hasattr(pipeline, "predict_proba"):
            logger.warning(
                "Model '%s' lacks predict_proba; skipping ROC/PR.", model_name
            )
            continue

        y_score = pipeline.predict_proba(X_test)
        n_classes = y_score.shape[1]
        classes = sorted(set(y_test))

        ax_roc = axes[idx][0]
        ax_pr = axes[idx][1]

        for pl in show_prefixes:
            mask = X_test[prefix_length_col] == pl
            if mask.sum() == 0:
                continue
            y_true_g = y_test[mask]
            y_score_g = y_score[mask]

            for c in range(n_classes):
                if c not in classes:
                    continue
                y_bin = (y_true_g == c).astype(int)

                try:
                    if y_bin.nunique() > 1:
                        fpr, tpr, _ = roc_curve(y_bin, y_score_g[:, c])
                        auc_val = roc_auc_score(y_bin, y_score_g[:, c])
                        ax_roc.plot(
                            fpr,
                            tpr,
                            alpha=0.7,
                            label=f"Cls {c}, PL={pl} (AUC={auc_val:.2f})",
                        )
                except (ValueError, IndexError):
                    pass

                try:
                    if y_bin.nunique() > 1:
                        prec, rec, _ = precision_recall_curve(
                            y_bin, y_score_g[:, c]
                        )
                        ap_val = average_precision_score(y_bin, y_score_g[:, c])
                        ax_pr.plot(
                            rec,
                            prec,
                            alpha=0.7,
                            label=f"Cls {c}, PL={pl} (AP={ap_val:.2f})",
                        )
                except (ValueError, IndexError):
                    pass

        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax_roc.set_title(f"{model_name} — ROC Curves")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(fontsize=7, loc="lower right")

        ax_pr.set_title(f"{model_name} — PR Curves")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.legend(fontsize=7, loc="lower left")

    fig.suptitle("ROC and PR Curves", fontsize=16, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved ROC/PR curves plot: %s", output_path)


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


def report_metric_plots(
    artifact: ModelArtifact,
    report: EvaluationReport,
    output_dir: Path | None,
    *,
    dpi: int = 150,
) -> None:
    """Reporter: save one line-plot PNG per metric vs prefix length."""
    if output_dir is None:
        logger.warning("No output directory; skipping metric line plots.")
        return
    if not report.prefix_metrics:
        logger.warning(
            "No prefix metrics in report; skipping metric line plots."
        )
        return

    rows = [
        {"model": model, "prefix_length": pl, **metrics}
        for model, by_prefix in report.prefix_metrics.items()
        for pl, metrics in by_prefix.items()
    ]
    df = pd.DataFrame(rows).sort_values(["model", "prefix_length"])
    metric_cols = [c for c in df.columns if c not in ("model", "prefix_length")]

    plots_dir = output_dir / "reports"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for metric in metric_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=df,
            x="prefix_length",
            y=metric,
            hue="model",
            style="model",
            markers=True,
            linewidth=2,
            ax=ax,
        )
        ax.set_title(f"{metric.upper()} vs Prefix Length")
        ax.set_xlabel("Prefix Length")
        ax.set_ylabel(metric.upper())
        ax.legend(title="Model", loc="best")
        fig.tight_layout()
        out = plots_dir / f"{metric}_vs_prefix.png"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        logger.info("Saved %s plot: %s", metric, out)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate required assessment plots from a pipeline run."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint.joblib from a pipeline run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for figures (default: <checkpoint_dir>/figures/).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution (default: 300).",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint file not found: {args.checkpoint}")

    output_dir = args.output or args.checkpoint.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    state = joblib.load(args.checkpoint)
    if state.features is None:
        sys.exit("Pipeline state has no features — cannot generate plots.")
    X_test = state.features.X_test
    y_test = state.features.y_test

    csv_path = args.checkpoint.parent / "reports" / "evaluation_report.csv"
    if not csv_path.exists():
        sys.exit(f"Evaluation report not found: {csv_path}")
    df = pd.read_csv(csv_path)

    task = detect_task(set(df.columns))
    metric = select_primary_metric(task, set(df.columns))
    logger.info("Detected task: %s, primary metric: %s", task, metric)

    _plot_metric_vs_prefix(
        csv_path, metric, output_dir / "metric_vs_prefix.png", dpi=args.dpi
    )

    if task == "regression":
        _plot_error_distribution(
            state.trained_models,
            X_test,
            y_test,
            _PREFIX_LENGTH_COL,
            output_dir / "error_distribution.png",
            dpi=args.dpi,
        )
        _plot_predicted_vs_actual(
            state.trained_models,
            X_test,
            y_test,
            _PREFIX_LENGTH_COL,
            output_dir / "predicted_vs_actual.png",
            dpi=args.dpi,
        )
    elif task == "classification":
        has_proba = any(
            hasattr(m, "predict_proba") for m in state.trained_models.values()
        )
        if has_proba:
            _plot_roc_pr_curves(
                state.trained_models,
                X_test,
                y_test,
                _PREFIX_LENGTH_COL,
                output_dir / "roc_pr_curves.png",
                dpi=args.dpi,
            )
        else:
            logger.warning(
                "No model supports predict_proba — skipping ROC/PR curves"
            )

    logger.info("All plots saved to %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

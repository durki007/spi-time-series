import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)

from spi_time_series.config import TaskType
from spi_time_series.data.schemas import (
    EvaluationReport,
    FeatureSet,
    ModelArtifact,
)

logger = logging.getLogger(__name__)


def _compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    task: TaskType,
) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if task == "regression":
            return {
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(root_mean_squared_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred)),
                "median_ae": float(median_absolute_error(y_true, y_pred)),
            }
        if task == "classification":
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_true, y_pred)
                ),
                "f1_macro": float(
                    f1_score(y_true, y_pred, average="macro", zero_division=0)
                ),
                "f1_weighted": float(
                    f1_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    )
                ),
                "precision_macro": float(
                    precision_score(
                        y_true, y_pred, average="macro", zero_division=0
                    )
                ),
                "recall_macro": float(
                    recall_score(
                        y_true, y_pred, average="macro", zero_division=0
                    )
                ),
            }
        raise ValueError(f"Unknown task: {task}")


def evaluate_overfitting(
    artifact: ModelArtifact, features: FeatureSet, task: TaskType
) -> EvaluationReport:
    """Compute overall train and test metrics per model for overfitting detection."""
    comparison: dict[str, dict[str, dict[str, float]]] = {}

    for model_name, pipeline in artifact.models.items():
        y_pred_train = pd.Series(
            pipeline.predict(features.X_train), index=features.X_train.index
        )
        y_pred_test = pd.Series(
            pipeline.predict(features.X_test), index=features.X_test.index
        )

        train_m = _compute_metrics(features.y_train, y_pred_train, task)
        test_m = _compute_metrics(features.y_test, y_pred_test, task)
        comparison[model_name] = {"train": train_m, "test": test_m}

        for metric in train_m:
            logger.info(
                "  %s  %s  train=%.4f  test=%.4f  gap=%.4f",
                model_name,
                metric,
                train_m[metric],
                test_m[metric],
                test_m[metric] - train_m[metric],
            )

    return EvaluationReport(
        train_test_comparison=comparison,
        model_names=list(artifact.models),
    )


def report_overfitting(
    artifact: ModelArtifact,
    report: EvaluationReport,
    output_dir: Path | None,
) -> None:
    if output_dir is None or not report.train_test_comparison:
        return

    out_dir = output_dir / "overfitting"
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison = report.train_test_comparison
    models = list(comparison)

    rows = []
    for model, splits in comparison.items():
        for metric, tv in splits["train"].items():
            sv = splits["test"].get(metric, float("nan"))
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "train": round(tv, 4),
                    "test": round(sv, 4),
                    "gap (test-train)": round(sv - tv, 4),
                }
            )
    df = pd.DataFrame(rows)
    csv_path = out_dir / "train_vs_test.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Train vs test metrics saved to %s", csv_path)

    metrics = df["metric"].unique().tolist()
    ncols = min(len(metrics), 3)
    nrows = (len(metrics) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False
    )

    x = np.arange(len(models))
    width = 0.35

    for i, metric in enumerate(metrics):
        ax = axes[i // ncols][i % ncols]
        mdf = df[df["metric"] == metric].set_index("model")

        train_vals = [mdf.loc[m, "train"] for m in models]
        test_vals = [mdf.loc[m, "test"] for m in models]

        ax.bar(x - width / 2, train_vals, width, label="train", color="#4C72B0")
        ax.bar(x + width / 2, test_vals, width, label="test", color="#DD8452")

        ax.set_title(metric.upper(), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylabel(metric, fontsize=8)

    for j in range(len(metrics), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(
        "Train vs Test Metrics — Overfitting Check",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    plot_path = out_dir / "train_vs_test.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Train vs test plot saved to %s", plot_path)

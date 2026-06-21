"""Confusion matrix evaluator and reporter for classification tasks.

Metric choice: Matthews Correlation Coefficient (MCC).

    MCC = (TP·TN - FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

MCC is preferred over accuracy or F1 because it incorporates all four
confusion-matrix quadrants in a single balanced scalar (-1 = perfectly
anti-correlated, 0 = random, +1 = perfect), and remains meaningful under
class imbalance — a common characteristic of loan-outcome datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, matthews_corrcoef

from spi_time_series.config import TaskType
from spi_time_series.data.schemas import (
    EvaluationReport,
    FeatureSet,
    ModelArtifact,
)

logger = logging.getLogger(__name__)

# Consistent colours for the four quadrants across all plots.
_STAT_COLORS = {
    "tp": "#2ecc71",  # green  — correct positive
    "tn": "#3498db",  # blue   — correct negative
    "fp": "#e74c3c",  # red    — false alarm
    "fn": "#e67e22",  # orange — missed positive
}
_STAT_LABELS = {"tp": "TP", "tn": "TN", "fp": "FP", "fn": "FN"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _per_class_stats(
    cm: np.ndarray, classes: list
) -> dict[Any, dict[str, int]]:
    """Derive per-class TP/FP/FN/TN from a (K×K) confusion matrix."""
    total = int(cm.sum())
    stats: dict[Any, dict[str, int]] = {}
    for i, cls in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        tn = total - tp - fp - fn
        stats[cls] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return stats


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def evaluate_confusion_matrix(
    artifact: ModelArtifact, features: FeatureSet, task: TaskType
) -> EvaluationReport:
    """Compute confusion matrices, per-class TP/TN/FP/FN, and MCC.

    Runs on both the training split and the test split so overfitting is
    visible at a glance in the report.  Returns an empty report for
    regression tasks.
    """
    if task != "classification":
        return EvaluationReport()

    all_labels: list = sorted(
        set(features.y_train.tolist()) | set(features.y_test.tolist())
    )
    model_metrics: dict[str, dict[str, Any]] = {}

    for model_name, pipeline in artifact.models.items():
        y_pred_train = pd.Series(
            pipeline.predict(features.X_train), index=features.X_train.index
        )
        y_pred_test = pd.Series(
            pipeline.predict(features.X_test), index=features.X_test.index
        )

        cm_train = confusion_matrix(
            features.y_train, y_pred_train, labels=all_labels
        )
        cm_test = confusion_matrix(
            features.y_test, y_pred_test, labels=all_labels
        )
        mcc_train = float(matthews_corrcoef(features.y_train, y_pred_train))
        mcc_test = float(matthews_corrcoef(features.y_test, y_pred_test))

        stats_train = _per_class_stats(cm_train, all_labels)
        stats_test = _per_class_stats(cm_test, all_labels)

        entry: dict[str, Any] = {
            "train_confusion_matrix": cm_train,
            "test_confusion_matrix": cm_test,
            "train_mcc": mcc_train,
            "test_mcc": mcc_test,
            "classes": all_labels,
        }
        for cls, s in stats_train.items():
            for stat, val in s.items():
                entry[f"train_{stat}_c{cls}"] = val
        for cls, s in stats_test.items():
            for stat, val in s.items():
                entry[f"test_{stat}_c{cls}"] = val

        model_metrics[model_name] = entry
        logger.info(
            "Model '%s': train MCC=%.4f  test MCC=%.4f",
            model_name,
            mcc_train,
            mcc_test,
        )

    return EvaluationReport(
        model_metrics=model_metrics,
        model_names=list(artifact.models),
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_cm_heatmaps(
    cm_train: np.ndarray,
    cm_test: np.ndarray,
    classes: list,
    model_name: str,
    output_path: Path,
    *,
    dpi: int = 150,
) -> None:
    """Side-by-side annotated heatmaps: train (left) and test (right)."""
    labels = [str(c) for c in classes]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, cm, split in zip(
        axes, [cm_train, cm_test], ["Train", "Test"], strict=True
    ):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            linewidths=0.5,
            cbar=True,
            ax=ax,
        )
        ax.set_title(f"{model_name} — {split}", fontsize=11)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")

    fig.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix heatmaps: %s", output_path)


def _plot_counts_bar(
    stats_train: dict[Any, dict[str, int]],
    stats_test: dict[Any, dict[str, int]],
    model_name: str,
    output_path: Path,
    *,
    dpi: int = 150,
) -> None:
    """Grouped bar chart of TP/TN/FP/FN per class, train vs test."""
    stat_keys = ["tp", "tn", "fp", "fn"]
    classes = list(stats_train.keys())
    n_classes = len(classes)

    fig, axes = plt.subplots(
        1, n_classes, figsize=(6 * n_classes, 5), squeeze=False
    )

    for col, cls in enumerate(classes):
        ax = axes[0][col]
        x = np.arange(len(stat_keys))
        width = 0.35

        train_vals = [stats_train[cls][k] for k in stat_keys]
        test_vals = [stats_test[cls][k] for k in stat_keys]
        bar_colors = [_STAT_COLORS[k] for k in stat_keys]
        y_max = max(train_vals + test_vals, default=1)
        offset = y_max * 0.02 + 1

        bars_tr = ax.bar(
            x - width / 2,
            train_vals,
            width,
            color=bar_colors,
            alpha=0.90,
            label="Train",
        )
        bars_te = ax.bar(
            x + width / 2,
            test_vals,
            width,
            color=bar_colors,
            alpha=0.45,
            label="Test",
            hatch="//",
        )

        for bar, val in zip(bars_tr, train_vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + offset,
                str(val),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        for bar, val in zip(bars_te, test_vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + offset,
                str(val),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([_STAT_LABELS[k] for k in stat_keys], fontsize=11)
        ax.set_title(f"Class {cls}", fontsize=11)
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.set_ylim(0, y_max * 1.18)

    fig.suptitle(
        f"{model_name} — TP / TN / FP / FN  (Train vs Test)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved TP/TN/FP/FN bar chart: %s", output_path)


def _plot_mcc_comparison(
    cm_metrics: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    dpi: int = 150,
) -> None:
    """Bar chart comparing MCC across all models for train and test splits."""
    models = list(cm_metrics.keys())
    train_mccs = [cm_metrics[m]["train_mcc"] for m in models]
    test_mccs = [cm_metrics[m]["test_mcc"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(models)), 5))
    ax.bar(
        x - width / 2,
        train_mccs,
        width,
        label="Train",
        color="#4C72B0",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        test_mccs,
        width,
        label="Test",
        color="#DD8452",
        alpha=0.85,
    )

    all_vals = train_mccs + test_mccs
    offset = max((abs(v) for v in all_vals), default=0.1) * 0.03 + 0.01
    for i, (tv, sv) in enumerate(zip(train_mccs, test_mccs, strict=True)):
        ax.text(
            i - width / 2,
            tv + offset,
            f"{tv:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
        ax.text(
            i + width / 2,
            sv + offset,
            f"{sv:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(-1.1, 1.25)
    ax.set_ylabel("MCC")
    ax.set_title(
        "Matthews Correlation Coefficient — Train vs Test", fontsize=12
    )
    ax.legend(fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved MCC comparison: %s", output_path)


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


def report_confusion_matrix(
    artifact: ModelArtifact,
    report: EvaluationReport,
    output_dir: Path | None,
) -> None:
    """Generate confusion matrix plots and a summary CSV.

    Outputs (under ``output_dir/confusion_matrix/``):
    * ``confusion_matrices_{model}.png``  — annotated heatmaps (train | test)
    * ``counts_{model}.png``              — TP/TN/FP/FN bars per class
    * ``mcc_comparison.png``              — MCC for all models (train | test)
    * ``confusion_matrix_summary.csv``    — raw counts + MCC, one row per
                                            (model, split, class)
    """
    if output_dir is None:
        logger.warning(
            "No output directory provided; skipping confusion matrix report."
        )
        return

    cm_metrics = {
        m: v
        for m, v in report.model_metrics.items()
        if "train_confusion_matrix" in v
    }
    if not cm_metrics:
        logger.info(
            "No confusion matrix data in report — "
            "add evaluate_confusion_matrix to the pipeline."
        )
        return

    out_dir = output_dir / "confusion_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name, metrics in cm_metrics.items():
        classes: list = metrics["classes"]

        stats_train = {
            cls: {
                k: metrics[f"train_{k}_c{cls}"]
                for k in ("tp", "tn", "fp", "fn")
            }
            for cls in classes
        }
        stats_test = {
            cls: {
                k: metrics[f"test_{k}_c{cls}"] for k in ("tp", "tn", "fp", "fn")
            }
            for cls in classes
        }

        _plot_cm_heatmaps(
            metrics["train_confusion_matrix"],
            metrics["test_confusion_matrix"],
            classes,
            model_name,
            out_dir / f"confusion_matrices_{model_name}.png",
        )
        _plot_counts_bar(
            stats_train,
            stats_test,
            model_name,
            out_dir / f"counts_{model_name}.png",
        )

    _plot_mcc_comparison(cm_metrics, out_dir / "mcc_comparison.png")

    rows: list[dict[str, Any]] = []
    for model_name, metrics in cm_metrics.items():
        for split in ("train", "test"):
            for cls in metrics["classes"]:
                rows.append(
                    {
                        "model": model_name,
                        "split": split,
                        "class": cls,
                        "tp": metrics[f"{split}_tp_c{cls}"],
                        "tn": metrics[f"{split}_tn_c{cls}"],
                        "fp": metrics[f"{split}_fp_c{cls}"],
                        "fn": metrics[f"{split}_fn_c{cls}"],
                        "mcc": round(metrics[f"{split}_mcc"], 4),
                    }
                )

    csv_path = out_dir / "confusion_matrix_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("Confusion matrix summary saved to %s", csv_path)

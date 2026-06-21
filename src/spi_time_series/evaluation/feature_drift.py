import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

from spi_time_series.config import TaskType
from spi_time_series.data.schemas import (
    EvaluationReport,
    FeatureSet,
    ModelArtifact,
)
from spi_time_series.evaluation.metrics import _PREFIX_LENGTH_COL

logger = logging.getLogger(__name__)

_STATS_COLUMNS = [
    "feature",
    "train_mean",
    "train_std",
    "train_median",
    "train_p25",
    "train_p75",
    "test_mean",
    "test_std",
    "test_median",
    "test_p25",
    "test_p75",
    "ks_stat",
    "ks_pvalue",
    "drift_detected",
]


def evaluate_feature_drift(
    artifact: ModelArtifact, features: FeatureSet, task: TaskType
) -> EvaluationReport:
    """Compare feature distributions between train and test splits."""
    X_train = features.X_train
    X_test = features.X_test
    feature_cols = [c for c in X_test.columns if c != _PREFIX_LENGTH_COL]

    drift_stats: dict[str, dict[str, Any]] = {}
    for col in feature_cols:
        train_vals = X_train[col].dropna().to_numpy()
        test_vals = X_test[col].dropna().to_numpy()
        ks_stat, ks_pvalue = stats.ks_2samp(train_vals, test_vals)
        drift_stats[col] = {
            "train_mean": float(train_vals.mean()),
            "train_std": float(train_vals.std()),
            "train_median": float(pd.Series(train_vals).median()),
            "train_p25": float(pd.Series(train_vals).quantile(0.25)),
            "train_p75": float(pd.Series(train_vals).quantile(0.75)),
            "test_mean": float(test_vals.mean()),
            "test_std": float(test_vals.std()),
            "test_median": float(pd.Series(test_vals).median()),
            "test_p25": float(pd.Series(test_vals).quantile(0.25)),
            "test_p75": float(pd.Series(test_vals).quantile(0.75)),
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "drift_detected": bool(ks_pvalue < 0.05),
        }

    train_prefix_means = (
        X_train.groupby(_PREFIX_LENGTH_COL)[feature_cols].mean().reset_index()
    )
    test_prefix_means = (
        X_test.groupby(_PREFIX_LENGTH_COL)[feature_cols].mean().reset_index()
    )

    prefix_means_records: list[dict[str, Any]] = []
    for df, split in [
        (train_prefix_means, "train"),
        (test_prefix_means, "test"),
    ]:
        for _, row in df.iterrows():
            pl = int(row[_PREFIX_LENGTH_COL])
            for col in feature_cols:
                prefix_means_records.append(
                    {
                        "feature": col,
                        "prefix_length": pl,
                        "split": split,
                        "mean": row[col],
                    }
                )

    prefix_lengths = sorted(
        int(pl) for pl in X_test.groupby(_PREFIX_LENGTH_COL).groups
    )

    return EvaluationReport(
        feature_drift={
            "stats": drift_stats,
            "prefix_means": prefix_means_records,
        },
        model_names=[],
        prefix_lengths=prefix_lengths,
    )


def report_feature_drift(
    artifact: ModelArtifact, report: EvaluationReport, output_dir: Path | None
) -> None:
    if output_dir is None or not report.feature_drift:
        return

    drift_dir = output_dir / "feature_drift"
    drift_dir.mkdir(parents=True, exist_ok=True)

    _save_stats_csv(report.feature_drift["stats"], drift_dir)
    _save_prefix_means_csv(report.feature_drift["prefix_means"], drift_dir)
    _save_drift_plots(report.feature_drift, drift_dir)


def _save_stats_csv(
    stats_dict: dict[str, dict[str, Any]], output_dir: Path
) -> None:
    rows = [{"feature": feat, **vals} for feat, vals in stats_dict.items()]
    df = pd.DataFrame(rows, columns=_STATS_COLUMNS).sort_values(
        "ks_stat", ascending=False
    )
    df.to_csv(output_dir / "feature_drift_stats.csv", index=False)
    logger.info(
        "Feature drift stats saved to %s",
        output_dir / "feature_drift_stats.csv",
    )


def _save_prefix_means_csv(
    records: list[dict[str, Any]], output_dir: Path
) -> None:
    df = pd.DataFrame(
        records, columns=["feature", "prefix_length", "split", "mean"]
    )
    df.to_csv(output_dir / "feature_prefix_means.csv", index=False)
    logger.info(
        "Feature prefix means saved to %s",
        output_dir / "feature_prefix_means.csv",
    )


def _save_drift_plots(
    drift_data: dict[str, Any],
    output_dir: Path,
    features_per_page: int = 6,
    ncols: int = 3,
    dpi: int = 150,
) -> None:
    means_df = pd.DataFrame(
        drift_data["prefix_means"],
        columns=["feature", "prefix_length", "split", "mean"],
    )
    features = list(drift_data["stats"].keys())

    pages = math.ceil(len(features) / features_per_page)
    for page_idx in range(pages):
        page_features = features[
            page_idx * features_per_page : (page_idx + 1) * features_per_page
        ]
        nrows = math.ceil(len(page_features) / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 5, nrows * 3.5),
            squeeze=False,
        )

        for i, feat in enumerate(page_features):
            ax = axes[i // ncols][i % ncols]
            feat_df = means_df[means_df["feature"] == feat]

            for split, linestyle, color in [
                ("train", "--", "#1f77b4"),
                ("test", "-", "#ff7f0e"),
            ]:
                split_df = feat_df[feat_df["split"] == split].sort_values(
                    "prefix_length"
                )
                ax.plot(
                    split_df["prefix_length"],
                    split_df["mean"],
                    linestyle=linestyle,
                    color=color,
                    linewidth=1.8,
                    label=split,
                )

            ks_stat = drift_data["stats"][feat]["ks_stat"]
            drift_flag = drift_data["stats"][feat]["drift_detected"]
            short_name = feat.split("__", 1)[-1] if "__" in feat else feat
            title = f"{short_name}\nKS={ks_stat:.3f}" + (
                " *" if drift_flag else ""
            )
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Prefix length", fontsize=8)
            ax.set_ylabel("Mean value", fontsize=8)
            ax.tick_params(labelsize=7)

            if i == 0:
                ax.legend(fontsize=8)

        for j in range(len(page_features), nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)

        fig.suptitle(
            f"Feature distribution: train vs test (page {page_idx + 1}/{pages})",
            fontsize=11,
        )
        fig.tight_layout()

        out_path = output_dir / f"feature_drift_plots_page{page_idx + 1}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Feature drift plot saved to %s", out_path)

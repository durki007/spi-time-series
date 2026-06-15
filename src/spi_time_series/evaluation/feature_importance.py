import logging
from pathlib import Path
from typing import Any

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from tqdm import tqdm

from spi_time_series.config import TaskType
from spi_time_series.data.schemas import (
    EvaluationReport,
    FeatureSet,
    ModelArtifact,
)

_PREFIX_LENGTH_COL = "BasicControlFlowFeatures__prefix_length"
logger = logging.getLogger(__name__)


def evaluate_feature_importance(
    artifact: ModelArtifact, features: FeatureSet, task: TaskType
) -> EvaluationReport:
    feature_importance: dict[str, dict[str, Any]] = {}
    groups: dict = features.X_test.groupby(_PREFIX_LENGTH_COL).groups
    prefix_lengths: list[int] = sorted(int(pl) for pl in groups)
    model_names: list[str] = list(artifact.models)

    for model_name, pipeline in artifact.models.items():
        logger.info("Evaluate feature importance for model %s", model_name)
        # Feature Importance per model
        importance = permutation_importance(
            pipeline,
            features.X_test,
            features.y_test,
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )

        feature_importance[model_name] = {
            "feature": list(features.X_test.columns),
            "importance_mean": list(importance.importances_mean),
            "importance_std": list(importance.importances_std),
        }

    return EvaluationReport(
        model_metrics=feature_importance,
        model_names=model_names,
        prefix_lengths=prefix_lengths,
    )


def evaluate_feature_importance_per_prefix(
    artifact: ModelArtifact, features: FeatureSet, task: TaskType
) -> EvaluationReport:
    prefix_feature_importance: dict[str, dict[int, dict[str, Any]]] = {}

    groups: dict = features.X_test.groupby(_PREFIX_LENGTH_COL).groups
    prefix_lengths: list[int] = sorted(int(pl) for pl in groups)
    model_names: list[str] = list(artifact.models)

    for model_name, pipeline in artifact.models.items():
        # feature importance per prefix
        per_model_metrics: dict[int, dict[str, Any]] = {}
        for pl_val, group_idx in tqdm(
            groups.items(),
            desc=f"Feature Importance per prefix for model: {model_name}",
        ):
            y_test_g = features.y_test.loc[group_idx]
            X_test_g = features.X_test.loc[group_idx]

            per_model_metrics[int(pl_val)] = {"n_prefixes": len(y_test_g)}

            importance_g = permutation_importance(
                pipeline,
                X_test_g,
                y_test_g,
                n_repeats=10,
                random_state=42,
                n_jobs=-1,
            )

            per_model_metrics[int(pl_val)].update(
                {
                    "feature": list(features.X_test.columns),
                    "importance_mean": list(importance_g.importances_mean),
                    "importance_std": list(importance_g.importances_std),
                }
            )

        prefix_feature_importance[model_name] = per_model_metrics

    return EvaluationReport(
        prefix_metrics=prefix_feature_importance,
        model_names=model_names,
        prefix_lengths=prefix_lengths,
    )


def report_feature_importance(
    artifact: ModelArtifact, report: EvaluationReport, output_dir: Path | None
):
    if output_dir is None:
        return
    reports_dir = output_dir / "feature_importance"
    reports_dir.mkdir(parents=True, exist_ok=True)

    columns = ["feature", "importance_mean", "importance_std"]

    # model importance
    for model, metrics in report.model_metrics.items():
        if any(col not in metrics for col in columns):
            continue

        df = pd.DataFrame({col: metrics[col] for col in columns}).sort_values(
            "importance_mean",
            ascending=False,
        )

        save_feature_importance_plot(
            df, reports_dir / f"{model}_feature_importance.png"
        )


def report_prefix_importance_visualizations(
    artifact: ModelArtifact,
    report: EvaluationReport,
    output_dir: Path | None,
) -> None:
    """Reporter: generate heatmap and trajectory plots for per-prefix feature
    importance and save them under ``output_dir / "feature_importance"``.

    This reporter expects the evaluation report to contain per-prefix feature
    importance metrics (populated by ``evaluate_feature_importance_per_prefix``).
    When ``output_dir`` is ``None`` or the report contains no ``prefix_metrics``,
    the function logs a warning and returns early.
    """
    if output_dir is None:
        logger.warning(
            "No output directory provided; skipping prefix importance "
            "visualizations."
        )
        return
    if not report.prefix_metrics:
        logger.info(
            "No per-prefix feature importance data found; "
            "skipping visualization generation."
        )
        return

    importance_df: pd.DataFrame = _prefix_importance_to_dataframe(report)
    vis_dir: Path = output_dir / "feature_importance"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for model_name in report.prefix_metrics:
        model_df: pd.DataFrame = importance_df.query(f"model == '{model_name}'")
        if model_df.empty:
            logger.warning(
                "No per-prefix importance rows for model '%s'; skipping.",
                model_name,
            )
            continue

        heatmap_path: Path = save_prefix_importance_heatmap(
            model_df,
            vis_dir / f"{model_name}_heatmap.png",
        )
        logger.info("Prefix importance heatmap saved to %s", heatmap_path)

        trajectory_path: Path = save_prefix_importance_trajectories(
            model_df,
            vis_dir / f"{model_name}_trajectories.png",
            smooth=True,
        )
        logger.info(
            "Prefix importance trajectories saved to %s", trajectory_path
        )


def _prefix_importance_to_dataframe(report: EvaluationReport):
    records = []

    for model, prefix_data in report.prefix_metrics.items():
        for prefix_length, metrics in prefix_data.items():
            for feature, mean, std in zip(  # type: ignore[call-overload]
                metrics["feature"],
                metrics["importance_mean"],
                metrics["importance_std"],
                strict=True,
            ):
                records.append(
                    {
                        "model": model,
                        "prefix_length": prefix_length,
                        "feature": feature,
                        "importance_mean": mean,
                        "importance_std": std,
                    }
                )

    return pd.DataFrame(records)


def save_feature_importance_plot(
    importance: pd.DataFrame,
    output_path: str | Path,
    *,
    top_n: int | None = None,
    figsize: tuple[int, int] = (10, 8),
    title: str = "Permutation Feature Importance",
    dpi: int = 300,
) -> Path:
    """
    Save permutation feature importance plot to disk.

    Returns
    -------
    Path
        Path to the saved image.
    """
    output_path = Path(output_path)

    df = importance.copy()

    if top_n is not None:
        df = df.head(top_n)

    # Most important feature at the top
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(
        y=df["feature"],
        width=df["importance_mean"],
        xerr=df["importance_std"],
        capsize=3,
    )

    ax.set_xlabel("Mean Importance")
    ax.set_ylabel("Feature")
    ax.set_title(title)

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


def save_prefix_importance_heatmap(
    importance_df: pd.DataFrame,
    output_path: str | Path,
    *,
    top_n: int = 15,
    figsize: tuple[int, int] = (18, 8),
    dpi: int = 300,
    title: str = "Feature Importance by Prefix Length",
):
    """
    Heatmap of feature importance across prefix lengths.

    Parameters
    ----------
    importance_df:
        Long dataframe with columns:
        feature, prefix_length, importance_mean
    """

    output_path = Path(output_path)

    # Select globally important features
    top_features = (
        importance_df.groupby("feature")["importance_mean"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    df = importance_df[importance_df["feature"].isin(top_features)]

    heatmap_df = df.pivot_table(
        index="feature",
        columns="prefix_length",
        values="importance_mean",
        aggfunc="mean",
    ).fillna(0)

    # Most important at top
    heatmap_df = heatmap_df.loc[
        heatmap_df.mean(axis=1).sort_values(ascending=False).index
    ]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        heatmap_df,
        cmap="viridis",
        ax=ax,
        cbar_kws={"label": "Permutation Importance"},
    )

    ax.set_xlabel("Prefix Length")
    ax.set_ylabel("Feature")
    ax.set_title(title)

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_prefix_importance_trajectories(
    importance_df: pd.DataFrame,
    output_path: str | Path,
    *,
    top_n: int = 10,
    smooth: bool = False,
    rolling_window: int = 5,
    figsize: tuple[int, int] = (14, 8),
    dpi: int = 300,
    title: str = "Feature Importance Evolution Across Prefix Lengths",
):
    """
    Line plot showing importance trajectories over prefix lengths.
    """

    output_path = Path(output_path)

    top_features = (
        importance_df.groupby("feature")["importance_mean"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    df = importance_df[importance_df["feature"].isin(top_features)].copy()

    if smooth:
        df = (
            df.sort_values(["feature", "prefix_length"])
            .groupby("feature", group_keys=False)
            .apply(
                lambda g: g.assign(
                    importance_mean=g["importance_mean"]
                    .rolling(
                        rolling_window,
                        center=True,
                        min_periods=1,
                    )
                    .mean()
                )
            )
        )

    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=df,
        x="prefix_length",
        y="importance_mean",
        hue="feature",
        linewidth=2,
        ax=ax,
    )

    ax.set_xlabel("Prefix Length")
    ax.set_ylabel("Permutation Importance")
    ax.set_title(title)

    ax.legend(
        title="Feature",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path

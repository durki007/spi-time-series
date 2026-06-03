from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance

from spi_time_series.config import TaskType
from spi_time_series.data.schemas import (
    EvaluationReport,
    FeatureSet,
    ModelArtifact,
)


def evaluate_feature_importance(
    artifact: ModelArtifact, features: FeatureSet, task: TaskType
) -> EvaluationReport:
    feature_importance: dict[str, dict[str, float]] = {}
    model_names: list[str] = list(artifact.models)
    prefix_lengths: list[int] = []

    for model_name, pipeline in artifact.models.items():
        # Feature Importance
        importance = permutation_importance(
            pipeline,
            features.X_test,
            features.y_test,
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )

        feature_importance[model_name] = {
            "feature": features.X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }

    return EvaluationReport(
        model_metrics=feature_importance,
        model_names=model_names,
        prefix_lengths=prefix_lengths,
    )


def report_feature_importance(
    artifact: ModelArtifact, report: EvaluationReport, output_dir: Path | None
):
    if output_dir is None:
        return

    columns = ["feature", "importance_mean", "importance_std"]
    for model, metrics in report.model_metrics.items():
        if any(col not in metrics for col in columns):
            continue

        df = pd.DataFrame({col: metrics[col] for col in columns}).sort_values(
            "importance_mean",
            ascending=False,
        )

        reports_dir = output_dir / "feature_importance"
        reports_dir.mkdir(parents=True, exist_ok=True)
        save_feature_importance_plot(
            df, reports_dir / f"{model}_feature_importance.png"
        )


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

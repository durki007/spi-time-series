from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from spi_time_series.config.schema import AblationConfig, RunConfig
from spi_time_series.data.schemas import (
    EvaluationReport,
    PrefixFeature,
)
from spi_time_series.data.types import FeatureExtractor
from spi_time_series.evaluation.metrics import evaluate
from spi_time_series.features.extraction import extract_features_builder
from spi_time_series.features.log_based_features import (
    ActivityCountFeatures,
    BasicControlFlowFeatures,
    FinancialFeatures,
    TemporalFeatures,
    WaitingStateFeatures,
)
from spi_time_series.features.targets import (
    CLASSIFICATION_TARGETS,
    remaining_time_target,
)
from spi_time_series.features.time_series_features import (
    ActiveCaseCountFeature,
    DecisionRateFeature,
    FinancialVolumeFeature,
)
from spi_time_series.pipeline import PipelineBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FEATURE_CLASSES: dict[str, type] = {
    "BasicControlFlowFeatures": BasicControlFlowFeatures,
    "WaitingStateFeatures": WaitingStateFeatures,
    "ActivityCountFeatures": ActivityCountFeatures,
    "TemporalFeatures": TemporalFeatures,
    "FinancialFeatures": FinancialFeatures,
    "ActiveCaseCountFeature": ActiveCaseCountFeature,
    "FinancialVolumeFeature": FinancialVolumeFeature,
    "DecisionRateFeature": DecisionRateFeature,
}


class _ConstantFeature:
    """PrefixFeature that returns a single constant column (dummy baseline)."""

    def __init__(self) -> None:
        self.feature_names = ["__dummy_constant__"]

    def __call__(
        self, prefix: np.ndarray, col_idx_mapping: dict[str, int]
    ) -> pd.Series:
        return pd.Series([0.0], index=self.feature_names)

    def fit(
        self,
        event_log: Iterable,
        col_idx_mapping: dict[str, int],
        **config_kwargs,
    ) -> None:
        pass

    def name(self) -> str:
        return "ConstantFeature"


def _build_extractor(
    feature_names: list[str], config: RunConfig
) -> FeatureExtractor:
    feature_list: list[PrefixFeature] = []
    for name in feature_names:
        cls = _FEATURE_CLASSES.get(name)
        if cls is None:
            logger.warning("Unknown feature '%s' — skipping", name)
            continue
        if name == "BasicControlFlowFeatures":
            feature_list.append(
                cls(
                    one_hot_encode_categorical=config.features.one_hot_encode_categorical
                )  # type: ignore[operator]
            )
        else:
            feature_list.append(cls())

    if not feature_list:
        feature_list.append(_ConstantFeature())

    if config.task == "regression":
        target = remaining_time_target
    else:
        target = CLASSIFICATION_TARGETS[config.data.outcome_class]

    return extract_features_builder(
        feature_list,
        target,
        exclude_features=config.features.exclude_features,
    )


def _aggregate_metric(
    report: EvaluationReport, metric: str, model_name: str
) -> float:
    pm = report.prefix_metrics.get(model_name, {})
    if not pm:
        return float("nan")
    total_weight: float = 0.0
    weighted_sum: float = 0.0
    for pl, m in pm.items():
        if metric in m:
            count: int = report.prefix_counts.get(pl, 1)
            weighted_sum += m[metric] * count
            total_weight += count
    if total_weight == 0:
        return float("nan")
    return weighted_sum / total_weight


def _lower_is_better(metric: str) -> bool:
    return metric in ("mae", "rmse", "median_ae")


def _combo_key(name: str, features: list[str], config: RunConfig) -> str:
    return str(
        joblib.hash(
            {
                "combo_name": name,
                "features": features,
                "models": {
                    n: m.model_dump(mode="json")
                    for n, m in config.models.items()
                },
                "search": config.search.model_dump(mode="json"),
                "data": config.data.model_dump(mode="json"),
                "prefix": config.prefix.model_dump(mode="json"),
                "task": config.task,
                "pca": config.pca_config.model_dump(mode="json"),
                "exclude_features": config.features.exclude_features,
            }
        )[:12]
    )


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------


@dataclass
class ComboResult:
    name: str
    group: str
    report: EvaluationReport


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _write_ablation_report(
    results: list[ComboResult],
    metric: str,
    output_dir: Path,
) -> Path:
    rows: list[dict] = []
    for cr in results:
        report = cr.report
        for model_name in report.model_names:
            val = _aggregate_metric(report, metric, model_name)
            rows.append(
                {
                    "combo": cr.name,
                    "group": cr.group,
                    "model": model_name,
                    "prefix_length": 0,
                    "metric": metric,
                    "metric_value": val,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No metrics found for ablation report.")
        return output_dir / "ablation_report.csv"

    lower_better = _lower_is_better(metric)
    df = df.sort_values("metric_value", ascending=lower_better).reset_index(
        drop=True
    )
    df["rank"] = range(1, len(df) + 1)

    path = output_dir / "ablation_report.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Ablation report saved to %s", path)
    return path


def _write_per_prefix_metrics(
    results: list[ComboResult],
    metric: str,
    output_dir: Path,
) -> Path:
    rows: list[dict] = []
    for cr in results:
        report = cr.report
        for model_name in report.model_names:
            pm = report.prefix_metrics.get(model_name, {})
            for pl, m in pm.items():
                if metric in m:
                    rows.append(
                        {
                            "combo": cr.name,
                            "group": cr.group,
                            "model": model_name,
                            "prefix_length": pl,
                            "metric": metric,
                            "metric_value": m[metric],
                        }
                    )

    path = output_dir / "per_prefix_metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Per-prefix metrics saved to %s", path)
    return path


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot_metric_vs_prefix(
    results: list[ComboResult],
    metric: str,
    output_dir: Path,
    *,
    dpi: int = 300,
) -> Path:
    rows: list[dict] = []
    for cr in results:
        report = cr.report
        for model_name in report.model_names:
            pm = report.prefix_metrics.get(model_name, {})
            for pl, m in pm.items():
                if metric in m:
                    rows.append(
                        {
                            "combo": cr.name,
                            "group": cr.group,
                            "prefix_length": pl,
                            "metric_value": m[metric],
                        }
                    )

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No data for metric-vs-prefix plot.")
        return output_dir / "figures" / "metric_vs_prefix.png"

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=df,
        x="prefix_length",
        y="metric_value",
        hue="combo",
        style="group",
        markers=True,
        linewidth=2,
        ax=ax,
    )
    ax.set_title(f"{metric.upper()} vs Prefix Length")
    ax.set_xlabel("Prefix Length")
    ax.set_ylabel(metric.upper())
    ax.legend(title="Feature Set", loc="best")
    fig.tight_layout()

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "metric_vs_prefix.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Metric-vs-prefix plot saved to %s", path)
    return path


# ---------------------------------------------------------------------------
# AblationRunner
# ---------------------------------------------------------------------------


class AblationRunner:
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.ablation_config: AblationConfig = config.ablation  # type: ignore[assignment]

    def run(
        self,
        output_dir: Path,
        *,
        force: bool = False,
        n_jobs: int = 1,
        search_config=None,
    ) -> Path:
        ablation_cfg = self.ablation_config

        all_combos: list[ComboResult] = []
        combo_specs: list[tuple[str, list[str], str]] = []

        for c in ablation_cfg.log.combinations:
            combo_specs.append((c.name, c.features, "log"))

        if ablation_cfg.ts is not None:
            for c in ablation_cfg.ts.combinations:
                merged = ablation_cfg.ts.base + c.features
                combo_specs.append((c.name, merged, "ts"))

        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        for name, features, group in tqdm(
            combo_specs, desc="Running ablation", unit="combo"
        ):
            key = _combo_key(name, features, self.config)
            checkpoint_file = checkpoints_dir / f"{name}.joblib"

            if not force and checkpoint_file.exists():
                try:
                    cached = joblib.load(checkpoint_file)
                    if cached.get("key") == key:
                        logger.info(
                            "Skipping '%s' (restored from checkpoint)", name
                        )
                        all_combos.append(cached["result"])
                        continue
                    else:
                        logger.info(
                            "Checkpoint for '%s' is stale — re-running", name
                        )
                except Exception as exc:
                    logger.warning(
                        "Could not load checkpoint for '%s' (%s) — re-running",
                        name,
                        exc,
                    )

            logger.info(
                "Running '%s' (%s, %d features)…", name, group, len(features)
            )

            builder = PipelineBuilder.from_config(self.config)
            extractor = _build_extractor(features, self.config)
            pipeline = (
                builder.with_feature_extractor(extractor)
                .add_evaluator(evaluate)
                .build()
            )
            pipeline.fit(
                search_config=search_config,
                n_jobs=n_jobs,
                force=force,
            )
            report = pipeline.evaluate(output_dir=None)

            result = ComboResult(name=name, group=group, report=report)
            all_combos.append(result)

            joblib.dump({"key": key, "result": result}, checkpoint_file)
            logger.info("Checkpoint saved for '%s' (%s)", name, checkpoint_file)

        metric = ablation_cfg.metric
        _write_ablation_report(all_combos, metric, output_dir)
        _write_per_prefix_metrics(all_combos, metric, output_dir)
        _plot_metric_vs_prefix(all_combos, metric, output_dir)

        logger.info(
            "Ablation complete — %d combos evaluated in %s",
            len(all_combos),
            output_dir,
        )
        return output_dir

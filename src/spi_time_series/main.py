"""CLI entry point: python -m spi_time_series.main"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from spi_time_series.config.loader import save_config
from spi_time_series.config.schema import RunConfig
from spi_time_series.data.schemas import (
    EvaluationReport,
    ModelArtifact,
    PrefixFeature,
)
from spi_time_series.data.types import FeatureExtractor
from spi_time_series.evaluation.feature_importance import (
    _prefix_importance_to_dataframe,
    evaluate_feature_importance,
    evaluate_feature_importance_per_prefix,
    report_feature_importance,
    save_prefix_importance_heatmap,
    save_prefix_importance_trajectories,
)
from spi_time_series.evaluation.metrics import (
    _make_model_comparison_reporter,
    evaluate,
)
from spi_time_series.features.extraction import extract_features_builder
from spi_time_series.features.log_based_features import (
    BasicControlFlowFeatures,
    InteractionFeatures,
    OfferFeatures,
    WaitingStateFeatures,
)
from spi_time_series.features.targets import (
    outcome_target,
    remaining_time_target,
)
from spi_time_series.features.time_series_features import ActiveCaseCountFeature
from spi_time_series.pipeline import PipelineBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spi_time_series.main",
        description="Run the SPI time-series prediction pipeline.",
    )
    parser.add_argument(
        "config", type=Path, help="Path to RunConfig YAML file."
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dot-notation config override, e.g. --override search.n_iter=5. Repeatable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved config and exit without running anything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all stages, ignoring any existing checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        metavar="PATH",
        help="Directory for evaluation artefacts and saved config. Defaults to 'output/<config_name>/'.",
    )
    _default_jobs = max(1, (os.cpu_count() or 1) - 1)
    parser.add_argument(
        "--jobs",
        type=int,
        default=_default_jobs,
        metavar="N",
        help=f"Number of parallel jobs for training. Defaults to available cores - 1 ({_default_jobs}).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Config override helpers
# ---------------------------------------------------------------------------


def _coerce(value: str) -> Any:
    if value.lower() == "null":
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _apply_overrides(raw: dict, overrides: list[str]) -> dict:
    for item in overrides:
        if "=" not in item:
            raise ValueError(
                f"Invalid override '{item}': expected KEY=VALUE format "
                f"(e.g. search.n_iter=5)."
            )
        key, _, value = item.partition("=")
        parts = key.split(".")
        d = raw
        for part in parts[:-1]:
            if part not in d or not isinstance(d[part], dict):
                d[part] = {}
            d = d[part]
        d[parts[-1]] = _coerce(value)
    return raw


# ---------------------------------------------------------------------------
# Default feature extractor
# ---------------------------------------------------------------------------


def _build_default_feature_extractor(config: RunConfig) -> FeatureExtractor:
    """Build a default feature extractor for CLI use.

    Instantiates features listed in config.features.enabled_features with a generic target:
    - regression:     remaining event count — float(len(trace) - end_idx)
    - classification: binary flag for whether more events follow the prefix
    """
    feature_list: list[PrefixFeature] = []
    for name in config.features.enabled_features:
        if name == "BasicControlFlowFeatures":
            feature_list.append(
                BasicControlFlowFeatures(
                    one_hot_encode_categorical=config.features.one_hot_encode_categorical
                )
            )
        elif name == "OfferFeatures":
            feature_list.append(OfferFeatures())
        elif name == "InteractionFeatures":
            feature_list.append(InteractionFeatures())
        elif name == "WaitingStateFeatures":
            feature_list.append(WaitingStateFeatures())
        elif name == "ActiveCaseCountFeature":
            feature_list.append(ActiveCaseCountFeature())

    if config.task == "regression":
        return extract_features_builder(
            feature_list,
            remaining_time_target,
            exclude_features=config.features.exclude_features,
        )
    else:
        return extract_features_builder(
            feature_list,
            outcome_target,
            exclude_features=config.features.exclude_features,
        )


# ---------------------------------------------------------------------------
# Report reporter
# ---------------------------------------------------------------------------


def _save_report(
    artifact: ModelArtifact,
    report: EvaluationReport,
    output_dir: Path | None,
) -> None:
    """Reporter: write EvaluationReport to a CSV in output_dir/reports/."""
    if output_dir is None:
        return

    rows = [
        {"model": model, "prefix_length": pl, **metrics}
        for model, by_prefix in report.prefix_metrics.items()
        for pl, metrics in by_prefix.items()
    ]
    df = (
        pd.DataFrame(rows)
        .sort_values(["model", "prefix_length"])
        .reset_index(drop=True)
    )
    metric_cols = [c for c in df.columns if c not in ("model", "prefix_length")]
    df[metric_cols] = df[metric_cols].round(4)

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "evaluation_report.csv"
    df.to_csv(path, index=False)
    logger.info("Evaluation report saved to %s", path)


def _save_prefix_importance_visualizations(
    artifact: ModelArtifact,
    report: EvaluationReport,
    output_dir: Path | None,
) -> None:
    """Reporter: generate heatmap and trajectory plots for per-prefix feature
    importance and save them under ``output_dir / "feature_importance"``.

    This reporter expects the evaluation report to contain per-prefix feature
    importance metrics (populated by ``evaluate_feature_importance_per_prefix``).
    When ``output_dir`` is ``None`` or the report contains no ``prefix_metrics``,
    the function returns silently.
    """
    if output_dir is None:
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


# ---------------------------------------------------------------------------
# Stage key computation
# ---------------------------------------------------------------------------


def _compute_extract_key(config: RunConfig) -> str:
    """Hash the config sections that affect the extract stage."""
    return str(
        joblib.hash(
            {
                "data": config.data.model_dump(mode="json"),
                "prefix": config.prefix.model_dump(mode="json"),
                "features": config.features.model_dump(mode="json"),
            }
        )[:8]
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    args = _parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    raw: dict = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _apply_overrides(raw, args.override)

    try:
        config = RunConfig.model_validate(raw)
    except Exception as exc:
        logger.error("Config validation failed: %s", exc)
        sys.exit(1)

    if args.dry_run:
        print(config.to_yaml())
        return None

    output_dir = args.output_dir or Path("output") / config_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(output_dir / "run.log")
    file_handler.setFormatter(
        logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    pipeline = (
        PipelineBuilder.from_config(config)
        .with_feature_extractor(_build_default_feature_extractor(config))
        .add_evaluator(evaluate)
        .add_reporter(_save_report)
        .add_evaluator(evaluate_feature_importance)
        .add_reporter(report_feature_importance)
        .add_evaluator(evaluate_feature_importance_per_prefix)
        .add_reporter(_save_prefix_importance_visualizations)
        .add_reporter(_make_model_comparison_reporter(config.task))
        .build()
    )

    checkpoint_path = output_dir / "checkpoint.joblib"

    if not args.force and checkpoint_path.exists():
        try:
            pipeline.restore_state(joblib.load(checkpoint_path))
            logger.info("Restored pipeline state from %s", checkpoint_path)
        except Exception as exc:
            logger.warning(
                "Could not load checkpoint at %s (%s) — running from scratch.",
                checkpoint_path,
                exc,
            )

    pipeline.fit(
        extract_key=_compute_extract_key(config),
        force=args.force,
        n_jobs=args.jobs,
        search_config=config.search,
    )

    joblib.dump(pipeline.extract_state(), checkpoint_path)
    logger.info("Pipeline state saved to %s", checkpoint_path)

    if pipeline.is_fitted:
        pipeline.evaluate(output_dir=output_dir)

    save_config(config, output_dir / "run_config.yaml")
    logger.info("Resolved config saved to %s", output_dir / "run_config.yaml")


if __name__ == "__main__":
    main()

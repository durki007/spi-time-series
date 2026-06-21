"""CLI entry point: python -m spi_time_series.main"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
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
from spi_time_series.evaluation.feature_drift import (
    evaluate_feature_drift,
    report_feature_drift,
)
from spi_time_series.evaluation.feature_importance import (
    evaluate_feature_importance,
    evaluate_feature_importance_per_prefix,
    report_feature_importance,
)
from spi_time_series.evaluation.feature_importance import (
    report_prefix_importance_visualizations as _save_prefix_importance_visualizations,
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
    CLASSIFICATION_TARGETS,
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
# YAML duplicate key detection
# ---------------------------------------------------------------------------


def _warn_duplicate_keys(config_path: Path) -> None:
    import yaml as _yaml

    tracker: dict[int, dict[str, int]] = {}

    class _DupTracker(_yaml.SafeLoader):
        pass

    def _construct_mapping(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            line = key_node.start_mark.line + 1
            tracker.setdefault(line, {})
            tracker[line][str(key)] = tracker[line].get(str(key), 0) + 1
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    _DupTracker.add_constructor(
        _yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        _construct_mapping,
    )
    try:
        _yaml.load(config_path.read_text(), Loader=_DupTracker)
    except Exception:
        return
    duplicates = {
        k for line in tracker.values() for k, cnt in line.items() if cnt > 1
    }
    if duplicates:
        logger.warning(
            "Duplicate YAML key(s) in %s: %s (last value wins)",
            config_path.name,
            ", ".join(sorted(duplicates)),
        )


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
    return extract_features_builder(
        feature_list,
        CLASSIFICATION_TARGETS[config.data.outcome_class],
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
        logger.warning(
            "No output directory provided; skipping evaluation report."
        )
        return

    rows = [
        {
            "model": model,
            "prefix_length": pl,
            "n_prefixes": report.prefix_counts[pl],
            **metrics,
        }
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


def _save_predictions(
    artifact: ModelArtifact,
    report: EvaluationReport,
    output_dir: Path | None,
) -> None:
    if output_dir is None:
        logger.warning(
            "No output directory provided; skipping evaluation report."
        )
        return

    if report.feature_set is None:
        logger.warning("No feature set available; skipping prediction export.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    feature_set = report.feature_set

    # -----------------
    # Train dataset
    # -----------------
    train_df = feature_set.X_train.copy()
    train_df[artifact.target_col] = feature_set.y_train
    train_df["Trace_ID"] = feature_set.trace_ids_train

    train_path = output_dir / "train.csv"
    train_df.to_csv(train_path, index=False, sep=";")

    # -----------------
    # Test dataset
    # -----------------
    test_df = feature_set.X_test.copy()
    test_df[artifact.target_col] = feature_set.y_test
    test_df["Trace_ID"] = feature_set.trace_ids_test

    # Add predictions from each model
    for model_name, y_pred in report.model_predictions.items():
        test_df[f"{model_name}_prediction"] = y_pred

    test_path = output_dir / "test.csv"
    test_df.to_csv(test_path, index=False, sep=";")

    logger.info("Saved training data to %s", train_path)
    logger.info("Saved test data and predictions to %s", test_path)


# ---------------------------------------------------------------------------
# Stage key computation
# ---------------------------------------------------------------------------


def _apply_best_params_to_config(
    config: RunConfig, best_params: dict[str, dict]
) -> RunConfig:
    """Return a new RunConfig with found hyperparameters baked into each model.

    For every model that was searched, its ``params`` field is updated with the
    best values found and its ``param_grid`` is cleared.  This makes the saved
    run_config.yaml self-contained: re-running it skips the search stage
    automatically because param_grid is empty.
    """
    import numpy as np

    def _to_scalar(v: Any) -> int | float | str | bool | None:
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        return v  # type: ignore[no-any-return]

    updated_models = {}
    for name, model_cfg in config.models.items():
        if name not in best_params:
            updated_models[name] = model_cfg
            continue
        merged = {
            **model_cfg.params,
            **{k: _to_scalar(v) for k, v in best_params[name].items()},
        }
        updated_models[name] = model_cfg.model_copy(
            update={"params": merged, "param_grid": {}}
        )
    return config.model_copy(update={"models": updated_models})


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

    random.seed(42)
    np.random.seed(42)

    args = _parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    raw: dict = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _warn_duplicate_keys(config_path)
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
        .add_evaluator(evaluate_feature_drift)
        .add_reporter(report_feature_drift)
        .add_evaluator(evaluate_feature_importance)
        .add_reporter(report_feature_importance)
        .add_evaluator(evaluate_feature_importance_per_prefix)
        .add_reporter(_save_prefix_importance_visualizations)
        .add_reporter(_make_model_comparison_reporter(config.task))
        .add_reporter(_save_predictions)
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

    resolved_config = _apply_best_params_to_config(config, pipeline.best_params)
    save_config(resolved_config, output_dir / "run_config.yaml")
    logger.info("Resolved config saved to %s", output_dir / "run_config.yaml")


if __name__ == "__main__":
    main()

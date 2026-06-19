"""Prediction demo: load a checkpoint and predict on a prefix.

Usage:
    # Test-set replay mode (picks a random held-out test prefix)
    python -m spi_time_series.evaluation.prototype \\
        --config configs/classification.yaml \\
        --checkpoint results/classification/checkpoint.joblib

    # New prefix mode (predict on raw event data)
    python -m spi_time_series.evaluation.prototype \\
        --config configs/classification.yaml \\
        --checkpoint results/classification/checkpoint.joblib \\
        --prefix-csv sample_prefix.csv \\
        --case-id CASE_123
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.pipeline import Pipeline as SklearnPipeline

from spi_time_series.config.schema import RunConfig
from spi_time_series.evaluation.shap_explainability import (
    _expected_for_class,
    _save_waterfall,
    _shap_for_class,
    _unwrap_pipeline,
)
from spi_time_series.pipeline.state import PipelineState

logger = logging.getLogger(__name__)

_SHAP_MAX_DISPLAY = 20


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m spi_time_series.evaluation.prototype",
        description="Load a checkpoint and predict on a prefix.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to RunConfig YAML file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint.joblib file.",
    )
    parser.add_argument(
        "--prefix-csv",
        type=Path,
        default=None,
        help="CSV file with raw events for a new prefix. "
        "If omitted, a random test-set prefix is used.",
    )
    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help="Case identifier for the new prefix (default: CSV filename stem).",
    )
    parser.add_argument(
        "--prefix-length",
        type=int,
        default=None,
        help="Number of events to use as prefix (default: all events in CSV).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("prototype_output"),
        help="Directory to save SHAP plots (default: prototype_output/).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (default: first model in checkpoint).",
    )
    return parser.parse_args(argv)


def _load_config(config_path):
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return RunConfig.model_validate(raw)


def _get_model_name(state, requested=None):
    if requested and requested in state.trained_models:
        return requested
    names = list(state.trained_models.keys())
    if not names:
        raise ValueError("No trained models in checkpoint.")
    return names[0]


def _build_columns_ordered(col_idx_mapping):
    return [
        c for c, _ in sorted(col_idx_mapping.items(), key=lambda item: item[1])
    ]


def _extract_features(
    prefix_array, fitted_features, col_idx_mapping, feature_names
):
    n_features = len(feature_names)
    row = np.empty(n_features, dtype=np.float32)
    offset = 0
    for feature in fitted_features:
        vec = feature(prefix_array, col_idx_mapping)
        n = len(vec)
        row[offset : offset + n] = vec
        offset += n
    return pd.DataFrame([row], columns=feature_names)


def _save_shap_waterfall(
    pipeline,
    X_single,
    feature_names,
    output_path,
    is_classification,
):
    preprocessor, estimator = _unwrap_pipeline(pipeline)
    try:
        explainer = shap.TreeExplainer(estimator)
    except Exception:
        logger.warning("TreeExplainer not available — skipping SHAP waterfall.")
        return None

    if preprocessor is not None:
        X_transformed = preprocessor.transform(X_single)
        try:
            shap_feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            shap_feature_names = [
                f"f{i}" for i in range(X_transformed.shape[1])
            ]
    else:
        X_transformed = X_single.values
        shap_feature_names = list(feature_names)

    shap_vals = explainer.shap_values(X_transformed)

    is_multiclass = isinstance(shap_vals, list) or (
        isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3
    )

    if is_multiclass:
        n_classes = (
            len(shap_vals)
            if isinstance(shap_vals, list)
            else shap_vals.shape[2]
        )
        for cls in range(n_classes):
            vals = _shap_for_class(shap_vals, cls)
            base = _expected_for_class(explainer, cls)
            stem = output_path.stem
            _save_waterfall(
                vals[0],
                base,
                X_transformed[0],
                shap_feature_names,
                output_path.with_stem(f"{stem}_cls{cls}"),
            )
    else:
        base = _expected_for_class(explainer, 0)
        _save_waterfall(
            shap_vals[0],
            base,
            X_transformed[0],
            shap_feature_names,
            output_path,
        )

    vals = _shap_for_class(shap_vals, 1 if is_multiclass else 0)
    if vals.ndim > 1:
        vals = vals[0]
    idx = np.argsort(np.abs(vals))[::-1][:_SHAP_MAX_DISPLAY]
    return [(shap_feature_names[i], float(vals[i])) for i in idx]


def _predict_row(pipeline, X_row, is_classification):
    y_pred = pipeline.predict(X_row)[0]
    result = [f"Predicted: {y_pred}"]
    if is_classification:
        proba = pipeline.predict_proba(X_row)[0]
        classes = pipeline.classes_
        for cls, p in zip(classes, proba, strict=True):
            result.append(f"  P({cls}) = {p:.4f}")
    return y_pred, result


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)

    config = _load_config(args.config)
    state: PipelineState = joblib.load(str(args.checkpoint))

    model_name = _get_model_name(state, args.model)
    pipeline = state.trained_models[model_name]

    raw_estimator = (
        pipeline.named_steps.get("model", pipeline)
        if isinstance(pipeline, SklearnPipeline)
        else pipeline
    )
    is_classification = hasattr(raw_estimator, "predict_proba")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Prediction Demo")
    print(f"{'=' * 60}")
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model:      {model_name}")
    print(f"Task:       {config.task}")
    print(f"{'=' * 60}\n")

    if args.prefix_csv:
        _demo_new_prefix(
            args, state, pipeline, model_name, is_classification, output_dir
        )
    else:
        _demo_test_replay(
            state, pipeline, model_name, is_classification, output_dir
        )


def _demo_test_replay(
    state, pipeline, model_name, is_classification, output_dir
):
    if state.features is None:
        print("ERROR: Checkpoint has no feature set. Run pipeline first.")
        sys.exit(1)

    fs = state.features
    idx = np.random.default_rng().integers(0, len(fs.X_test))
    X_row = fs.X_test.iloc[idx : idx + 1]
    y_actual = fs.y_test.iloc[idx]
    case_id = fs.trace_ids_test.iloc[idx]

    _, pred_lines = _predict_row(pipeline, X_row, is_classification)

    print("Mode:  test-set replay")
    print(f"Case:  {case_id}")
    for line in pred_lines:
        print(line)
    print(f"Actual: {y_actual}")
    print()

    waterfall_path = output_dir / f"{model_name}_{case_id}_shap_waterfall.png"
    top_features = _save_shap_waterfall(
        pipeline, X_row, fs.feature_names, waterfall_path, is_classification
    )
    if top_features:
        print("Top-5 features driving this prediction:")
        for name, val in top_features[:5]:
            direction = "+" if val >= 0 else ""
            print(f"  {name}: {direction}{val:.4f}")
        print(f"\nSHAP waterfall saved to: {waterfall_path}")
    else:
        print("(SHAP waterfall not available for this model)")


def _demo_new_prefix(
    args, state, pipeline, model_name, is_classification, output_dir
):
    if state.fitted_features is None:
        print(
            "ERROR: Checkpoint has no fitted feature objects. "
            "Re-run the pipeline with the updated code, or omit "
            "--prefix-csv to use test-set replay mode."
        )
        sys.exit(1)
    if state.fitted_col_idx_mapping is None:
        print(
            "ERROR: Checkpoint has no column index mapping. "
            "Re-run the pipeline first."
        )
        sys.exit(1)
    if state.features is None:
        print("ERROR: Checkpoint has no feature set. Run pipeline first.")
        sys.exit(1)

    if not args.prefix_csv.exists():
        print(f"ERROR: Prefix CSV not found: {args.prefix_csv}")
        sys.exit(1)

    df = pd.read_csv(args.prefix_csv)

    col_idx = state.fitted_col_idx_mapping
    expected_cols = set(col_idx.keys())
    provided_cols = set(df.columns)

    unknown = provided_cols - expected_cols
    if unknown:
        print(f"Warning: ignoring unknown columns: {sorted(unknown)}")

    missing = expected_cols - provided_cols
    if missing:
        print(
            f"Note: filling {len(missing)} missing columns with defaults: {sorted(missing)}"
        )
        for col in sorted(missing):
            df[col] = "" if col != "time:timestamp" else pd.NaT

    ts_col = "time:timestamp"

    if ts_col in df.columns and df[ts_col].notna().any():
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.sort_values(ts_col).reset_index(drop=True)

    case_id = args.case_id or args.prefix_csv.stem

    n = args.prefix_length if args.prefix_length else len(df)
    n = min(n, len(df))
    prefix_df = df.iloc[:n]

    ordered_cols = _build_columns_ordered(col_idx)
    prefix_array = prefix_df[ordered_cols].to_numpy()

    X_row = _extract_features(
        prefix_array,
        state.fitted_features,
        col_idx,
        state.features.feature_names,
    )

    _, pred_lines = _predict_row(pipeline, X_row, is_classification)

    activities = (
        prefix_df["concept:name"].tolist()
        if "concept:name" in prefix_df.columns
        else [str(i) for i in range(n)]
    )

    print("Mode:   new prefix")
    print(f"Case:   {case_id}")
    print(f"Prefix: {n} events: {', '.join(str(a) for a in activities)}")
    for line in pred_lines:
        print(line)
    print()

    waterfall_path = output_dir / f"{model_name}_{case_id}_shap_waterfall.png"
    top_features = _save_shap_waterfall(
        pipeline,
        X_row,
        state.features.feature_names,
        waterfall_path,
        is_classification,
    )
    if top_features:
        print("Top-5 features driving this prediction:")
        for name, val in top_features[:5]:
            direction = "+" if val >= 0 else ""
            print(f"  {name}: {direction}{val:.4f}")
        print(f"\nSHAP waterfall saved to: {waterfall_path}")
    else:
        print("(SHAP waterfall not available for this model)")


if __name__ == "__main__":
    main()

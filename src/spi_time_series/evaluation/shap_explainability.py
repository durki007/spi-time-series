"""SHAP explainability reporter for the pipeline.

Generates per-model plots:
    - Summary bar: global feature importance ranking
    - Summary dot: feature importance with impact direction
    - Waterfall: local explanations for selected test instances
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.pipeline import Pipeline as SklearnPipeline

from spi_time_series.data.schemas import EvaluationReport, ModelArtifact

logger = logging.getLogger(__name__)

_SHAP_SAMPLE_SIZE = 500
_SHAP_MAX_DISPLAY = 20


def _unwrap_pipeline(pipeline):
    """Split into (preprocessor, final_estimator)."""
    if not isinstance(pipeline, SklearnPipeline) or len(pipeline.steps) == 0:
        return None, pipeline
    pre_steps = pipeline.steps[:-1]
    estimator = pipeline.steps[-1][1]
    if pre_steps:
        return SklearnPipeline(pre_steps), estimator
    return None, estimator


def report_shap(
    artifact: ModelArtifact,
    report: EvaluationReport,
    output_dir: Path | None,
) -> None:
    """Reporter: compute SHAP values and save summary + waterfall plots."""
    if output_dir is None:
        return
    if report.feature_set is None:
        logger.warning("No feature set available; skipping SHAP.")
        return

    X_test = report.feature_set.X_test
    y_test = report.feature_set.y_test
    preds = report.model_predictions
    feature_names = X_test.columns.tolist()

    shap_dir = output_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    for model_name, pipeline in artifact.models.items():
        logger.info("SHAP for model: %s", model_name)

        preprocessor, estimator = _unwrap_pipeline(pipeline)
        try:
            explainer = shap.TreeExplainer(estimator)
        except Exception:
            logger.warning(
                "TreeExplainer failed for %s — skipping.", model_name
            )
            continue

        n = min(_SHAP_SAMPLE_SIZE, len(X_test))
        X_sample = X_test.iloc[:n]
        y_sample = y_test.iloc[:n]

        if preprocessor is not None:
            X_transformed = preprocessor.transform(X_sample)
        else:
            X_transformed = X_sample.values

        try:
            shap_feature_names = list(
                preprocessor.get_feature_names_out()
                if preprocessor
                else feature_names
            )
        except Exception:
            shap_feature_names = [
                f"f{i}" for i in range(X_transformed.shape[1])
            ]

        shap_vals = explainer.shap_values(X_transformed)
        is_multiclass = isinstance(shap_vals, list) or (
            isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3
        )
        n_classes = (
            len(shap_vals)
            if isinstance(shap_vals, list)
            else shap_vals.shape[2]
            if is_multiclass
            else 1
        )

        _save_summary(
            shap_vals,
            X_transformed,
            shap_feature_names,
            shap_dir / f"{model_name}_shap_summary_bar.png",
            plot_type="bar",
        )
        _save_summary(
            shap_vals,
            X_transformed,
            shap_feature_names,
            shap_dir / f"{model_name}_shap_summary_dot.png",
            plot_type="dot",
        )
        _save_waterfalls(
            explainer,
            shap_vals,
            X_transformed,
            y_sample,
            preds.get(model_name),
            shap_feature_names,
            is_multiclass,
            n_classes,
            shap_dir,
            model_name,
        )

    logger.info("SHAP plots saved to %s", shap_dir)


def _shap_for_class(shap_vals, class_idx=1):
    """Extract SHAP values for a given class index."""
    if isinstance(shap_vals, list):
        return shap_vals[class_idx]
    if shap_vals.ndim == 3:
        return shap_vals[:, :, class_idx]
    return shap_vals


def _expected_for_class(explainer, class_idx=0):
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)) and np.ndim(ev) >= 1:
        return float(ev[class_idx])
    return float(ev)


def _save_summary(
    shap_vals,
    X_values,
    feature_names,
    path,
    *,
    plot_type,
):
    vals = _shap_for_class(shap_vals, class_idx=1)

    shap.summary_plot(
        vals,
        X_values,
        feature_names=feature_names,
        plot_type=plot_type,
        max_display=_SHAP_MAX_DISPLAY,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close("all")
    logger.info("  saved: %s", path.name)


def _save_waterfalls(
    explainer,
    shap_vals,
    X_values,
    y_sample,
    y_pred,
    feature_names,
    is_multiclass,
    n_classes,
    shap_dir,
    model_name,
):
    if is_multiclass:
        for class_idx in range(n_classes):
            vals = _shap_for_class(shap_vals, class_idx)
            base = _expected_for_class(explainer, class_idx)
            idx = _find_correct(y_sample, y_pred, class_idx)
            _save_waterfall(
                vals[idx],
                base,
                X_values[idx],
                feature_names,
                shap_dir / f"{model_name}_shap_waterfall_cls{class_idx}.png",
            )
    else:
        base = _expected_for_class(explainer, 0)
        idx = 0
        _save_waterfall(
            shap_vals[idx],
            base,
            X_values[idx],
            feature_names,
            shap_dir / f"{model_name}_shap_waterfall.png",
        )


def _find_correct(y_true, y_pred, target_class):
    if y_pred is None:
        return 0
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if target_class is None:
        return 0  # regression: just use first
    for i in range(len(yt)):
        if yt[i] == target_class and yp[i] == target_class:
            return i
    return 0


def _save_waterfall(
    values,
    base_value,
    data,
    feature_names,
    path,
):
    shap.waterfall_plot(
        shap.Explanation(
            values=values,
            base_values=base_value,
            data=data,
            feature_names=feature_names,
        ),
        max_display=_SHAP_MAX_DISPLAY,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close("all")
    logger.info("  saved: %s", path.name)

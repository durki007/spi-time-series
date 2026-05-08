import logging
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.base import BaseEstimator

from spi_time_series.data import Dataset
from spi_time_series.data.schemas import (
    EvaluationReport,
    ModelArtifact,
    RawData,
)
from spi_time_series.data.types import (
    Evaluator,
    FeatureExtractor,
    Preprocessor,
    Reporter,
    Splitter,
)
from spi_time_series.models import train

logger = logging.getLogger(__name__)


def _merge_evaluations(reports: list[EvaluationReport]) -> EvaluationReport:
    """Deep-merge a list of evaluation reports into one, unioning all models, prefix lengths, and metrics."""
    if not reports:
        raise ValueError("No evaluators produced a report.")
    merged = EvaluationReport(metrics={}, model_names=[], prefix_lengths=[])
    for report in reports:
        for model, by_prefix in report.metrics.items():
            merged.metrics.setdefault(model, {})
            for prefix, metrics in by_prefix.items():
                merged.metrics[model].setdefault(prefix, {}).update(metrics)
        for name in report.model_names:
            if name not in merged.model_names:
                merged.model_names.append(name)
        for pl in report.prefix_lengths:
            if pl not in merged.prefix_lengths:
                merged.prefix_lengths.append(pl)
    return merged


@dataclass
class Pipeline:
    """Predictive process monitoring pipeline composed of swappable strategy callables.

    Attributes:
        dataset: Source of the raw event log.
        preprocessor: Cleans the raw event log and returns a flat EventLog.
        splitter: Partitions the cleaned log into train/test sets as PreprocessedData.
        feature_extractor: Builds labeled feature matrices from the split log.
        models: Named scikit-learn estimators to train and compare.
        evaluators: Functions that compute evaluation metrics; their results are merged.
        reporters: Functions that produce output artifacts (plots, CSVs, MLflow runs, etc.).
    """

    dataset: Dataset
    preprocessor: Preprocessor
    splitter: Splitter
    feature_extractor: FeatureExtractor
    models: dict[str, BaseEstimator]
    evaluators: list[Evaluator] = field(default_factory=list)
    reporters: list[Reporter] = field(default_factory=list)

    def run(self, output_dir: Path | None = None) -> EvaluationReport:
        """Execute the pipeline end-to-end and return the merged evaluation report."""
        logger.info("Pipeline started.")
        raw = RawData(event_log=self.dataset.log)

        cleaned = self.preprocessor(raw)
        logger.info("Preprocessing complete.")

        preprocessed = self.splitter(cleaned)
        logger.info("Splitting complete.")

        features = self.feature_extractor(preprocessed)
        logger.info("Feature extraction complete.")

        artifact: ModelArtifact = train(features, self.models)
        logger.info("Training complete.")

        reports = [e(artifact, features) for e in self.evaluators]
        evaluation = _merge_evaluations(reports)
        logger.info("Evaluation complete.")

        for reporter in self.reporters:
            reporter(artifact, evaluation, output_dir)
        logger.info("Pipeline finished.")

        return evaluation

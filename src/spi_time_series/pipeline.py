import logging
from pathlib import Path

from spi_time_series.data import Dataset
from spi_time_series.data.schemas import (
    EvaluationReport,
    ModelArtifact,
    RawData,
)
from spi_time_series.evaluation import evaluate
from spi_time_series.features import extract_features
from spi_time_series.models import train
from spi_time_series.preprocessing import preprocess

logger = logging.getLogger(__name__)


def ingest(data_dir: Path | None = None) -> RawData:
    """Load raw XES event log from disk, downloading if necessary."""

    logger.info("Ingesting data...")
    raw = RawData(event_log=Dataset(data_dir).log)
    logger.info("Ingestion complete. %d events loaded.", len(raw.event_log))
    return raw


def report(
    artifact: ModelArtifact,
    evaluation: EvaluationReport,
    output_dir: Path | None = None,
) -> None:
    """Produce comparative visualizations and feature importance plots."""
    raise NotImplementedError


def run(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
) -> EvaluationReport:
    """Execute the full predictive process monitoring pipeline end-to-end."""
    logger.info("Pipeline started.")
    raw = ingest(data_dir)
    preprocessed = preprocess(raw)
    logger.info("Preprocessing complete.")
    features = extract_features(preprocessed)
    logger.info("Feature extraction complete.")
    artifact = train(features)
    logger.info("Training complete.")
    evaluation = evaluate(artifact, features)
    logger.info("Evaluation complete.")
    report(artifact, evaluation, output_dir)
    logger.info("Pipeline finished.")
    return evaluation

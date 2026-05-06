import logging

from spi_time_series.data.schemas import EvaluationReport, FeatureSet, ModelArtifact

logger = logging.getLogger(__name__)


def evaluate(artifact: ModelArtifact, features: FeatureSet) -> EvaluationReport:
    """Compute prediction metrics for each model on the test set."""
    raise NotImplementedError

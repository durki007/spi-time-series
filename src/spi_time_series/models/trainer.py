import logging

from sklearn.base import BaseEstimator

from spi_time_series.data.schemas import FeatureSet, ModelArtifact

logger = logging.getLogger(__name__)


def train(
    features: FeatureSet, models: dict[str, BaseEstimator]
) -> ModelArtifact:
    """Fit baseline and extended models on training feature vectors."""
    raise NotImplementedError

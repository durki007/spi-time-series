import logging

from spi_time_series.data.schemas import FeatureSet, PreprocessedData

logger = logging.getLogger(__name__)


def extract_features(preprocessed: PreprocessedData) -> FeatureSet:
    """Build labeled feature vectors for train and test cases."""
    raise NotImplementedError

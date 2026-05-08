import logging

from spi_time_series.data.schemas import PreprocessedData, RawData

logger = logging.getLogger(__name__)


def preprocess(raw: RawData) -> PreprocessedData:
    """Clean the event log and split cases into train and test sets."""
    raise NotImplementedError

import logging
from collections.abc import Iterator

import numpy as np
import pandas as pd

from spi_time_series.data.schemas import (
    PreprocessedData,
    RawData,
    TraceSample,
    WindowGenerator,
)

logger = logging.getLogger(__name__)


# TODO implement
def clean_data(raw: RawData) -> RawData:
    """
    Clean Data:
        - remove incomplete traces
        - other cleaning operations
    """

    return raw


# TODO implement
def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split event log into train and test set.

    Returns (train_df, test_df)
    """
    return df, df.copy()


def build_traces(df: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame]]:
    """
    Yield temporally ordered event traces grouped by case ID.

    Args:
        data:
            Raw event log container with a pandas event log dataframe.

    Yields:
        tuple[str, pd.DataFrame]:
            A tuple containing:

            - The case identifier.
            - A dataframe representing the ordered trace for that case.

            The yielded trace dataframe preserves the original event
            columns and has a reset zero-based index.
    """
    df = df.sort_values(["case:concept:name", "time:timestamp"])

    for case_id, trace in df.groupby("case:concept:name", sort=False):
        yield case_id, trace.reset_index(drop=True)


def sliding_window_factory(
    min_length: int = 1,
    max_length: int | None = None,
) -> WindowGenerator:
    """
    Create a sliding window generator for trace prefixes.

    The generated function yields temporally ordered indice of subtraces
    constrained by the provided minimum and maximum lengths.

    Args:
        min_length:
            Minimum number of events required in a window.

        max_length:
            Maximum number of events allowed in a window. Set to None for no maximum length.

    Returns:
        A callable that generates trace windows from a trace numpy array.
    """

    def sliding_window(trace: np.ndarray) -> Iterator[pd.DataFrame]:
        n_events = trace.shape[0]

        for end_idx in range(1, n_events + 1):
            start_idx = (
                0 if max_length is None else max(end_idx - max_length, 0)
            )

            if end_idx - start_idx < min_length:
                continue

            yield start_idx, end_idx

    return sliding_window


def _build_trace_samples(
    df: pd.DataFrame,
    window_generator: WindowGenerator,
) -> Iterator[TraceSample]:
    """
    Generate trace samples from an event log.
    Yields:
        TraceSample
    """
    for case_id, trace in build_traces(df):
        data = trace.to_numpy()
        yield TraceSample(
            case_id=case_id, data=data, prefix_indexes=window_generator(data)
        )


def preprocess(
    raw: RawData,
    prefix_generator: WindowGenerator | None = None,
) -> PreprocessedData:
    """
    End-to-end preprocessing pipeline for event logs.

    Cleans raw event data, splits it into train and test sets, and
    generates prefix samples with associated targets for both splits.

    Steps:
        1. Clean raw event log
        2. Split into train and test sets
        3. Generate prefix samples with targets

    Args:
        raw:
            Raw event log input.

        prefix_generator:
            Optional window generator for creating prefixes.
            If None, a default sliding window strategy is used.

    Returns:
        PreprocessedData:
            Structured dataset containing train and test prefix streams
            ready for feature extraction or modeling.
    """
    cleaned_data = clean_data(raw)
    train_df, test_df = split_data(cleaned_data.event_log)

    col_idx = {c: i for i, c in enumerate(train_df.columns)}

    if prefix_generator is None:
        prefix_generator = sliding_window_factory()

    preprocessed_data = PreprocessedData(
        train_log=_build_trace_samples(train_df, prefix_generator),
        num_train_cases=len(train_df["case:concept:name"].unique()),
        test_log=_build_trace_samples(test_df, prefix_generator),
        num_test_cases=len(test_df["case:concept:name"].unique()),
        col_idx=col_idx,
    )

    return preprocessed_data

import logging
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pm4py

from spi_time_series.data.schemas import (
    PreprocessedData,
    RawData,
    TraceSample,
    WindowGenerator,
)

logger = logging.getLogger(__name__)


def clean_event_log(
    df: pd.DataFrame,
    valid_end_activities: list[str] | None = None,
    top_k_variants: int | None = None,
) -> pd.DataFrame:
    """
    Clean the event log dataframe by:
    - Formatting it for PM4Py processing.
    - Optionally filtering to the top K variants.
    - Optionally filtering for valid end activities.
    - Sorting events by case ID and timestamp.
    """
    #
    # Ensure the dataframe has the necessary columns and is in the correct format for PM4Py.
    df = pm4py.format_dataframe(
        df,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )

    # Optionally filter to the top K variants to focus on the most common process paths.
    if top_k_variants:
        logger.info(f"Filtering to top {top_k_variants} variants.")
        df = pm4py.filter_variants_top_k(df, k=top_k_variants)

    # Optionally filter for valid end activities to ensure only complete cases are retained.
    if valid_end_activities:
        logger.info(
            f"Filtering for valid end activities: {valid_end_activities}"
        )
        df = pm4py.filter_end_activities(df, valid_end_activities)

    df = df.sort_values(by=["case:concept:name", "time:timestamp"])
    logger.info(f"Remaining events after cleaning: {len(df)}")

    return df


def preprocess_time_series(event_log_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a continuous hourly timestamp grid for a time-series event log.

    The grid spans the full log from the minimum start time to the maximum
    end time and uses a 1-hour frequency. The returned frame includes
    active case counts for each timestamp.
    """
    required_columns = {"case_id", "start_time", "end_time"}
    missing_columns = required_columns.difference(event_log_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"Missing required columns: {missing}")

    if event_log_df.empty:
        return pd.DataFrame(
            {
                "timestamp": pd.DatetimeIndex([], name="timestamp"),
                "active_cases": pd.Series(dtype="int64"),
            }
        )

    start_times = pd.to_datetime(event_log_df["start_time"])
    end_times = pd.to_datetime(event_log_df["end_time"])

    min_start_time = start_times.min()
    max_end_time = end_times.max()

    timestamp_grid = pd.date_range(
        start=min_start_time, end=max_end_time, freq="1h"
    )

    time_series = pd.DataFrame({"timestamp": timestamp_grid}).set_index(
        "timestamp"
    )
    start_values = start_times.to_numpy()
    end_values = end_times.to_numpy()

    time_series["active_cases"] = [
        int(((start_values <= ts) & (end_values >= ts)).sum())
        for ts in time_series.index
    ]
    time_series["active_cases"] = time_series["active_cases"].ffill()

    return time_series.reset_index()


def split_data(
    df: pd.DataFrame, split_quantile: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split event log into train and test sets using a strict temporal cutoff.

    To prevent data leakage, the split is based on the timestamps of events.
    Cases that overlap the split boundary are discarded to ensure strict separation.

    Returns (train_df, test_df)
    """
    # Determine the cutoff time based on the specified quantile of case start times.
    case_starts = df.groupby("case:concept:name")["time:timestamp"].min()
    case_ends = df.groupby("case:concept:name")["time:timestamp"].max()
    cutoff_time = case_starts.quantile(split_quantile)
    logger.info(f"Splitting cases at cutoff time: {cutoff_time}")

    # Assign cases to train or test based on their start and end times relative to the cutoff.
    train_ids = case_ends[case_ends < cutoff_time].index
    test_ids = case_starts[case_starts >= cutoff_time].index

    # Discard cases that overlap the cutoff to prevent data leakage
    train_df = df[df["case:concept:name"].isin(train_ids)]
    test_df = df[df["case:concept:name"].isin(test_ids)]

    logger.info(
        f"Split successful. \nTrain: {len(train_ids)} cases, \nTest: {len(test_ids)} cases."
    )

    return train_df, test_df


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
        1. Clean the raw event log data to ensure quality and consistency.
        2. Split the cleaned event log into training and testing sets based on a temporal cutoff.
        3. Generate prefix samples for both training and testing sets using the provided window generator and target generator.

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
    cleaned_df = clean_event_log(raw.event_log)

    case_times = (
        cleaned_df.groupby("case:concept:name")["time:timestamp"]
        .agg(["min", "max"])
        .reset_index()
        .rename(
            columns={
                "case:concept:name": "case_id",
                "min": "start_time",
                "max": "end_time",
            }
        )
    )

    time_series_df = preprocess_time_series(case_times)

    train_df, test_df = split_data(cleaned_df)

    col_idx = {c: i for i, c in enumerate(train_df.columns)}

    # Use a default sliding window generator if none is provided.
    if prefix_generator is None:
        prefix_generator = sliding_window_factory()

    # Build prefix samples for both training and testing sets.
    preprocessed_data = PreprocessedData(
        train_log=_build_trace_samples(train_df, prefix_generator),
        num_train_cases=len(train_df["case:concept:name"].unique()),
        test_log=_build_trace_samples(test_df, prefix_generator),
        num_test_cases=len(test_df["case:concept:name"].unique()),
        col_idx=col_idx,
        time_series=time_series_df,
    )

    return preprocessed_data

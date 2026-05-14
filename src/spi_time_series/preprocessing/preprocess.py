import logging
from collections.abc import Iterator

import pandas as pd
import pm4py

from spi_time_series.data.schemas import (
    PrefixSample,
    PreprocessedData,
    RawData,
    TargetGenerator,
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


def clean_data(
    raw: RawData,
    valid_ends: list[str] | None = None,
    top_k_variants: int | None = None,
) -> RawData:
    """
    Clean the raw event log data by applying the clean_event_log function.
    Args:
        raw:
            Raw event log input.

        valid_ends:
            Optional list of valid end activities to filter the event log.
             If None, no filtering based on end activities is applied.
    """
    cleaned_log = clean_event_log(
        raw.event_log,
        valid_end_activities=valid_ends,
        top_k_variants=top_k_variants,
    )
    return RawData(event_log=cleaned_log)


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
    cutoff_time = case_starts.quantile(split_quantile)
    logger.info(f"Splitting cases at cutoff time: {cutoff_time}")

    # Compute the start and end times for each case to determine their relation to the cutoff.
    case_ends = df.groupby("case:concept:name")["time:timestamp"].max()
    case_starts = df.groupby("case:concept:name")["time:timestamp"].min()

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

    The generated function yields temporally ordered subtraces
    constrained by the provided minimum and maximum lengths.

    Args:
        min_length:
            Minimum number of events required in a window.

        max_length:
            Maximum number of events allowed in a window. Set to None for no maximum length.

    Returns:
        A callable that generates trace windows from a trace dataframe.
    """

    def sliding_window(trace: pd.DataFrame) -> Iterator[pd.DataFrame]:
        n_events = len(trace)

        for end_idx in range(1, n_events + 1):
            start_idx = (
                0 if max_length is None else max(end_idx - max_length, 0)
            )

            if end_idx - start_idx < min_length:
                continue

            yield trace.iloc[start_idx:end_idx]

    return sliding_window


def _build_prefixes(
    df: pd.DataFrame,
    window_generator: WindowGenerator,
    target_generator: TargetGenerator,
) -> Iterator[PrefixSample]:
    """
    Generate prefix samples from an event log.

    Iterates over traces, applies a window generator to create prefixes,
    and computes a target value for each prefix.

    Yields:
        PrefixSample:
            Case ID, prefix dataframe, and associated target value.
    """
    for case_id, trace in build_traces(df):
        for prefix in window_generator(trace):
            target = target_generator(trace, prefix)

            yield PrefixSample(case_id=case_id, prefix=prefix, target=target)


def preprocess(
    raw: RawData,
    target_generator: TargetGenerator,
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

        target_generator:
            Function used to compute target values for each prefix.

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

    # Use a default sliding window generator if none is provided.
    if prefix_generator is None:
        prefix_generator = sliding_window_factory()

    # Build prefix samples for both training and testing sets.
    preprocessed_data = PreprocessedData(
        train_log=_build_prefixes(train_df, prefix_generator, target_generator),
        test_log=_build_prefixes(test_df, prefix_generator, target_generator),
        activity_col="concept:name",
        timestamp_col="time:timestamp",
    )

    return preprocessed_data

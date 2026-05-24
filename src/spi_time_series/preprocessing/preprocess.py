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


def clean_data(
    raw: RawData,
    valid_ends: list[str] | None = None,
    top_k_variants: int | None = None,
) -> RawData:
    """
    Clean the raw event log data by formatting it for PM4Py,
    optionally filtering to the top-K variants and valid end activities,
    and sorting by case ID and timestamp.

    Args:
        raw: Raw event log input.
        valid_ends: Optional list of valid end activities to filter the event log.
        top_k_variants: Optional number of top variants to keep.
    """
    df = pm4py.format_dataframe(
        raw.event_log,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )

    if top_k_variants:
        logger.info("Filtering to top %s variants.", top_k_variants)
        df = pm4py.filter_variants_top_k(df, k=top_k_variants)

    if valid_ends:
        logger.info("Filtering for valid end activities: %s", valid_ends)
        df = pm4py.filter_end_activities(df, valid_ends)

    df = df.sort_values(by=["case:concept:name", "time:timestamp"])
    logger.info("Remaining events after cleaning: %d", len(df))

    return RawData(event_log=df)


def split_data(
    df: pd.DataFrame, split_quantile: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split event log into train and test sets using a strict temporal cutoff.

    To prevent data leakage, the split is based on the timestamps of events.
    Cases that overlap the split boundary are discarded to ensure strict separation.

    Returns (train_df, test_df)
    """
    case_starts = df.groupby("case:concept:name")["time:timestamp"].min()
    cutoff_time = case_starts.quantile(split_quantile)
    logger.info("Splitting cases at cutoff time: %s", cutoff_time)

    case_ends = df.groupby("case:concept:name")["time:timestamp"].max()
    case_starts = df.groupby("case:concept:name")["time:timestamp"].min()

    train_ids = case_ends[case_ends < cutoff_time].index
    test_ids = case_starts[case_starts >= cutoff_time].index

    train_df = df[df["case:concept:name"].isin(train_ids)]
    test_df = df[df["case:concept:name"].isin(test_ids)]

    logger.info(
        "Split successful. \nTrain: %d cases, \nTest: %d cases.",
        len(train_ids),
        len(test_ids),
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


def clean_time_series(
    df: pd.DataFrame, id_col: str = "series_id", time_col: str = "timestamp"
) -> pd.DataFrame:
    """
    Prepare a time series dataframe for processing.

    Ensures the timestamp column is datetime-like, sorts by `id_col` and
    `time_col`, and returns the cleaned frame.
    """
    if time_col not in df.columns:
        raise KeyError(f"Timestamp column '{time_col}' not found in dataframe")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)

    if id_col not in df.columns:
        df[id_col] = "_global"

    df = df.sort_values([id_col, time_col])
    logger.info("Prepared time series with %d rows.", len(df))
    return df


def split_time_series(
    df: pd.DataFrame,
    id_col: str = "series_id",
    time_col: str = "timestamp",
    split_quantile: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split multiple time series (grouped by `id_col`) into train and test
    sets using a temporal cutoff computed from series start times.

    Returns (train_df, test_df) containing only the series assigned to each
    split (no overlapping series across splits).
    """
    starts = df.groupby(id_col)[time_col].min()
    cutoff = starts.quantile(split_quantile)
    logger.info("Time-series split cutoff: %s", cutoff)

    ends = df.groupby(id_col)[time_col].max()

    train_ids = ends[ends < cutoff].index
    test_ids = starts[starts >= cutoff].index

    train_df = df[df[id_col].isin(train_ids)]
    test_df = df[df[id_col].isin(test_ids)]

    logger.info(
        "Split time series: %d train series, %d test series.",
        len(train_ids),
        len(test_ids),
    )
    return train_df, test_df


def _build_series_samples(
    df: pd.DataFrame,
    window_generator: WindowGenerator,
    id_col: str,
    time_col: str,
) -> Iterator[TraceSample]:
    """
    Generate TraceSample objects for each series in a time-series dataframe.
    """
    df = df.sort_values([id_col, time_col])
    for sid, series in df.groupby(id_col, sort=False):
        data = series.to_numpy()
        yield TraceSample(
            case_id=sid, data=data, prefix_indexes=window_generator(data)
        )


def preprocess(
    raw: RawData | pd.DataFrame,
    prefix_generator: WindowGenerator | None = None,
    *,
    mode: str = "log",
    id_col: str = "series_id",
    time_col: str = "timestamp",
    valid_end_activities: list[str] | None = None,
    top_k_variants: int | None = None,
) -> PreprocessedData:
    """
    End-to-end preprocessing for event logs or time series.

    Args:
        raw:
            For `mode="log"`, pass RawData. For `mode="time_series"`, pass a
            pandas DataFrame with `id_col` and `time_col`.
        prefix_generator:
            Optional window generator for creating prefixes.
        mode:
            "log" (default) or "time_series".
        id_col:
            Series identifier column (time-series mode only).
        time_col:
            Timestamp column (time-series mode only).
        valid_end_activities:
            Optional list of valid end activities to filter the event log.
        top_k_variants:
            Optionally filter to the top K variants.
    """
    if prefix_generator is None:
        prefix_generator = sliding_window_factory()

    if mode == "log":
        if not isinstance(raw, RawData):
            raise TypeError("mode='log' expects RawData")
        cleaned = clean_data(
            raw,
            valid_ends=valid_end_activities,
            top_k_variants=top_k_variants,
        )
        train_df, test_df = split_data(cleaned.event_log)
        col_idx = {c: i for i, c in enumerate(train_df.columns)}
        return PreprocessedData(
            train_log=_build_trace_samples(train_df, prefix_generator),
            num_train_cases=len(train_df["case:concept:name"].unique()),
            test_log=_build_trace_samples(test_df, prefix_generator),
            num_test_cases=len(test_df["case:concept:name"].unique()),
            col_idx=col_idx,
        )

    if mode == "time_series":
        if not isinstance(raw, pd.DataFrame):
            raise TypeError("mode='time_series' expects a pandas DataFrame")
        cleaned = clean_time_series(raw, id_col=id_col, time_col=time_col)
        train_df, test_df = split_time_series(
            cleaned, id_col=id_col, time_col=time_col
        )
        col_idx = {c: i for i, c in enumerate(train_df.columns)}
        return PreprocessedData(
            train_log=_build_series_samples(
                train_df, prefix_generator, id_col=id_col, time_col=time_col
            ),
            num_train_cases=len(train_df[id_col].unique()),
            test_log=_build_series_samples(
                test_df, prefix_generator, id_col=id_col, time_col=time_col
            ),
            num_test_cases=len(test_df[id_col].unique()),
            col_idx=col_idx,
        )

    raise ValueError("mode must be 'log' or 'time_series'")

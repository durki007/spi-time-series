import logging

import numpy as np
import pandas as pd
import pm4py

from spi_time_series.data.constants import OUTCOME_EVENTS
from spi_time_series.data.schemas import (
    TraceSample,
    WindowGenerator,
)

logger = logging.getLogger(__name__)


def clean_event_log(
    df: pd.DataFrame,
    filter_valid_outcomes: bool = False,
    valid_end_activities: list[str] | None = None,
    top_k_variants: int | None = None,
) -> pd.DataFrame:
    """
    Clean the event log dataframe by:
    - Formatting it for PM4Py processing.
    - Optionally filtering traces that don't contain at least one OUTCOME event (-> incomplete case)
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

    # Filter out cases, that don't have a outcome event
    if filter_valid_outcomes:
        logger.info(
            f"Filtering cases without any of {OUTCOME_EVENTS}. They are likely incomplete"
        )
        outcome_events_set = set(OUTCOME_EVENTS)
        cases_with_outcome = df.groupby("case:concept:name")[
            "concept:name"
        ].apply(lambda x: bool(set(x) & outcome_events_set))

        valid_cases = cases_with_outcome[cases_with_outcome].index

        df = df[df["case:concept:name"].isin(valid_cases)]

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


def filter_dev_cases(
    df: pd.DataFrame, dev_quantile: float = 0.1
) -> pd.DataFrame:
    case_starts = df.groupby("case:concept:name")["time:timestamp"].min()
    cutoff = case_starts.quantile(dev_quantile)
    dev_ids = case_starts[case_starts <= cutoff].index
    logger.info(
        f"Dev mode: keeping {len(dev_ids)} cases (≤ {dev_quantile:.0%} quantile, cutoff {cutoff})"
    )
    return df[df["case:concept:name"].isin(dev_ids)]


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
    case_stats = df.groupby("case:concept:name")["time:timestamp"].agg(
        ["min", "max"]
    )
    case_starts = case_stats["min"]
    case_ends = case_stats["max"]
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
            Minimum number of events required in a window (must be >= 1).

        max_length:
            Maximum number of events allowed in a window. Set to None for
            no maximum length. If provided, must be >= min_length.

    Returns:
        A callable that generates trace windows from a trace numpy array.

    Raises:
        ValueError: If ``min_length < 1`` or ``max_length < min_length``.
    """
    if min_length < 1:
        raise ValueError(f"min_length must be >= 1, got {min_length}")
    if max_length is not None and max_length < min_length:
        raise ValueError(
            f"max_length ({max_length}) must be >= min_length ({min_length})"
        )

    logger.info(
        "Creating sliding window generator (min_length=%d, max_length=%s)",
        min_length,
        max_length,
    )

    def sliding_window(trace: np.ndarray) -> np.ndarray:
        n_events: int = trace.shape[0]

        logger.debug(
            "Building windows for trace with %d events "
            "(min_length=%d, max_length=%s)",
            n_events,
            min_length,
            max_length,
        )

        end_idx = np.arange(min_length, n_events + 1)

        if max_length is None:
            start_idx = np.zeros_like(end_idx)
        else:
            start_idx = np.maximum(end_idx - max_length, 0)

        windows = np.column_stack((start_idx, end_idx))

        logger.debug(
            "Generated %d windows for trace with %d events",
            windows.shape[0],
            n_events,
        )

        return windows

    return sliding_window


def _build_trace_samples(
    df: pd.DataFrame,
    window_generator: WindowGenerator,
) -> list[TraceSample]:
    """
    Generate trace samples from an event log.

    Yields:
        TraceSample
    """
    logger.info("Building trace samples from event log (%d rows)", len(df))

    traces: list[TraceSample] = []
    for case_id, trace in df.groupby("case:concept:name", sort=False):
        data = trace.to_numpy()
        prefix_indexes = window_generator(data)
        traces.append(
            TraceSample(
                case_id=str(case_id),
                data=data,
                prefix_indexes=prefix_indexes,
            )
        )

    logger.info(
        "Built %d trace samples across %d unique cases",
        len(traces),
        df["case:concept:name"].nunique(),
    )

    return traces

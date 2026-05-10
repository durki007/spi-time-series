import logging
from collections.abc import Callable, Iterator
from typing import Any

import pandas as pd

from spi_time_series.data.schemas import RawData

logger = logging.getLogger(__name__)

type WindowGenerator = Callable[[list[dict]], Iterator[list[dict]]]


def build_traces(data: RawData) -> dict[str, list[dict]]:
    """
    Build ordered event traces per case.

    Returns:
        dict mapping case_id -> ordered list of event dictionaries
    """
    logger.info("Building traces from event log")

    df = data.event_log.sort_values(["case:concept:name", "time:timestamp"])

    traces = {
        case_id: group.to_dict(orient="records")
        for case_id, group in df.groupby("case:concept:name", sort=False)
    }

    return traces


def sliding_window_factory(
    min_length: int,
    max_length: int,
) -> WindowGenerator:
    """
    Generate windows for a single trace.
    """

    def sliding_window(trace: list[dict]):
        for end_idx in range(len(trace) + 1):
            start_idx = max(end_idx - max_length, 0)

            if end_idx - start_idx < min_length:
                continue
            print(start_idx, end_idx)
            yield trace[start_idx:end_idx]

    return sliding_window


def build_prefixes(
    traces: dict[str, list[dict]],
    window_generator: WindowGenerator | None = None,
    features: dict[str, Callable[[list[dict]], Any]] | None = None,
) -> pd.DataFrame:
    """
    Generate a prefix-based tabular dataset from event traces.

    This function converts case-level event traces into a machine-learning
    ready dataset using a configurable windowing strategy. For each trace,
    multiple prefix windows are generated, and each window is transformed into
    a single training instance.

    Feature values can be optionally computed over each prefix window using
    user-defined aggregation functions.

    Each resulting row corresponds to one prefix instance and contains:
    - the case identifier
    - the sequence of activities in the prefix window
    - optional aggregated feature values derived from the prefix

    Parameters
    ----------
    traces:
        Dictionary mapping case identifiers to ordered event traces.
        Each trace is a list of event dictionaries sorted chronologically.

    window_generator:
        Callable that generates windows for a given trace. This allows customization
        of the windowing strategy (e.g., sliding window, full prefix, exponential windows).
        per default sliding window strategy with minimum window length of 2 and maximum length 5

    features:
        Dictionary mapping feature names to aggregation functions.
        Each function receives a prefix window (list of event dictionaries)
        and returns a scalar or str feature value (e.g., count, duration, average).

    Returns
    -------
    pandas.DataFrame
        A DataFrame where each row represents a prefix instance. Columns include:

        - "case:concept:name": case identifier
        - "prefix": ordered list of activity names in the prefix window
        - additional feature columns defined in `features`

    Notes
    -----
    - Prefix generation is delegated to `window_generator`, enabling flexible
      windowing strategies.
    - The function assumes that all traces are already chronologically ordered.
    - If no features are provided, only structural prefix data is returned.
    """
    logger.info("Building prefix dataset")

    if window_generator is None:
        window_generator = sliding_window_factory(min_length=2, max_length=5)

    if features is None:
        features = {}

    rows = []

    for case_id, trace in traces.items():
        # iterate over prefix windows
        for window in window_generator(trace):
            row = {
                "case:concept:name": case_id,
                "prefix": [event["concept:name"] for event in window],
            }

            # add aggregated features
            row.update(
                {
                    feature_name: func(window)
                    for feature_name, func in features.items()
                }
            )

            rows.append(row)

    return pd.DataFrame(rows)

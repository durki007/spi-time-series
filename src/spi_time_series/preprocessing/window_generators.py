import logging

import numpy as np

from spi_time_series.data.constants import OUTCOME_EVENTS
from spi_time_series.data.schemas import (
    WindowGenerator,
)

logger = logging.getLogger(__name__)


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

    def sliding_window(
        trace: np.ndarray, col_idx_mapping: dict[str, int]
    ) -> np.ndarray:
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


def outcome_window_factory(
    min_length: int = 1, max_length: int | None = None
) -> WindowGenerator:
    """
    Create a prefix window generator for outcome prediction tasks.

    The generated function yields temporally ordered indices of trace prefixes,
    stopping before the first occurrence of an outcome event. This prevents
    generating prefixes that already contain information revealing the outcome.

    Args:
        min_length:
            Minimum number of events required in a window (must be >= 1).

        max_length:
            Maximum number of events allowed in a window. Set to None for
            no maximum length. If provided, must be >= min_length.

    Returns:
        A callable that generates trace windows from a trace numpy array,
        bounded by the first outcome event.

    Raises:
        ValueError: If ``min_length < 1`` or ``max_length < min_length``.
    """
    if min_length < 1:
        raise ValueError(f"min_length must be >= 1, got {min_length}")
    if max_length is not None and max_length < min_length:
        raise ValueError(
            f"max_length ({max_length}) must be >= min_length ({min_length})"
        )

    def window_function(
        trace: np.ndarray, col_idx_mapping: dict[str, int]
    ) -> np.ndarray:
        n_events: int = trace.shape[0]

        activities = trace[:, col_idx_mapping["concept:name"]]
        mask = np.isin(activities, OUTCOME_EVENTS)
        idx_outcome_event = n_events + 1
        if mask.any():
            idx_outcome_event = np.argmax(mask)  # index of first outcome event

        end_idx = np.arange(min_length, idx_outcome_event)

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

    return window_function

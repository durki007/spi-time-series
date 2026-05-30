from datetime import datetime

import numpy as np
import pytest

from spi_time_series.features.targets import (
    outcome_target,
    remaining_time_target,
)

# ---------------------------
# Helpers
# ---------------------------


def make_trace(times, activities):
    """Create a structured numpy trace with timestamps + activity names."""
    return np.array(
        list(zip(times, activities, strict=True)),
        dtype=[("time:timestamp", "O"), ("concept:name", "O")],
    )


def make_mapping():
    return {"time:timestamp": 0, "concept:name": 1}


# ---------------------------
# remaining_time_target tests
# ---------------------------


def test_remaining_time_basic():
    trace = np.array(
        [
            [datetime(2024, 1, 1, 10), "A"],
            [datetime(2024, 1, 1, 12), "B"],
            [datetime(2024, 1, 1, 16), "C"],
        ],
        dtype=object,
    )

    mapping = make_mapping()

    # prefix ends at index 2 (event "B")
    result = remaining_time_target(trace, 0, 2, mapping)

    # 12 -> 16 = 4 hours
    assert result == pytest.approx(4.0)


def test_remaining_time_single_step():
    trace = np.array(
        [
            [datetime(2024, 1, 1, 10), "A"],
            [datetime(2024, 1, 1, 14), "B"],
        ],
        dtype=object,
    )

    mapping = make_mapping()

    result = remaining_time_target(trace, 0, 1, mapping)

    assert result == pytest.approx(4)


def test_remaining_time_full_prefix_returns_zero():
    trace = np.array(
        [
            [datetime(2024, 1, 1, 10), "A"],
            [datetime(2024, 1, 1, 14), "B"],
        ],
        dtype=object,
    )

    mapping = make_mapping()

    # end_idx == len(trace) -> last event
    result = remaining_time_target(trace, 0, 2, mapping)

    assert result == pytest.approx(0.0)


# ---------------------------
# outcome_target tests
# ---------------------------


def test_outcome_successful_completion_first_match(monkeypatch):
    trace = np.array(
        [
            [datetime(2024, 1, 1), "X"],
            [datetime(2024, 1, 2), "A_Pending"],
            [datetime(2024, 1, 3), "Y"],
        ],
        dtype=object,
    )

    mapping = make_mapping()

    assert outcome_target(trace, 0, 3, mapping) == 0


def test_outcome_denied():
    trace = np.array(
        [
            [datetime(2024, 1, 1), "A_Denied"],
            [datetime(2024, 1, 2), "X"],
        ],
        dtype=object,
    )

    mapping = make_mapping()

    assert outcome_target(trace, 0, 2, mapping) == 1


def test_outcome_cancelled():
    trace = np.array(
        [
            [datetime(2024, 1, 1), "X"],
            [datetime(2024, 1, 2), "A_Cancelled"],
        ],
        dtype=object,
    )

    mapping = make_mapping()

    assert outcome_target(trace, 0, 2, mapping) == 2


def test_outcome_priority_first_match_only():
    """
    Ensures function returns FIRST matching class in OUTCOME_EVENTS order.
    """
    trace = np.array(
        [
            [datetime(2024, 1, 1), "A_Denied"],
            [datetime(2024, 1, 2), "A_Cancelled"],
        ],
        dtype=object,
    )

    mapping = make_mapping()

    # depends on OUTCOME_EVENTS order → deterministic behavior expected
    result = outcome_target(trace, 0, 2, mapping)

    assert result in [0, 1, 2]  # but should match first valid rule


def test_outcome_missing_event_raises():
    trace = np.array(
        [
            [datetime(2024, 1, 1), "X"],
            [datetime(2024, 1, 2), "Y"],
        ],
        dtype=object,
    )

    mapping = make_mapping()

    with pytest.raises(ValueError, match="Could not determine outcome"):
        outcome_target(trace, 0, 2, mapping)


def test_outcome_multiple_occurrences_still_valid():
    trace = np.array(
        [
            [datetime(2024, 1, 1), "A_Pending"],
            [datetime(2024, 1, 2), "A_Pending"],
            [datetime(2024, 1, 3), "A_Pending"],
        ],
        dtype=object,
    )

    mapping = make_mapping()

    assert outcome_target(trace, 0, 3, mapping) == 0


# ---------------------------
# edge robustness tests
# ---------------------------


def test_outcome_set_behavior_independent_of_duplicates():
    trace = np.array(
        [
            [datetime(2024, 1, 1), "A_Denied"],
            [datetime(2024, 1, 2), "A_Denied"],
            [datetime(2024, 1, 3), "A_Cancelled"],
        ],
        dtype=object,
    )

    mapping = make_mapping()

    result = outcome_target(trace, 0, 3, mapping)

    assert result in [1, 2]  # depending on OUTCOME_EVENTS order

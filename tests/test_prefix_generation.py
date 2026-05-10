import pandas as pd
import pytest

from spi_time_series.data.schemas import RawData
from spi_time_series.preprocessing.prefix_generation import (
    build_prefixes,
    build_traces,
    sliding_window_factory,
)

# =========================================================
# Fixtures
# =========================================================


@pytest.fixture
def raw_event_log():
    return pd.DataFrame(
        [
            {
                "case:concept:name": "C1",
                "time:timestamp": 1,
                "concept:name": "A",
            },
            {
                "case:concept:name": "C1",
                "time:timestamp": 2,
                "concept:name": "B",
            },
            {
                "case:concept:name": "C1",
                "time:timestamp": 3,
                "concept:name": "C",
            },
            {
                "case:concept:name": "C2",
                "time:timestamp": 1,
                "concept:name": "X",
            },
            {
                "case:concept:name": "C2",
                "time:timestamp": 2,
                "concept:name": "Y",
            },
        ]
    )


# =========================================================
# build_traces
# =========================================================


def test_build_traces_orders_events(raw_event_log):
    data = RawData(raw_event_log)

    traces = build_traces(data)

    assert "C1" in traces
    assert len(traces["C1"]) == 3

    assert [e["concept:name"] for e in traces["C1"]] == ["A", "B", "C"]


# =========================================================
# sliding_window_factory
# =========================================================


def test_sliding_window_factory_generates_windows():
    trace = [
        {"concept:name": "A"},
        {"concept:name": "B"},
        {"concept:name": "C"},
    ]

    window_fn = sliding_window_factory(min_length=1, max_length=2)

    windows = list(window_fn(trace))

    assert len(windows) == 3
    assert all(isinstance(w, list) for w in windows)
    assert all(len(w) >= 1 for w in windows)


# =========================================================
# build_prefixes (basic structure)
# =========================================================


def test_build_prefixes_basic():
    traces = {
        "C1": [
            {"concept:name": "A"},
            {"concept:name": "B"},
            {"concept:name": "C"},
        ]
    }

    df = build_prefixes(traces)

    assert isinstance(df, pd.DataFrame)
    assert "case:concept:name" in df.columns
    assert "prefix" in df.columns
    assert len(df) > 0


# =========================================================
# build_prefixes with features
# =========================================================


def test_build_prefixes_with_features():
    traces = {
        "C1": [
            {"concept:name": "A"},
            {"concept:name": "B"},
            {"concept:name": "C"},
        ]
    }

    features = {
        "length": lambda w: len(w),
    }

    df = build_prefixes(traces, features=features)

    assert "length" in df.columns
    assert all(df["length"] > 0)


# =========================================================
# edge cases
# =========================================================


def test_empty_trace_returns_empty_df():
    traces: dict[str, list[dict]] = {"C1": []}

    df = build_prefixes(traces)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0

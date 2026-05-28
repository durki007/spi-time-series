import pandas as pd
import pytest

from spi_time_series.data.schemas import RawData, TraceSample
from spi_time_series.features.time_series_features import preprocess_time_series
from spi_time_series.preprocessing.preprocess import (
    _build_trace_samples,
    preprocess,
    sliding_window_factory,
)

# ----------------------------
# Fixtures
# ----------------------------


@pytest.fixture
def sample_log():
    df = pd.DataFrame(
        [
            {
                "case:concept:name": "A",
                "time:timestamp": "2023-01-01 10:00:00",
                "concept:name": "x",
            },
            {
                "case:concept:name": "A",
                "time:timestamp": "2023-01-01 11:00:00",
                "concept:name": "y",
            },
            {
                "case:concept:name": "B",
                "time:timestamp": "2023-01-02 10:00:00",
                "concept:name": "z",
            },
            {
                "case:concept:name": "B",
                "time:timestamp": "2023-01-02 11:00:00",
                "concept:name": "w",
            },
            {
                "case:concept:name": "C",
                "time:timestamp": "2023-01-03 10:00:00",
                "concept:name": "v",
            },
            {
                "case:concept:name": "C",
                "time:timestamp": "2023-01-03 11:00:00",
                "concept:name": "u",
            },
            {
                "case:concept:name": "D",
                "time:timestamp": "2023-01-04 10:00:00",
                "concept:name": "t",
            },
            {
                "case:concept:name": "D",
                "time:timestamp": "2023-01-04 11:00:00",
                "concept:name": "s",
            },
            {
                "case:concept:name": "E",
                "time:timestamp": "2023-01-07 10:00:00",
                "concept:name": "r",
            },
            {
                "case:concept:name": "E",
                "time:timestamp": "2023-01-07 11:00:00",
                "concept:name": "q",
            },
        ]
    )
    # pm4py requires proper datetime columns
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
    return df


@pytest.fixture
def raw(sample_log):
    return RawData(event_log=sample_log)


# ----------------------------
# Tests: sliding window
# ----------------------------


def test_sliding_window_factory_basic():
    df = pd.DataFrame(
        {
            "concept:name": ["a", "b", "c"],
        }
    )

    window = sliding_window_factory(min_length=1, max_length=2)

    result = [w for w in window(df)]

    # expected windows:
    # [a], [a,b], [b,c]
    assert len(result) == 3


def test_sliding_window_min_length():
    df = pd.DataFrame({"x": [1, 2, 3, 4]})

    window = sliding_window_factory(min_length=2, max_length=None)

    result = list(window(df))

    # only windows length >= 2
    assert all(len(w) >= 2 for w in result)


# ----------------------------
# Tests: _build_trace_samples
# ----------------------------


def test_build_trace_samples_generates_samples(sample_log):
    window = sliding_window_factory(min_length=1, max_length=2)

    samples = list(_build_trace_samples(sample_log, window))

    assert all(isinstance(s, TraceSample) for s in samples)

    # check case preservation across all 5 cases
    assert {s.case_id for s in samples} == {"A", "B", "C", "D", "E"}


# ----------------------------
# Tests: preprocess integration
# ----------------------------


def test_preprocess_pipeline(raw):
    result = preprocess(raw)

    train = list(result.train_log)
    test = list(result.test_log)

    assert len(train) > 0
    assert len(test) > 0

    # structure checks
    assert isinstance(train[0], TraceSample)
    assert isinstance(test[0], TraceSample)


def test_preprocess_time_series_builds_hourly_grid():
    event_log_df = pd.DataFrame(
        {
            "case_id": ["A", "B"],
            "start_time": [
                "2023-01-01 10:15:00+00:00",
                "2023-01-01 12:00:00+00:00",
            ],
            "end_time": [
                "2023-01-01 13:45:00+00:00",
                "2023-01-01 15:00:00+00:00",
            ],
        }
    )

    result = preprocess_time_series(event_log_df)

    expected_timestamps = pd.date_range(
        start=pd.Timestamp("2023-01-01 10:15:00+00:00"),
        end=pd.Timestamp("2023-01-01 15:00:00+00:00"),
        freq="1h",
    )

    expected_active_cases = [1, 1, 2, 2, 1]

    assert result.columns.tolist() == ["timestamp", "active_cases"]
    assert result["timestamp"].equals(
        expected_timestamps.to_series().reset_index(drop=True)
    )
    assert result["active_cases"].tolist() == expected_active_cases

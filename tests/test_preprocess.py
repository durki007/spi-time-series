import pandas as pd
import pytest

# Corrected imports: Pull schemas from their actual location
from spi_time_series.data.schemas import PrefixSample, RawData
from spi_time_series.preprocessing.preprocess import (
    _build_prefixes,
    build_traces,
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
# Tests: build_traces
# ----------------------------


def test_build_traces_orders_and_groups(sample_log):
    traces = list(build_traces(sample_log))

    # we have 5 cases in the fixture
    assert len(traces) == 5

    case_ids = {t[0] for t in traces}
    assert case_ids == {"A", "B", "C", "D", "E"}

    trace_a = next(df for cid, df in traces if cid == "A")
    assert trace_a["concept:name"].tolist() == ["x", "y"]


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
# Tests: _build_prefixes
# ----------------------------


def test_build_prefixes_generates_samples(sample_log):
    def target_fn(trace, prefix):
        return len(prefix)

    window = sliding_window_factory(min_length=1, max_length=2)

    samples = list(_build_prefixes(sample_log, window, target_fn))

    assert all(isinstance(s, PrefixSample) for s in samples)
    assert all(isinstance(s.target, int) for s in samples)

    # check case preservation across all 5 cases
    assert {s.case_id for s in samples} == {"A", "B", "C", "D", "E"}


# ----------------------------
# Tests: preprocess integration
# ----------------------------


def test_preprocess_pipeline(raw):
    def target_fn(trace, prefix):
        return prefix["concept:name"].iloc[-1]

    result = preprocess(raw, target_fn)

    train = list(result.train_log)
    test = list(result.test_log)

    # Because we expanded the fixture and spaced the dates out,
    # we guarantee both train and test sets have data.
    assert len(train) > 0
    assert len(test) > 0

    # structure checks
    assert isinstance(train[0], PrefixSample)
    assert isinstance(test[0], PrefixSample)

    # columns preserved
    assert result.activity_col == "concept:name"
    assert result.timestamp_col == "time:timestamp"

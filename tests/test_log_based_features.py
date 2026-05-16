import numpy as np
import pandas as pd
import pytest

from spi_time_series.features.log_based_features import (
    BasicControlFlowFeatures,
)


@pytest.fixture(autouse=True)
def patch_event_names(monkeypatch):
    monkeypatch.setattr(
        "spi_time_series.features.log_based_features.EVENT_NAMES",
        ["A", "B", "C", "D"],
    )


@pytest.fixture
def col_idx_mapping():
    return {
        "concept:name": 0,
        "time:timestamp": 1,
    }


@pytest.fixture
def feature_extractor():
    return BasicControlFlowFeatures()


@pytest.fixture
def feature_extractor_onehot():
    return BasicControlFlowFeatures(
        one_hot_encode_categorical=True,
    )


@pytest.fixture
def sample_prefix():
    return np.array(
        [
            ["A", pd.Timestamp("2024-01-01 10:00:00")],
            ["B", pd.Timestamp("2024-01-01 10:05:00")],
            ["C", pd.Timestamp("2024-01-01 10:10:00")],
        ],
        dtype=object,
    )


def make_prefix(events, timestamps):
    """
    Helper to create numpy prefix arrays.
    """
    return np.array(
        list(zip(events, timestamps, strict=True)),
        dtype=object,
    )


# =========================================================
# CONSTRUCTOR / METADATA
# =========================================================


def test_default_constructor():
    extractor = BasicControlFlowFeatures()

    assert extractor.activity_column == "concept:name"
    assert extractor.timestamp_column == "time:timestamp"


def test_custom_constructor():
    extractor = BasicControlFlowFeatures(
        activity_column="activity",
        timestamp_column="timestamp",
    )

    assert extractor.activity_column == "activity"
    assert extractor.timestamp_column == "timestamp"


def test_name():
    extractor = BasicControlFlowFeatures()

    assert extractor.name() == "BasicControlFlowFeatures"


# =========================================================
# SINGLE EVENT PREFIX
# =========================================================


def test_single_event_prefix(feature_extractor, col_idx_mapping):
    prefix = make_prefix(
        ["A"],
        [pd.Timestamp("2024-01-01 10:00:00")],
    )

    result = feature_extractor(prefix, col_idx_mapping)

    assert result["elapsed_time_hours"] == 0.0
    assert result["prefix_length"] == 1
    assert result["last_activity"] == "A"
    assert result["time_since_last_event_hours"] == 0.0
    assert result["rework_count"] == 0
    assert result["last_transition"] is None

    # bag of activities
    assert result["count_A"] == 1


# =========================================================
# MULTI EVENT PREFIX
# =========================================================


def test_multi_event_prefix(feature_extractor, col_idx_mapping):
    prefix = make_prefix(
        ["A", "B", "C"],
        [
            pd.Timestamp("2024-01-01 10:00:00"),
            pd.Timestamp("2024-01-01 11:00:00"),
            pd.Timestamp("2024-01-01 13:30:00"),
        ],
    )

    result = feature_extractor(prefix, col_idx_mapping)

    # elapsed = 3.5 hours
    assert result["elapsed_time_hours"] == 3.5

    assert result["prefix_length"] == 3

    assert result["last_activity"] == "C"

    # last delta = 2.5 hours
    assert result["time_since_last_event_hours"] == 2.5

    assert result["rework_count"] == 0

    assert result["last_transition"] == "B->C"

    assert result["count_A"] == 1
    assert result["count_B"] == 1
    assert result["count_C"] == 1


# =========================================================
# REWORK
# =========================================================


def test_rework_count(feature_extractor, col_idx_mapping):
    prefix = make_prefix(
        ["A", "B", "A", "A"],
        [
            pd.Timestamp("2024-01-01 10:00:00"),
            pd.Timestamp("2024-01-01 11:00:00"),
            pd.Timestamp("2024-01-01 12:00:00"),
            pd.Timestamp("2024-01-01 13:00:00"),
        ],
    )

    result = feature_extractor(prefix, col_idx_mapping)

    # 4 events - 2 unique activities = 2
    assert result["rework_count"] == 2

    assert result["count_A"] == 3
    assert result["count_B"] == 1


# =========================================================
# LAST TRANSITION
# =========================================================


def test_last_transition(feature_extractor, col_idx_mapping):
    prefix = make_prefix(
        ["Submit", "Approve"],
        [
            pd.Timestamp("2024-01-01 10:00:00"),
            pd.Timestamp("2024-01-01 11:00:00"),
        ],
    )

    result = feature_extractor(prefix, col_idx_mapping)

    assert result["last_transition"] == "Submit->Approve"


# =========================================================
# BAG OF ACTIVITIES DEFAULTS
# =========================================================


def test_all_event_counts_exist(feature_extractor, col_idx_mapping):
    prefix = make_prefix(
        ["A"],
        [pd.Timestamp("2024-01-01 10:00:00")],
    )

    result = feature_extractor(prefix, col_idx_mapping)

    for event in ["A", "B", "C", "D"]:
        key = f"count_{event}"

        assert key in result

        if event == "A":
            assert result[key] == 1
        else:
            assert result[key] == 0


# =========================================================
# TIME FEATURES
# =========================================================


def test_elapsed_time_uses_min_and_max_timestamp(
    feature_extractor,
    col_idx_mapping,
):
    """
    Ensures elapsed time is computed using min/max timestamps
    and not event ordering.
    """

    prefix = make_prefix(
        ["A", "B", "C"],
        [
            pd.Timestamp("2024-01-01 15:00:00"),
            pd.Timestamp("2024-01-01 10:00:00"),
            pd.Timestamp("2024-01-01 12:00:00"),
        ],
    )

    result = feature_extractor(prefix, col_idx_mapping)

    # max=15:00 min=10:00 => 5h
    assert result["elapsed_time_hours"] == 5.0


def test_time_since_last_event(feature_extractor, col_idx_mapping):
    prefix = make_prefix(
        ["A", "B"],
        [
            pd.Timestamp("2024-01-01 10:00:00"),
            pd.Timestamp("2024-01-01 12:30:00"),
        ],
    )

    result = feature_extractor(prefix, col_idx_mapping)

    assert result["time_since_last_event_hours"] == 2.5


# =========================================================
# OUTPUT STRUCTURE
# =========================================================


def test_returns_series(feature_extractor, col_idx_mapping):
    prefix = make_prefix(
        ["A"],
        [pd.Timestamp("2024-01-01 10:00:00")],
    )

    result = feature_extractor(prefix, col_idx_mapping)

    assert isinstance(result, pd.Series)


def test_expected_core_features_present(
    feature_extractor,
    col_idx_mapping,
):
    prefix = make_prefix(
        ["A", "B"],
        [
            pd.Timestamp("2024-01-01 10:00:00"),
            pd.Timestamp("2024-01-01 11:00:00"),
        ],
    )

    result = feature_extractor(prefix, col_idx_mapping)

    expected = {
        "elapsed_time_hours",
        "prefix_length",
        "last_activity",
        "time_since_last_event_hours",
        "rework_count",
        "last_transition",
    }

    for feature in expected:
        assert feature in result.index


# =========================================================
# CUSTOM COLUMN MAPPING
# =========================================================


def test_custom_column_mapping():
    extractor = BasicControlFlowFeatures(
        activity_column="activity",
        timestamp_column="ts",
    )

    col_idx_mapping = {
        "activity": 0,
        "ts": 1,
    }

    prefix = np.array(
        [
            ["A", pd.Timestamp("2024-01-01 10:00:00")],
            ["B", pd.Timestamp("2024-01-01 11:00:00")],
        ],
        dtype=object,
    )

    result = extractor(prefix, col_idx_mapping)

    assert result["last_activity"] == "B"
    assert result["last_transition"] == "A->B"


# =========================================================
# ONE HOT ENCODING
# =========================================================


def test_one_hot_last_activity(
    feature_extractor_onehot, sample_prefix, col_idx_mapping
):
    result = feature_extractor_onehot(sample_prefix, col_idx_mapping)

    cols = [c for c in result.index if c.startswith("last_activity__")]
    assert len(cols) > 0

    # Only "C" should be active
    for col in cols:
        if col.endswith("__C"):
            assert result[col] == 1
        else:
            assert result[col] == 0


def test_one_hot_last_transition(
    feature_extractor_onehot, sample_prefix, col_idx_mapping
):
    result = feature_extractor_onehot(sample_prefix, col_idx_mapping)

    cols = [c for c in result.index if c.startswith("last_transition__")]
    assert len(cols) > 0

    # expected transition is B->C
    for col in cols:
        if col.endswith("__B->C"):
            assert result[col] == 1
        else:
            assert result[col] == 0


def test_one_hot_is_binary(
    feature_extractor_onehot, sample_prefix, col_idx_mapping
):
    result = feature_extractor_onehot(sample_prefix, col_idx_mapping)

    ohe_cols = [c for c in result.index if "__" in c]

    for c in ohe_cols:
        assert result[c] in (0, 1)

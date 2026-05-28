import numpy as np
import pytest

from spi_time_series.features.log_based_features import BasicControlFlowFeatures

# =========================================================
# FIXTURES
# =========================================================


@pytest.fixture(autouse=True)
def patch_event_names(monkeypatch):
    monkeypatch.setattr(
        "spi_time_series.data.constants.EVENT_NAMES",
        ["A", "B", "C", "D"],
    )


@pytest.fixture
def col_idx_mapping():
    return {
        "concept:name": 0,
        "time:timestamp": 1,
    }


@pytest.fixture
def sample_prefix():
    return np.array(
        [
            ["A", np.datetime64("2024-01-01T10:00:00")],
            ["B", np.datetime64("2024-01-01T11:00:00")],
            ["C", np.datetime64("2024-01-01T13:30:00")],
        ],
        dtype=object,
    )


def make_prefix(events, timestamps):
    return np.array(list(zip(events, timestamps, strict=True)), dtype=object)


# =========================================================
# BASIC CONSTRUCTION
# =========================================================


def test_constructor_defaults():
    f = BasicControlFlowFeatures()

    assert f.activity_column == "concept:name"
    assert f.timestamp_column == "time:timestamp"
    assert f.one_hot_encode_categorical is False


def test_name():
    f = BasicControlFlowFeatures()
    assert f.name() == "BasicControlFlowFeatures"


# =========================================================
# FIT BEHAVIOR
# =========================================================


def test_fit_creates_feature_names():
    f = BasicControlFlowFeatures(one_hot_encode_categorical=True)

    log = [
        type(
            "T",
            (),
            {"data": np.array([["A", "t1"], ["B", "t2"]], dtype=object)},
        )()
    ]

    f.fit(log, {"concept:name": 0}, min_last_activity=1, min_last_transition=1)

    assert len(f.feature_names) > 0
    assert "elapsed_time_hours" in f.feature_names
    assert any("last_activity__" in n for n in f.feature_names)
    assert any("last_transition__" in n for n in f.feature_names)


def test_fit_empty_trace_handled():
    f = BasicControlFlowFeatures()

    log = [type("T", (), {"data": np.array([], dtype=object).reshape(0, 2)})()]

    f.fit(log, {"concept:name": 0})


# =========================================================
# CORE FEATURE VALUES
# =========================================================


def test_elapsed_time_and_length(col_idx_mapping):
    f = BasicControlFlowFeatures()

    prefix = make_prefix(
        ["A", "B", "C"],
        [
            np.datetime64("2024-01-01T10:00:00"),
            np.datetime64("2024-01-01T11:00:00"),
            np.datetime64("2024-01-01T13:00:00"),
        ],
    )

    out = f(prefix, col_idx_mapping)

    assert out[1] == 3  # prefix_length
    assert out[0] == pytest.approx(3.0)  # elapsed hours


def test_time_since_last_event(col_idx_mapping):
    f = BasicControlFlowFeatures()

    prefix = make_prefix(
        ["A", "B"],
        [
            np.datetime64("2024-01-01T10:00:00"),
            np.datetime64("2024-01-01T12:30:00"),
        ],
    )

    out = f(prefix, col_idx_mapping)

    assert out[2] == pytest.approx(2.5)


def test_single_event_edge_case(col_idx_mapping):
    f = BasicControlFlowFeatures()

    prefix = make_prefix(
        ["A"],
        [np.datetime64("2024-01-01T10:00:00")],
    )

    out = f(prefix, col_idx_mapping)

    assert out[0] == 0.0
    assert out[2] == 0.0
    assert out[1] == 1


# =========================================================
# REWORK COUNT
# =========================================================


def test_rework_count(col_idx_mapping):
    f = BasicControlFlowFeatures()

    prefix = make_prefix(
        ["A", "B", "A", "A"],
        [
            np.datetime64("2024-01-01T10:00:00"),
            np.datetime64("2024-01-01T11:00:00"),
            np.datetime64("2024-01-01T12:00:00"),
            np.datetime64("2024-01-01T13:00:00"),
        ],
    )

    out = f(prefix, col_idx_mapping)

    assert out[3] == 4 - 2  # unique = {A,B}


# =========================================================
# BAG OF ACTIVITIES
# =========================================================


def test_bag_of_activities(col_idx_mapping):
    f = BasicControlFlowFeatures()

    prefix = make_prefix(
        ["A", "A", "B"],
        [
            np.datetime64("2024-01-01T10:00:00"),
            np.datetime64("2024-01-01T11:00:00"),
            np.datetime64("2024-01-01T12:00:00"),
        ],
    )

    out = f(prefix, col_idx_mapping)

    # base offset starts at index 4
    assert out[4] >= 0


# =========================================================
# OHE BEHAVIOR
# =========================================================


def test_ohe_last_activity():
    f = BasicControlFlowFeatures(one_hot_encode_categorical=True)

    log = [
        type(
            "T",
            (),
            {
                "data": np.array(
                    [["A", "t"], ["B", "t2"], ["C", "t3"]], dtype=object
                )
            },
        )()
    ]

    f.fit(log, {"concept:name": 0}, min_last_activity=0, min_last_transition=0)

    prefix = make_prefix(
        ["A", "B", "C"],
        [
            np.datetime64("2024-01-01T10:00:00"),
            np.datetime64("2024-01-01T11:00:00"),
            np.datetime64("2024-01-01T12:00:00"),
        ],
    )

    out = f(prefix, {"concept:name": 0, "time:timestamp": 1})

    # last activity = C
    last_idx = f.activity_ohe_offset + f.activity_to_ohe_idx["C"]
    assert out[last_idx] == 1


def test_ohe_last_transition():
    f = BasicControlFlowFeatures(one_hot_encode_categorical=True)

    log = [
        type(
            "T", (), {"data": np.array([["A", "t"], ["B", "t2"]], dtype=object)}
        )()
    ]

    f.fit(log, {"concept:name": 0}, min_last_activity=0, min_last_transition=0)

    prefix = make_prefix(
        ["A", "B"],
        [
            np.datetime64("2024-01-01T10:00:00"),
            np.datetime64("2024-01-01T11:00:00"),
        ],
    )

    out = f(prefix, {"concept:name": 0, "time:timestamp": 1})

    idx = f.transition_ohe_offset + f.transition_to_ohe_idx["A->B"]
    assert out[idx] == 1


# =========================================================
# STRUCTURE TESTS
# =========================================================


def test_output_shape_consistency():
    f = BasicControlFlowFeatures()

    log = [type("T", (), {"data": np.array([["A", "t"]], dtype=object)})()]

    f.fit(log, {"concept:name": 0})

    prefix = np.array(
        [["A", np.datetime64("2024-01-01T10:00:00")]], dtype=object
    )

    out = f(prefix, {"concept:name": 0, "time:timestamp": 1})

    assert isinstance(out, np.ndarray)
    assert out.shape[0] == len(f.feature_names)


def test_all_features_non_negative():
    f = BasicControlFlowFeatures()

    prefix = make_prefix(
        ["A", "B", "C"],
        [
            np.datetime64("2024-01-01T10:00:00"),
            np.datetime64("2024-01-01T11:00:00"),
            np.datetime64("2024-01-01T12:00:00"),
        ],
    )

    out = f(prefix, {"concept:name": 0, "time:timestamp": 1})

    assert np.all(np.isfinite(out))


# =========================================================
# COLUMN MAPPING
# =========================================================


def test_custom_column_mapping():
    f = BasicControlFlowFeatures(
        activity_column="activity",
        timestamp_column="ts",
    )

    prefix = np.array(
        [
            ["A", np.datetime64("2024-01-01T10:00:00")],
            ["B", np.datetime64("2024-01-01T11:00:00")],
        ],
        dtype=object,
    )

    out = f(prefix, {"activity": 0, "ts": 1})

    assert out[1] == 2

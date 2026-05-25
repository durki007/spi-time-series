import numpy as np
import pandas as pd
import pytest

from spi_time_series.features.time_series_features import (
    InterEventTimeStatsFeature,
    LastDeltaFeature,
    LastValueFeature,
    MeanDiffFeature,
    MeanValueFeature,
    PrefixLengthFeature,
    RatioFeature,
    SlopeFeature,
    StdValueFeature,
    UniqueCountFeature,
)


def _sample_prefix():
    columns = [
        "time:timestamp",
        "OfferedAmount",
        "case:RequestedAmount",
        "CreditScore",
        "OfferID",
    ]
    data = [
        [pd.Timestamp("2020-01-02T00:00:00Z"), 200.0, 1000.0, 600.0, "O2"],
        [pd.Timestamp("2020-01-01T00:00:00Z"), 100.0, 1000.0, 650.0, "O1"],
        [pd.Timestamp("2020-01-03T00:00:00Z"), np.nan, 1000.0, np.nan, None],
    ]
    prefix = np.array(data, dtype=object)
    col_idx = {c: i for i, c in enumerate(columns)}
    return prefix, col_idx


def _trend_prefix():
    columns = ["time:timestamp", "OfferedAmount"]
    data = [
        [pd.Timestamp("2020-01-01T00:00:00Z"), 10.0],
        [pd.Timestamp("2020-01-01T01:00:00Z"), 20.0],
        [pd.Timestamp("2020-01-01T02:00:00Z"), 40.0],
    ]
    prefix = np.array(data, dtype=object)
    col_idx = {c: i for i, c in enumerate(columns)}
    return prefix, col_idx


def test_last_value_feature_uses_timestamp_order():
    prefix, col_idx = _sample_prefix()
    feature = LastValueFeature(columns=["OfferedAmount"])

    out = feature(prefix, col_idx)

    assert out["OfferedAmount_last"] == 200.0


def test_mean_value_feature_ignores_nan():
    prefix, col_idx = _sample_prefix()
    feature = MeanValueFeature(columns=["OfferedAmount"])

    out = feature(prefix, col_idx)

    assert out["OfferedAmount_mean"] == 150.0


def test_std_value_feature_sample_std():
    prefix, col_idx = _sample_prefix()
    feature = StdValueFeature(columns=["OfferedAmount"])

    out = feature(prefix, col_idx)

    assert out["OfferedAmount_std"] == pytest.approx(70.710678, rel=1e-6)


def test_ratio_feature_uses_last_values():
    prefix, col_idx = _sample_prefix()
    feature = RatioFeature(
        numerator_col="case:RequestedAmount",
        denominator_col="OfferedAmount",
        suffix="last",
    )

    out = feature(prefix, col_idx)

    assert out["case:RequestedAmount_to_OfferedAmount_last"] == 5.0


def test_unique_count_feature_counts_unique_offer_ids():
    prefix, col_idx = _sample_prefix()
    feature = UniqueCountFeature(column="OfferID")

    out = feature(prefix, col_idx)

    assert out["OfferID_unique_count"] == 2.0


def test_prefix_length_feature_counts_rows():
    prefix, col_idx = _sample_prefix()
    feature = PrefixLengthFeature()

    out = feature(prefix, col_idx)

    assert out["prefix_length"] == 3.0


def test_mean_diff_feature():
    prefix, col_idx = _trend_prefix()
    feature = MeanDiffFeature(columns=["OfferedAmount"])

    out = feature(prefix, col_idx)

    assert out["OfferedAmount_diff_mean"] == 15.0


def test_last_delta_feature():
    prefix, col_idx = _trend_prefix()
    feature = LastDeltaFeature(columns=["OfferedAmount"])

    out = feature(prefix, col_idx)

    assert out["OfferedAmount_last_delta"] == 20.0


def test_slope_feature_per_hour():
    prefix, col_idx = _trend_prefix()
    feature = SlopeFeature(columns=["OfferedAmount"])

    out = feature(prefix, col_idx)

    assert out["OfferedAmount_slope_per_hour"] == pytest.approx(15.0)


def test_inter_event_time_stats():
    prefix, col_idx = _trend_prefix()
    feature = InterEventTimeStatsFeature()

    out = feature(prefix, col_idx)

    assert out["inter_event_time_mean_hours"] == pytest.approx(1.0)
    assert out["inter_event_time_std_hours"] == pytest.approx(0.0)

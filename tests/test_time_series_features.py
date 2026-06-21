import pandas as pd

from spi_time_series.features.time_series_features import (
    align_features_to_prefixes,
    extract_time_series_features,
)


def test_extract_time_series_features_builds_columns_and_resets_index():
    time_series_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2023-01-01 00:00:00+00:00", periods=8, freq="1h"
            ),
            "active_cases": [1, 2, 3, 4, 3, 2, 1, 0],
        }
    )

    result = extract_time_series_features(time_series_df)

    expected_columns = {
        "timestamp",
        "active_cases",
        "active_cases__window_mean_8h",
        "active_cases__trend_8h",
    }

    assert expected_columns.issubset(result.columns)
    assert result["timestamp"].dtype == "datetime64[ns, UTC]"
    assert result.isna().sum().sum() == 0


def test_align_features_to_prefixes_merges_backward():
    prefix_df = pd.DataFrame(
        {
            "case_id": ["A", "B"],
            "prefix_length": [2, 3],
            "last_event_timestamp": pd.to_datetime(
                ["2023-01-01 02:30:00+00:00", "2023-01-01 05:00:00+00:00"]
            ),
        }
    )

    feature_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2023-01-01 02:00:00+00:00", "2023-01-01 04:00:00+00:00"]
            ),
            "active_cases": [10, 20],
        }
    )

    merged = align_features_to_prefixes(prefix_df, feature_df)

    assert "timestamp" not in merged.columns
    assert merged["active_cases"].tolist() == [10, 20]

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
        # rolling windows for multiple scales
        "active_cases__window_mean_1h",
        "active_cases__window_max_1h",
        "active_cases__window_std_1h",
        "active_cases__window_mean_6h",
        "active_cases__window_max_6h",
        "active_cases__window_std_6h",
        "active_cases__window_mean_12h",
        "active_cases__window_max_12h",
        "active_cases__window_std_12h",
        "active_cases__window_mean_24h",
        "active_cases__window_max_24h",
        "active_cases__window_std_24h",
        # lags
        "active_cases__lag_1h",
        "active_cases__lag_6h",
        # calendar context
        "time_series_hour",
        "time_series_dayofweek",
        "time_series_is_weekend",
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

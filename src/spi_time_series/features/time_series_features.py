import pandas as pd


def extract_time_series_features(time_series_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling window statistics for active case counts.

    Expects a dataframe containing an `active_cases` column and either a
    `timestamp` column or a DatetimeIndex. Returns the enriched dataframe
    with `timestamp` as a column.
    """
    if "active_cases" not in time_series_df.columns:
        raise KeyError("Missing required column: active_cases")

    df = time_series_df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" not in df.columns:
            raise KeyError("Missing required column: timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index)

    df = df.sort_index()

    df["time_series_hour"] = df.index.hour
    df["time_series_dayofweek"] = df.index.dayofweek
    df["time_series_is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    for window in ("1h", "6h", "12h", "24h"):
        rolling_window = df["active_cases"].rolling(window)
        df[f"active_cases__window_mean_{window}"] = rolling_window.mean()
        df[f"active_cases__window_max_{window}"] = rolling_window.max()
        df[f"active_cases__window_std_{window}"] = rolling_window.std()

    df["active_cases__lag_1h"] = df["active_cases"].shift(1)
    df["active_cases__lag_6h"] = df["active_cases"].shift(6)

    df = df.bfill()
    # replace any remaining NaNs (e.g., std over a single-sample window) with 0
    df = df.fillna(0)

    return df.reset_index()


def align_features_to_prefixes(
    prefix_df: pd.DataFrame, feature_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Sort prefix and feature dataframes and merge time-series features.
    """
    if "last_event_timestamp" not in prefix_df.columns:
        raise KeyError("Missing required column: last_event_timestamp")
    if "timestamp" not in feature_df.columns:
        raise KeyError("Missing required column: timestamp")

    sorted_prefix_df = prefix_df.sort_values(
        "last_event_timestamp"
    ).reset_index(drop=True)
    sorted_feature_df = feature_df.sort_values("timestamp").reset_index(
        drop=True
    )

    merged_df = pd.merge_asof(
        sorted_prefix_df,
        sorted_feature_df,
        left_on="last_event_timestamp",
        right_on="timestamp",
        direction="backward",
    )

    return merged_df.drop(columns=["timestamp"])

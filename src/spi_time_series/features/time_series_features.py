from collections.abc import Iterable

import numpy as np
import pandas as pd

from spi_time_series.data.schemas import TraceSample


class ActiveCaseCountFeature:
    """
    Feature extractor that computes the number of active cases at the time of the last event in the prefix.
    """

    time_series: pd.DataFrame
    cleaned_event_log: pd.DataFrame
    featured_df: pd.DataFrame

    def __init__(self):
        self.feature_names = ["active_cases"]

    def name(self) -> str:
        return "active_cases"

    def fit(
        self,
        event_log: Iterable[TraceSample],
        col_idx_mapping: dict[str, int],
        **config_kwargs,
    ):
        cleaned_log = config_kwargs.get("cleaned_log")
        if cleaned_log is None:
            raise ValueError(
                "ActiveCaseCountFeature.fit() requires 'cleaned_log' kwarg "
                "(a cleaned event log DataFrame)."
            )
        self.cleaned_event_log = cleaned_log
        self.time_series = self._preprocess_time_series()
        self.featured_df = extract_time_series_features(self.time_series)
        self.featured_df = self.featured_df.sort_values(
            "timestamp"
        ).reset_index(drop=True)
        self.feature_names = [
            col for col in self.featured_df.columns if col != "timestamp"
        ]
        self._ts_array = self.featured_df["timestamp"].to_numpy()
        self._features_array = self.featured_df[self.feature_names].to_numpy(
            dtype=np.float32
        )

    def _preprocess_time_series(self) -> pd.DataFrame:
        case_times = (
            self.cleaned_event_log.groupby("case:concept:name")[
                "time:timestamp"
            ]
            .agg(["min", "max"])
            .reset_index()
            .rename(
                columns={
                    "case:concept:name": "case_id",
                    "min": "start_time",
                    "max": "end_time",
                }
            )
        )
        return preprocess_time_series(case_times)

    def __call__(
        self, prefix: np.ndarray, col_idx_mapping: dict[str, int]
    ) -> np.ndarray:
        ts = pd.to_datetime(prefix[-1, col_idx_mapping["time:timestamp"]])

        idx = np.searchsorted(self._ts_array, ts, side="right") - 1
        if idx < 0:
            idx = 0
        return self._features_array[idx]


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
    df["active_cases__trend_1h"] = (
        df["active_cases"] - df["active_cases__lag_1h"]
    )
    df["active_cases__lag_6h"] = df["active_cases"].shift(6)
    df["active_cases__trend_6h"] = (
        df["active_cases"] - df["active_cases__lag_6h"]
    )
    df["active_cases__lag_12h"] = df["active_cases"].shift(12)
    df["active_cases__trend_12h"] = (
        df["active_cases"] - df["active_cases__lag_12h"]
    )
    df["active_cases__lag_24h"] = df["active_cases"].shift(24)
    df["active_cases__trend_24h"] = (
        df["active_cases"] - df["active_cases__lag_24h"]
    )

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


def preprocess_time_series(event_log_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a continuous hourly timestamp grid for a time-series event log.

    The grid spans the full log from the minimum start time to the maximum
    end time and uses a 1-hour frequency. The returned frame includes
    active case counts for each timestamp.
    """
    required_columns = {"case_id", "start_time", "end_time"}
    missing_columns = required_columns.difference(event_log_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"Missing required columns: {missing}")

    if event_log_df.empty:
        return pd.DataFrame(
            {
                "timestamp": pd.DatetimeIndex([], name="timestamp"),
                "active_cases": pd.Series(dtype="int64"),
            }
        )

    start_times = pd.to_datetime(event_log_df["start_time"])
    end_times = pd.to_datetime(event_log_df["end_time"])

    min_start_time = start_times.min()
    max_end_time = end_times.max()

    timestamp_grid = pd.date_range(
        start=min_start_time, end=max_end_time, freq="1h"
    )

    events = pd.concat(
        [
            pd.Series(1, index=start_times, dtype="int64"),
            pd.Series(-1, index=end_times, dtype="int64"),
        ]
    )

    events = events.sort_index()
    active_cases = events.cumsum()

    active_cases = active_cases[~active_cases.index.duplicated(keep="last")]

    active_df = pd.DataFrame(
        {"timestamp": active_cases.index, "active_cases": active_cases.values}
    )
    grid_df = pd.DataFrame({"timestamp": timestamp_grid})

    time_series = pd.merge_asof(
        grid_df,
        active_df,
        on="timestamp",
        direction="backward",
    )

    time_series["active_cases"] = (
        time_series["active_cases"].fillna(0).astype(int)
    )

    return time_series

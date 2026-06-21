import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd

from spi_time_series.data.schemas import TraceSample

logger = logging.getLogger(__name__)


class ActiveCaseCountFeature:
    time_series: pd.DataFrame
    cleaned_event_log: pd.DataFrame
    featured_df: pd.DataFrame
    _ts_array: np.ndarray
    _features_array: np.ndarray

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

        if self._ts_array.size == 0:
            return np.zeros(len(self.feature_names), dtype=np.float32)
        idx = np.searchsorted(self._ts_array, ts, side="right") - 1
        if idx < 0:
            idx = 0
        result: np.ndarray = self._features_array[idx]
        return result


def _fill_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    full_grid = pd.date_range(
        df["timestamp"].min(), df["timestamp"].max(), freq="1h"
    )
    value_cols = [c for c in df.columns if c != "timestamp"]
    df = df.set_index("timestamp").reindex(full_grid).reset_index()
    df = df.rename(columns={"index": "timestamp"})
    for col in value_cols:
        df[col] = df[col].fillna(0)
    return df


def extract_time_series_features(time_series_df: pd.DataFrame) -> pd.DataFrame:
    df = time_series_df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index)

    df = df.sort_index()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        df[f"{col}__window_mean_8h"] = df[col].rolling("8h").mean()
        df[f"{col}__trend_8h"] = df[col] - df[col].shift(8)

    df = df.bfill()
    df = df.fillna(0)

    return df.reset_index()


def align_features_to_prefixes(
    prefix_df: pd.DataFrame, feature_df: pd.DataFrame
) -> pd.DataFrame:
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
    required_columns = {"case_id", "start_time", "end_time"}
    missing_columns = required_columns.difference(event_log_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"Missing required columns: {missing}")

    if event_log_df.empty:
        logger.info("Empty event log — returning empty time series")
        return pd.DataFrame(
            {
                "timestamp": pd.DatetimeIndex([], name="timestamp"),
                "active_cases": pd.Series(dtype="int64"),
            }
        )

    logger.info("Preprocessing time series for %d cases", len(event_log_df))

    start_times = pd.to_datetime(event_log_df["start_time"])
    end_times = pd.to_datetime(event_log_df["end_time"])

    min_start_time = start_times.min()
    max_end_time = end_times.max()

    span_hours = (max_end_time - min_start_time).total_seconds() / 3600.0
    if span_hours > 876_000:
        raise ValueError(
            f"Timestamp span of {span_hours:.0f} hours exceeds maximum "
            f"allowed range (876,000 hours). "
            f"Check for corrupted timestamps (min={min_start_time}, "
            f"max={max_end_time})."
        )

    logger.debug("Time range: %s → %s", min_start_time, max_end_time)

    timestamp_grid = pd.date_range(
        start=min_start_time, end=max_end_time, freq="1h"
    )

    logger.debug("Built timestamp grid: %d hourly steps", len(timestamp_grid))

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

    logger.info(
        "Finished time series preprocessing: %d rows, max active cases = %d",
        len(time_series),
        time_series["active_cases"].max(),
    )

    return time_series


class _BaseGlobalTSFeature:
    _ts_array: np.ndarray
    _features_array: np.ndarray
    feature_names: list[str]

    def fit(
        self,
        event_log: Iterable[TraceSample],
        col_idx_mapping: dict[str, int],
        **config_kwargs,
    ):
        cleaned_log = config_kwargs.get("cleaned_log")
        if cleaned_log is None:
            raise ValueError(
                f"{type(self).__name__}.fit() requires 'cleaned_log' kwarg."
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

    def __call__(
        self, prefix: np.ndarray, col_idx_mapping: dict[str, int]
    ) -> np.ndarray:
        ts = pd.to_datetime(prefix[-1, col_idx_mapping["time:timestamp"]])
        if self._ts_array.size == 0:
            return np.zeros(len(self.feature_names), dtype=np.float32)
        idx = np.searchsorted(self._ts_array, ts, side="right") - 1
        if idx < 0:
            idx = 0
        return self._features_array[idx]

    def _preprocess_time_series(self) -> pd.DataFrame:
        raise NotImplementedError


class FinancialVolumeFeature(_BaseGlobalTSFeature):
    def name(self) -> str:
        return "financial_volume"

    def _preprocess_time_series(self) -> pd.DataFrame:
        df = self.cleaned_event_log.copy()
        df["hour"] = df["time:timestamp"].dt.floor("h")
        result = (
            df.groupby("hour")
            .agg(
                total_withdrawal_amount=("FirstWithdrawalAmount", "sum"),
                mean_withdrawal_amount=("FirstWithdrawalAmount", "mean"),
                total_offer_amount=("OfferedAmount", "sum"),
                mean_offer_amount=("OfferedAmount", "mean"),
                total_monthly_cost=("MonthlyCost", "sum"),
                mean_monthly_cost=("MonthlyCost", "mean"),
            )
            .reset_index()
        )
        result = result.rename(columns={"hour": "timestamp"})
        return _fill_hourly_grid(result)


class DecisionRateFeature(_BaseGlobalTSFeature):
    def name(self) -> str:
        return "decision_rate"

    def _preprocess_time_series(self) -> pd.DataFrame:
        df = self.cleaned_event_log.copy()
        df["hour"] = df["time:timestamp"].dt.floor("h")
        df["is_accepted"] = df["concept:name"] == "A_Accepted"
        df["is_denied"] = df["concept:name"] == "A_Denied"
        df["is_cancelled"] = df["concept:name"].isin(
            ["A_Cancelled", "O_Cancelled"]
        )
        df["is_pending"] = df["concept:name"] == "A_Pending"

        result = (
            df.groupby("hour")
            .agg(
                num_accepted=("is_accepted", "sum"),
                num_denied=("is_denied", "sum"),
                num_cancelled=("is_cancelled", "sum"),
                num_pending=("is_pending", "sum"),
            )
            .reset_index()
        )

        total = result[
            ["num_accepted", "num_denied", "num_cancelled", "num_pending"]
        ].sum(axis=1)
        result["accept_ratio"] = result["num_accepted"] / total.replace(
            0, np.nan
        )
        result["accept_ratio"] = result["accept_ratio"].fillna(0)
        result = result.rename(columns={"hour": "timestamp"})
        return _fill_hourly_grid(result[["timestamp", "accept_ratio"]])

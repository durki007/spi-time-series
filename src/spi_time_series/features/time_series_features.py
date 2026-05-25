from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from spi_time_series.data.schemas import PrefixFeature


def _align_prefix_by_timestamp(
    prefix: np.ndarray,
    col_idx_mapping: dict[str, int],
    time_col: str,
) -> Any:
    if time_col not in col_idx_mapping:
        return prefix

    if prefix.size == 0:
        return prefix

    time_idx = col_idx_mapping[time_col]
    times = pd.to_datetime(prefix[:, time_idx], utc=True, errors="coerce")

    if times.isna().all():
        return prefix

    mask = ~times.isna()
    prefix = prefix[mask]
    if prefix.size == 0:
        return prefix

    order = np.argsort(times[mask].values)
    return prefix[order]


def _get_numeric_values(
    prefix: np.ndarray,
    col_idx_mapping: dict[str, int],
    col_name: str,
) -> Any:
    if col_name not in col_idx_mapping:
        return None

    idx = col_idx_mapping[col_name]
    series = pd.Series(prefix[:, idx])
    values = pd.to_numeric(series, errors="coerce")
    return values.to_numpy(dtype=float, na_value=np.nan)


def _last_non_nan(values: np.ndarray | None) -> float:
    if values is None:
        return float("nan")
    not_nan = values[~np.isnan(values)]
    if not_nan.size == 0:
        return float("nan")
    return float(not_nan[-1])


def _nanmean(values: np.ndarray | None) -> float:
    if values is None or np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def _nanstd(values: np.ndarray | None) -> float:
    if values is None:
        return float("nan")
    count = np.sum(~np.isnan(values))
    if count < 2:
        return float("nan")
    return float(np.nanstd(values, ddof=1))


def _get_time_hours(
    prefix: np.ndarray,
    col_idx_mapping: dict[str, int],
    time_col: str,
) -> Any:
    if time_col not in col_idx_mapping:
        return None

    idx = col_idx_mapping[time_col]
    times = pd.to_datetime(prefix[:, idx], utc=True, errors="coerce")
    if times.isna().all():
        return None

    times = times[~times.isna()]
    if len(times) == 0:
        return None

    base = times[0]
    hours = (times - base) / pd.Timedelta(hours=1)
    return hours.to_numpy(dtype=float)


@dataclass(frozen=True)
class LastValueFeature:
    columns: list[str]
    time_col: str = "time:timestamp"

    def name(self) -> str:
        return "ts_last"

    def __call__(
        self, prefix: np.ndarray, col_idx_mapping: dict[str, int]
    ) -> pd.Series:
        aligned = _align_prefix_by_timestamp(
            prefix, col_idx_mapping, self.time_col
        )
        out: dict[str, float] = {}
        for col in self.columns:
            values = _get_numeric_values(aligned, col_idx_mapping, col)
            out[f"{col}_last"] = _last_non_nan(values)
        return pd.Series(out)


@dataclass(frozen=True)
class MeanValueFeature:
    columns: list[str]
    time_col: str = "time:timestamp"

    def name(self) -> str:
        return "ts_mean"

    def __call__(
        self, prefix: np.ndarray, col_idx_mapping: dict[str, int]
    ) -> pd.Series:
        aligned = _align_prefix_by_timestamp(
            prefix, col_idx_mapping, self.time_col
        )
        out: dict[str, float] = {}
        for col in self.columns:
            values = _get_numeric_values(aligned, col_idx_mapping, col)
            out[f"{col}_mean"] = _nanmean(values)
        return pd.Series(out)


@dataclass(frozen=True)
class StdValueFeature:
    columns: list[str]
    time_col: str = "time:timestamp"

    def name(self) -> str:
        return "ts_std"

    def __call__(
        self, prefix: np.ndarray, col_idx_mapping: dict[str, int]
    ) -> pd.Series:
        aligned = _align_prefix_by_timestamp(
            prefix, col_idx_mapping, self.time_col
        )
        out: dict[str, float] = {}
        for col in self.columns:
            values = _get_numeric_values(aligned, col_idx_mapping, col)
            out[f"{col}_std"] = _nanstd(values)
        return pd.Series(out)


@dataclass(frozen=True)
class PrefixLengthFeature:
    time_col: str = "time:timestamp"

    def name(self) -> str:
        return "ts_meta"

    def __call__(
        self, prefix: np.ndarray, col_idx_mapping: dict[str, int]
    ) -> pd.Series:
        aligned = _align_prefix_by_timestamp(
            prefix, col_idx_mapping, self.time_col
        )
        return pd.Series({"prefix_length": float(len(aligned))})


@dataclass(frozen=True)
class MeanDiffFeature:
    columns: list[str]
    time_col: str = "time:timestamp"

    def name(self) -> str:
        return "ts_diff_mean"

    def __call__(
        self, prefix: np.ndarray, col_idx_mapping: dict[str, int]
    ) -> pd.Series:
        aligned = _align_prefix_by_timestamp(
            prefix, col_idx_mapping, self.time_col
        )
        out: dict[str, float] = {}
        for col in self.columns:
            values = _get_numeric_values(aligned, col_idx_mapping, col)
            if values is None:
                out[f"{col}_diff_mean"] = float("nan")
                continue
            values = values[~np.isnan(values)]
            if values.size < 2:
                out[f"{col}_diff_mean"] = float("nan")
                continue
            out[f"{col}_diff_mean"] = float(np.mean(np.diff(values)))
        return pd.Series(out)


@dataclass(frozen=True)
class LastDeltaFeature:
    columns: list[str]
    time_col: str = "time:timestamp"

    def name(self) -> str:
        return "ts_last_delta"

    def __call__(
        self, prefix: np.ndarray, col_idx_mapping: dict[str, int]
    ) -> pd.Series:
        aligned = _align_prefix_by_timestamp(
            prefix, col_idx_mapping, self.time_col
        )
        out: dict[str, float] = {}
        for col in self.columns:
            values = _get_numeric_values(aligned, col_idx_mapping, col)
            if values is None:
                out[f"{col}_last_delta"] = float("nan")
                continue
            values = values[~np.isnan(values)]
            if values.size < 2:
                out[f"{col}_last_delta"] = float("nan")
                continue
            out[f"{col}_last_delta"] = float(values[-1] - values[-2])
        return pd.Series(out)


@dataclass(frozen=True)
class SlopeFeature:
    columns: list[str]
    time_col: str = "time:timestamp"

    def name(self) -> str:
        return "ts_slope"

    def __call__(
        self, prefix: np.ndarray, col_idx_mapping: dict[str, int]
    ) -> pd.Series:
        aligned = _align_prefix_by_timestamp(
            prefix, col_idx_mapping, self.time_col
        )
        time_hours = _get_time_hours(aligned, col_idx_mapping, self.time_col)
        out: dict[str, float] = {}
        for col in self.columns:
            values = _get_numeric_values(aligned, col_idx_mapping, col)
            if values is None or time_hours is None:
                out[f"{col}_slope_per_hour"] = float("nan")
                continue
            mask = ~np.isnan(values)
            if np.sum(mask) < 2:
                out[f"{col}_slope_per_hour"] = float("nan")
                continue
            x = time_hours[mask]
            y = values[mask]
            x_mean = float(np.mean(x))
            y_mean = float(np.mean(y))
            denom = float(np.sum((x - x_mean) ** 2))
            if denom == 0:
                out[f"{col}_slope_per_hour"] = float("nan")
                continue
            slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
            out[f"{col}_slope_per_hour"] = slope
        return pd.Series(out)


@dataclass(frozen=True)
class InterEventTimeStatsFeature:
    time_col: str = "time:timestamp"

    def name(self) -> str:
        return "ts_inter_event"

    def __call__(
        self, prefix: np.ndarray, col_idx_mapping: dict[str, int]
    ) -> pd.Series:
        aligned = _align_prefix_by_timestamp(
            prefix, col_idx_mapping, self.time_col
        )
        time_hours = _get_time_hours(aligned, col_idx_mapping, self.time_col)
        if time_hours is None or time_hours.size < 2:
            return pd.Series(
                {
                    "inter_event_time_mean_hours": float("nan"),
                    "inter_event_time_std_hours": float("nan"),
                }
            )
        diffs = np.diff(time_hours)
        mean_val = float(np.mean(diffs)) if diffs.size > 0 else float("nan")
        std_val = (
            float(np.std(diffs, ddof=1)) if diffs.size > 1 else float("nan")
        )
        return pd.Series(
            {
                "inter_event_time_mean_hours": mean_val,
                "inter_event_time_std_hours": std_val,
            }
        )


@dataclass(frozen=True)
class RatioFeature:
    numerator_col: str
    denominator_col: str
    time_col: str = "time:timestamp"
    suffix: str = "ratio"

    def name(self) -> str:
        return "ts_ratio"

    def __call__(
        self, prefix: np.ndarray, col_idx_mapping: dict[str, int]
    ) -> pd.Series:
        aligned = _align_prefix_by_timestamp(
            prefix, col_idx_mapping, self.time_col
        )
        numerator = _get_numeric_values(
            aligned, col_idx_mapping, self.numerator_col
        )
        denominator = _get_numeric_values(
            aligned, col_idx_mapping, self.denominator_col
        )
        num_val = _last_non_nan(numerator)
        denom_val = _last_non_nan(denominator)
        if np.isnan(num_val) or np.isnan(denom_val) or denom_val == 0:
            ratio = float("nan")
        else:
            ratio = float(num_val / denom_val)
        key = f"{self.numerator_col}_to_{self.denominator_col}_{self.suffix}"
        return pd.Series({key: ratio})


@dataclass(frozen=True)
class UniqueCountFeature:
    column: str
    time_col: str = "time:timestamp"

    def name(self) -> str:
        return "ts_count"

    def __call__(
        self, prefix: np.ndarray, col_idx_mapping: dict[str, int]
    ) -> pd.Series:
        aligned = _align_prefix_by_timestamp(
            prefix, col_idx_mapping, self.time_col
        )
        if self.column not in col_idx_mapping:
            return pd.Series({f"{self.column}_unique_count": float("nan")})
        idx = col_idx_mapping[self.column]
        series = pd.Series(aligned[:, idx]).dropna()
        return pd.Series(
            {f"{self.column}_unique_count": float(series.nunique())}
        )


def default_time_series_features(
    numeric_columns: Iterable[str] | None = None,
    *,
    time_col: str = "time:timestamp",
    requested_col: str = "case:RequestedAmount",
    offered_col: str = "OfferedAmount",
    credit_col: str = "CreditScore",
    offer_id_col: str = "OfferID",
) -> list[PrefixFeature]:
    if numeric_columns is None:
        numeric_columns = [offered_col, requested_col, credit_col]
    columns = list(numeric_columns)

    return [
        PrefixLengthFeature(time_col=time_col),
        LastValueFeature(columns=columns, time_col=time_col),
        MeanValueFeature(columns=columns, time_col=time_col),
        StdValueFeature(columns=columns, time_col=time_col),
        MeanDiffFeature(columns=columns, time_col=time_col),
        LastDeltaFeature(columns=columns, time_col=time_col),
        SlopeFeature(columns=columns, time_col=time_col),
        InterEventTimeStatsFeature(time_col=time_col),
        RatioFeature(
            numerator_col=requested_col,
            denominator_col=offered_col,
            time_col=time_col,
            suffix="last",
        ),
        UniqueCountFeature(column=offer_id_col, time_col=time_col),
    ]

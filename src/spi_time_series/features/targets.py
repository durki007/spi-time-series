import numpy as np

from spi_time_series.data.constants import OUTCOME_EVENTS


def remaining_time_target(
    trace: np.ndarray,
    start_idx: int,
    end_idx: int,
    col_idx_mapping: dict[str, int],
) -> float:
    current_time = trace[end_idx - 1, col_idx_mapping["time:timestamp"]]
    completion_time = trace[-1, col_idx_mapping["time:timestamp"]]
    remaining_hours = (completion_time - current_time) / np.timedelta64(1, "h")
    return float(remaining_hours)


def _outcome_label(
    trace: np.ndarray,
    col_idx_mapping: dict[str, int],
    positive_set: set[str],
    negative_set: set[str],
) -> int:
    activities = set(trace[:, col_idx_mapping["concept:name"]])
    for activity in positive_set:
        if activity in activities:
            return 1
    for activity in negative_set:
        if activity in activities:
            return 0
    raise ValueError(
        f"Could not determine outcome for case. Activities: {activities}"
    )


def outcome_target(
    trace: np.ndarray,
    start_idx: int,
    end_idx: int,
    col_idx_mapping: dict[str, int],
) -> int:
    activities = set(trace[:, col_idx_mapping["concept:name"]])
    for class_id, activity in enumerate(OUTCOME_EVENTS):
        if activity in activities:
            return class_id
    raise ValueError(
        f"Could not determine outcome for case. Activities: {activities}"
    )


def binary_outcome_target(
    trace: np.ndarray,
    start_idx: int,
    end_idx: int,
    col_idx_mapping: dict[str, int],
) -> int:
    return _outcome_label(
        trace,
        col_idx_mapping,
        positive_set={"A_Cancelled", "A_Denied"},
        negative_set={"A_Pending"},
    )


CLASSIFICATION_TARGETS = {
    "3class": outcome_target,
    "2class": binary_outcome_target,
}

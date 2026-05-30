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

    remaining_hours = (completion_time - current_time).total_seconds() / 3600.0

    return float(remaining_hours)


def outcome_target(
    trace: np.ndarray,
    start_idx: int,
    end_idx: int,
    col_idx_mapping: dict[str, int],
) -> float:
    """Predict final case outcome: successful, denied, or cancelled."""
    activities = set(trace[:, col_idx_mapping["concept:name"]])
    for class_id, activity in enumerate(OUTCOME_EVENTS):
        if activity in activities:
            return class_id

    raise ValueError(
        f"Could not determine outcome for case. Activities: {(activities)}"
    )

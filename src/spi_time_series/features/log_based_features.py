import numpy as np
import pandas as pd

from spi_time_series.data.constants import EVENT_NAMES


class BasicControlFlowFeatures:
    """
    Collection of common control-flow features for predictive process mining.

    Features:
    - elapsed_time_hours
    - prefix_length
    - last_activity
    - bag-of-activities counts
    """

    def __init__(
        self,
        activity_column: str = "concept:name",
        timestamp_column: str = "time:timestamp",
    ):
        self.activity_column = activity_column
        self.timestamp_column = timestamp_column

    def name(self):
        return "BasicControlFlowFeatures"

    def __call__(
        self, prefix: np.ndarray, col_idx_mapping: dict[str, int]
    ) -> pd.Series:
        features = {}
        timestamp_idx = col_idx_mapping[self.timestamp_column]
        activity_idx = col_idx_mapping[self.activity_column]

        # ---------------------------------------------------------
        # ELAPSED TIME
        # ---------------------------------------------------------

        if len(prefix) > 1:
            total_seconds = (
                prefix[:, timestamp_idx].max() - prefix[:, timestamp_idx].min()
            ).total_seconds()

            features["elapsed_time_hours"] = total_seconds / (60 * 60)
        else:
            features["elapsed_time_hours"] = 0.0

        # ---------------------------------------------------------
        # PREFIX LENGTH
        # ---------------------------------------------------------

        features["prefix_length"] = prefix.shape[0]

        # ---------------------------------------------------------
        # LAST ACTIVITY
        # ---------------------------------------------------------

        if len(prefix) > 0:
            features["last_activity"] = prefix[-1][activity_idx]
        else:
            features["last_activity"] = None

        # ---------------------------------------------------------
        # TIME SINCE LAST EVENT
        # ---------------------------------------------------------

        if len(prefix) > 1:
            delta_seconds = (
                prefix[-1][timestamp_idx] - prefix[-2][timestamp_idx]
            ).total_seconds()

            features["time_since_last_event_hours"] = delta_seconds / (60 * 60)
        else:
            features["time_since_last_event_hours"] = 0.0

        # ---------------------------------------------------------
        # REWORK COUNT
        # ---------------------------------------------------------

        unique_activities = len(np.unique(prefix[:, activity_idx]))

        features["rework_count"] = prefix.shape[0] - unique_activities

        # ---------------------------------------------------------
        # LAST TRANSITION
        # ---------------------------------------------------------

        if len(prefix) >= 2:
            prev_activity = prefix[-2][activity_idx]
            last_activity = prefix[-1][activity_idx]

            features["last_transition"] = f"{prev_activity}->{last_activity}"
        else:
            features["last_transition"] = None

        # ---------------------------------------------------------
        # BAG-OF-ACTIVITIES
        # ---------------------------------------------------------

        counts = {event: 0 for event in EVENT_NAMES}  # set defaults
        counts.update(
            dict(
                zip(
                    *np.unique(prefix[:, activity_idx], return_counts=True),
                    strict=True,
                )
            )
        )  # update with actual counts
        counts = {
            f"count_{k}": v for k, v in counts.items()
        }  # add prefix to feature names

        features.update(counts)

        return pd.Series(features)

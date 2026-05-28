from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np

from spi_time_series.data.constants import EVENT_NAMES
from spi_time_series.data.schemas import PrefixFeature, TraceSample


def _hours(t2, t1):
    return (t2 - t1) / np.timedelta64(1, "h")


class BasicControlFlowFeatures(PrefixFeature):
    """
    Collection of common control-flow features for predictive process mining.
    """

    def __init__(
        self,
        activity_column: str = "concept:name",
        timestamp_column: str = "time:timestamp",
        one_hot_encode_categorical: bool = False,
    ):
        self.activity_column = activity_column
        self.timestamp_column = timestamp_column
        self.one_hot_encode_categorical = one_hot_encode_categorical

        # OHE mappings
        self.activity_to_ohe_idx: dict[str, int] = {}
        self.transition_to_ohe_idx: dict[str, int] = {}

        # fixed feature names/order
        self.feature_names: list[str] = []

        # feature offsets
        self.base_feature_count = 0
        self.activity_ohe_offset = 0
        self.transition_ohe_offset = 0

        # feature names
        self.feature_names = [
            "elapsed_time_hours",
            "prefix_length",
            "time_since_last_event_hours",
            "rework_count",
        ]

        # bag-of-activities
        self.feature_names.extend(f"count_{event}" for event in EVENT_NAMES)
        self.base_feature_count = len(self.feature_names)

    # ---------------------------------------------------------
    # METADATA
    # ---------------------------------------------------------

    def name(self):
        return "BasicControlFlowFeatures"

    # ---------------------------------------------------------
    # FIT
    # ---------------------------------------------------------

    def fit(
        self,
        event_log: Iterable[TraceSample],
        col_idx_mapping: dict[str, int],
        min_last_activity: int = 50,
        min_last_transition: int = 750,
        **kwargs: Any,
    ):
        """Determine and set feature names of the class and initialize one hot encoders."""
        activity_idx = col_idx_mapping[self.activity_column]

        activity_counter: Counter[str] = Counter()
        transition_counter: Counter[str] = Counter()

        for trace in event_log:
            data = trace.data

            if len(data) == 0:
                continue

            activities = data[:, activity_idx]

            activity_counter.update(activities)

            if len(activities) >= 2:
                transitions = (
                    f"{a}->{b}"
                    for a, b in zip(
                        activities[:-1], activities[1:], strict=True
                    )
                )

                transition_counter.update(transitions)

        # ---------------------------------------------------------
        # Frequent categories
        # ---------------------------------------------------------

        frequent_activities = sorted(
            k for k, v in activity_counter.items() if v >= min_last_activity
        )

        frequent_transitions = sorted(
            k for k, v in transition_counter.items() if v >= min_last_transition
        )

        # ---------------------------------------------------------
        # OHE mappings
        # ---------------------------------------------------------

        self.activity_to_ohe_idx = {
            act: i for i, act in enumerate(frequent_activities)
        }

        self.transition_to_ohe_idx = {
            tr: i for i, tr in enumerate(frequent_transitions)
        }

        # ---------------------------------------------------------
        # Feature names
        # ---------------------------------------------------------

        feature_names = self.feature_names

        # ---------------------------------------------------------
        # OHE feature names
        # ---------------------------------------------------------

        self.activity_ohe_offset = len(feature_names)

        if self.one_hot_encode_categorical:
            feature_names.extend(
                f"last_activity__{act}" for act in frequent_activities
            )

        self.transition_ohe_offset = len(feature_names)

        if self.one_hot_encode_categorical:
            feature_names.extend(
                f"last_transition__{tr}" for tr in frequent_transitions
            )

        self.feature_names = feature_names

    # ---------------------------------------------------------
    # EXTRACT
    # ---------------------------------------------------------

    def __call__(
        self,
        prefix: np.ndarray,
        col_idx_mapping: dict[str, int],
    ) -> np.ndarray:

        timestamp_idx = col_idx_mapping[self.timestamp_column]
        activity_idx = col_idx_mapping[self.activity_column]

        out = np.zeros(len(self.feature_names), dtype=np.float32)

        # ---------------------------------------------------------
        # PREFIX LENGTH
        # ---------------------------------------------------------

        prefix_len = prefix.shape[0]

        # ---------------------------------------------------------
        # ELAPSED TIME
        # ---------------------------------------------------------

        if prefix_len > 1:
            elapsed_hours = _hours(
                prefix[-1][timestamp_idx], prefix[0][timestamp_idx]
            )
            since_last_hours = _hours(
                prefix[-1][timestamp_idx], prefix[-2][timestamp_idx]
            )

        else:
            elapsed_hours = 0.0
            since_last_hours = 0.0

        out[0] = elapsed_hours
        out[1] = prefix_len
        out[2] = since_last_hours

        # ---------------------------------------------------------
        # REWORK COUNT
        # ---------------------------------------------------------

        activities = prefix[:, activity_idx]

        # rework count
        unique_activities = len(set(activities))
        out[3] = prefix_len - unique_activities

        # ---------------------------------------------------------
        # BAG OF ACTIVITIES
        # ---------------------------------------------------------

        counts = Counter(activities)

        offset = 4

        for event in EVENT_NAMES:
            out[offset] = counts.get(event, 0)
            offset += 1

        # ---------------------------------------------------------
        # LAST ACTIVITY / TRANSITION OHE
        # ---------------------------------------------------------

        if self.one_hot_encode_categorical and prefix_len > 0:
            last_activity = activities[-1]

            activity_ohe_idx = self.activity_to_ohe_idx.get(last_activity)

            if activity_ohe_idx is not None:
                out[self.activity_ohe_offset + activity_ohe_idx] = 1.0

            # transition

            if prefix_len >= 2:
                prev_activity = activities[-2]

                transition = f"{prev_activity}->{last_activity}"

                transition_ohe_idx = self.transition_to_ohe_idx.get(transition)

                if transition_ohe_idx is not None:
                    out[self.transition_ohe_offset + transition_ohe_idx] = 1.0

        return out

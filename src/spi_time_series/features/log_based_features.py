import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from spi_time_series.data.constants import EVENT_NAMES


class BasicControlFlowFeatures:
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

        self.ohe_last_activity: OneHotEncoder | None = None
        self.ohe_last_transition: OneHotEncoder | None = None

        if self.one_hot_encode_categorical:
            self._init_encoders()

    # ---------------------------------------------------------
    # INITIALIZE ENCODERS
    # ---------------------------------------------------------

    def _init_encoders(self):
        self.ohe_last_activity = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

        self.ohe_last_transition = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

        # Fit on known space
        self.ohe_last_activity.fit(np.array(EVENT_NAMES).reshape(-1, 1))

        transitions = [f"{a}->{b}" for a in EVENT_NAMES for b in EVENT_NAMES]

        self.ohe_last_transition.fit(np.array(transitions).reshape(-1, 1))

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
            features["elapsed_time_hours"] = total_seconds / 3600
        else:
            features["elapsed_time_hours"] = 0.0

        # ---------------------------------------------------------
        # PREFIX LENGTH
        # ---------------------------------------------------------

        features["prefix_length"] = prefix.shape[0]

        # ---------------------------------------------------------
        # LAST ACTIVITY
        # ---------------------------------------------------------

        last_activity = prefix[-1][activity_idx] if len(prefix) > 0 else None
        features["last_activity"] = last_activity

        # ---------------------------------------------------------
        # TIME SINCE LAST EVENT
        # ---------------------------------------------------------

        if len(prefix) > 1:
            delta_seconds = (
                prefix[-1][timestamp_idx] - prefix[-2][timestamp_idx]
            ).total_seconds()
            features["time_since_last_event_hours"] = delta_seconds / 3600
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
            last_transition = f"{prev_activity}->{last_activity}"
        else:
            last_transition = None

        features["last_transition"] = last_transition

        # ---------------------------------------------------------
        # BAG OF ACTIVITIES
        # ---------------------------------------------------------

        counts = {event: 0 for event in EVENT_NAMES}
        counts.update(
            dict(
                zip(
                    *np.unique(prefix[:, activity_idx], return_counts=True),
                    strict=True,
                )
            )
        )

        for k, v in counts.items():
            features[f"count_{k}"] = v

        # ---------------------------------------------------------
        # ONE HOT ENCODING of categorical features
        # ---------------------------------------------------------

        if (
            self.one_hot_encode_categorical
            and self.ohe_last_activity is not None
            and self.ohe_last_transition is not None
        ):
            # last_activity OHE
            if last_activity is not None:
                vec = self.ohe_last_activity.transform([[last_activity]])
                for i, col in enumerate(self.ohe_last_activity.categories_[0]):
                    features[f"last_activity__{col}"] = vec[0, i]
            else:
                for col in self.ohe_last_activity.categories_[0]:
                    features[f"last_activity__{col}"] = 0

            # last_transition OHE
            if last_transition is not None:
                vec = self.ohe_last_transition.transform([[last_transition]])
                for i, col in enumerate(
                    self.ohe_last_transition.categories_[0]
                ):
                    features[f"last_transition__{col}"] = vec[0, i]
            else:
                for col in self.ohe_last_transition.categories_[0]:
                    features[f"last_transition__{col}"] = 0

        return pd.Series(features)

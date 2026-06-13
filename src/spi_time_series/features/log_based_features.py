from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np

from spi_time_series.data.constants import EVENT_NAMES
from spi_time_series.data.schemas import TraceSample


def _hours(t2, t1):
    return (t2 - t1) / np.timedelta64(1, "h")


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


class OfferFeatures:
    def __init__(
        self,
        activity_column="concept:name",
        offered_amount_column="OfferedAmount",
        monthly_cost_column="MonthlyCost",
        num_terms_column="NumberOfTerms",
        requested_amount_column="RequestedAmount",
    ):
        self.activity_column = activity_column

        self.offered_amount_column = offered_amount_column
        self.monthly_cost_column = monthly_cost_column
        self.num_terms_column = num_terms_column
        self.requested_amount_column = requested_amount_column

        self.feature_names = [
            "num_offers",
            "requested_amount",
            "offered_amount",
            "monthly_cost",
            "num_terms",
        ]

    def name(self):
        return "OfferFeatures"

    def fit(self, *args, **kwargs):
        pass

    def __call__(self, prefix, col_idx_mapping):

        activity_idx = col_idx_mapping[self.activity_column]
        requested_idx = col_idx_mapping.get(self.requested_amount_column)
        offered_idx = col_idx_mapping.get(self.offered_amount_column)
        monthly_cost_idx = col_idx_mapping.get(self.monthly_cost_column)
        terms_idx = col_idx_mapping.get(self.num_terms_column)

        out = np.zeros(len(self.feature_names), dtype=np.float32)
        activities = prefix[:, activity_idx]

        #
        # number of offers
        #

        out[0] = np.sum(activities == "O_Create Offer")

        #
        # latest known financial values
        #

        def last_valid(idx):
            if idx is None:
                return 0.0

            values = prefix[:, idx]

            for value in reversed(values):
                if value not in (None, "", np.nan):
                    try:
                        v = float(value)
                        if np.isnan(value):
                            continue
                        return v
                    except Exception:
                        pass

            return 0.0

        out[1] = last_valid(requested_idx)
        out[2] = last_valid(offered_idx)
        out[3] = last_valid(monthly_cost_idx)
        out[4] = last_valid(terms_idx)

        return out


class InteractionFeatures:
    def __init__(
        self,
        activity_column="concept:name",
    ):
        self.activity_column = activity_column

        self.feature_names = [
            "num_calls",
            "num_incomplete_cycles",
            "num_incomplete",
            "num_validating",
        ]

    def name(self):
        return "InteractionFeatures"

    def fit(self, *args, **kwargs):
        pass

    def __call__(self, prefix, col_idx_mapping):

        activity_idx = col_idx_mapping[self.activity_column]
        activities = prefix[:, activity_idx]
        out = np.zeros(len(self.feature_names), dtype=np.float32)

        call_events = {
            "W_Call after offers",
            "W_Call incomplete files",
        }

        out[0] = sum(act in call_events for act in activities)
        incomplete_count = np.sum(activities == "A_Incomplete")
        validating_count = np.sum(activities == "A_Validating")

        out[1] = min(incomplete_count, validating_count)
        out[2] = incomplete_count
        out[3] = validating_count

        return out


class WaitingStateFeatures:
    def __init__(
        self,
        activity_column="concept:name",
        timestamp_column="time:timestamp",
    ):
        self.activity_column = activity_column
        self.timestamp_column = timestamp_column

        self.feature_names = [
            "hours_since_offer_sent",
            "hours_since_incomplete",
            "is_waiting_for_customer",
            "is_waiting_for_bank",
        ]

    def name(self):
        return "WaitingStateFeatures"

    def fit(self, *args, **kwargs):
        pass

    def __call__(self, prefix, col_idx_mapping):

        activity_idx = col_idx_mapping[self.activity_column]
        timestamp_idx = col_idx_mapping[self.timestamp_column]

        activities = prefix[:, activity_idx]

        current_time = prefix[-1][timestamp_idx]

        out = np.zeros(len(self.feature_names), dtype=np.float32)

        #
        # hours since O_Sent
        #

        offer_indices = np.where(activities == "O_Sent")[0]

        if len(offer_indices):
            idx = offer_indices[-1]

            out[0] = _hours(
                current_time,
                prefix[idx][timestamp_idx],
            )

        #
        # hours since A_Incomplete
        #

        incomplete_indices = np.where(activities == "A_Incomplete")[0]

        if len(incomplete_indices):
            idx = incomplete_indices[-1]

            out[1] = _hours(
                current_time,
                prefix[idx][timestamp_idx],
            )

        #
        # state flags
        #

        last_activity = activities[-1]

        waiting_for_customer_states = {
            "O_Sent",
            "A_Complete",
            "A_Incomplete",
        }

        waiting_for_bank_states = {
            "A_Submitted",
            "A_Concept",
            "A_Accepted",
            "A_Validating",
            "O_Returned",
        }

        out[2] = float(last_activity in waiting_for_customer_states)
        out[3] = float(last_activity in waiting_for_bank_states)

        return out

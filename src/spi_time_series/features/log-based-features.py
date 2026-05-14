from abc import ABC, abstractmethod

import pandas as pd

from spi_time_series.data.schemas import EventLog


class PrefixFeature(ABC):
    @abstractmethod
    def eval(self, prefix: EventLog) -> pd.Series:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__


class BasicControlFlowFeatures(PrefixFeature):
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

    def eval(self, prefix: EventLog) -> pd.Series:
        features = {}

        # ---------------------------------------------------------
        # ELAPSED TIME
        # ---------------------------------------------------------

        if len(prefix) > 1:
            total_seconds = (
                prefix[self.timestamp_column].max()
                - prefix[self.timestamp_column].min()
            ).total_seconds()

            features["elapsed_time_hours"] = total_seconds / (60 * 60)
        else:
            features["elapsed_time_hours"] = 0.0

        # ---------------------------------------------------------
        # PREFIX LENGTH
        # ---------------------------------------------------------

        features["prefix_length"] = len(prefix)

        # ---------------------------------------------------------
        # LAST ACTIVITY
        # ---------------------------------------------------------

        if len(prefix) > 0:
            features["last_activity"] = prefix.iloc[-1][self.activity_column]
        else:
            features["last_activity"] = None

        # ---------------------------------------------------------
        # BAG-OF-ACTIVITIES
        # ---------------------------------------------------------

        counts = (
            prefix[self.activity_column].value_counts().add_prefix("count_")
        )

        features.update(counts.to_dict())

        return pd.Series(features)

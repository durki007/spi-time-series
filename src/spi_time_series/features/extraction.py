import logging
from collections.abc import Callable, Iterator

import pandas as pd
from tqdm import tqdm

from spi_time_series.data.schemas import (
    FeatureSet,
    PrefixFeature,
    PreprocessedData,
    TargetGenerator,
    TraceSample,
)

logger = logging.getLogger(__name__)


def generate_feature_matrix(
    samples: Iterator[TraceSample],
    features: list[PrefixFeature],
    target_generator: TargetGenerator,
    col_idx_mapping: dict[str, int],
    num_cases: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Convert TraceSamples into feature matrix + labels.
    """

    rows = []
    targets: list[str | float] = []

    seen_cases = set()
    pbar = tqdm(total=num_cases, desc="Processing cases")

    for sample in samples:
        for start_idx, end_idx in sample.prefix_indexes:
            feature_row = {}
            feature_input = sample.data[start_idx:end_idx]

            # evaluate all features
            for feature in features:
                # continue
                out = feature(feature_input, col_idx_mapping)

                # prefix feature namespace isolation, prevents collisions between feature classes
                prefixed = {f"{feature.name()}__{k}": v for k, v in out.items()}
                feature_row.update(prefixed)

            rows.append(feature_row)
            targets.append(
                target_generator(
                    trace=sample.data, start_idx=start_idx, end_idx=end_idx
                )
            )

        # update progress bar
        if sample.case_id not in seen_cases:
            seen_cases.add(sample.case_id)
            pbar.update(1)
    pbar.close()

    X = pd.DataFrame(rows)
    y = pd.Series(targets)

    feature_names = list(X.columns)

    return X, y, feature_names


def extract_features_builder(
    features: list[PrefixFeature],
    target_generator: TargetGenerator,
) -> Callable[[PreprocessedData], FeatureSet]:

    def extract_features(data: PreprocessedData) -> FeatureSet:
        X_train, y_train, feature_names = generate_feature_matrix(
            samples=data.train_log,
            features=features,
            target_generator=target_generator,
            col_idx_mapping=data.col_idx,
            num_cases=data.num_train_cases,
        )

        X_test, y_test, _ = generate_feature_matrix(
            samples=data.test_log,
            features=features,
            target_generator=target_generator,
            col_idx_mapping=data.col_idx,
            num_cases=data.num_test_cases,
        )

        return FeatureSet(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
        )

    return extract_features

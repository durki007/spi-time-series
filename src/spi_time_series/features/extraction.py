import logging
from collections.abc import Callable, Iterable

import numpy as np
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
    samples: Iterable[TraceSample],
    features: list[PrefixFeature],
    target_generator: TargetGenerator,
    col_idx_mapping: dict[str, int],
    num_cases: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:

    # ---------------------------------------------------------
    # PRECOMPUTE FEATURE NAMES
    # ---------------------------------------------------------

    feature_names: list[str] = []

    for feature in features:
        feature_names.extend(
            f"{feature.name()}__{name}" for name in feature.feature_names
        )

    n_features = len(feature_names)

    # ---------------------------------------------------------
    # STORAGE
    # ---------------------------------------------------------

    rows = []
    targets = []

    seen_cases = set()

    pbar = tqdm(total=num_cases, desc="Processing cases")

    # ---------------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------------

    for sample in samples:
        data = sample.data

        for start_idx, end_idx in sample.prefix_indexes:
            prefix_data = data[start_idx:end_idx]

            # preallocate row
            row = np.empty(n_features, dtype=np.float32)
            offset = 0

            # evaluate features
            for feature in features:
                vec = feature(
                    prefix_data,
                    col_idx_mapping,
                )
                n = len(vec)
                row[offset : offset + n] = vec
                offset += n

            rows.append(row)

            targets.append(
                target_generator(
                    trace=data,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    col_idx_mapping=col_idx_mapping,
                )
            )

        if sample.case_id not in seen_cases:
            seen_cases.add(sample.case_id)
            pbar.update(1)

    pbar.close()

    # ---------------------------------------------------------
    # FINALIZE
    # ---------------------------------------------------------

    X = pd.DataFrame(
        np.vstack(rows),
        columns=feature_names,
    )

    y = pd.Series(targets)

    return X, y, feature_names


def extract_features_builder(
    features: list[PrefixFeature],
    target_generator: TargetGenerator,
    feature_kwargs: dict[str, dict] | None = None,
    *,
    drop_features: list[str] | None = None,
) -> Callable[[PreprocessedData], FeatureSet]:
    """Create a feature extractor callable with optional column pruning.

    Parameters
    ----------
    features:
        Ordered list of prefix features to compute.
    target_generator:
        Function that produces a target label for each prefix window.
    feature_kwargs:
        Optional per-feature keyword-argument overrides keyed by feature name.
    drop_features:
        Column names to remove from ``X_train`` and ``X_test`` after
        extraction.  Columns that are not present in the dataframe are
        logged as a warning and skipped.  Pass ``None`` or an empty list
        to keep all columns.

    Returns
    -------
    Callable[[PreprocessedData], FeatureSet]
        A function that accepts preprocessed data and returns a FeatureSet.
    """
    if feature_kwargs is None:
        feature_kwargs = {}

    if drop_features is None:
        drop_features = []

    # Normalize to a set for efficient lookup
    _drop: set[str] = set(drop_features)

    def extract_features(data: PreprocessedData) -> FeatureSet:
        # fit features on training data
        for feature in features:
            kwargs = {
                **feature_kwargs.get(feature.name(), {}),
                "cleaned_log": data.cleaned_log,
            }
            logger.info("Fitting %s", feature.name())
            feature.fit(data.train_log, data.col_idx, **kwargs)

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

        # ---------------------------------------------------------
        # PRUNE LOW-IMPORTANCE FEATURES
        # ---------------------------------------------------------
        if _drop:
            for side, df in (("train", X_train), ("test", X_test)):
                missing: set[str] = _drop - set(df.columns)
                present: set[str] = _drop & set(df.columns)
                if missing:
                    logger.warning(
                        "drop_features: %d column(s) not found in X_%s: %s",
                        len(missing),
                        side,
                        sorted(missing),
                    )
                if present:
                    logger.info(
                        "Dropping %d feature(s) from X_%s: %s",
                        len(present),
                        side,
                        sorted(present),
                    )
                    df.drop(columns=list(present), inplace=True)

            # Recompute feature_names after pruning
            feature_names = list(X_train.columns)

        return FeatureSet(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
        )

    return extract_features

import logging
from collections.abc import Callable, Iterable
from multiprocessing import Pool, cpu_count

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


_worker_features: list[PrefixFeature] | None = None
_worker_target_generator: TargetGenerator | None = None
_worker_col_idx_mapping: dict[str, int] | None = None


def _init_worker(features, target_generator, col_idx_mapping):
    global _worker_features
    global _worker_target_generator
    global _worker_col_idx_mapping

    _worker_features = features
    _worker_target_generator = target_generator
    _worker_col_idx_mapping = col_idx_mapping


def _process_sample(sample: TraceSample):
    rows = []
    targets = []

    data = sample.data
    if (
        _worker_features is None
        or _worker_target_generator is None
        or _worker_col_idx_mapping is None
    ):
        raise ValueError(
            "Feature Extraction worker is not properly initialized!"
        )

    n_features = sum(len(feature.feature_names) for feature in _worker_features)
    trace_id = data[0, _worker_col_idx_mapping["case:concept:name"]]

    for start_idx, end_idx in sample.prefix_indexes:
        prefix_data = data[start_idx:end_idx]

        row = np.empty(n_features, dtype=np.float32)

        offset = 0
        for feature in _worker_features:
            vec = feature(
                prefix_data,
                _worker_col_idx_mapping,
            )

            n = len(vec)
            row[offset : offset + n] = vec
            offset += n

        rows.append(row)

        targets.append(
            _worker_target_generator(
                trace=data,
                start_idx=start_idx,
                end_idx=end_idx,
                col_idx_mapping=_worker_col_idx_mapping,
            )
        )

    return rows, targets, trace_id


def generate_feature_matrix(
    samples: Iterable[TraceSample],
    features: list[PrefixFeature],
    target_generator: TargetGenerator,
    col_idx_mapping: dict[str, int],
):
    # ---------------------------------------------------------
    # PRECOMPUTE FEATURE NAMES
    # ---------------------------------------------------------

    feature_names: list[str] = []

    for feature in features:
        feature_names.extend(
            f"{feature.name()}__{name}" for name in feature.feature_names
        )

    # Materialize samples because Pool needs a finite iterable
    samples = list(samples)

    # ---------------------------------------------------------
    # PARALLEL PROCESSING
    # ---------------------------------------------------------

    all_rows = []
    all_targets = []
    all_trace_ids = []

    with Pool(
        cpu_count(),
        _init_worker,
        initargs=(
            features,
            target_generator,
            col_idx_mapping,
        ),
    ) as pool:
        chunk_size = max(1, len(samples) // (cpu_count() * 4))
        results = pool.imap_unordered(
            _process_sample,
            samples,
            chunksize=chunk_size,
        )

        for rows, targets, trace_id in tqdm(
            results,
            total=len(samples),
            desc="Processing cases",
        ):
            all_rows.extend(rows)
            all_targets.extend(targets)
            all_trace_ids.extend(len(rows) * [trace_id])

    # ---------------------------------------------------------
    # FINALIZE
    # ---------------------------------------------------------

    X = pd.DataFrame(
        np.vstack(all_rows),
        columns=feature_names,
    )

    y = pd.Series(all_targets)
    trace_ids = pd.Series(all_trace_ids)

    return X, y, feature_names, trace_ids


def extract_features_builder(
    features: list[PrefixFeature],
    target_generator: TargetGenerator,
    feature_kwargs: dict[str, dict] | None = None,
    *,
    exclude_features: list[str] | None = None,
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
    exclude_features:
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

    if exclude_features is None:
        exclude_features = []

    # Normalize to a set for efficient lookup
    _drop: set[str] = set(exclude_features)

    def extract_features(data: PreprocessedData) -> FeatureSet:
        # fit features on training data
        for feature in features:
            kwargs = {
                **feature_kwargs.get(feature.name(), {}),
                "cleaned_log": data.cleaned_log,
            }
            logger.info("Fitting %s", feature.name())
            feature.fit(data.train_log, data.col_idx, **kwargs)

        X_train, y_train, feature_names, trace_ids_train = (
            generate_feature_matrix(
                samples=data.train_log,
                features=features,
                target_generator=target_generator,
                col_idx_mapping=data.col_idx,
            )
        )

        X_test, y_test, _, trace_ids_test = generate_feature_matrix(
            samples=data.test_log,
            features=features,
            target_generator=target_generator,
            col_idx_mapping=data.col_idx,
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
                        "exclude_features: %d column(s) not found in X_%s: %s",
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
            trace_ids_train=trace_ids_train,
            trace_ids_test=trace_ids_test,
        )

    return extract_features

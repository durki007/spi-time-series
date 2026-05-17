import logging
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, cast

import joblib

logger = logging.getLogger(__name__)
T = TypeVar("T")


def checkpoint[T](
    path: Path | str, compute: Callable[[], T], params: dict | None = None
) -> T:
    path = Path(path)
    if params is not None:
        h = joblib.hash(params)[:8]
        path = path.with_stem(f"{path.stem}__{h}")
    if path.exists():
        logger.info("Loading checkpoint: %s", path)
        return cast(T, joblib.load(path))
    logger.info("Checkpoint not found — computing: %s", path)
    result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, path)
    logger.info("Checkpoint saved: %s", path)
    return result

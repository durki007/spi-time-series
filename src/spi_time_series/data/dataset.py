import logging
import zipfile
from pathlib import Path

import pandas as pd
import pm4py
import requests

logger = logging.getLogger(__name__)

_DOWNLOAD_URL = "https://data.4tu.nl/ndownloader/items/34c3f44b-3101-4ea9-8281-e38905c68b8d/versions/1"
_XES_FILENAME = "BPI Challenge 2017.xes.gz"
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"


class Dataset:
    """BPI Challenge 2017 event log, loaded as a pandas DataFrame.

    Downloads the dataset from 4TU.ResearchData on first use and caches it
    locally as an XES file. Subsequent instantiations read from the cache.

    Attributes:
        data_dir: Directory where the raw XES file is stored.
        log: Event log as a DataFrame with one row per event.
    """

    def __init__(self, data_dir: Path | None = None):
        """Load the dataset, downloading it first if not already cached.

        Args:
            data_dir: Directory to store / read the raw data file.
                Defaults to ``<repo_root>/data/raw``.
        """
        self.data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log: pd.DataFrame = self._load()

    def _xes_path(self) -> Path:
        matches = list(self.data_dir.rglob(_XES_FILENAME))
        if matches:
            return matches[0]
        return self.data_dir / _XES_FILENAME

    def _load(self) -> pd.DataFrame:
        if not self._xes_path().exists():
            logger.info("Dataset not found at %s — downloading.", self.data_dir)
            self._download()
        else:
            logger.info("Dataset found at %s.", self._xes_path())
        logger.info("Reading XES log...")
        df = pm4py.read_xes(str(self._xes_path()))
        if not isinstance(df, pd.DataFrame):
            df = pm4py.convert_to_dataframe(df)

        logger.info("Done. Loaded %d events.", len(df))
        return df

    def _download(self) -> None:
        zip_path = self.data_dir / "data.zip"
        logger.info("Downloading from %s ...", _DOWNLOAD_URL)
        response = requests.get(_DOWNLOAD_URL, stream=True, timeout=300)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Extracting archive...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.data_dir)
        zip_path.unlink()

        xes_files = list(self.data_dir.rglob(_XES_FILENAME))
        if not xes_files:
            raise FileNotFoundError(
                f"Expected file '{_XES_FILENAME}' not found after "
                f"extraction in {self.data_dir}. "
                f"Archive may contain unexpected folder structure."
            )
        logger.info("Extraction complete.")

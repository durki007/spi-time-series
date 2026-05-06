import logging
import zipfile
from pathlib import Path

import pandas as pd
import pm4py
import requests

logger = logging.getLogger(__name__)

_DOWNLOAD_URL = "https://data.4tu.nl/ndownloader/items/34c3f44b-3101-4ea9-8281-e38905c68b8d/versions/1"
_XES_FILENAME = "BPI Challenge 2017.xes.gz"
_DEFAULT_DATA_DIR = Path(__file__).parents[3] / "data" / "raw"


class Dataset:
    def __init__(self, data_dir: Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log: pd.DataFrame = self._load()

    def _xes_path(self) -> Path:
        return self.data_dir / _XES_FILENAME

    def _load(self) -> pd.DataFrame:
        if not self._xes_path().exists():
            logger.info("Dataset not found at %s — downloading.", self.data_dir)
            self._download()
        else:
            logger.info("Dataset found at %s.", self._xes_path())
        logger.info("Reading XES log...")
        df = pm4py.read_xes(str(self._xes_path()))
        logger.info("Done. Loaded %d events.", len(df))
        return df

    def _download(self) -> None:
        zip_path = self.data_dir / "data.zip"
        logger.info("Downloading from %s ...", _DOWNLOAD_URL)
        response = requests.get(_DOWNLOAD_URL, stream=True)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Extracting archive...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.data_dir)
        zip_path.unlink()
        logger.info("Extraction complete.")

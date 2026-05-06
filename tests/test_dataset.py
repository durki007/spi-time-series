import io
import zipfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from spi_time_series.data import Dataset
from spi_time_series.data.dataset import _DOWNLOAD_URL, _XES_FILENAME

_FAKE_LOG = pd.DataFrame(
    {"concept:name": ["A", "B"], "case:concept:name": ["1", "1"]}
)


def _make_zip(filename: str) -> bytes:
    """Build a minimal in-memory zip containing an empty file at `filename`."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(filename, "")
    return buf.getvalue()


@pytest.fixture
def mock_pm4py():
    with patch(
        "spi_time_series.data.dataset.pm4py.read_xes", return_value=_FAKE_LOG
    ) as m:
        yield m


@pytest.fixture
def mock_requests():
    response = MagicMock()
    response.iter_content.return_value = [_make_zip(_XES_FILENAME)]
    response.raise_for_status.return_value = None
    with patch(
        "spi_time_series.data.dataset.requests.get", return_value=response
    ) as m:
        yield m, response


def test_log_is_dataframe(tmp_path, mock_pm4py):
    (tmp_path / _XES_FILENAME).touch()
    ds = Dataset(data_dir=tmp_path)
    assert isinstance(ds.log, pd.DataFrame)


def test_no_download_when_file_exists(tmp_path, mock_pm4py, mock_requests):
    mock_get, _ = mock_requests
    (tmp_path / _XES_FILENAME).touch()
    Dataset(data_dir=tmp_path)
    mock_get.assert_not_called()


def test_download_triggered_when_file_missing(
    tmp_path, mock_pm4py, mock_requests
):
    mock_get, _ = mock_requests
    Dataset(data_dir=tmp_path)
    mock_get.assert_called_once()
    assert "4tu.nl" in mock_get.call_args[0][0]


def test_zip_removed_after_download(tmp_path, mock_pm4py, mock_requests):
    Dataset(data_dir=tmp_path)
    assert not (tmp_path / "data.zip").exists()


def test_xes_file_extracted_after_download(tmp_path, mock_pm4py, mock_requests):
    Dataset(data_dir=tmp_path)
    assert (tmp_path / _XES_FILENAME).exists()


@pytest.mark.integration
def test_dataset_url_is_reachable():
    response = requests.head(_DOWNLOAD_URL, allow_redirects=True, timeout=10)
    assert response.status_code == 200


def test_http_error_propagates(tmp_path):
    response = MagicMock()
    response.raise_for_status.side_effect = Exception("HTTP 404")
    with patch(
        "spi_time_series.data.dataset.requests.get", return_value=response
    ):
        with pytest.raises(Exception, match="HTTP 404"):
            Dataset(data_dir=tmp_path)

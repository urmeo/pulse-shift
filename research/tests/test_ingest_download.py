from http.client import HTTPResponse
from io import BytesIO
from unittest.mock import Mock

import pandas as pd
import pytest

from pulseshift.ingest import _download, _write_csv


def _response(body, headers):
    socket = Mock()
    socket.makefile.return_value = BytesIO(
        b"HTTP/1.1 200 OK\r\n" + headers + b"Connection: close\r\n\r\n" + body
    )
    response = HTTPResponse(socket)
    response.begin()
    return response


@pytest.mark.parametrize("headers", [b"Content-Length: 10\r\n", b""])
def test_truncated_download_can_retry_and_cache_complete_response(
    tmp_path, monkeypatch, headers
):
    body = b"0123456789"
    opener = Mock(
        side_effect=[
            _response(b"hi", b"Content-Length: 10\r\n"),
            _response(body, headers),
        ]
    )
    monkeypatch.setattr("pulseshift.ingest.urllib.request.urlopen", opener)
    target = tmp_path / "source.zip"
    url = "https://example.test/source.zip"

    with pytest.raises(RuntimeError, match="received 2 of 10 bytes"):
        _download(url, target)
    assert not target.exists()
    assert not target.with_name("source.zip.part").exists()

    assert _download(url, target) == target
    assert target.read_bytes() == body
    assert not target.with_name("source.zip.part").exists()
    assert _download(url, target) == target
    assert opener.call_count == 2


def test_failed_cache_write_preserves_previous_cache_and_removes_partial_file(
    tmp_path, monkeypatch
):
    cache = tmp_path / "hourly.csv"
    frame = pd.DataFrame({"rides": [100]})
    _write_csv(frame, cache)
    previous = cache.read_bytes()

    def interrupted_write(self, path, **kwargs):
        path.write_text("partial")
        raise OSError("interrupted")

    monkeypatch.setattr(pd.DataFrame, "to_csv", interrupted_write)
    with pytest.raises(OSError, match="interrupted"):
        _write_csv(frame, cache)
    assert cache.read_bytes() == previous
    assert not cache.with_name("hourly.csv.part").exists()

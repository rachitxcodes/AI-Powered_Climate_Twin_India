from datetime import date

import pytest
import requests

from ingestion.clients.mosdac_client import MOSDACClient


class FakeResponse:
    def __init__(self, *, status_code=200, payload=None, chunks=()):
        self.status_code = status_code
        self.payload = payload
        self.chunks = chunks
        self.text = "fake response"

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        assert chunk_size == 1024 * 1024
        return iter(self.chunks)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append((url, params, timeout))
        return next(self.responses)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def test_search_records_rejects_reversed_dates_without_transport(monkeypatch):
    def fail_if_called():
        raise AssertionError("HTTP session must not be created")

    monkeypatch.setattr("ingestion.clients.mosdac_client.requests.Session", fail_if_called)

    with pytest.raises(ValueError, match="end_date cannot be earlier"):
        MOSDACClient().search_records(date(2024, 6, 19), date(2024, 6, 18))


def test_search_records_aggregates_pages_with_expected_offsets(monkeypatch):
    session = FakeSession(
        [
            FakeResponse(payload={"entries": [{"id": 1}], "totalResults": 2}),
            FakeResponse(payload={"entries": [{"id": 2}], "totalResults": 2}),
        ]
    )
    monkeypatch.setattr(
        "ingestion.clients.mosdac_client.requests.Session", lambda: session
    )

    records = MOSDACClient(timeout=17).search_records(
        date(2024, 6, 18), date(2024, 6, 18)
    )

    assert records == [{"id": 1}, {"id": 2}]
    assert [call[1]["startIndex"] for call in session.calls] == [1, 101]
    assert all(call[2] == 17 for call in session.calls)


@pytest.mark.parametrize(
    "message",
    [
        "Data unavailable for given parameters",
        ["Data unavailable for given parameters"],
    ],
)
def test_search_records_treats_documented_no_data_response_as_empty(
    monkeypatch, message
):
    session = FakeSession([FakeResponse(status_code=500, payload={"message": message})])
    monkeypatch.setattr(
        "ingestion.clients.mosdac_client.requests.Session", lambda: session
    )

    assert MOSDACClient().search_records(date(2024, 6, 19), date(2024, 6, 19)) == []


def test_search_records_raises_for_other_http_errors(monkeypatch):
    session = FakeSession([FakeResponse(status_code=503, payload={"message": "down"})])
    monkeypatch.setattr(
        "ingestion.clients.mosdac_client.requests.Session", lambda: session
    )

    with pytest.raises(requests.HTTPError, match="HTTP 503"):
        MOSDACClient().search_records(date(2024, 6, 18), date(2024, 6, 18))


def test_search_records_rejects_invalid_or_unexpected_json(monkeypatch):
    invalid_json = FakeSession([FakeResponse(payload=ValueError("not json"))])
    monkeypatch.setattr(
        "ingestion.clients.mosdac_client.requests.Session", lambda: invalid_json
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        MOSDACClient().search_records(date(2024, 6, 18), date(2024, 6, 18))

    unexpected_json = FakeSession([FakeResponse(payload=[{"id": 1}])])
    monkeypatch.setattr(
        "ingestion.clients.mosdac_client.requests.Session", lambda: unexpected_json
    )

    with pytest.raises(ValueError, match="expected a JSON object"):
        MOSDACClient().search_records(date(2024, 6, 18), date(2024, 6, 18))


def test_get_access_token_sends_credentials_and_requires_token(monkeypatch):
    calls = []

    def fake_post(url, *, json, timeout):
        calls.append((url, json, timeout))
        return FakeResponse(payload={"access_token": "token-123"})

    monkeypatch.setattr("ingestion.clients.mosdac_client.requests.post", fake_post)
    client = MOSDACClient(timeout=9)

    assert client._get_access_token("user", "secret") == "token-123"
    assert calls == [
        (
            client.TOKEN_URL,
            {"username": "user", "password": "secret"},
            9,
        )
    ]

    monkeypatch.setattr(
        "ingestion.clients.mosdac_client.requests.post",
        lambda url, **kwargs: FakeResponse(payload={}),
    )

    with pytest.raises(RuntimeError, match="without returning an access token"):
        client._get_access_token("user", "secret")


def test_download_with_token_writes_final_file_and_uses_bearer_token(
    monkeypatch, tmp_path
):
    calls = []

    def fake_get(url, *, headers, params, stream, timeout):
        calls.append((url, headers, params, stream, timeout))
        return FakeResponse(chunks=[b"first", b"", b"second"])

    monkeypatch.setattr("ingestion.clients.mosdac_client.requests.get", fake_get)
    client = MOSDACClient()

    destination = client._download_with_token(
        {"id": 7, "identifier": "nested/sample.h5"},
        "token-123",
        tmp_path,
    )

    assert destination == tmp_path / "sample.h5"
    assert destination.read_bytes() == b"firstsecond"
    assert not (tmp_path / "sample.h5.part").exists()
    assert calls == [
        (
            client.DOWNLOAD_URL,
            {"Authorization": "Bearer token-123"},
            {"id": 7},
            True,
            (10, 120),
        )
    ]


@pytest.mark.parametrize(
    "response, error_type",
    [
        (FakeResponse(chunks=[]), RuntimeError),
        (FakeResponse(status_code=502, chunks=[b"partial"]), requests.HTTPError),
    ],
)
def test_download_with_token_removes_partial_file_after_empty_or_failed_download(
    monkeypatch, tmp_path, response, error_type
):
    monkeypatch.setattr(
        "ingestion.clients.mosdac_client.requests.get",
        lambda url, **kwargs: response,
    )

    with pytest.raises(error_type):
        MOSDACClient()._download_with_token(
            {"id": 7, "identifier": "sample.h5"}, "token-123", tmp_path
        )

    assert not (tmp_path / "sample.h5").exists()
    assert not (tmp_path / "sample.h5.part").exists()


def test_download_records_reports_downloaded_skipped_and_failed(monkeypatch, tmp_path):
    existing = tmp_path / "already.h5"
    existing.write_bytes(b"existing")
    client = MOSDACClient()
    calls = []

    monkeypatch.setattr(client, "_get_access_token", lambda username, password: "token")

    def fake_download(record, access_token, output_dir):
        calls.append((record, access_token, output_dir))
        if record["identifier"] == "broken.h5":
            raise RuntimeError("download failed")
        path = output_dir / record["identifier"]
        path.write_bytes(b"downloaded")
        return path

    monkeypatch.setattr(client, "_download_with_token", fake_download)

    result = client.download_records(
        records=[
            {"id": 1, "identifier": "new.h5"},
            {"id": 2, "identifier": "already.h5"},
            {"id": 3, "identifier": "broken.h5"},
            {"id": 4},
        ],
        username="user",
        password="secret",
        output_dir=tmp_path,
    )

    assert result == {
        "downloaded": ["new.h5"],
        "skipped": ["already.h5"],
        "failed": [
            {"filename": "broken.h5", "error": "download failed"},
            {"record": {"id": 4}, "error": "Missing filename identifier"},
        ],
    }
    assert [call[0]["identifier"] for call in calls] == ["new.h5", "broken.h5"]

from unittest.mock import Mock

import pytest
import requests

from camie_tagger.immich import ImmichClient, ImmichError, trigger_scan


@pytest.fixture
def client():
    client = ImmichClient("https://immich.example///", "test-secret", timeout=7)
    yield client
    client._session.close()


@pytest.fixture
def api_request(client, monkeypatch):
    response = Mock(spec=requests.Response)
    response.status_code = 200
    response.text = ""
    request = Mock(return_value=response)
    monkeypatch.setattr(client._session, "request", request)
    return request


def test_client_sets_authentication_headers_and_timeout(client):
    assert client.base_url == "https://immich.example"
    assert client.timeout == 7
    assert client._session.headers["x-api-key"] == "test-secret"
    assert client._session.headers["Accept"] == "application/json"


@pytest.mark.parametrize("status, expected", [(200, True), (204, False)])
def test_ping_checks_response_status(client, api_request, status, expected):
    api_request.return_value.status_code = status
    assert client.ping() is expected
    api_request.assert_called_once_with(
        "GET", "https://immich.example/api/server/ping", json=None, timeout=7
    )


@pytest.mark.parametrize("method, path", [("libraries", "/libraries"), ("tags", "/tags")])
def test_list_endpoints_return_json(client, api_request, method, path):
    payload = [{"id": "example-id"}]
    api_request.return_value.json.return_value = payload
    assert getattr(client, method)() == payload
    api_request.assert_called_once_with(
        "GET", f"https://immich.example/api{path}", json=None, timeout=7
    )


@pytest.mark.parametrize(
    "method, arguments, verb, path, payload, status",
    [
        ("scan_library", ("library-id",), "POST", "/libraries/library-id/scan", None, 202),
        ("start_sidecar_job", (), "PUT", "/jobs/sidecar", {"command": "start", "force": True}, 204),
        ("delete_tag", ("tag-id",), "DELETE", "/tags/tag-id", None, 204),
    ],
)
def test_write_endpoints_send_expected_requests(
    client, api_request, method, arguments, verb, path, payload, status
):
    api_request.return_value.status_code = status
    assert getattr(client, method)(*arguments) == status
    api_request.assert_called_once_with(
        verb, f"https://immich.example/api{path}", json=payload, timeout=7
    )


def test_sidecar_force_flag_can_be_disabled(client, api_request):
    client.start_sidecar_job(force=False)
    assert api_request.call_args.kwargs["json"] == {"command": "start", "force": False}


@pytest.mark.parametrize("status", [401, 403, 404, 429, 500, 503])
def test_http_errors_include_status_and_bounded_detail(client, api_request, status):
    api_request.return_value.status_code = status
    api_request.return_value.text = "  " + "x" * 300 + "  "
    with pytest.raises(ImmichError) as raised:
        client.libraries()
    assert str(raised.value) == f"GET /libraries returned {status}: " + "x" * 200
    assert "test-secret" not in str(raised.value)


@pytest.mark.parametrize("error_type", [requests.Timeout, requests.ConnectionError])
def test_transport_errors_are_wrapped(client, api_request, error_type):
    error = error_type("connection failed")
    api_request.side_effect = error
    with pytest.raises(ImmichError, match="GET /server/ping failed") as raised:
        client.ping()
    assert raised.value.__cause__ is error


def test_trigger_scan_scans_all_libraries_before_sidecars():
    client = Mock(spec=ImmichClient)
    assert trigger_scan(client, ["first", "second"], force_sidecar=False) is True
    assert client.method_calls == [
        ("scan_library", ("first",), {}),
        ("scan_library", ("second",), {}),
        ("start_sidecar_job", (), {"force": False}),
    ]


def test_trigger_scan_without_libraries_still_starts_sidecars(caplog):
    client = Mock(spec=ImmichClient)
    assert trigger_scan(client, []) is True
    client.scan_library.assert_not_called()
    client.start_sidecar_job.assert_called_once_with(force=True)
    assert "No library IDs configured" in caplog.text


def test_trigger_scan_continues_after_a_library_failure():
    client = Mock(spec=ImmichClient)
    client.scan_library.side_effect = [ImmichError("unavailable"), 202]
    assert trigger_scan(client, ["first", "second"]) is False
    assert client.scan_library.call_count == 2
    client.start_sidecar_job.assert_called_once_with(force=True)


def test_trigger_scan_reports_sidecar_failure():
    client = Mock(spec=ImmichClient)
    client.start_sidecar_job.side_effect = ImmichError("unavailable")
    assert trigger_scan(client, ["library-id"]) is False
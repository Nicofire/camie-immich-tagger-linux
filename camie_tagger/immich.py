"""Immich REST API client.

Only the endpoints this tool needs are implemented. The API key is sent in the
x-api-key header and is never written to logs.
"""

from __future__ import annotations

import requests

from .logging_setup import get_logger

DEFAULT_TIMEOUT = 30


class ImmichError(RuntimeError):
    """Raised when an Immich API call fails."""


class ImmichClient:
    def __init__(self, base_url: str, api_key: str, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {"x-api-key": api_key, "Accept": "application/json"}
        )

    def _request(self, method: str, path: str, payload: dict | None = None) -> requests.Response:
        url = f"{self.base_url}/api{path}"
        try:
            response = self._session.request(
                method, url, json=payload, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise ImmichError(f"{method} {path} failed: {exc}") from exc
        if response.status_code >= 400:
            detail = response.text.strip()[:200]
            raise ImmichError(
                f"{method} {path} returned {response.status_code}: {detail}"
            )
        return response

    def ping(self) -> bool:
        response = self._request("GET", "/server/ping")
        return response.status_code == 200

    def libraries(self) -> list[dict]:
        return self._request("GET", "/libraries").json()

    def scan_library(self, library_id: str) -> int:
        return self._request("POST", f"/libraries/{library_id}/scan").status_code

    def start_sidecar_job(self, force: bool = True) -> int:
        payload = {"command": "start", "force": force}
        return self._request("PUT", "/jobs/sidecar", payload).status_code

    def tags(self) -> list[dict]:
        return self._request("GET", "/tags").json()

    def delete_tag(self, tag_id: str) -> int:
        return self._request("DELETE", f"/tags/{tag_id}").status_code


def trigger_scan(client: ImmichClient, library_ids: list[str], force_sidecar: bool = True) -> bool:
    """Rescan the external libraries, then have Immich discover the sidecars."""
    log = get_logger()
    ok = True

    if not library_ids:
        log.warning(
            "No library IDs configured; skipping library scan. "
            "Set IMMICH_LIBRARY_IDS to the External Library UUIDs."
        )
    for library_id in library_ids:
        try:
            status = client.scan_library(library_id)
            log.info("Library scan %s -> %s", library_id, status)
        except ImmichError as exc:
            ok = False
            log.error("Library scan failed for %s: %s", library_id, exc)

    try:
        status = client.start_sidecar_job(force=force_sidecar)
        log.info("Sidecar job -> %s", status)
        log.info("Immich will now discover the sidecars and extract the tags.")
    except ImmichError as exc:
        ok = False
        log.error("Sidecar job failed: %s", exc)

    return ok

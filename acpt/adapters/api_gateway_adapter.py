"""HTTP adapter exposing ACP services through an API gateway."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional
from urllib.parse import urljoin

import requests

from acpt.utils import get_logger


class ApiGatewayAdapter:
    """Lightweight REST client for orchestrator-to-gateway interactions."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        default_headers: Optional[Mapping[str, str]] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/") + "/"
        self._timeout = timeout
        self._headers = dict(default_headers or {})
        self._session = session or requests.Session()
        self._logger = get_logger(self.__class__.__name__)

    def dispatch(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json: Optional[Any] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> requests.Response:
        """Dispatch an HTTP request and return the response object."""

        url = urljoin(self._base_url, path.lstrip("/"))
        merged_headers = {**self._headers, **(headers or {})}
        self._logger.info("API %s %s", method.upper(), url)

        response = self._session.request(
            method=method.upper(),
            url=url,
            params=params,
            json=json,
            headers=merged_headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response

    def close(self) -> None:
        """Close the underlying HTTP session."""

        self._session.close()

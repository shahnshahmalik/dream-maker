"""HTTP client with rate limiting and exponential retry."""

from __future__ import annotations

import threading
import time
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


class RateLimiter:
    def __init__(self, max_per_second: float):
        self._interval = 1.0 / max_per_second
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            if elapsed < self._interval:
                time.sleep(self._interval - elapsed)
            self._last = time.monotonic()


class HttpClient:
    def __init__(
        self,
        base_url: str,
        *,
        headers: dict[str, str] | None = None,
        max_per_second: float = 25.0,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._limiter = RateLimiter(max_per_second)
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers or {},
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=8))
    def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._limiter.wait()
        resp = self._client.request(method, path, json=json, params=params)
        if resp.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{resp.status_code}: {resp.text[:500]}",
                request=resp.request,
                response=resp,
            )
        if not resp.content:
            return {}
        data = resp.json()
        if isinstance(data, dict):
            return data
        return {"data": data}

    def get(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return self.request("POST", path, **kwargs)

    def put(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return self.request("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return self.request("DELETE", path, **kwargs)

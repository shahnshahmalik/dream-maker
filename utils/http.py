"""HTTP client with rate limiting, exponential retry, and circuit breaker."""

from __future__ import annotations

import threading
import time
from enum import Enum, auto
from typing import Any

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)


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


class CircuitState(Enum):
    CLOSED = auto()       # Normal operation
    OPEN = auto()         # Rejecting calls
    HALF_OPEN = auto()    # Probing with a single request


class CircuitBreaker:
    """Prevents hammering a failing downstream service.

    - CLOSED → OPEN after ``failure_threshold`` consecutive 5xx errors
    - OPEN → HALF_OPEN after ``recovery_timeout`` seconds
    - HALF_OPEN → CLOSED on success, → OPEN on failure
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        name: str = "default",
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._transition()
            return self._state

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN

    def _transition(self) -> None:
        """Check if OPEN → HALF_OPEN transition is due."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN

    def check(self) -> None:
        """Raise CircuitOpenError if the circuit is open."""
        self._transition()
        if self._state == CircuitState.OPEN:
            remaining = self.recovery_timeout - (time.monotonic() - self._last_failure_time)
            raise CircuitOpenError(
                f"Circuit '{self.name}' is OPEN for {remaining:.0f}s more "
                f"({self._failure_count} consecutive failures)"
            )


class CircuitOpenError(Exception):
    """Raised when the circuit breaker blocks a call."""
    pass


class HttpClient:
    def __init__(
        self,
        base_url: str,
        *,
        headers: dict[str, str] | None = None,
        max_per_second: float = 25.0,
        timeout: float = 30.0,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._limiter = RateLimiter(max_per_second)
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers or {},
            timeout=timeout,
        )
        self._breaker = circuit_breaker or CircuitBreaker(name=base_url)

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._breaker

    def close(self) -> None:
        self._client.close()

    def health_check(self, path: str = "/v2/fundlimit") -> bool:
        """Lightweight health probe — returns True if service is reachable."""
        try:
            resp = self._client.get(path, timeout=10.0)
            return resp.status_code < 500
        except Exception:
            return False

    def _should_retry(exception: BaseException) -> bool:
        """Retry on server errors (5xx) and connection errors, not client errors (4xx)."""
        if isinstance(exception, CircuitOpenError):
            return False
        if isinstance(exception, httpx.HTTPStatusError):
            return exception.response.status_code >= 500
        if isinstance(exception, (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError)):
            return True
        return False

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=30.0),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError)),
        reraise=True,
    )
    def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Check circuit breaker before making the call
        self._breaker.check()

        self._limiter.wait()
        try:
            resp = self._client.request(method, path, json=json, params=params)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as e:
            self._breaker.record_failure()
            raise

        if resp.status_code >= 500:
            self._breaker.record_failure()
            raise httpx.HTTPStatusError(
                f"{resp.status_code}: {resp.text[:500]}",
                request=resp.request,
                response=resp,
            )
        elif resp.status_code >= 400:
            # Client errors (4xx) — don't trip circuit breaker, don't retry
            raise httpx.HTTPStatusError(
                f"{resp.status_code}: {resp.text[:500]}",
                request=resp.request,
                response=resp,
            )

        # Success — reset circuit
        self._breaker.record_success()

        if not resp.content:
            return {}
        data = resp.json()
        if isinstance(data, dict):
            return data
        return {"data": data}

    # Convenience methods
    def get(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return self.request("POST", path, **kwargs)

    def put(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return self.request("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return self.request("DELETE", path, **kwargs)

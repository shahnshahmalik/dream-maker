"""Append-only trade audit log."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SECRET_RE = re.compile(
    r"(token|key|secret|password|authorization)[\"']?\s*[:=]\s*[\"']?[\w\-./]+",
    re.IGNORECASE,
)


class TradeLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        action: str,
        symbol: str,
        provider: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "symbol": symbol,
            "provider": provider,
            "details": details or {},
            "reason": self._redact(reason),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    @staticmethod
    def _redact(text: str) -> str:
        return _SECRET_RE.sub(r"\1=***REDACTED***", text)

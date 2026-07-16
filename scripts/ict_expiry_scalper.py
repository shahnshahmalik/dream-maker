"""Expiry ICT Sniper — Tuesday 1-minute Order Block + Liquidity Sweep + OTE.

Thin wrapper: all logic lives in scalping/ict_engine.py and scalping/ict_config.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scalping.ict_config import ExpiryICTConfig  # noqa: E402
from scalping.ict_engine import ICTEngine         # noqa: E402


def main() -> None:
    ICTEngine(ExpiryICTConfig).run()


if __name__ == "__main__":
    main()

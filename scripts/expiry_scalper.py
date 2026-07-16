"""
Expiry Day Sniper v3 — 1-minute precision.

High-conviction only: primary trigger + HTF align + volume on closed bar.
Percent trail (no absolute ₹ buffer). Daily max loss + consecutive-loss halt.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scalping import ExpirySniperConfig, SniperEngine  # noqa: E402


def main() -> None:
    SniperEngine(ExpirySniperConfig).run()


if __name__ == "__main__":
    main()

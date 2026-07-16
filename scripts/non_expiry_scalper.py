"""
Non-Expiry Day Sniper v2 — 5-minute precision.

High-conviction only: primary trigger + HTF align + volume on closed bar.
VIX + OI wall gates. Next-week ATM. Percent trail. Hard daily risk caps.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scalping import NonExpirySniperConfig, SniperEngine  # noqa: E402


def main() -> None:
    SniperEngine(NonExpirySniperConfig).run()


if __name__ == "__main__":
    main()

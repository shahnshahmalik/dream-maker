import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HEARTBEAT = Path("/data/state/heartbeat.json")
MAX_AGE_SECONDS = 900

if not HEARTBEAT.exists():
    sys.exit(1)

payload = json.loads(HEARTBEAT.read_text(encoding="utf-8"))
ts = datetime.fromisoformat(payload["timestamp"])
age = (datetime.now(timezone.utc) - ts).total_seconds()
sys.exit(0 if age <= MAX_AGE_SECONDS else 1)

"""Archive NESO's official carbon intensity forecast for the next day, as published at
the issue time.

The Carbon Intensity API keeps one forecast per half hour: the latest, revised until just
before that half hour. A day-ahead model can only be compared fairly with the forecast
NESO had published a day ahead, so it has to be saved when it is published. A scheduled
GitHub Action (.github/workflows/archive.yml) runs this every day.

Standard library only, so the Action needs no installs.

python -m carbon_intensity_forecast.archive OUT_DIR
-> OUT_DIR/YYYY/MM/YYYY-MM-DD.json for the issue day (UTC); an existing file is kept.
"""

import json
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

API = "https://api.carbonintensity.org.uk"
ATTEMPTS = 4


def get_json(url: str) -> dict:
    """GET with retries (the API occasionally answers 5xx)."""
    for attempt in range(ATTEMPTS):
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except OSError:
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(10 * 2**attempt)
    raise AssertionError("unreachable")


def snapshot(now: datetime) -> dict:
    """The next 48 hours of forecasts as published at `now`, with the time they were read."""
    start = now.strftime("%Y-%m-%dT%H:%MZ")
    data = get_json(f"{API}/intensity/{start}/fw48h")["data"]
    if len(data) < 90:  # 96 half hours expected; a short answer is an API fault
        raise ValueError(f"only {len(data)} half hours in the 48-hour forecast")
    return {
        "source": f"{API}/intensity/{{now}}/fw48h",
        "licence": "CC BY 4.0, National Energy System Operator (NESO)",
        "fetched_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "periods": [
            {"from": d["from"], "to": d["to"], "forecast": d["intensity"]["forecast"]}
            for d in data
        ],
    }


def main(out_dir: str) -> None:
    now = datetime.now(UTC)
    path = Path(out_dir) / now.strftime("%Y/%m/%Y-%m-%d.json")
    if path.exists():
        print(f"{path} exists: today's forecast is already archived")
        return
    record = snapshot(now)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"{path}: {len(record['periods'])} half hours, read at {record['fetched_at']}")


if __name__ == "__main__":
    main(sys.argv[1])

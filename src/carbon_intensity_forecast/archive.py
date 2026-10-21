"""Archive, at the issue time, what was known about the next day: NESO's official carbon
intensity forecast and the weather forecasts the model uses.

- NESO: the Carbon Intensity API keeps one forecast per half hour, the latest, revised until
  just before that half hour. A day-ahead model can only be compared fairly with the
  forecast NESO had published a day ahead, so it has to be saved when it is published.
- Weather: the model's inputs exactly as Open-Meteo served them at the issue time (latest
  run and fixed-lead forecasts for the next UTC day), so that live forecasts can be
  reproduced and the downloaded history checked against what was really available.

A scheduled GitHub Action (.github/workflows/archive.yml) runs this every day. Standard
library only, so the Action needs no installs.

python -m carbon_intensity_forecast.archive OUT_DIR
-> OUT_DIR/{neso,weather}/YYYY/MM/YYYY-MM-DD.json for the issue day (UTC); existing files
   are kept.
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

NESO_API = "https://api.carbonintensity.org.uk"
WEATHER = json.loads((Path(__file__).resolve().parents[2] / "configs/weather.json").read_text())
ATTEMPTS = 4


def get_json(url: str) -> dict | list:
    """GET with retries (both APIs occasionally answer 5xx)."""
    for attempt in range(ATTEMPTS):
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except OSError:
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(10 * 2**attempt)
    raise AssertionError("unreachable")


def neso_snapshot(now: datetime) -> dict:
    """The next 48 hours of NESO forecasts as published at `now`."""
    data = get_json(f"{NESO_API}/intensity/{now:%Y-%m-%dT%H:%MZ}/fw48h")["data"]
    if len(data) < 90:  # 96 half hours expected; a short answer is an API fault
        raise ValueError(f"only {len(data)} half hours in the 48-hour forecast")
    return {
        "source": f"{NESO_API}/intensity/{{now}}/fw48h",
        "licence": "CC BY 4.0, National Energy System Operator (NESO)",
        "fetched_at": f"{now:%Y-%m-%dT%H:%M:%SZ}",
        "periods": [
            {"from": d["from"], "to": d["to"], "forecast": d["intensity"]["forecast"]} for d in data
        ],
    }


def weather_snapshot(now: datetime) -> dict:
    """Hourly weather forecasts for the next UTC day at every point, as served at `now`:
    the latest run (day0) and the fixed-lead ones the model is trained on."""
    day = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    names = list(WEATHER["points"])
    leads = [0, *WEATHER["leads"]]
    hourly = [f"{v}_previous_day{k}" for v in WEATHER["variables"] for k in leads]
    query = urllib.parse.urlencode({
        "latitude": ",".join(str(WEATHER["points"][n][0]) for n in names),
        "longitude": ",".join(str(WEATHER["points"][n][1]) for n in names),
        "start_date": day,
        "end_date": day,
        "hourly": ",".join(hourly),
        "timezone": "UTC",
    })  # fmt: skip
    answer = get_json(f"{WEATHER['api']}?{query}")
    points = {name: location["hourly"] for name, location in zip(names, answer, strict=True)}
    if any(len(p["time"]) != 24 for p in points.values()):
        raise ValueError("expected 24 hours per point")
    return {
        "source": WEATHER["api"],
        "licence": "CC BY 4.0, Open-Meteo",
        "fetched_at": f"{now:%Y-%m-%dT%H:%M:%SZ}",
        "valid_day": day,
        "points": points,
    }


def main(out_dir: str) -> None:
    now = datetime.now(UTC)
    for kind, take in [("neso", neso_snapshot), ("weather", weather_snapshot)]:
        path = Path(out_dir) / kind / f"{now:%Y/%m/%Y-%m-%d}.json"
        if path.exists():
            print(f"{path} exists: already archived today")
            continue
        record = take(now)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1) + "\n")
        print(f"{path}: read at {record['fetched_at']}")


if __name__ == "__main__":
    main(sys.argv[1])

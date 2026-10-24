"""Archive, at the issue time, what was known about the next day: NESO's official carbon
intensity forecast and the weather forecasts the model uses.

- NESO: the Carbon Intensity API keeps one forecast per half hour, the latest, revised until
  just before that half hour. A day-ahead model can only be compared fairly with the
  forecast NESO had published a day ahead, so it has to be saved when it is published.
- Weather: the model's inputs exactly as Open-Meteo served them at the issue time (latest
  run and fixed-lead forecasts for the next UTC day and the hour after it), so that live
  forecasts can be reproduced and the downloaded history checked against what was really
  available.
- Intensity: the actual values of the last 9 days as published at the issue time (they are
  revised later), for the same reason.

A scheduled GitHub Action (.github/workflows/archive.yml) runs this every day. Standard
library only, so the Action needs no installs.

python -m carbon_intensity_forecast.archive OUT_DIR
-> OUT_DIR/{neso,weather,intensity}/YYYY/MM/YYYY-MM-DD.json for the issue day (UTC). It
   runs many times a morning (GitHub starts scheduled runs hours late); the read closest to
   09:00 UTC is kept. Each kind is attempted even if another fails; the exit status is 1 if
   any failed, after the others are written.
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
ISSUE_HOUR_UTC = 9  # task.issue_hour_utc in configs/config.yaml (a test keeps them equal)


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
    after = (now + timedelta(days=2)).strftime("%Y-%m-%d")
    names = list(WEATHER["points"])
    leads = [0, *WEATHER["leads"]]
    hourly = [f"{v}_previous_day{k}" for v in WEATHER["variables"] for k in leads]
    query = urllib.parse.urlencode({
        "latitude": ",".join(str(WEATHER["points"][n][0]) for n in names),
        "longitude": ",".join(str(WEATHER["points"][n][1]) for n in names),
        "start_date": day,
        "end_date": after,
        "hourly": ",".join(hourly),
        "timezone": "UTC",
    })  # fmt: skip
    answer = get_json(f"{WEATHER['api']}?{query}")
    points = {name: location["hourly"] for name, location in zip(names, answer, strict=True)}
    if any(len(p["time"]) != 48 for p in points.values()):
        raise ValueError("expected 48 hours per point")
    return {
        "source": WEATHER["api"],
        "licence": "CC BY 4.0, Open-Meteo",
        "fetched_at": f"{now:%Y-%m-%dT%H:%M:%SZ}",
        "valid_day": day,
        "points": points,
    }


def intensity_snapshot(now: datetime) -> dict:
    """The actual values of the last 9 days, as published at `now`."""
    start = now - timedelta(days=9)
    data = get_json(f"{NESO_API}/intensity/{start:%Y-%m-%dT%H:%MZ}/{now:%Y-%m-%dT%H:%MZ}")["data"]
    return {
        "source": f"{NESO_API}/intensity/{{now - 9 days}}/{{now}}",
        "licence": "CC BY 4.0, National Energy System Operator (NESO)",
        "fetched_at": f"{now:%Y-%m-%dT%H:%M:%SZ}",
        "periods": [{"from": d["from"], "actual": d["intensity"].get("actual")} for d in data],
    }


def distance(fetched_at: str, now: datetime) -> timedelta:
    """How far a read is from the day's issue time."""
    read = datetime.strptime(fetched_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    return abs(read - now.replace(hour=ISSUE_HOUR_UTC, minute=0, second=0, microsecond=0))


def main(out_dir: str, now: datetime | None = None) -> None:
    """Keep, for each kind and day, the read closest to the issue time: a new read replaces
    the saved one only if it is closer (the times of all reads are listed in "reads_at")."""
    now = now or datetime.now(UTC)
    failed = []
    for kind, take in [("neso", neso_snapshot), ("weather", weather_snapshot),
                       ("intensity", intensity_snapshot)]:  # fmt: skip
        path = Path(out_dir) / kind / f"{now:%Y/%m/%Y-%m-%d}.json"
        saved = json.loads(path.read_text()) if path.exists() else None
        stamp = f"{now:%Y-%m-%dT%H:%M:%SZ}"
        if saved and distance(saved["fetched_at"], now) <= distance(stamp, now):
            saved["reads_at"] = [*saved.get("reads_at", [saved["fetched_at"]]), stamp]
            path.write_text(json.dumps(saved, indent=1) + "\n")
            print(f"{path}: kept the read at {saved['fetched_at']} (closer to the issue time)")
            continue
        try:
            record = take(now)
        except Exception as exc:  # noqa: BLE001 - keep the others; fail at the end
            print(f"{kind}: failed: {exc!r}", file=sys.stderr)
            failed.append(kind)
            continue
        record["reads_at"] = [*(saved.get("reads_at", [saved["fetched_at"]]) if saved else []),
                              record["fetched_at"]]  # fmt: skip
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1) + "\n")
        print(f"{path}: read at {record['fetched_at']}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1])

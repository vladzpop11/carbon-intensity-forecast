"""Download the raw data, check it against its contracts (schemas.py) and save it.

- data/raw/intensity.csv: national carbon intensity every half hour since 2018: the actual
  value and the API's latest forecast (revised until just before each half hour, so it is
  not a day-ahead benchmark; see archive.py).
- data/raw/weather.csv: hourly weather forecasts at fixed lead times for the points in
  configs/config.yaml, from when both leads are archived.

A download that breaks a contract is not saved. Re-running replaces both files with a
fresh download up to now.

python -m carbon_intensity_forecast.ingest
"""

import time
from datetime import UTC, datetime, timedelta

import httpx
import pandas as pd

from carbon_intensity_forecast.artifacts import write_json
from carbon_intensity_forecast.config import load_config, project_path
from carbon_intensity_forecast.schemas import (
    INTENSITY,
    INTENSITY_LIMITS,
    WEATHER_LIMITS,
    clean,
    weather_schema,
)

INTENSITY_CHUNK = timedelta(days=30)  # the API refuses ranges over 31 days
WEATHER_CHUNK = timedelta(days=92)
ATTEMPTS = 4


class Client(httpx.Client):
    """An HTTP client with its own timeout and number of attempts: patient for the daily
    download, brief for a live request."""

    def __init__(self, timeout: float = 60.0, attempts: int = ATTEMPTS):
        super().__init__(headers={"Accept": "application/json"}, timeout=timeout)
        self.attempts = attempts


def get(client: httpx.Client, url: str, params: dict | None = None) -> dict:
    """GET JSON with retries on network errors and 5xx/429 answers."""
    attempts = getattr(client, "attempts", ATTEMPTS)
    for attempt in range(attempts):
        try:
            response = client.get(url, params=params)
            if response.status_code < 500 and response.status_code != 429:
                response.raise_for_status()
                return response.json()
        except httpx.TransportError:
            if attempt == attempts - 1:
                raise
        if attempt == attempts - 1:
            response.raise_for_status()
        time.sleep(5 * 2**attempt)
    raise AssertionError("unreachable")


def fetch_intensity(client: httpx.Client, api: str, start: datetime, end: datetime) -> pd.DataFrame:
    rows = []
    t = start
    while t < end:
        stop = min(t + INTENSITY_CHUNK, end)
        span = f"{t:%Y-%m-%dT%H:%MZ}/{stop:%Y-%m-%dT%H:%MZ}"
        for d in get(client, f"{api}/intensity/{span}")["data"]:
            i = d["intensity"]  # either value can be absent or null for some half hours
            rows.append((d["from"], i.get("actual"), i.get("forecast")))
        t = stop
    frame = pd.DataFrame(rows, columns=["period_start", "actual", "forecast_latest"])
    frame["period_start"] = pd.to_datetime(frame["period_start"], utc=True)
    # Consecutive ranges share their boundary half hour; the API also returns the period
    # ending at the start time.
    frame = frame.drop_duplicates("period_start").sort_values("period_start")
    frame = frame[(frame["period_start"] >= start) & (frame["period_start"] < end)]
    return frame.reset_index(drop=True)


def fetch_weather(client: httpx.Client, cfg: dict, start: datetime, end: datetime) -> pd.DataFrame:
    w = cfg["weather"]
    names = list(w["points"])
    lats = ",".join(str(w["points"][n][0]) for n in names)
    lons = ",".join(str(w["points"][n][1]) for n in names)
    hourly = [f"{v}_previous_day{k}" for v in w["variables"] for k in w["leads"]]
    parts = []
    t = start
    while t < end:
        stop = min(t + WEATHER_CHUNK, end)
        params = {
            "latitude": lats,
            "longitude": lons,
            "start_date": f"{t:%Y-%m-%d}",
            "end_date": f"{stop - timedelta(days=1):%Y-%m-%d}",
            "hourly": ",".join(hourly),
            "timezone": "UTC",
        }
        answer = get(client, w["api"], params)
        for name, location in zip(names, answer, strict=True):
            part = pd.DataFrame(location["hourly"])
            part.insert(1, "point", name)
            parts.append(part)
        t = stop
    frame = pd.concat(parts, ignore_index=True)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.rename(columns={h: h.replace("_previous_day", "_day") for h in hourly})
    return frame.sort_values(["time", "point"]).reset_index(drop=True)


def complete_grid(frame: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    """Every half hour from start to end, with missing ones as empty rows."""
    grid = pd.date_range(start, end, freq="30min", inclusive="left", name="period_start")
    return frame.set_index("period_start").reindex(grid).reset_index()


def main() -> None:
    cfg = load_config()
    data = cfg["data"]
    raw = project_path(data["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    with Client() as client:
        start = datetime.fromisoformat(data["intensity_start"]).replace(tzinfo=UTC)
        intensity = fetch_intensity(client, data["intensity_api"], start, now)
        returned = len(intensity)
        intensity = complete_grid(intensity, start, now)
        intensity, intensity_faults = clean(intensity, INTENSITY_LIMITS)
        intensity = INTENSITY.validate(intensity)
        # Weather through the hour after the end of tomorrow (a 23:30 target uses 00:00 too):
        # forecasts for those hours exist already.
        weather_end = (now + timedelta(days=3)).replace(hour=0)
        w = cfg["weather"]
        start = datetime.fromisoformat(w["start"]).replace(tzinfo=UTC)
        weather = fetch_weather(client, cfg, start, weather_end)
        limits = {f"{v}_day{k}": WEATHER_LIMITS[v] for v in w["variables"]
                  for k in w["leads"]}  # fmt: skip
        weather, weather_faults = clean(weather, limits)
        weather = weather_schema(w["variables"], w["leads"]).validate(weather)
    intensity.to_csv(raw / "intensity.csv", index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    weather.to_csv(raw / "weather.csv", index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    quality = {
        "downloaded_at": f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}",
        "intensity": {
            "half_hours": len(intensity),
            "first": str(intensity["period_start"].min()),
            "last": str(intensity["period_start"].max()),
            "not_returned_by_the_api": len(intensity) - returned,
            "without_actual": int(intensity["actual"].isna().sum()),
            "faults_set_to_missing": intensity_faults,
        },
        "weather": {
            "rows": len(weather),
            "points": int(weather["point"].nunique()),
            "missing_values": {c: int(weather[c].isna().sum()) for c in limits},
            "faults_set_to_missing": {c: f["count"] for c, f in weather_faults.items()},
        },
    }
    write_json(project_path("reports/data_quality.json"), quality)
    q, faults = quality["intensity"], intensity_faults["actual"]["count"]
    print(f"intensity: {q['half_hours']:,} half hours to {q['last']}; {q['without_actual']:,} "
          f"without an actual value ({q['not_returned_by_the_api']:,} not returned, "
          f"{faults} faults); weather: {len(weather):,} rows")  # fmt: skip


if __name__ == "__main__":
    main()

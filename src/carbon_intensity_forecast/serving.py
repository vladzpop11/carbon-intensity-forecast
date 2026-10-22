"""Live forecasts: fetch what was known at the issue time, check it against the data
contracts, build the features with the same code as training, and predict.

The forecast for day D+1 is issued at 09:00 UTC on day D (task.issue_hour_utc). Features
only use values published by then, so asking later in the day gives the same forecast.
"""

from datetime import UTC, datetime

import httpx
import numpy as np
import pandas as pd

from carbon_intensity_forecast import features as F
from carbon_intensity_forecast.ingest import Client, complete_grid, fetch_intensity, fetch_weather
from carbon_intensity_forecast.schemas import (
    INTENSITY,
    INTENSITY_LIMITS,
    WEATHER_LIMITS,
    ContractError,
    clean,
    weather_schema,
)

PUBLISH_MARGIN = pd.Timedelta("15min")  # the issue-time inputs are complete by then
HISTORY = pd.Timedelta("9D")  # covers the week-before lag and the last day's statistics


def latest_issue_day(now: datetime, cfg: dict) -> pd.Timestamp:
    """The issue day of the most recent forecast: today once 09:00 UTC (plus a margin) has
    passed, yesterday before that."""
    now = pd.Timestamp(now).tz_convert("UTC")
    today = now.normalize()
    issued = today + pd.Timedelta(hours=cfg["task"]["issue_hour_utc"]) + PUBLISH_MARGIN
    return today if now >= issued else today - pd.Timedelta("1D")


def live_inputs(client: httpx.Client, cfg: dict, issue_day: pd.Timestamp, now: datetime):
    """Intensity and weather around one issue day, checked against the contracts.
    Raises schemas.ContractError or a Pandera SchemaError when they fail."""
    start = (issue_day - HISTORY).to_pydatetime()
    end = pd.Timestamp(now).tz_convert("UTC").floor("30min").to_pydatetime()
    intensity = fetch_intensity(client, cfg["data"]["intensity_api"], start, end)
    intensity = complete_grid(intensity, start, end)
    intensity, _ = clean(intensity, INTENSITY_LIMITS)
    intensity = INTENSITY.validate(intensity)
    # Weather for the target day and the first hour after it (a 23:30 target averages
    # 23:00 and 00:00).
    w = cfg["weather"]
    target_day = (issue_day + pd.Timedelta("1D")).to_pydatetime()
    weather = fetch_weather(client, cfg, target_day, target_day + pd.Timedelta("2D"))
    limits = {f"{v}_day{k}": WEATHER_LIMITS[v] for v in w["variables"] for k in w["leads"]}
    weather, _ = clean(weather, limits)
    weather = weather_schema(w["variables"], w["leads"]).validate(weather)
    return intensity, weather


# Inputs a live forecast must have for every half hour (same_time_issue_day is missing by
# design after 08:00, and same_time_week_before may be a source gap the model tolerates).
REQUIRED = ["last_known", "mean_last_24h", "min_last_24h", "max_last_24h", "same_time_latest"]
REQUIRED_PREFIXES = ("wind_speed_100m_", "shortwave_radiation_", "temperature_2m_")


def require_complete(table: pd.DataFrame) -> None:
    """The completeness contract: refuse a forecast that would be made from missing inputs."""
    columns = [c for c in table.columns if c in REQUIRED or c.startswith(REQUIRED_PREFIXES)]
    missing = sorted(c for c in columns if table[c].isna().any())
    if missing:
        raise ContractError(f"incomplete inputs: {', '.join(missing[:6])}"
                            f"{' and more' if len(missing) > 6 else ''}")  # fmt: skip


def predict(model, columns: list[str], cfg: dict, issue_day: pd.Timestamp,
            intensity: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:  # fmt: skip
    """The 48 half hours of the target day with the model's forecast (and the baseline),
    after the completeness contract."""
    table = F.build(pd.DatetimeIndex([issue_day]), intensity, weather, cfg)
    require_complete(table)
    table["forecast"] = np.asarray(model.predict(table[columns]), dtype=float)
    return table[["issue_time", "target_time", "lead_hours", "forecast", "same_time_latest"]]


# Live calls get a short time budget (20 s per call, two attempts): a request should fail
# fast rather than hang.
LIVE_TIMEOUT, LIVE_ATTEMPTS = 20.0, 2


def forecast_for(model, columns: list[str], cfg: dict, issue_day: pd.Timestamp,
                 now: datetime) -> pd.DataFrame:  # fmt: skip
    """The forecast table for one issue day, from live inputs read at `now`."""
    with Client(timeout=LIVE_TIMEOUT, attempts=LIVE_ATTEMPTS) as client:
        intensity, weather = live_inputs(client, cfg, issue_day, now)
    return predict(model, columns, cfg, issue_day, intensity, weather)


def forecast_now(model, columns: list[str], cfg: dict, now: datetime | None = None):
    """-> (issue day, forecast table) for the most recent issue time."""
    now = now or datetime.now(UTC)
    issue_day = latest_issue_day(now, cfg)
    return issue_day, forecast_for(model, columns, cfg, issue_day, now)

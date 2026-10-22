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
from carbon_intensity_forecast.ingest import complete_grid, fetch_intensity, fetch_weather
from carbon_intensity_forecast.schemas import (
    INTENSITY,
    INTENSITY_LIMITS,
    WEATHER_LIMITS,
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


def predict(model, columns: list[str], cfg: dict, issue_day: pd.Timestamp,
            intensity: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:  # fmt: skip
    """The 48 half hours of the target day with the model's forecast."""
    table = F.build(pd.DatetimeIndex([issue_day]), intensity, weather, cfg)
    table["forecast"] = np.asarray(model.predict(table[columns]), dtype=float)
    return table[["issue_time", "target_time", "lead_hours", "forecast"]]


def forecast_now(model, columns: list[str], cfg: dict, now: datetime | None = None):
    """-> (issue day, forecast table) for the most recent issue time."""
    now = now or datetime.now(UTC)
    issue_day = latest_issue_day(now, cfg)
    with httpx.Client(headers={"Accept": "application/json"}) as client:
        intensity, weather = live_inputs(client, cfg, issue_day, now)
    return issue_day, predict(model, columns, cfg, issue_day, intensity, weather)

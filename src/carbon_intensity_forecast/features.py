"""Features for the day-ahead task: one row per (issue day, target half hour), built only
from what was known at the issue time. The same code builds training rows, backtest rows
and the live forecast.

For issue day D the issue time is D at task.issue_hour_utc, and the targets are the 48 half
hours of D+1 (00:00-23:30 UTC), 15 to 38.5 hours ahead.

- Calendar of the target: time of day, day of week, bank holiday (England), season.
- Carbon intensity known at the issue time: actual values up to an hour before it
  (task.actual_delay_minutes), the same half hour on earlier days, recent levels.
- Weather forecasts at the target time, at the shortest lead that had been published by
  the issue time (task.weather_availability_delay_hours; see configs/config.yaml).
"""

import holidays
import numpy as np
import pandas as pd

HALF_HOUR = pd.Timedelta("30min")
HOUR = pd.Timedelta("1h")
DAY = pd.Timedelta("1D")
WIND_POINTS = ["scotland_north", "scotland_south", "dogger_bank", "hornsea",
               "east_anglia_offshore", "irish_sea"]  # fmt: skip
SOLAR_POINTS = ["southern_england", "london", "birmingham"]
CITY_POINTS = ["london", "birmingham", "manchester"]


def targets(issue_days: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """The (issue day, target time) grid with issue times and leads."""
    task = cfg["task"]
    days = pd.DatetimeIndex(issue_days).normalize()
    slots = np.arange(task["horizon_half_hours"])
    grid = pd.DataFrame({
        "issue_day": np.repeat(days, len(slots)),
        "slot": np.tile(slots, len(days)),
    })  # fmt: skip
    grid["issue_time"] = grid["issue_day"] + pd.Timedelta(hours=task["issue_hour_utc"])
    grid["target_time"] = grid["issue_day"] + DAY + grid["slot"] * HALF_HOUR
    grid["lead_hours"] = (grid["target_time"] - grid["issue_time"]) / HOUR
    return grid


def calendar(grid: pd.DataFrame) -> pd.DataFrame:
    t = grid["target_time"]
    years = range(t.dt.year.min(), t.dt.year.max() + 1)
    bank = holidays.country_holidays("GB", subdiv="ENG", years=years)
    day_of_year = t.dt.dayofyear
    return pd.DataFrame({
        "hour": t.dt.hour + t.dt.minute / 60,
        "day_of_week": t.dt.dayofweek,
        "bank_holiday": t.dt.date.map(lambda d: d in bank).astype(int),
        "season_sin": np.sin(2 * np.pi * day_of_year / 365.25),
        "season_cos": np.cos(2 * np.pi * day_of_year / 365.25),
    }, index=grid.index)  # fmt: skip


def known_intensity(grid: pd.DataFrame, actual: pd.Series, cfg: dict) -> pd.DataFrame:
    """Carbon intensity features from values published by the issue time. `actual` is
    indexed by period start on the full half-hour grid."""
    delay = pd.Timedelta(minutes=cfg["task"]["actual_delay_minutes"])
    last_known = grid["issue_time"] - HALF_HOUR - delay  # start of the last half hour known

    def at(times: pd.Series) -> np.ndarray:
        values = actual.reindex(pd.DatetimeIndex(times)).to_numpy()
        return np.where(times.to_numpy() <= last_known.to_numpy(), values, np.nan)

    # Rolling statistics of the past day, for each distinct issue time.
    day_mean = actual.rolling("1D", min_periods=24).mean()
    day_min = actual.rolling("1D", min_periods=24).min()
    day_max = actual.rolling("1D", min_periods=24).max()
    lk = pd.DatetimeIndex(last_known)
    features = pd.DataFrame({
        # The same clock time on the issue day (known only for early slots), the day before,
        # and a week before the target.
        "same_time_issue_day": at(grid["target_time"] - DAY),
        "same_time_day_before": at(grid["target_time"] - 2 * DAY),
        "same_time_week_before": at(grid["target_time"] - 7 * DAY),
        "last_known": at(last_known),
        "mean_last_24h": day_mean.reindex(lk).to_numpy(),
        "min_last_24h": day_min.reindex(lk).to_numpy(),
        "max_last_24h": day_max.reindex(lk).to_numpy(),
    }, index=grid.index)  # fmt: skip
    features["same_time_latest"] = features["same_time_issue_day"].fillna(
        features["same_time_day_before"]
    )
    return features


def weather_lead(target_hour: pd.Series, issue_time: pd.Series, cfg: dict) -> np.ndarray:
    """The shortest lead (1 or 2 days) whose final forecast for `target_hour` had been
    published by the issue time."""
    cutoff = issue_time - pd.Timedelta(hours=cfg["task"]["weather_availability_delay_hours"])
    day1_ok = (target_hour - DAY) <= cutoff
    if not ((target_hour - 2 * DAY) <= cutoff).all():
        raise ValueError("a target is too far ahead for the longest weather lead")
    return np.where(day1_ok, 1, 2)


def weather_features(grid: pd.DataFrame, weather: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Weather at the target time, at the lead allowed by the issue time. Hourly forecasts
    are averaged to the half hour (xx:30 = mean of xx:00 and the next hour); both hours use
    the lead allowed for the later one."""
    w = cfg["weather"]
    wide = weather.pivot(index="time", columns="point")
    t0 = grid["target_time"].dt.floor("1h")
    t1 = grid["target_time"].dt.ceil("1h")  # the same hour for xx:00, the next for xx:30
    lead = weather_lead(t1, grid["issue_time"], cfg)
    out = {"weather_lead_days": lead}
    for variable in w["variables"]:
        for point in w["points"]:
            values = np.full(len(grid), np.nan)
            for k in w["leads"]:
                column = wide[(f"{variable}_day{k}", point)]
                v0 = column.reindex(pd.DatetimeIndex(t0)).to_numpy()
                v1 = column.reindex(pd.DatetimeIndex(t1)).to_numpy()
                values = np.where(lead == k, (v0 + v1) / 2, values)
            out[f"{variable}_{point}"] = values
    frame = pd.DataFrame(out, index=grid.index)
    wind = frame[[f"wind_speed_100m_{p}" for p in WIND_POINTS]]
    frame["wind_mean"] = wind.mean(axis=1)
    # Turbine output grows with the cube of wind speed up to its rated speed (about 45 km/h).
    frame["wind_power_proxy"] = (wind.clip(upper=45) ** 3).mean(axis=1) / 45**3
    frame["solar_mean"] = frame[[f"shortwave_radiation_{p}" for p in SOLAR_POINTS]].mean(axis=1)
    frame["temperature_mean"] = frame[[f"temperature_2m_{p}" for p in CITY_POINTS]].mean(axis=1)
    return frame


def build(issue_days, intensity: pd.DataFrame, weather: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """grid + features + the target (`actual`, missing for future or unrecorded half hours)."""
    actual = intensity.set_index("period_start")["actual"]
    grid = targets(issue_days, cfg)
    table = pd.concat([grid, calendar(grid), known_intensity(grid, actual, cfg),
                       weather_features(grid, weather, cfg)], axis=1)  # fmt: skip
    table["actual"] = actual.reindex(pd.DatetimeIndex(grid["target_time"])).to_numpy()
    return table


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns
            if c not in {"issue_day", "issue_time", "target_time", "actual"}]  # fmt: skip


def load_raw(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    from carbon_intensity_forecast.config import project_path

    raw = project_path(cfg["data"]["raw_dir"])
    intensity = pd.read_csv(raw / "intensity.csv", parse_dates=["period_start"])
    weather = pd.read_csv(raw / "weather.csv", parse_dates=["time"])
    return intensity, weather

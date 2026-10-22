"""Data contracts for everything the pipeline ingests (Pandera), at two levels:

- Structure (times on the grid, unique, in order; types; columns): a batch that breaks it
  is not used at all. Ingestion stops and reports what failed.
- Values: a value outside physical limits is a fault in the source (the API has some). It
  is set to missing and counted (clean()), unless faults exceed MAX_FAULT_SHARE of the
  batch's last 30 days: then the feed itself is broken and the batch is refused.
- Completeness, for a live forecast (serving.require_complete): every input the model relies
  on must be present, or the forecast is refused rather than made from gaps.

Limits are physical, with a margin, not the observed range, so that a genuinely unusual
day passes and only broken data fails.
"""

import pandas as pd
import pandera.pandas as pa

MAX_FAULT_SHARE = 0.01
FAULT_WINDOW_DAYS = 30


class ContractError(ValueError):
    """A batch the pipeline must not use."""


def _on_the_half_hour(s: pd.Series) -> pd.Series:
    return (s.dt.minute % 30 == 0) & (s.dt.second == 0)


def _on_the_hour(s: pd.Series) -> pd.Series:
    return (s.dt.minute == 0) & (s.dt.second == 0)


# National carbon intensity, gCO2/kWh, one row per half hour (period start, UTC). Since
# 2018 GB's genuine values have stayed between about 20 and 550 (nuclear, biomass and
# imports never all stop), so 5-1000 allows anything real. The source has faults: zeros in
# `actual` during estimation outages (2023, 2026) and forecasts up to 13,579 (2018-19).
# `actual` is also missing for some half hours (the API's own gaps): counted, not filled.
INTENSITY_LIMITS = {"actual": (5, 1000), "forecast_latest": (5, 1000)}
INTENSITY = pa.DataFrameSchema(
    {
        "period_start": pa.Column(
            pd.DatetimeTZDtype(tz="UTC"),
            pa.Check(_on_the_half_hour, error="not on a half hour"),
            unique=True,
        ),
        "actual": pa.Column(float, pa.Check.in_range(*INTENSITY_LIMITS["actual"]), nullable=True),
        "forecast_latest": pa.Column(
            float, pa.Check.in_range(*INTENSITY_LIMITS["forecast_latest"]), nullable=True
        ),
    },
    checks=pa.Check(
        lambda df: df["period_start"].is_monotonic_increasing, error="periods out of order"
    ),
    strict=True,
    coerce=True,
)

# Weather forecasts at fixed lead times, hourly (valid time, UTC), one column per variable
# and lead. Limits: wind at 100 m up to 250 km/h, shortwave radiation 0-1400 W/m2 (above
# the solar constant is impossible at the surface), temperature -40 to 45 C in GB.
WEATHER_LIMITS = {
    "wind_speed_100m": (0, 250),
    "shortwave_radiation": (0, 1400),
    "temperature_2m": (-40, 45),
}


def weather_schema(variables: list[str], leads: list[int]) -> pa.DataFrameSchema:
    columns = {
        "time": pa.Column(
            pd.DatetimeTZDtype(tz="UTC"), pa.Check(_on_the_hour, error="not on the hour")
        ),
        "point": pa.Column(str),
    }
    for variable in variables:
        low, high = WEATHER_LIMITS[variable]
        for lead in leads:
            columns[f"{variable}_day{lead}"] = pa.Column(
                float, pa.Check.in_range(low, high), nullable=True
            )
    return pa.DataFrameSchema(columns, unique=["time", "point"], strict=True, coerce=True)


def clean(frame: pd.DataFrame, limits: dict[str, tuple[float, float]],
          max_share: float = MAX_FAULT_SHARE) -> tuple[pd.DataFrame, dict]:  # fmt: skip
    """Set values outside `limits` to missing. -> (cleaned frame, faults per column with
    the times affected). Refuses the batch when faults exceed MAX_FAULT_SHARE of a column."""
    frame = frame.copy()
    faults = {}
    time_column = "period_start" if "period_start" in frame else "time"
    # The share is judged on the most recent 30 days, so a broken stretch cannot hide in a
    # long history (the full download covers 2018 to now).
    recent = frame[time_column] >= frame[time_column].max() - pd.Timedelta(days=FAULT_WINDOW_DAYS)
    for column, (low, high) in limits.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        bad = values.notna() & ((values < low) | (values > high))
        share = bad[recent].mean()
        if share > max_share:
            raise ContractError(
                f"{column}: {share:.1%} of the last {FAULT_WINDOW_DAYS} days' values outside "
                f"{low}-{high}: the feed is broken"
            )
        faults[column] = {
            "count": int(bad.sum()),
            "times": frame.loc[bad, time_column].astype(str).tolist(),
        }
        frame.loc[bad, column] = float("nan")
    return frame, faults

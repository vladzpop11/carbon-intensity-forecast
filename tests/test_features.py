"""The features use only what was known at the issue time: changing anything published
later must not change them."""

import numpy as np
import pandas as pd
import pytest

from carbon_intensity_forecast import features as F
from carbon_intensity_forecast.config import load_config

ISSUE_DAY = pd.Timestamp("2025-01-10", tz="UTC")


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def data(cfg):
    rng = np.random.default_rng(0)
    times = pd.date_range("2024-12-20", "2025-01-13", freq="30min", tz="UTC", inclusive="left")
    intensity = pd.DataFrame({"period_start": times, "actual": rng.uniform(50, 300, len(times))})
    hours = pd.date_range("2024-12-20", "2025-01-13", freq="1h", tz="UTC", inclusive="left")
    w = cfg["weather"]
    rows = []
    for point in w["points"]:
        part = pd.DataFrame({"time": hours, "point": point})
        for v in w["variables"]:
            for k in w["leads"]:
                part[f"{v}_day{k}"] = rng.uniform(0, 30, len(hours))
        rows.append(part)
    return intensity, pd.concat(rows, ignore_index=True)


def test_nothing_published_after_the_issue_time_is_used(cfg, data) -> None:
    intensity, weather = data
    days = pd.DatetimeIndex([ISSUE_DAY])
    before = F.build(days, intensity, weather, cfg)
    issue = ISSUE_DAY + pd.Timedelta(hours=cfg["task"]["issue_hour_utc"])

    # Actual values: published 30 minutes after their half hour ends.
    delay = pd.Timedelta(minutes=cfg["task"]["actual_delay_minutes"])
    late = intensity["period_start"] + pd.Timedelta("30min") + delay > issue
    changed_intensity = intensity.assign(actual=np.where(late, 9999.0, intensity["actual"]))
    # Weather: lead k's final forecast for time t comes from a run made 24k hours before t,
    # published about `delay` hours later.
    cutoff = issue - pd.Timedelta(hours=cfg["task"]["weather_availability_delay_hours"])
    changed_weather = weather.copy()
    for k in cfg["weather"]["leads"]:
        unpublished = weather["time"] - pd.Timedelta(days=k) > cutoff
        for v in cfg["weather"]["variables"]:
            changed_weather.loc[unpublished, f"{v}_day{k}"] = 9999.0

    after = F.build(days, changed_intensity, changed_weather, cfg)
    columns = F.feature_columns(before)
    pd.testing.assert_frame_equal(before[columns], after[columns])
    assert (after["actual"] == 9999.0).all()  # the targets themselves are in the future


def test_features_do_change_when_known_data_changes(cfg, data) -> None:
    """The test above would pass trivially if the features ignored the data."""
    intensity, weather = data
    days = pd.DatetimeIndex([ISSUE_DAY])
    before = F.build(days, intensity, weather, cfg)
    early = intensity.assign(actual=intensity["actual"] + 1.0)
    shifted = weather.assign(**{c: weather[c] + 1.0 for c in weather.columns if "_day" in c})
    after = F.build(days, early, shifted, cfg)
    assert not np.allclose(before["last_known"], after["last_known"])
    assert not np.allclose(before["wind_mean"], after["wind_mean"])


def test_the_shortest_published_weather_lead_is_used(cfg) -> None:
    grid = F.targets(pd.DatetimeIndex([ISSUE_DAY]), cfg)
    lead = F.weather_lead(grid["target_time"].dt.ceil("1h"), grid["issue_time"], cfg)
    # Issue 09:00 with a 6-hour delay: day-1 forecasts made by 03:00, so targets to 03:00.
    day1 = grid.loc[lead == 1, "target_time"]
    assert day1.max() == ISSUE_DAY + pd.Timedelta("1D") + pd.Timedelta("3h")
    assert (lead[grid["target_time"].dt.hour >= 4] == 2).all()


def test_backtest_training_targets_end_before_the_first_issue_time(cfg) -> None:
    """Every month's model is trained only on targets published before the month's first
    forecast is issued (09:00 on the day before the month)."""
    from carbon_intensity_forecast.backtest import split

    days = pd.date_range("2025-01-01", "2025-03-31", freq="1D", tz="UTC")
    table = F.targets(days, cfg).assign(actual=1.0)
    month = pd.Period("2025-03", freq="M")
    train, test = split(table, month)
    first_issue = table.loc[test, "issue_time"].min()
    delay = pd.Timedelta(minutes=cfg["task"]["actual_delay_minutes"])
    published = table.loc[train, "target_time"] + pd.Timedelta("30min") + delay
    assert published.max() <= first_issue
    assert table.loc[test, "target_time"].min() == pd.Timestamp("2025-03-01", tz="UTC")

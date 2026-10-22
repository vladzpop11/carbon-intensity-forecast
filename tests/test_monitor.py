"""Monitoring: drift parsing, the seasonal reference, scoring complete days, and the
retraining decision (SQLite in memory instead of Postgres; no MLflow, no network)."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from carbon_intensity_forecast import db, monitor
from carbon_intensity_forecast.config import load_config

CFG = load_config()
NOW = datetime(2026, 11, 20, 10, tzinfo=UTC)


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setenv("MONITORING_DATABASE_URL", "sqlite://")
    return db.engine(CFG)


def test_drift_direction_depends_on_the_test() -> None:
    metrics = {
        "DriftedColumnsCount(drift_share=0.5)": {"count": 2, "share": 0.5},
        "ValueDrift(column=a,method=K-S p_value,threshold=0.05)": 0.01,  # p-value: drift
        "ValueDrift(column=b,method=K-S p_value,threshold=0.05)": 0.40,  # no drift
        "ValueDrift(column=c,method=PSI,threshold=0.25)": 0.30,  # distance: drift
        "ValueDrift(column=d,method=Wasserstein distance (normed),threshold=0.1)": 0.05,
    }
    assert monitor.drifted_columns(metrics) == ["a", "c"]


def test_the_seasonal_reference_spans_years_and_leaves_a_gap() -> None:
    days = pd.date_range("2024-03-09", "2026-10-31", freq="1D", tz="UTC")
    window = pd.date_range("2026-10-01", "2026-10-28", freq="1D", tz="UTC")
    ref = monitor.same_season_days(days, window, 45)
    assert {d.year for d in ref} == {2024, 2025, 2026}
    assert ref.max() < window[0] - pd.Timedelta("30D")
    assert all(
        min(abs(d.dayofyear - window[14].dayofyear), 365 - abs(d.dayofyear - window[14].dayofyear))
        <= 45
        for d in ref
    )


def put_forecasts(e, day: str, values: dict[str, float]) -> None:
    times = pd.date_range(day, periods=48, freq="30min", tz="UTC")
    rows = [
        pd.DataFrame(
            {
                "issue_day": (times[0] - pd.Timedelta("1D")).date(),
                "target_time": times,
                "method": m,
                "value": v,
                "source": "test",
            }
        )
        for m, v in values.items()
    ]
    db.replace(e, db.FORECASTS, pd.concat(rows))


def test_only_complete_days_are_scored_on_common_half_hours(engine, monkeypatch) -> None:
    put_forecasts(engine, "2026-11-01", {"model": 110.0, "baseline": 150.0, "neso": 105.0})
    put_forecasts(engine, "2026-11-02", {"model": 110.0, "baseline": 150.0})
    times = pd.date_range("2026-11-01", periods=48 + 40, freq="30min", tz="UTC")  # day 2 partial

    def fake_fetch(client, api, start, end):
        return pd.DataFrame({"period_start": times, "actual": 100.0})

    monkeypatch.setattr(monitor, "fetch_intensity", fake_fetch)
    scores = monitor.score(engine, None, CFG, NOW).set_index("method")
    assert set(scores["target_day"].astype(str)) == {"2026-11-01"}
    assert scores.loc["model", "mae"] == 10.0 and scores.loc["neso", "mae"] == 5.0
    assert scores.loc["baseline", "half_hours"] == 48


@pytest.mark.parametrize(
    ("daily_mae", "trained_through", "trigger"),
    [([20.0] * 7, "2026-11-10", None),            # fine
     ([30.0] * 7, "2026-11-10", "performance"),   # 30 > 1.3 x 20.65
     ([30.0] * 3, "2026-11-10", None),            # too few scored days to judge
     ([20.0] * 7, "2026-10-01", "staleness")],    # trained through 50 days ago
)  # fmt: skip
def test_retraining_triggers(engine, monkeypatch, daily_mae, trained_through, trigger) -> None:
    monkeypatch.setattr(
        monitor, "champion", lambda cfg: (None, "3", pd.Timestamp(trained_through, tz="UTC"))
    )
    days = pd.date_range("2026-11-01", periods=len(daily_mae), freq="1D").date
    db.replace(
        engine,
        db.DAILY_SCORES,
        pd.DataFrame({"target_day": days, "method": "model", "mae": daily_mae, "half_hours": 48}),
    )
    got, row = monitor.decide(engine, CFG, NOW, {"share": 0.4, "detected": False})
    assert got == trigger, row["reason"]


def test_no_automatic_retraining_during_the_cooldown(engine, monkeypatch) -> None:
    monkeypatch.setattr(
        monitor, "champion", lambda cfg: (None, "3", pd.Timestamp("2026-10-01", tz="UTC"))
    )
    first, _ = monitor.decide(engine, CFG, NOW, {"share": 0.4, "detected": False})
    second, row = monitor.decide(
        engine, CFG, NOW.replace(day=22), {"share": 0.4, "detected": False}
    )
    assert first == "staleness" and second is None and "cooldown" in row["reason"]

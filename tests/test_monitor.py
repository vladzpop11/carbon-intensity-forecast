"""Monitoring: drift parsing, the seasonal reference, scoring complete days, and the
retraining decision (no MLflow, no network)."""

import os
from datetime import UTC, datetime

import pandas as pd
import pytest

from carbon_intensity_forecast import db, monitor
from carbon_intensity_forecast.config import load_config

CFG = load_config()
NOW = datetime(2026, 11, 20, 10, tzinfo=UTC)


# SQLite in memory always; Postgres too when TEST_DATABASE_URL is set (CI runs a service).
DATABASES = ["sqlite://"] + ([os.environ["TEST_DATABASE_URL"]]
                             if os.environ.get("TEST_DATABASE_URL") else [])  # fmt: skip


@pytest.fixture(params=DATABASES, ids=lambda url: url.split(":")[0])
def engine(request, monkeypatch):
    monkeypatch.setenv("MONITORING_DATABASE_URL", request.param)
    monkeypatch.setattr(monitor, "last_training", lambda cfg: None)  # no training yet
    e = db.engine(CFG)
    db.METADATA.drop_all(e)  # a clean database for every test
    db.METADATA.create_all(e)
    return e


def test_drift_direction_depends_on_the_test() -> None:
    metrics = {
        "DriftedColumnsCount(drift_share=0.5)": {"count": 2, "share": 0.5},
        "ValueDrift(column=a,method=K-S p_value,threshold=0.05)": 0.01,  # p-value: drift
        "ValueDrift(column=b,method=K-S p_value,threshold=0.05)": 0.40,  # no drift
        "ValueDrift(column=c,method=PSI,threshold=0.25)": 0.30,  # distance: drift
        "ValueDrift(column=d,method=Wasserstein distance (normed),threshold=0.1)": 0.05,
    }
    assert monitor.drifted_columns(metrics) == ["a", "c"]


def test_the_seasonal_reference_is_the_same_season_in_earlier_years() -> None:
    days = pd.date_range("2024-03-09", "2026-10-31", freq="1D", tz="UTC")
    window = pd.date_range("2026-10-01", "2026-10-28", freq="1D", tz="UTC")
    ref = monitor.same_season_days(days, window, 45)
    assert {d.year for d in ref} == {2024, 2025}  # not this year's late summer
    assert ref.max() < window[0] - pd.Timedelta("300D")
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


def test_the_cooldown_runs_from_the_last_completed_training(engine, monkeypatch) -> None:
    monkeypatch.setattr(
        monitor, "champion", lambda cfg: (None, "3", pd.Timestamp("2026-10-01", tz="UTC"))
    )
    first, _ = monitor.decide(engine, CFG, NOW, {"share": 0.4, "detected": False})
    assert first == "staleness"
    # The training it started failed (no new version): the next day it tries again.
    again, _ = monitor.decide(engine, CFG, NOW.replace(day=21), {"share": 0.4, "detected": False})
    assert again == "staleness"
    # A training completed (a version registered on the 21st): cooldown.
    monkeypatch.setattr(monitor, "last_training",
                        lambda cfg: pd.Timestamp("2026-11-21 10:00", tz="UTC"))  # fmt: skip
    later, row = monitor.decide(engine, CFG, NOW.replace(day=22), {"share": 0.4, "detected": False})
    assert later is None and "cooldown" in row["reason"]


def test_a_complete_day_without_neso_is_scored_for_the_others(engine, monkeypatch) -> None:
    put_forecasts(engine, "2026-11-01", {"model": 110.0, "baseline": 150.0, "neso": 105.0})
    put_forecasts(engine, "2026-11-02", {"model": 120.0, "baseline": 150.0})  # no NESO archive
    times = pd.date_range("2026-11-01", periods=96, freq="30min", tz="UTC")
    monkeypatch.setattr(
        monitor,
        "fetch_intensity",
        lambda *a: pd.DataFrame({"period_start": times, "actual": 100.0}),
    )
    scores = monitor.score(engine, None, CFG, NOW)
    day2 = scores[scores["target_day"].astype(str) == "2026-11-02"].set_index("method")
    assert set(day2.index) == {"model", "baseline"} and day2.loc["model", "mae"] == 20.0


def test_faulty_actual_values_are_not_scored_as_truth(engine, monkeypatch) -> None:
    put_forecasts(engine, "2026-11-01", {"model": 110.0, "baseline": 150.0})
    times = pd.date_range("2026-11-01", periods=48, freq="30min", tz="UTC")
    actual = [100.0] * 47 + [0.0]  # an estimation outage reported as zero
    monkeypatch.setattr(
        monitor,
        "fetch_intensity",
        lambda *a: pd.DataFrame({"period_start": times, "actual": actual}),
    )
    assert monitor.score(engine, None, CFG, NOW).empty  # the day is now incomplete


class Response:
    def __init__(self, record):
        self.status_code, self.record = 200, record

    def raise_for_status(self):
        pass

    def json(self):
        return self.record


@pytest.mark.parametrize(
    ("read_at", "method"), [("2026-11-01T09:20:00Z", "neso"), ("2026-11-01T12:03:19Z", "neso_late")]
)
def test_neso_read_late_is_kept_apart(engine, read_at, method) -> None:
    put_forecasts(engine, "2026-11-02", {"model": 110.0})  # issue day 2026-11-01
    periods = [{"from": f"{t:%Y-%m-%dT%H:%MZ}", "forecast": 100}
               for t in pd.date_range("2026-11-02", periods=48, freq="30min")]  # fmt: skip
    client = type("C", (), {"get": lambda self, url, timeout: Response(
        {"fetched_at": read_at, "periods": periods})})()  # fmt: skip
    monitor.neso(engine, client, CFG)
    stored = db.read(engine, "select distinct method from forecasts where method like 'neso%'")
    assert list(stored["method"]) == [method]


def test_a_day_the_source_never_completes_is_reported(engine, monkeypatch) -> None:
    put_forecasts(engine, "2026-11-01", {"model": 110.0, "baseline": 150.0})
    times = pd.date_range("2026-11-01", periods=46, freq="30min", tz="UTC")  # 2 never published
    monkeypatch.setattr(
        monitor,
        "fetch_intensity",
        lambda *a: pd.DataFrame({"period_start": times, "actual": 100.0}),
    )
    monitor.score(engine, None, CFG, NOW)  # 19 days later
    unscored = db.read(engine, "select target_day, missing_half_hours from unscored_days")
    assert unscored["missing_half_hours"].tolist() == [2]

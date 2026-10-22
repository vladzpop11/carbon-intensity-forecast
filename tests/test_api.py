"""The API with a stub model and stub live inputs (no network, no MLflow)."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from carbon_intensity_forecast import api, serving
from carbon_intensity_forecast.schemas import ContractError


class StubModel:
    def __init__(self, uri):
        self.uri, self.version, self.columns, self.model = uri, "7", ["a", "b"], object()

    def refresh(self):
        pass


def table(issue_day):
    times = pd.date_range(issue_day + pd.Timedelta("1D"), periods=48, freq="30min")
    return pd.DataFrame({
        "issue_time": issue_day + pd.Timedelta("9h"),
        "target_time": times,
        "lead_hours": 15 + 0.5 * pd.RangeIndex(48),
        "forecast": 100.0,
    })  # fmt: skip


@pytest.fixture
def client(monkeypatch):
    calls = []

    def fake_forecast(model, columns, cfg, now=None):
        calls.append(1)
        day = pd.Timestamp("2026-10-22", tz="UTC")
        return day, table(day)

    monkeypatch.setattr(api, "ServedModel", StubModel)
    monkeypatch.setattr(serving, "forecast_now", fake_forecast)
    monkeypatch.setattr(serving, "latest_issue_day",
                        lambda now, cfg: pd.Timestamp("2026-10-22", tz="UTC"))  # fmt: skip
    with TestClient(api.app) as c:
        yield c, calls


def test_health(client) -> None:
    c, _ = client
    assert c.get("/health").json()["model_version"] == "7"


def test_forecast_has_48_half_hours_and_is_cached(client) -> None:
    c, calls = client
    first = c.get("/forecast").json()
    second = c.get("/forecast").json()
    assert first == second and len(calls) == 1
    assert first["target_day"] == "2026-10-23" and len(first["half_hours"]) == 48
    assert first["half_hours"][0]["period_start"].startswith("2026-10-23T00:00")


def test_inputs_that_break_a_contract_give_503(client, monkeypatch) -> None:
    c, _ = client

    def broken(*args, **kwargs):
        raise ContractError("actual: 5% of values outside 5-1000: the feed is broken")

    monkeypatch.setattr(serving, "forecast_now", broken)
    monkeypatch.setattr(serving, "latest_issue_day",
                        lambda now, cfg: pd.Timestamp("2026-10-23", tz="UTC"))  # fmt: skip
    response = c.get("/forecast")
    assert response.status_code == 503 and "contracts" in response.json()["detail"]
    assert "input_contract_failures_total 1.0" in c.get("/metrics").text


def test_the_latest_issue_day_switches_after_09_15_utc() -> None:
    from carbon_intensity_forecast.config import load_config

    cfg = load_config()
    day = pd.Timestamp("2026-10-22", tz="UTC")
    assert serving.latest_issue_day(day + pd.Timedelta("9h14min"), cfg) == day - pd.Timedelta("1D")
    assert serving.latest_issue_day(day + pd.Timedelta("9h15min"), cfg) == day

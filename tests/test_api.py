"""The API with stub models and stub live inputs (no network, no MLflow)."""

import threading
import time

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from carbon_intensity_forecast import api, serving
from carbon_intensity_forecast.schemas import ContractError

ISSUE_DAY = pd.Timestamp("2026-10-22", tz="UTC")


def snapshot(version="7", trained_through="2026-10-20"):
    return api.Snapshot(object(), version, ["a", "b"], pd.Timestamp(trained_through, tz="UTC"),
                        "models:/m@champion")  # fmt: skip


class StubServed:
    def __init__(self, uri, fallback):
        self.snapshot = snapshot()
        self.stop = threading.Event()

    def watch(self):
        pass


def table(issue_day, value=100.0):
    times = pd.date_range(issue_day + pd.Timedelta("1D"), periods=48, freq="30min")
    return pd.DataFrame({
        "issue_time": issue_day + pd.Timedelta("9h"),
        "target_time": times,
        "lead_hours": 15 + 0.5 * pd.RangeIndex(48),
        "forecast": value,
    })  # fmt: skip


@pytest.fixture
def client(monkeypatch):
    calls = []

    def fake_forecast(model, columns, cfg, issue_day, now):
        calls.append(issue_day)
        time.sleep(0.2)  # long enough for concurrent requests to overlap
        return table(issue_day)

    monkeypatch.setattr(api, "ServedModel", StubServed)
    monkeypatch.setattr(serving, "forecast_for", fake_forecast)
    monkeypatch.setattr(serving, "latest_issue_day", lambda now, cfg: ISSUE_DAY)
    with TestClient(api.app) as c:
        yield c, calls


def test_health(client) -> None:
    c, _ = client
    body = c.get("/health").json()
    assert body["model_version"] == "7" and body["trained_through"] == "2026-10-20"


def test_an_issued_forecast_is_kept_even_when_the_champion_changes(client) -> None:
    c, calls = client
    first = c.get("/forecast").json()
    api.app.state.served.snapshot = snapshot(version="8")
    second = c.get("/forecast").json()
    assert first == second and len(calls) == 1 and first["model_version"] == "7"
    assert first["target_day"] == "2026-10-23" and len(first["half_hours"]) == 48


def test_concurrent_requests_share_one_computation(client) -> None:
    c, calls = client
    threads = [threading.Thread(target=c.get, args=("/forecast",)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1


def test_a_model_never_forecasts_an_issue_day_it_was_trained_through(client) -> None:
    c, calls = client
    api.app.state.served.snapshot = snapshot(trained_through="2026-10-22")
    response = c.get("/forecast")
    assert response.status_code == 503 and "trained through" in response.json()["detail"]
    assert calls == []


@pytest.mark.parametrize(
    ("error", "detail", "metric"),
    [(ContractError("feed is broken"), "contracts", "input_contract_failures_total 1.0"),
     (httpx.ConnectError("down"), "unavailable", "upstream_failures_total 1.0")],
)  # fmt: skip
def test_failures_give_503_are_counted_and_not_retried_at_once(client, monkeypatch, error,
                                                               detail, metric) -> None:  # fmt: skip
    c, _ = client
    attempts = []

    def broken(*args):
        attempts.append(1)
        raise error

    monkeypatch.setattr(serving, "forecast_for", broken)
    for _ in range(2):
        response = c.get("/forecast")
        assert response.status_code == 503 and detail in response.json()["detail"]
    assert len(attempts) == 1  # the failure is remembered for a minute
    assert metric in c.get("/metrics").text


def test_only_get_and_head_reach_the_api_and_its_metrics(client) -> None:
    c, _ = client
    assert c.request("X0RANDOM", "/forecast").status_code == 405
    assert c.post("/forecast").status_code == 405
    assert "X0RANDOM" not in c.get("/metrics").text


def test_the_latest_issue_day_switches_after_09_15_utc() -> None:
    from carbon_intensity_forecast.config import load_config

    cfg = load_config()
    day = pd.Timestamp("2026-10-22", tz="UTC")
    assert serving.latest_issue_day(day + pd.Timedelta("9h14min"), cfg) == day - pd.Timedelta("1D")
    assert serving.latest_issue_day(day + pd.Timedelta("9h15min"), cfg) == day


def test_the_api_leaves_the_fallback_once_the_registry_answers(monkeypatch) -> None:
    loaded = []

    def fake_load(uri):
        loaded.append(uri)
        if uri.startswith("models:/") and len(loaded) == 1:
            raise ConnectionError("MLflow not up yet")
        return api.Snapshot(object(), "3", ["a"], None, uri)

    monkeypatch.setattr(api, "load_snapshot", fake_load)
    monkeypatch.setattr(api, "model_info", lambda uri: (uri, "3", None))
    monkeypatch.setattr(api, "RELOAD_SECONDS", 0)
    served = api.ServedModel("models:/m@champion", "/app/models/champion")
    assert served.snapshot.uri == "/app/models/champion"
    served.stop.wait = lambda timeout: len(loaded) > 2  # one check, then stop
    served.watch()
    assert served.snapshot.uri == "models:/m@champion"  # same version, now from the registry

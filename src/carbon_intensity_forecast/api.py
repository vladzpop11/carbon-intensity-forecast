"""The forecast API.

- GET /forecast: the forecast issued at 09:00 UTC for the next day's 48 half hours. It is
  computed once per issue day from live data and then kept: like a published forecast, it
  does not change when a new champion is loaded later.
- GET /health: the model being served.
- GET /metrics: Prometheus metrics.

The model comes from MODEL_URI: a registry alias (default models:/<name>@champion, with
MLFLOW_TRACKING_URI) or a folder exported from the registry (make export-model). If the
alias cannot be loaded at start (empty registry, MLflow down), the API serves
FALLBACK_MODEL_URI (the champion exported into the image) and logs it. A background thread
checks every RELOAD_SECONDS whether the alias has moved and swaps in the new champion.

A model never forecasts an issue day it was trained through: that forecast is refused
(503) unless it was already issued with an earlier model.
"""

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import mlflow
import pandas as pd
import pandera.errors
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from pydantic import BaseModel

from carbon_intensity_forecast import serving
from carbon_intensity_forecast.config import load_config
from carbon_intensity_forecast.schemas import ContractError

log = logging.getLogger("carbon_intensity_forecast.api")
RELOAD_SECONDS = int(os.environ.get("RELOAD_SECONDS", "600"))
FAILURE_CACHE_SECONDS = 60
LATENCY_BUCKETS = (0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60)

FORECASTS = Counter("forecasts_computed_total", "Forecasts computed (cache misses)")
FORECAST_SECONDS = Histogram("forecast_compute_seconds", "Time to fetch inputs and predict",
                             buckets=LATENCY_BUCKETS)  # fmt: skip
CONTRACT_FAILURES = Counter("input_contract_failures_total", "Live inputs refused by contracts")
UPSTREAM_FAILURES = Counter("upstream_failures_total", "Live inputs unavailable (source APIs)")
RELOAD_FAILURES = Counter("model_reload_failures_total", "Failed checks for a new champion")
MODEL_VERSION = Gauge("model_version", "Registry version of the model being served")


class HalfHour(BaseModel):
    period_start: datetime
    forecast_gco2_per_kwh: float
    lead_hours: float


class Forecast(BaseModel):
    issue_time: datetime
    target_day: str
    model_version: str
    half_hours: list[HalfHour]


@dataclass(frozen=True)
class Snapshot:
    """Everything about the served model, swapped as one."""

    model: object
    version: str
    columns: list[str]
    trained_through: pd.Timestamp | None
    uri: str


def model_info(uri: str) -> tuple[str, str, pd.Timestamp | None]:
    """-> (URI pinned to a version, version, trained_through) for an alias or a folder."""
    if uri.startswith("models:/") and "@" in uri:
        name, alias = uri.removeprefix("models:/").split("@")
        v = mlflow.MlflowClient().get_model_version_by_alias(name, alias)
        trained = v.tags.get("trained_through")
        return (
            f"models:/{name}/{v.version}",
            v.version,
            pd.Timestamp(trained, tz="UTC") if trained else None,
        )
    folder = Path(uri)
    version_file = folder / "registry_version.txt"
    version = version_file.read_text().strip() if version_file.exists() else "local"
    trained_file = folder / "trained_through.txt"
    trained = (
        pd.Timestamp(trained_file.read_text().strip(), tz="UTC") if trained_file.exists() else None
    )
    return uri, version, trained


def load_snapshot(uri: str) -> Snapshot:
    pinned, version, trained = model_info(uri)
    model = mlflow.pyfunc.load_model(pinned)
    columns = model.metadata.get_input_schema().input_names()
    return Snapshot(model, version, columns, trained, uri)


class ServedModel:
    """The current snapshot; a background thread swaps in a new champion."""

    def __init__(self, uri: str, fallback: str | None):
        self.uri = uri
        try:
            self.snapshot = load_snapshot(uri)
        except Exception:
            if not fallback or fallback == uri:
                raise
            log.exception("could not load %s; serving the fallback %s", uri, fallback)
            self.snapshot = load_snapshot(fallback)
        MODEL_VERSION.set(float(self.snapshot.version) if self.snapshot.version.isdigit() else -1)
        self.stop = threading.Event()

    def watch(self) -> None:
        """Reload when the alias moves (registry URIs only); failures keep the current model."""
        if not self.uri.startswith("models:/"):
            return
        while not self.stop.wait(RELOAD_SECONDS):
            try:
                _, version, _ = model_info(self.uri)
                if version != self.snapshot.version:
                    self.snapshot = load_snapshot(self.uri)
                    MODEL_VERSION.set(float(version))
                    log.info("now serving model version %s", version)
            except Exception:
                RELOAD_FAILURES.inc()
                log.exception("checking for a new champion failed; keeping version %s",
                              self.snapshot.version)  # fmt: skip


class Forecasts:
    """Issued forecasts, one per issue day, computed once (a lock per issue day, so
    concurrent requests share one computation); recent failures are remembered briefly."""

    def __init__(self):
        self.lock = threading.Lock()
        self.issued: dict[pd.Timestamp, Forecast] = {}
        self.failed: dict[pd.Timestamp, tuple[float, HTTPException]] = {}
        self.day_locks: dict[pd.Timestamp, threading.Lock] = {}

    def get(self, issue_day: pd.Timestamp, now: datetime, snapshot: Snapshot, cfg: dict):
        with self.lock:
            if issue_day in self.issued:
                return self.issued[issue_day]
            day_lock = self.day_locks.setdefault(issue_day, threading.Lock())
        with day_lock:
            with self.lock:
                if issue_day in self.issued:
                    return self.issued[issue_day]
                failure = self.failed.get(issue_day)
                if failure and time.monotonic() - failure[0] < FAILURE_CACHE_SECONDS:
                    raise failure[1]
            try:
                forecast = compute(issue_day, now, snapshot, cfg)
            except HTTPException as exc:
                with self.lock:
                    self.failed[issue_day] = (time.monotonic(), exc)
                raise
            with self.lock:
                self.issued = {
                    d: f for d, f in self.issued.items() if d >= issue_day - pd.Timedelta("7D")
                }
                self.issued[issue_day] = forecast
                self.failed.pop(issue_day, None)
            return forecast


def compute(issue_day: pd.Timestamp, now: datetime, s: Snapshot, cfg: dict) -> Forecast:
    if s.trained_through is not None and s.trained_through >= issue_day:
        raise HTTPException(503, f"model version {s.version} was trained through "
                                 f"{s.trained_through:%Y-%m-%d}: it cannot forecast issue day "
                                 f"{issue_day:%Y-%m-%d}")  # fmt: skip
    start = time.perf_counter()
    try:
        table = serving.forecast_for(s.model, s.columns, cfg, issue_day, now)
    except (ContractError, pandera.errors.SchemaError) as exc:
        CONTRACT_FAILURES.inc()
        log.warning(
            "live inputs for issue day %s failed their contracts: %s", issue_day.date(), exc
        )
        raise HTTPException(503, "live inputs failed their data contracts") from exc
    except httpx.HTTPError as exc:
        UPSTREAM_FAILURES.inc()
        log.warning("live inputs for issue day %s unavailable: %r", issue_day.date(), exc)
        raise HTTPException(503, "live inputs are unavailable; try again later") from exc
    FORECAST_SECONDS.observe(time.perf_counter() - start)
    FORECASTS.inc()
    return Forecast(
        issue_time=table["issue_time"].iloc[0],
        target_day=f"{(issue_day + pd.Timedelta('1D')):%Y-%m-%d}",
        model_version=s.version,
        half_hours=[HalfHour(period_start=t, forecast_gco2_per_kwh=round(f, 2), lead_hours=lead)
                    for t, f, lead in zip(table["target_time"], table["forecast"],
                                          table["lead_hours"], strict=True)],
    )  # fmt: skip


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    uri = os.environ.get("MODEL_URI", f"models:/{cfg['registry']['model_name']}@champion")
    app.state.cfg = cfg
    app.state.served = ServedModel(uri, os.environ.get("FALLBACK_MODEL_URI"))
    app.state.forecasts = Forecasts()
    watcher = threading.Thread(target=app.state.served.watch, daemon=True)
    watcher.start()
    yield
    app.state.served.stop.set()


app = FastAPI(title="Carbon Intensity Forecast", version="1.0", lifespan=lifespan)


instrumentator = Instrumentator(excluded_handlers=["/metrics"])
instrumentator.add(metrics.default(latency_lowr_buckets=LATENCY_BUCKETS))
instrumentator.instrument(app).expose(app)


# Added after the instrumentation, so it runs first (Starlette's last-added middleware is the
# outermost) and a request with another method never reaches the metrics.
@app.middleware("http")
async def only_reads(request: Request, call_next):
    """Only GET and HEAD exist; refusing other methods first also keeps the metrics' method
    label bounded."""
    if request.method not in {"GET", "HEAD"}:
        return PlainTextResponse("Method Not Allowed", status_code=405)
    return await call_next(request)


@app.get("/health")
def health() -> dict:
    s = app.state.served.snapshot
    trained = None if s.trained_through is None else f"{s.trained_through:%Y-%m-%d}"
    return {"status": "ok", "model_uri": s.uri, "model_version": s.version,
            "trained_through": trained, "features": len(s.columns)}  # fmt: skip


@app.get("/forecast", response_model=Forecast)
def forecast() -> Forecast:
    now = datetime.now(UTC)  # one reading of the clock per request
    issue_day = serving.latest_issue_day(now, app.state.cfg)
    snapshot = app.state.served.snapshot  # read once per request
    return app.state.forecasts.get(issue_day, now, snapshot, app.state.cfg)

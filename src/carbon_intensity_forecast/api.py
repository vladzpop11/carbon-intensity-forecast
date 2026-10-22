"""The forecast API.

- GET /forecast: the most recent day-ahead forecast (issued at 09:00 UTC for the next day),
  computed from live data with the champion model and cached per (model version, issue day).
- GET /health: the model being served.
- GET /metrics: Prometheus metrics (requests, latency, forecasts, model version, input
  contract failures).

The model comes from MODEL_URI: a registry alias (default models:/<name>@champion, with
MLFLOW_TRACKING_URI) or a local folder exported from the registry (make export-model), for
places that cannot reach MLflow. With an alias, the API checks every RELOAD_SECONDS
whether it has moved and loads the new champion.
"""

import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd
import pandera.errors
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from carbon_intensity_forecast import serving
from carbon_intensity_forecast.config import load_config
from carbon_intensity_forecast.schemas import ContractError

RELOAD_SECONDS = int(os.environ.get("RELOAD_SECONDS", "600"))

FORECASTS = Counter("forecasts_computed_total", "Forecasts computed (cache misses)")
FORECAST_SECONDS = Histogram("forecast_compute_seconds", "Time to fetch inputs and predict")
CONTRACT_FAILURES = Counter("input_contract_failures_total", "Live inputs refused by contracts")
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


class ServedModel:
    """The loaded model, its version and input columns; reloaded when the alias moves."""

    def __init__(self, uri: str):
        self.uri = uri
        self.lock = threading.Lock()
        self.model = self.version = self.columns = None
        self.checked = 0.0
        self.load()

    def current_version(self) -> str:
        if self.uri.startswith("models:/") and "@" in self.uri:
            name, alias = self.uri.removeprefix("models:/").split("@")
            return mlflow.MlflowClient().get_model_version_by_alias(name, alias).version
        meta = Path(self.uri) / "registry_version.txt"
        return meta.read_text().strip() if meta.exists() else "local"

    def load(self) -> None:
        version = self.current_version()
        model = mlflow.pyfunc.load_model(self.uri)
        self.model, self.version = model, version
        self.columns = model.metadata.get_input_schema().input_names()
        MODEL_VERSION.set(float(version) if version.isdigit() else -1)
        self.checked = time.monotonic()

    def refresh(self) -> None:
        """Reload if the alias now points to another version (checked at most every
        RELOAD_SECONDS; a failed check keeps serving the loaded model)."""
        if time.monotonic() - self.checked < RELOAD_SECONDS:
            return
        with self.lock:
            self.checked = time.monotonic()
            try:
                if self.current_version() != self.version:
                    self.load()
            except Exception:  # noqa: BLE001 - registry unreachable: keep the loaded model
                pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    uri = os.environ.get("MODEL_URI", f"models:/{cfg['registry']['model_name']}@champion")
    app.state.cfg = cfg
    app.state.served = ServedModel(uri)
    app.state.cache = {}
    app.state.cache_lock = threading.Lock()
    yield


app = FastAPI(title="Carbon Intensity Forecast", version="1.0", lifespan=lifespan)
Instrumentator(excluded_handlers=["/metrics"]).instrument(app).expose(app)


@app.get("/health")
def health() -> dict:
    served = app.state.served
    return {"status": "ok", "model_uri": served.uri, "model_version": served.version,
            "features": len(served.columns)}  # fmt: skip


@app.get("/forecast", response_model=Forecast)
def forecast() -> Forecast:
    served = app.state.served
    served.refresh()
    cfg = app.state.cfg
    issue_day = serving.latest_issue_day(datetime.now(UTC), cfg)
    key = (served.version, issue_day)
    with app.state.cache_lock:
        cached = app.state.cache.get(key)
    if cached is None:
        start = time.perf_counter()
        try:
            _, table = serving.forecast_now(served.model, served.columns, cfg)
        except (ContractError, pandera.errors.SchemaError) as exc:
            CONTRACT_FAILURES.inc()
            raise HTTPException(503, "live inputs failed their data contracts") from exc
        except Exception as exc:  # noqa: BLE001 - upstream APIs unreachable or malformed
            raise HTTPException(503, "live inputs are unavailable; try again later") from exc
        FORECAST_SECONDS.observe(time.perf_counter() - start)
        FORECASTS.inc()
        cached = Forecast(
            issue_time=table["issue_time"].iloc[0],
            target_day=f"{(issue_day + pd.Timedelta('1D')):%Y-%m-%d}",
            model_version=served.version,
            half_hours=[HalfHour(period_start=t, forecast_gco2_per_kwh=round(f, 2),
                                 lead_hours=lead)
                        for t, f, lead in zip(table["target_time"], table["forecast"],
                                              table["lead_hours"], strict=True)],
        )  # fmt: skip
        with app.state.cache_lock:
            app.state.cache = {k: v for k, v in app.state.cache.items() if k[1] >= issue_day}
            app.state.cache[key] = cached
    return cached

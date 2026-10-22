"""Daily monitoring (configs/config.yaml, monitoring):

1. Issue: store the champion's forecast and the baseline for every issue day since
   `first_live_issue_day` that has none yet. A forecast uses only what was published by
   09:00 on its issue day, so a day missed (the Mac was off) is filled in later with the
   same result; a model never forecasts an issue day it was trained on.
2. Benchmark: store NESO's day-ahead forecast as archived on that issue day (archive.yml).
3. Score: once a target day's 48 actual values are in, MAE per method on the half hours
   every method covers.
4. Drift (Evidently), a diagnostic: the model's inputs over the last 28 issue days against
   the same season in earlier years, with PSI. Report in reports/drift/.
5. Decide: the retraining triggers (performance, staleness); returns the trigger or None.

python -m carbon_intensity_forecast.monitor   (needs Postgres and MLflow: make up)
"""

import os
from datetime import UTC, datetime

import httpx
import mlflow
import numpy as np
import pandas as pd

from carbon_intensity_forecast import db, serving
from carbon_intensity_forecast import features as F
from carbon_intensity_forecast.config import load_config, project_path
from carbon_intensity_forecast.ingest import fetch_intensity

DAY = pd.Timedelta("1D")


def utc(value) -> pd.Timestamp:
    """A timestamp from the database in UTC (Postgres returns it aware, SQLite naive)."""
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


# Inputs worth watching for drift: weather and recent carbon intensity, not the calendar.
NOT_DRIFT = {"slot", "lead_hours", "hour", "day_of_week", "bank_holiday", "season_sin",
             "season_cos", "weather_lead_days"}  # fmt: skip


def champion(cfg: dict):
    name = cfg["registry"]["model_name"]
    version = mlflow.MlflowClient().get_model_version_by_alias(name, "champion")
    model = mlflow.pyfunc.load_model(f"models:/{name}@champion")
    trained = pd.Timestamp(version.tags["trained_through"], tz="UTC")
    return model, version.version, trained


def issue(e, client: httpx.Client, cfg: dict, now: datetime) -> list[str]:
    model, version, trained_through = champion(cfg)
    columns = model.metadata.get_input_schema().input_names()
    first = pd.Timestamp(cfg["monitoring"]["first_live_issue_day"], tz="UTC")
    last = serving.latest_issue_day(now, cfg)
    done = set(db.read(e, "select distinct issue_day from forecasts where method = 'model'")
               ["issue_day"].astype(str))  # fmt: skip
    notes = []
    for day in pd.date_range(first, last, freq="1D"):
        if f"{day:%Y-%m-%d}" in done:
            continue
        if trained_through >= day:
            notes.append(f"{day:%Y-%m-%d}: skipped, the champion was trained through it")
            continue
        intensity, weather = serving.live_inputs(client, cfg, day, now)
        table = F.build(pd.DatetimeIndex([day]), intensity, weather, cfg)
        table["model"] = np.asarray(model.predict(table[columns]), dtype=float)
        rows = []
        methods = [("model", "model", f"version {version}"),
                   ("baseline", "same_time_latest", "same_time_latest")]  # fmt: skip
        for method, column, source in methods:
            part = table[["target_time", column]].dropna()
            rows.append(pd.DataFrame({"issue_day": day.date(), "target_time": part["target_time"],
                                      "method": method, "value": part[column],
                                      "source": source}))  # fmt: skip
        db.replace(e, db.FORECASTS, pd.concat(rows))
        notes.append(f"{day:%Y-%m-%d}: forecast with version {version}")
    return notes


def neso(e, client: httpx.Client, cfg: dict) -> list[str]:
    """NESO's forecast for each issue day's target day, from the archive branch."""
    have = set(db.read(e, "select distinct issue_day from forecasts where method = 'neso'")
               ["issue_day"].astype(str))  # fmt: skip
    days = db.read(e, "select distinct issue_day from forecasts where method = 'model'")
    notes = []
    for day in pd.to_datetime(days["issue_day"]):
        if f"{day:%Y-%m-%d}" in have:
            continue
        url = f"{cfg['monitoring']['neso_archive_url']}/{day:%Y/%m/%Y-%m-%d}.json"
        response = client.get(url, timeout=30)
        if response.status_code == 404:
            notes.append(f"{day:%Y-%m-%d}: no NESO archive")
            continue
        response.raise_for_status()
        record = response.json()
        periods = pd.DataFrame(record["periods"])
        periods["target_time"] = pd.to_datetime(periods["from"], utc=True)
        target = day.tz_localize("UTC") + DAY
        inside = (periods["target_time"] >= target) & (periods["target_time"] < target + DAY)
        periods = periods[inside].dropna(subset=["forecast"])
        db.replace(e, db.FORECASTS, pd.DataFrame({
            "issue_day": day.date(), "target_time": periods["target_time"], "method": "neso",
            "value": periods["forecast"].astype(float), "source": f"read {record['fetched_at']}",
        }))  # fmt: skip
        notes.append(f"{day:%Y-%m-%d}: NESO forecast read at {record['fetched_at']}")
    return notes


def score(e, client: httpx.Client, cfg: dict, now: datetime) -> pd.DataFrame:
    """Store the actual values of the forecast days and score each complete day."""
    span = db.read(e, "select min(target_time) as a, max(target_time) as b from forecasts")
    if span["a"].isna().all():
        return pd.DataFrame()
    start = utc(span["a"].iloc[0]).to_pydatetime()
    end = min(utc(span["b"].iloc[0]) + pd.Timedelta("30min"),
              pd.Timestamp(now).floor("30min")).to_pydatetime()  # fmt: skip
    actual = fetch_intensity(client, cfg["data"]["intensity_api"], start, end)
    db.replace(e, db.ACTUALS, actual[["period_start", "actual"]])
    joined = db.read(e, """
        select f.target_time, f.method, f.value, a.actual
        from forecasts f join actuals a on a.period_start = f.target_time
        where a.actual is not null""")  # fmt: skip
    if joined.empty:
        return pd.DataFrame()
    joined["target_day"] = pd.to_datetime(joined["target_time"], utc=True).dt.date
    wide = joined.pivot_table(index=["target_day", "target_time"], columns="method",
                              values=["value", "actual"])  # fmt: skip
    rows = []
    for day, part in wide.groupby(level="target_day"):
        if len(part) < 48:  # the day is not complete yet
            continue
        values = part["value"].dropna()  # half hours every method covers
        actual = part["actual"].iloc[:, 0].loc[values.index]
        for method in values.columns:
            rows.append({"target_day": day, "method": method, "half_hours": len(values),
                         "mae": float((values[method] - actual).abs().mean())})  # fmt: skip
    scores = pd.DataFrame(rows)
    db.replace(e, db.DAILY_SCORES, scores)
    return scores


def same_season_days(days: pd.DatetimeIndex, window: pd.DatetimeIndex,
                     season_days: int = 45) -> pd.DatetimeIndex:  # fmt: skip
    """Days in `days` that fall within `season_days` of the window's middle calendar date in
    any year, ending at least 30 days before the window."""
    middle = window[len(window) // 2].dayofyear
    gap = np.abs(days.dayofyear - middle)
    near = np.minimum(gap, 365 - gap) <= season_days
    return days[near & (days < window[0] - pd.Timedelta("30D"))]


def drifted_columns(metrics: dict) -> list[str]:
    """Columns Evidently found drifted. It picks the test per column by sample size: for a
    p-value test drift is a value below the threshold, for a distance a value at or above
    it. Names look like "ValueDrift(column=x,method=K-S p_value,threshold=0.05)"."""
    out = []
    for name, value in metrics.items():
        if not name.startswith("ValueDrift("):
            continue
        args = dict(part.split("=", 1) for part in name[len("ValueDrift(") : -1].split(","))
        threshold = float(args["threshold"])
        drifted = value < threshold if "p_value" in args["method"] else value >= threshold
        if drifted:
            out.append(args["column"])
    return sorted(out)


def drift(e, cfg: dict, now: datetime) -> dict:
    """Evidently data drift of the inputs: last window against the same days a year
    earlier, from data/raw (refreshed by the pipeline's ingest step)."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    d = cfg["monitoring"]["drift"]
    intensity, weather = F.load_raw(cfg)
    last = serving.latest_issue_day(now, cfg) - DAY  # the last issue day with full inputs
    current_days = pd.date_range(last - (d["window_days"] - 1) * DAY, last, freq="1D")
    first = pd.Timestamp(cfg["weather"]["start"], tz="UTC") + DAY
    history = pd.date_range(first, current_days[0] - DAY, freq="1D")
    reference_days = same_season_days(history, current_days, d["season_days"])
    current = F.build(current_days, intensity, weather, cfg)
    reference = F.build(reference_days, intensity, weather, cfg)
    columns = [c for c in F.feature_columns(current) if c not in NOT_DRIFT]
    preset = DataDriftPreset(drift_share=d["share_threshold"], method=d["method"],
                             threshold=d["threshold"])  # fmt: skip
    snapshot = Report([preset]).run(
        reference_data=reference[columns], current_data=current[columns]
    )
    out = project_path("reports/drift")
    out.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out / f"{now:%Y-%m-%d}.html"))
    metrics = {m["metric_name"]: m["value"] for m in snapshot.dict()["metrics"]}
    share = next(v["share"] for k, v in metrics.items() if k.startswith("DriftedColumnsCount"))
    drifted = drifted_columns(metrics)
    row = {
        "run_day": now.date(),
        "share": float(share),
        "features": len(columns),
        "detected": share >= d["share_threshold"],
        "drifted": ",".join(sorted(drifted)),
    }
    db.replace(e, db.DRIFT, pd.DataFrame([row]))
    return row


def decide(e, cfg: dict, now: datetime, drift_row: dict) -> tuple[str | None, dict]:
    """The retraining triggers (configs/config.yaml, monitoring.retrain)."""
    r = cfg["monitoring"]["retrain"]
    _, version, trained_through = champion(cfg)
    scores = db.read(e, """select target_day, mae from daily_scores
                           where method = 'model' order by target_day""")  # fmt: skip
    recent = scores.tail(r["rolling_days"])
    rolling = float(recent["mae"].mean()) if len(recent) else None
    age = (pd.Timestamp(now).normalize() - trained_through).days
    last_retrain = db.read(e, "select max(run_at) as t from decisions where trigger is not null")
    since = None if last_retrain["t"].isna().all() else (
        pd.Timestamp(now) - utc(last_retrain["t"].iloc[0])).days  # fmt: skip
    trigger, reason = None, "no trigger"
    enough = len(recent) >= r["min_scored_days"]
    if since is not None and since < r["cooldown_days"]:
        reason = f"cooldown: last retraining {since} days ago"
    elif enough and rolling > r["max_mae_ratio"] * r["reference_mae"]:
        trigger = "performance"
        reason = f"rolling MAE {rolling:.1f} > {r['max_mae_ratio']} x {r['reference_mae']}"
    elif age > r["max_model_age_days"]:
        trigger = "staleness"
        reason = f"champion trained through {trained_through:%Y-%m-%d} ({age} days ago)"
    elif not enough:
        reason = f"{len(recent)} scored days (needs {r['min_scored_days']} to judge performance)"
    # Drift is reported with every decision, as context, not a trigger.
    reason += f"; drift share {drift_row['share']:.2f}"
    row = {"run_at": now, "trigger": trigger, "rolling_mae": rolling, "scored_days": len(recent),
           "model_version": version, "model_age_days": age, "reason": reason}  # fmt: skip
    db.replace(e, db.DECISIONS, pd.DataFrame([row]))
    return trigger, row


def main(now: datetime | None = None) -> str | None:
    cfg = load_config()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    now = now or datetime.now(UTC)
    e = db.engine(cfg)
    with httpx.Client(headers={"Accept": "application/json"}) as client:
        for note in issue(e, client, cfg, now) + neso(e, client, cfg):
            print(note)
        scores = score(e, client, cfg, now)
    if not scores.empty:
        print(scores.pivot(index="target_day", columns="method", values="mae").round(1))
    drift_row = drift(e, cfg, now)
    print(f"drift share {drift_row['share']:.2f} over {drift_row['features']} inputs "
          f"({'detected' if drift_row['detected'] else 'not detected'})")  # fmt: skip
    trigger, row = decide(e, cfg, now, drift_row)
    print(f"decision: {row['reason']}")
    return trigger


if __name__ == "__main__":
    main()

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
from carbon_intensity_forecast.ingest import Client, fetch_intensity
from carbon_intensity_forecast.schemas import INTENSITY_LIMITS, ContractError, clean

DAY = pd.Timedelta("1D")
# A target day still missing actual values this long after it ended is reported as unscored.
UNSCORED_AFTER = pd.Timedelta("2D")


def utc(value) -> pd.Timestamp:
    """A timestamp from the database in UTC (Postgres returns it aware, SQLite naive)."""
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


# Inputs worth watching for drift: weather and recent carbon intensity, not the calendar.
NOT_DRIFT = {"slot", "lead_hours", "hour", "day_of_week", "bank_holiday", "season_sin",
             "season_cos", "weather_lead_days"}  # fmt: skip


def champion(cfg: dict):
    """-> (model, version, trained_through) of the champion, loaded by its exact version so
    the version and the model cannot disagree."""
    name = cfg["registry"]["model_name"]
    v = mlflow.MlflowClient().get_model_version_by_alias(name, "champion")
    model = mlflow.pyfunc.load_model(f"models:/{name}/{v.version}")
    return model, v.version, pd.Timestamp(v.tags["trained_through"], tz="UTC")


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
        try:  # the same path as the API, completeness contract included
            intensity, weather = serving.live_inputs(client, cfg, day, now)
            table = serving.predict(model, columns, cfg, day, intensity, weather)
        except ContractError as exc:
            notes.append(f"{day:%Y-%m-%d}: not issued, inputs refused ({exc}); retried next run")
            continue
        rows = []
        methods = [("model", "forecast", f"version {version}"),
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
    """NESO's forecast for each issue day's target day, from the archive branch. It counts
    as the day-ahead benchmark ("neso") only if it was read within neso_max_read_delay of the
    issue time; a later read had hours more information and is kept as "neso_late"."""
    days = db.read(e, "select distinct issue_day from forecasts where method = 'model'")
    late_after = pd.Timedelta(minutes=cfg["monitoring"]["neso_max_read_delay_minutes"])
    issue_hour = pd.Timedelta(hours=cfg["task"]["issue_hour_utc"])
    notes = []
    for day in pd.to_datetime(days["issue_day"]):
        url = f"{cfg['monitoring']['neso_archive_url']}/{day:%Y/%m/%Y-%m-%d}.json"
        response = client.get(url, timeout=30)
        if response.status_code == 404:
            notes.append(f"{day:%Y-%m-%d}: no NESO archive")
            continue
        response.raise_for_status()
        record = response.json()
        issue_time = day.tz_localize("UTC") + issue_hour
        read_at = pd.Timestamp(record["fetched_at"])
        method = "neso" if read_at <= issue_time + late_after else "neso_late"
        periods = pd.DataFrame(record["periods"])
        periods["target_time"] = pd.to_datetime(periods["from"], utc=True)
        target = day.tz_localize("UTC") + DAY
        inside = (periods["target_time"] >= target) & (periods["target_time"] < target + DAY)
        periods = periods[inside].dropna(subset=["forecast"])
        with e.begin() as conn:  # one label per day, even if the rule changed since
            conn.execute(db.FORECASTS.delete().where(
                db.FORECASTS.c.issue_day == day.date(),
                db.FORECASTS.c.method.in_(["neso", "neso_late"])))  # fmt: skip
        db.replace(e, db.FORECASTS, pd.DataFrame({
            "issue_day": day.date(), "target_time": periods["target_time"], "method": method,
            "value": periods["forecast"].astype(float), "source": f"read {record['fetched_at']}",
        }))  # fmt: skip
        notes.append(f"{day:%Y-%m-%d}: NESO forecast read at {record['fetched_at']} ({method})")
    return notes


def score(e, client: httpx.Client, cfg: dict, now: datetime) -> pd.DataFrame:
    """Store the actual values of the forecast days (through the data contract) and score
    each complete day on the half hours every method present that day covers."""
    span = db.read(e, "select min(target_time) as a, max(target_time) as b from forecasts")
    if span["a"].isna().all():
        return pd.DataFrame()
    start = utc(span["a"].iloc[0]).to_pydatetime()
    end = min(utc(span["b"].iloc[0]) + pd.Timedelta("30min"),
              pd.Timestamp(now).floor("30min")).to_pydatetime()  # fmt: skip
    actual = fetch_intensity(client, cfg["data"]["intensity_api"], start, end)
    # Faults become missing values (that day is then not scored), never a refused batch: a
    # short window with one fault must not stall monitoring.
    actual, _ = clean(actual, {"actual": INTENSITY_LIMITS["actual"]}, max_share=1.0)
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
    rows, unscored = [], []
    grace_end = (pd.Timestamp(now).tz_convert("UTC") - UNSCORED_AFTER).date()
    for day, part in wide.groupby(level="target_day"):
        if len(part) < 48:  # the day is not complete yet, or the source has gaps
            if day < grace_end:
                unscored.append({"target_day": day, "missing_half_hours": 48 - len(part)})
            continue
        values = part["value"].dropna(axis=1, how="all").dropna()  # methods present that day
        if values.empty:
            continue
        actual = part["actual"].iloc[:, 0].loc[values.index]
        for method in values.columns:
            rows.append({"target_day": day, "method": method, "half_hours": len(values),
                         "mae": float((values[method] - actual).abs().mean())})  # fmt: skip
    scores = pd.DataFrame(rows)
    db.replace(e, db.DAILY_SCORES, scores)
    db.replace(e, db.UNSCORED, pd.DataFrame(unscored))
    return scores


def same_season_days(days: pd.DatetimeIndex, window: pd.DatetimeIndex,
                     season_days: int = 45) -> pd.DatetimeIndex:  # fmt: skip
    """Days in `days` within `season_days` of the window's middle calendar date in earlier
    years (at least 300 days before the window, so never the same season of this year)."""
    middle = window[len(window) // 2].dayofyear
    gap = np.abs(days.dayofyear - middle)
    near = np.minimum(gap, 365 - gap) <= season_days
    return days[near & (days < window[0] - pd.Timedelta("300D"))]


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


def last_training(cfg: dict) -> pd.Timestamp | None:
    """When the newest model version was registered (None before the first)."""
    name = cfg["registry"]["model_name"]
    versions = mlflow.MlflowClient().search_model_versions(f"name='{name}'")
    if not versions:
        return None
    return pd.Timestamp(max(v.creation_timestamp for v in versions), unit="ms", tz="UTC")


def decide(e, cfg: dict, now: datetime, drift_row: dict) -> tuple[str | None, dict]:
    """The retraining triggers (configs/config.yaml, monitoring.retrain)."""
    r = cfg["monitoring"]["retrain"]
    _, version, trained_through = champion(cfg)
    scores = db.read(e, """select target_day, mae from daily_scores
                           where method = 'model' order by target_day""")  # fmt: skip
    recent = scores.tail(r["rolling_days"])
    rolling = float(recent["mae"].mean()) if len(recent) else None
    age = (pd.Timestamp(now).normalize() - trained_through).days
    # The cooldown runs from the last completed training (a registered version), so a
    # training that failed is retried the next day.
    trained_at = last_training(cfg)
    since = None if trained_at is None else (pd.Timestamp(now) - trained_at).days
    trigger, reason = None, "no trigger"
    enough = len(recent) >= r["min_scored_days"]
    if since is not None and since < r["cooldown_days"]:
        reason = f"cooldown: last training {since} days ago"
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
    with Client() as client:
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

"""Production training with promotion gates (configs/config.yaml, registry).

1. An evaluation twin is trained on everything known before the holdout window (the last
   `holdout_days` target days) and forecasts the window.
2. Gates, on the same holdout rows: the twin must beat the baseline by the required skill
   and have a strictly lower MAE than the current champion (when there is one).
3. The same setup is refit on all known data and registered as a new model version (alias
   `challenger`). If the gates passed, the `champion` alias moves to it; the version keeps
   the gate results and the previous champion as tags, so a rollback can return to it.

Every run is logged to the MLflow experiment "production-training" and summarised in
reports/training.json.

python -m carbon_intensity_forecast.train
"""

import os
from datetime import UTC, datetime

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from carbon_intensity_forecast import features as F
from carbon_intensity_forecast.artifacts import write_json
from carbon_intensity_forecast.backtest import fit, mae
from carbon_intensity_forecast.config import load_config, project_path
from carbon_intensity_forecast.experiments import provenance

EXPERIMENT = "production-training"


def last_complete_day(intensity: pd.DataFrame) -> pd.Timestamp:
    """The last UTC day whose final half hour has an actual value."""
    known = intensity.loc[intensity["actual"].notna(), "period_start"].max()
    day = known.normalize()
    return day if known.hour == 23 and known.minute == 30 else day - pd.Timedelta("1D")


def champion_predictions(client: MlflowClient, name: str, rows: pd.DataFrame):
    """(version, predictions) of the current champion, or (None, None) without one."""
    try:
        version = client.get_model_version_by_alias(name, "champion")
    except MlflowException:
        return None, None
    model = mlflow.pyfunc.load_model(f"models:/{name}@champion")
    columns = model.metadata.get_input_schema().input_names()
    return version, np.asarray(model.predict(rows[columns]))


def gates(twin: float, baseline: float, champion: float | None, cfg: dict) -> dict:
    g = cfg["registry"]["gates"]
    skill = 1 - twin / baseline
    result = {"skill_vs_baseline": skill,
              "beats_baseline": skill >= g["min_skill_vs_baseline"]}  # fmt: skip
    if g["beat_champion"]:
        result["beats_champion"] = True if champion is None else twin < champion
    result["passed"] = all(v for k, v in result.items() if k.startswith("beats"))
    return result


def main() -> None:
    cfg = load_config()
    reg = cfg["registry"]
    name = reg["model_name"]
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    intensity, weather = F.load_raw(cfg)
    last_day = last_complete_day(intensity)
    first = pd.Timestamp(cfg["weather"]["start"], tz="UTC") + pd.Timedelta("1D")
    table = F.build(pd.date_range(first, last_day - pd.Timedelta("1D"), freq="1D"),
                    intensity, weather, cfg)  # fmt: skip
    columns = F.feature_columns(table)
    params = cfg["model"]["params"]
    known = table["actual"].notna()
    target_day = table["target_time"].dt.normalize()
    holdout_start = last_day - pd.Timedelta(days=reg["holdout_days"] - 1)
    holdout = ((target_day >= holdout_start) & known).to_numpy()
    # As in the backtest: training targets end before the first holdout issue time.
    before = ((target_day < holdout_start - pd.Timedelta("1D")) & known).to_numpy()

    rows = table.loc[holdout]
    twin = fit(table, before, params, columns)
    twin_mae = mae(twin.predict(rows[columns]), rows["actual"])
    base_rows = rows[reg["gates"]["baseline"]].notna()
    baseline_mae = mae(rows.loc[base_rows, reg["gates"]["baseline"]],
                       rows.loc[base_rows, "actual"])  # fmt: skip
    champion, champion_pred = champion_predictions(client, name, rows)
    champion_mae = None if champion is None else mae(champion_pred, rows["actual"])
    decision = gates(twin_mae, baseline_mae, champion_mae, cfg)

    final = fit(table, known.to_numpy(), params, columns)
    tags = {**provenance(), "trained_through": f"{last_day:%Y-%m-%d}",
            "holdout": f"{holdout_start:%Y-%m-%d}..{last_day:%Y-%m-%d}",
            "gates_passed": str(decision["passed"])}  # fmt: skip
    with mlflow.start_run(run_name=f"train-{last_day:%Y-%m-%d}") as run:
        mlflow.set_tags(tags)
        mlflow.log_params({**params, "features": len(columns), "train_rows": int(known.sum()),
                           "holdout_rows": int(holdout.sum())})  # fmt: skip
        metrics = {"holdout_mae_challenger": twin_mae, "holdout_mae_baseline": baseline_mae,
                   "skill_vs_baseline": decision["skill_vs_baseline"]}  # fmt: skip
        if champion_mae is not None:
            metrics["holdout_mae_champion"] = champion_mae
        mlflow.log_metrics(metrics)
        info = mlflow.lightgbm.log_model(final.booster_, name="model",
                                         input_example=table.loc[known, columns].head(5),
                                         registered_model_name=name)  # fmt: skip
    version = info.registered_model_version
    for key, value in {**tags, **{k: f"{v:.4f}" for k, v in metrics.items()}}.items():
        client.set_model_version_tag(name, version, key, value)
    client.set_registered_model_alias(name, "challenger", version)
    if decision["passed"]:
        if champion is not None:
            client.set_model_version_tag(name, version, "previous_champion", champion.version)
        client.set_registered_model_alias(name, "champion", version)

    record = {
        "run_at": f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}",
        "mlflow_run": run.info.run_id,
        "model_version": version,
        "previous_champion": None if champion is None else champion.version,
        "promoted": decision["passed"],
        **tags,
        "holdout_mae": {
            "challenger": twin_mae,
            "baseline": baseline_mae,
            "champion": champion_mae,
        },  # fmt: skip
        "gates": decision,
    }
    write_json(project_path("reports/training.json"), record)
    verdict = "promoted to champion" if decision["passed"] else "not promoted"
    print(f"version {version}: holdout MAE {twin_mae:.2f} (baseline {baseline_mae:.2f}, "
          f"champion {champion_mae if champion_mae is None else round(champion_mae, 2)}): "
          f"{verdict}")  # fmt: skip


if __name__ == "__main__":
    main()

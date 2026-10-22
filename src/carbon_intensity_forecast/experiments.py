"""Development experiments: backtest each candidate setup (and each ablation) on the
development months, log every run to MLflow, and apply the selection rule fixed in
configs/config.yaml (experiments) before they were run.

Each MLflow run records the settings, the git commit, the DVC version of the data, the
overall and per-lead MAE of the model and the baselines, and MAE per month (as steps).

python -m carbon_intensity_forecast.experiments  -> reports/experiments.json
"""

import os
import subprocess

import mlflow
import yaml

from carbon_intensity_forecast import features as F
from carbon_intensity_forecast.artifacts import write_json
from carbon_intensity_forecast.backtest import build_table, months, run
from carbon_intensity_forecast.config import load_config, project_path

EXPERIMENT = "development-backtest"


def provenance() -> dict:
    def git(*args: str) -> str:
        done = subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=project_path("."), check=True)  # fmt: skip
        return done.stdout.strip()

    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    data = yaml.safe_load(project_path("data/raw.dvc").read_text())["outs"][0]["md5"]
    return {"git_commit": commit, "git_dirty": str(bool(dirty)), "data_md5": data}


def choose(results: dict[str, float], margin: float) -> str:
    """The rule: lowest MAE, but `as_set` unless beaten by more than `margin`."""
    best = min(results, key=results.get)
    return best if results["as_set"] - results[best] > margin else "as_set"


def main() -> None:
    cfg = load_config()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(EXPERIMENT)
    period = months(*cfg["evaluation"]["development_months"])
    table = build_table(cfg, period)
    all_columns = F.feature_columns(table)
    experiments = cfg["experiments"]
    setups = {k: ("candidate", v) for k, v in experiments["candidates"].items()}
    setups |= {k: ("ablation", v) for k, v in experiments["ablations"].items()}
    tags = provenance()
    summary = {}
    for name, (kind, change) in setups.items():
        change = dict(change)
        drop = tuple(change.pop("drop_features_starting", []))
        columns = [c for c in all_columns if not c.startswith(drop)] if drop else all_columns
        params = {**cfg["model"]["params"], **change}
        print(f"--- {name} ({kind})")
        result = run(table, period, cfg, params, columns)
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({**tags, "kind": kind, "months": "development"})
            mlflow.log_params({**params, "features": len(columns),
                               "dropped": ",".join(drop) or "none"})  # fmt: skip
            mlflow.log_metrics({f"mae_{k}": v for k, v in result["overall_mae"].items()})
            for lead, values in result["by_lead"].items():
                mlflow.log_metric(f"mae_model_lead_{lead.replace(' ', '')}", values["model"])
            for step, month in enumerate(result["per_month"]):
                mlflow.log_metric("mae_model_month", month["model"], step=step)
            mlflow.log_dict(result, "backtest.json")
        summary[name] = {
            "kind": kind,
            "change": experiments[f"{kind}s"][name],
            "mae": result["overall_mae"]["model"],
            "by_lead": result["by_lead"],
        }
    candidates = {k: v["mae"] for k, v in summary.items() if v["kind"] == "candidate"}
    chosen = choose(candidates, cfg["experiments"]["margin"])
    write_json(project_path("reports/experiments.json"),
               {**tags, "rule": f"lowest MAE; as_set unless beaten by > "
                                f"{cfg['experiments']['margin']}", "chosen": chosen,
                "runs": summary})  # fmt: skip
    for name, v in summary.items():
        print(f"{name:16} {v['kind']:9} MAE {v['mae']:.2f}")
    print(f"chosen by the rule: {chosen}")


if __name__ == "__main__":
    main()

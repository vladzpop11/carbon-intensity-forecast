"""Rolling-origin backtest of the day-ahead model against the baselines.

For each month of target days: train on every row whose target day ended before the first
issue time of that month (expanding window), forecast the month, score it. Only the
development months are used unless --final-test is given (configs/config.yaml, evaluation).

python -m carbon_intensity_forecast.backtest [--final-test]
-> reports/backtest.json (or reports/final_test.json)
"""

import argparse
from datetime import UTC, datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

from carbon_intensity_forecast import features as F
from carbon_intensity_forecast.artifacts import write_json
from carbon_intensity_forecast.config import load_config, project_path


def months(first: str, last: str) -> pd.PeriodIndex:
    return pd.period_range(first, last, freq="M")


def split(table: pd.DataFrame, month: pd.Period) -> tuple[np.ndarray, np.ndarray]:
    """Rows to train on and rows to score for one month of target days. The first issue
    time of the month is 09:00 on the day before it starts; training targets must have
    ended (and been published) by then, so they stop two days before the month."""
    start = month.start_time.tz_localize("UTC")
    end = (month + 1).start_time.tz_localize("UTC")
    target = table["target_time"]
    known = table["actual"].notna()
    train = (target.dt.normalize() < start - pd.Timedelta("1D")) & known
    test = (target >= start) & (target < end) & known
    return train.to_numpy(), test.to_numpy()


def fit(table: pd.DataFrame, rows: np.ndarray, params: dict, columns: list[str]):
    model = lgb.LGBMRegressor(**params)
    model.fit(table.loc[rows, columns], table.loc[rows, "actual"])
    return model


def mae(predicted, actual) -> float:
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(actual))))


def run(table: pd.DataFrame, period: pd.PeriodIndex, cfg: dict, params: dict | None = None,
        columns: list[str] | None = None) -> dict:  # fmt: skip
    """Backtest one model setup (by default the configured one, on every feature)."""
    params = params or cfg["model"]["params"]
    columns = columns or F.feature_columns(table)
    per_month, scored = [], []
    for month in period:
        train, test = split(table, month)
        model = fit(table, train, params, columns)
        rows = table.loc[test].copy()
        rows["model"] = model.predict(rows[columns])
        scored.append(rows)
        entry = {"month": str(month), "train_rows": int(train.sum()), "test_rows": int(test.sum()),
                 "model": mae(rows["model"], rows["actual"])}  # fmt: skip
        for b in cfg["baselines"]:
            entry[b] = mae(rows[b], rows["actual"])
        per_month.append(entry)
        print(f"{month}: model {entry['model']:.1f}, "
              + ", ".join(f"{b} {entry[b]:.1f}" for b in cfg["baselines"]))  # fmt: skip
    rows = pd.concat(scored)
    overall = {name: mae(rows[name], rows["actual"]) for name in ["model", *cfg["baselines"]]}
    by_lead = {
        f"{lo:g}-{hi:g} h": {
            name: mae(part[name], part["actual"]) for name in ["model", *cfg["baselines"]]
        }
        for lo, hi in [(15, 21), (21, 27), (27, 33), (33, 39)]
        if len(part := rows[(rows["lead_hours"] >= lo) & (rows["lead_hours"] < hi)])
    }
    skill = {b: 1 - overall["model"] / overall[b] for b in cfg["baselines"]}
    return {"overall_mae": overall, "by_lead": by_lead, "per_month": per_month, "skill_vs": skill}


def build_table(cfg: dict, period: pd.PeriodIndex) -> pd.DataFrame:
    """Every issue day from the start of the weather data to the last target in `period`."""
    intensity, weather = F.load_raw(cfg)
    first = pd.Timestamp(cfg["weather"]["start"], tz="UTC") + pd.Timedelta("1D")
    last = period[-1].end_time.tz_localize("UTC").normalize() - pd.Timedelta("1D")
    return F.build(pd.date_range(first, last, freq="1D"), intensity, weather, cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final-test", action="store_true", help="score the final-test months, once"
    )
    args = parser.parse_args()
    cfg = load_config()
    key = "final_test_months" if args.final_test else "development_months"
    period = months(*cfg["evaluation"][key])
    out = project_path("reports/final_test.json" if args.final_test else "reports/backtest.json")
    if args.final_test and out.exists():
        raise SystemExit(f"{out} exists: the final test has been run already")
    table = build_table(cfg, period)
    result = run(table, period, cfg)
    write_json(out, {"run_at": f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}", "months": key,
                     "model": cfg["model"], **result})  # fmt: skip
    o = result["overall_mae"]
    print("overall MAE: " + ", ".join(f"{k} {v:.2f}" for k, v in o.items()))


if __name__ == "__main__":
    main()

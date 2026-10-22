"""How certain the headline numbers are, from the saved per-row predictions (no model is
rerun on the final test).

- 95% intervals by resampling whole days (a day's 48 half hours are strongly related, so
  resampling half hours would be too narrow): development MAE, final-test MAE and skill.
- The final test against the same calendar months of development (Jan-Sep 2025 vs 2026),
  since the month mix differs.
- The L1 choice: the paired gain over the original setting on development days (the
  original setting is backtested again on the development months only).

python -m carbon_intensity_forecast.uncertainty  -> reports/uncertainty.json
"""

import numpy as np
import pandas as pd

from carbon_intensity_forecast.artifacts import provenance, write_json
from carbon_intensity_forecast.backtest import build_table, months, run
from carbon_intensity_forecast.config import load_config, project_path

RESAMPLES = 2000


def by_day(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Per target day: the sum of absolute errors per column and the half-hour count, on
    rows where every column has a value."""
    rows = frame.dropna(subset=columns + ["actual"])
    errors = rows[columns].sub(rows["actual"], axis=0).abs()
    errors["day"] = pd.to_datetime(rows["target_time"], utc=True).dt.date
    daily = errors.groupby("day").sum()
    daily["n"] = errors.groupby("day").size()
    return daily


def interval(daily: pd.DataFrame, statistic, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    days = len(daily)
    values = [statistic(daily.iloc[rng.integers(0, days, days)]) for _ in range(RESAMPLES)]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def mae(daily: pd.DataFrame, column: str) -> float:
    return float(daily[column].sum() / daily["n"].sum())


def summary(frame: pd.DataFrame, baseline: str) -> dict:
    daily = by_day(frame, ["model", baseline])
    skill = lambda d: 1 - mae(d, "model") / mae(d, baseline)  # noqa: E731
    return {
        "days": len(daily),
        "model_mae": mae(daily, "model"),
        "model_mae_ci95": interval(daily, lambda d: mae(d, "model")),
        "skill_vs_baseline": skill(daily),
        "skill_ci95": interval(daily, skill),
    }


def main() -> None:
    cfg = load_config()
    reports = project_path("reports")
    dev = pd.read_csv(reports / "backtest_predictions.csv")
    final = pd.read_csv(reports / "final_test_predictions.csv")
    baseline = cfg["baselines"][0]
    month = lambda f: pd.to_datetime(f["target_time"], utc=True).dt.strftime("%m")  # noqa: E731
    same_months = dev[month(dev).isin(month(final).unique())]

    # The L1 choice against the original setting (development months only).
    period = months(*cfg["evaluation"]["development_months"])
    original = {**cfg["model"]["params"], "objective": "regression"}
    rerun = run(build_table(cfg, period), period, cfg, params=original)["predictions"]
    dev_rows = dev.assign(target_time=pd.to_datetime(dev["target_time"], utc=True))
    paired = dev_rows[["target_time", "actual", "model"]].merge(
        rerun[["target_time", "model"]].rename(columns={"model": "original"}), on="target_time"
    )
    daily = by_day(paired, ["model", "original"])
    gain = lambda d: mae(d, "original") - mae(d, "model")  # noqa: E731

    result = {
        **provenance(),
        "method": f"{RESAMPLES} resamples of whole target days, 95% percentile intervals",
        "development": summary(dev, baseline),
        "final_test": summary(final, baseline),
        "development_same_months_as_final_test": summary(same_months, baseline),
        "l1_gain_over_original_setting": {
            "mae_gain": gain(daily),
            "ci95": interval(daily, gain),
            "months_l1_better": int(
                sum(
                    gain(by_day(p, ["model", "original"])) > 0
                    for _, p in paired.groupby(
                        pd.to_datetime(paired["target_time"], utc=True).dt.strftime("%Y-%m")
                    )
                )
            ),  # fmt: skip
            "months": len(period),
        },
    }
    write_json(reports / "uncertainty.json", result)
    for key in ["development", "development_same_months_as_final_test", "final_test"]:
        r = result[key]
        print(f"{key}: MAE {r['model_mae']:.2f} {np.round(r['model_mae_ci95'], 2)}, "
              f"skill {r['skill_vs_baseline']:.3f} {np.round(r['skill_ci95'], 3)}")  # fmt: skip
    g = result["l1_gain_over_original_setting"]
    print(f"L1 gain {g['mae_gain']:.2f} {np.round(g['ci95'], 2)}, better in "
          f"{g['months_l1_better']} of {g['months']} months")  # fmt: skip


if __name__ == "__main__":
    main()

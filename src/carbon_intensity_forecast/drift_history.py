"""How often input drift would have been flagged in ordinary months, and whether it tracks
the model's error: the evidence behind using drift as a diagnostic, not a retraining
trigger (configs/config.yaml, monitoring).

For each month-end window of 28 issue days from April 2025, the share of drifted inputs
against two references, with Evidently's default tests and with PSI (threshold 0.25):
- same_days_last_year: the same 28 days a year earlier;
- same_season: every earlier issue day within 45 days of the window's calendar date.
Months of the final test are paired with the model's MAE there.

python -m carbon_intensity_forecast.drift_history  -> reports/drift_history.json
"""

import json
import warnings

import numpy as np
import pandas as pd

from carbon_intensity_forecast import features as F
from carbon_intensity_forecast.artifacts import write_json
from carbon_intensity_forecast.config import load_config, project_path
from carbon_intensity_forecast.monitor import NOT_DRIFT, same_season_days

DAY = pd.Timedelta("1D")


def share(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], **method) -> float:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snapshot = Report([DataDriftPreset(drift_share=0.5, **method)]).run(
            reference_data=reference[columns], current_data=current[columns]
        )
    return next(m["value"]["share"] for m in snapshot.dict()["metrics"]
                if m["metric_name"].startswith("DriftedColumnsCount"))  # fmt: skip


def main() -> None:
    cfg = load_config()
    intensity, weather = F.load_raw(cfg)
    first = pd.Timestamp(cfg["weather"]["start"], tz="UTC") + DAY
    days = pd.date_range(first, pd.Timestamp("2026-09-30", tz="UTC"), freq="1D")
    table = F.build(days, intensity, weather, cfg)
    columns = [c for c in F.feature_columns(table) if c not in NOT_DRIFT]
    final = json.loads(project_path("reports/final_test.json").read_text())
    month_mae = {m["month"]: m["model"] for m in final["per_month"]}
    rows = []
    for end in pd.date_range("2025-05-01", "2026-10-01", freq="MS", tz="UTC") - DAY:
        current_days = pd.date_range(end - 27 * DAY, end, freq="1D")
        current = table[table["issue_day"].isin(current_days)]
        references = {
            "same_days_last_year": table[table["issue_day"].isin(
                current_days - pd.DateOffset(years=1))],
            "same_season": table[table["issue_day"].isin(same_season_days(days, current_days))],
        }  # fmt: skip
        row = {"window_end": f"{end:%Y-%m-%d}", "model_mae": month_mae.get(f"{end:%Y-%m}")}
        for ref_name, reference in references.items():
            row[f"{ref_name}_default"] = share(reference, current, columns)
            row[f"{ref_name}_psi"] = share(reference, current, columns, method="psi",
                                           threshold=0.25)  # fmt: skip
        rows.append(row)
        print(row)
    frame = pd.DataFrame(rows)
    summary = {c: {"mean_share": float(frame[c].mean()),
                   "months_flagged": int((frame[c] >= 0.5).sum()), "months": len(frame)}
               for c in frame.columns if c.endswith(("_default", "_psi"))}  # fmt: skip
    scored = frame.dropna(subset=["model_mae"])
    corr = float(np.corrcoef(scored["same_season_psi"], scored["model_mae"])[0, 1])
    write_json(project_path("reports/drift_history.json"),
               {"summary": summary, "correlation_same_season_psi_vs_mae": corr,
                "windows": rows})  # fmt: skip
    print(summary, f"correlation with monthly MAE: {corr:.2f}")


if __name__ == "__main__":
    main()

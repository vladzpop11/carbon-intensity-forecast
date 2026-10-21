"""The data contracts and the ingestion helpers, on small constructed frames."""

import pandas as pd
import pytest

from carbon_intensity_forecast.schemas import INTENSITY, ContractError, clean


def frame(actual, forecast=None, start="2025-01-01T00:00Z"):
    times = pd.date_range(start, periods=len(actual), freq="30min")
    return pd.DataFrame({"period_start": times, "actual": actual,
                         "forecast_latest": forecast or [100.0] * len(actual)})  # fmt: skip


def test_faults_are_set_to_missing_and_counted() -> None:
    values = [150.0] * 199 + [0.0]  # one fault in 200 (0.5%)
    cleaned, faults = clean(frame(values), {"actual": (5, 1000), "forecast_latest": (5, 1000)})
    assert cleaned["actual"].isna().sum() == 1
    assert faults["actual"]["count"] == 1
    assert faults["actual"]["times"] == [str(cleaned["period_start"].iloc[-1])]
    INTENSITY.validate(cleaned)  # the cleaned batch meets the contract


def test_a_batch_with_too_many_faults_is_refused() -> None:
    values = [150.0] * 98 + [0.0, 13579.0]  # 2% faults: the feed is broken
    with pytest.raises(ContractError, match="feed is broken"):
        clean(frame(values), {"actual": (5, 1000)})


@pytest.mark.parametrize("break_it", ["duplicate", "off_grid", "out_of_order"])
def test_structural_problems_fail_the_contract(break_it) -> None:
    import pandera.errors

    f = frame([150.0, 160.0, 170.0])
    if break_it == "duplicate":
        f.loc[2, "period_start"] = f.loc[1, "period_start"]
    elif break_it == "off_grid":
        f.loc[1, "period_start"] += pd.Timedelta("10min")
    else:
        f = f.iloc[[0, 2, 1]]
    with pytest.raises(pandera.errors.SchemaError):
        INTENSITY.validate(f)


def test_missing_half_hours_become_empty_rows() -> None:
    from datetime import UTC, datetime

    from carbon_intensity_forecast.ingest import complete_grid

    f = frame([150.0, 160.0, 170.0]).drop(index=1)
    grid = complete_grid(f, datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, 2, tzinfo=UTC))
    assert len(grid) == 4 and grid["actual"].isna().sum() == 2


def test_archive_snapshots_keep_what_was_served(monkeypatch) -> None:
    from datetime import UTC, datetime

    from carbon_intensity_forecast import archive

    now = datetime(2026, 10, 22, 9, 5, tzinfo=UTC)
    periods = [{"from": f"t{i}", "to": f"t{i + 1}", "intensity": {"forecast": 100 + i}}
               for i in range(96)]  # fmt: skip
    monkeypatch.setattr(archive, "get_json", lambda url: {"data": periods})
    record = archive.neso_snapshot(now)
    assert record["fetched_at"] == "2026-10-22T09:05:00Z"
    assert [p["forecast"] for p in record["periods"][:2]] == [100, 101]

    monkeypatch.setattr(archive, "get_json", lambda url: {"data": periods[:10]})
    with pytest.raises(ValueError, match="only 10 half hours"):
        archive.neso_snapshot(now)

"""Promotion gates and the fair comparison with the champion (train.py)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException

from carbon_intensity_forecast import train
from carbon_intensity_forecast.config import load_config

CFG = load_config()
HOLDOUT_START = pd.Timestamp("2026-09-23", tz="UTC")
SETUP = {"params": {"objective": "regression_l1"}, "columns": ["a", "b"]}


@pytest.mark.parametrize(
    ("twin", "baseline", "champion", "passed"),
    [(20.0, 50.0, None, True),     # first model: only the baseline gate
     (40.0, 50.0, None, False),    # skill 20% < 30%
     (20.0, 50.0, 21.0, True),     # beats the champion
     (21.0, 50.0, 21.0, False)],   # a tie does not beat it
)  # fmt: skip
def test_gates(twin, baseline, champion, passed) -> None:
    assert train.gates(twin, baseline, champion, CFG)["passed"] is passed


class StubClient:
    def __init__(self, champion=None):
        self.champion = champion

    def get_model_version_by_alias(self, name, alias):
        if self.champion is None:
            raise MlflowException("no alias")
        return self.champion


def champion_trained_through(day: str, setup: dict):
    return SimpleNamespace(tags={"trained_through": day, "setup": train.json.dumps(setup)},
                           name="m", version="1")  # fmt: skip


def test_no_champion() -> None:
    _, mae, how = train.compare_with_champion(StubClient(), "m", None, None, None,
                                              HOLDOUT_START, SETUP)  # fmt: skip
    assert (mae, how) == (None, "no champion")


def test_a_champion_that_saw_the_window_with_the_same_setup_is_a_refresh() -> None:
    champion = champion_trained_through("2026-10-20", SETUP)
    _, mae, how = train.compare_with_champion(StubClient(champion), "m", None, None, None,
                                              HOLDOUT_START, SETUP)  # fmt: skip
    assert (mae, how) == (None, "same setup (refresh)")


def test_a_champion_that_saw_the_window_with_another_setup_is_retrained(monkeypatch) -> None:
    other = {"params": {"objective": "regression"}, "columns": ["a"]}
    champion = champion_trained_through("2026-10-20", other)
    seen = {}

    def fake_fit(table, rows, params, columns):
        seen.update(params=params, columns=columns)
        return SimpleNamespace(predict=lambda x: np.full(len(x), 100.0))

    monkeypatch.setattr(train, "fit", fake_fit)
    rows = pd.DataFrame({"a": [1.0, 2.0], "actual": [90.0, 110.0]})
    _, mae, how = train.compare_with_champion(StubClient(champion), "m", None, rows, None,
                                              HOLDOUT_START, SETUP)  # fmt: skip
    assert how == "champion setup retrained" and mae == 10.0
    assert seen == {"params": other["params"], "columns": ["a"]}

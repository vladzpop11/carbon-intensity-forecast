"""The monitoring store (Postgres in compose), read by Grafana.

- forecasts: one row per (issue day, target half hour, method): the model, NESO's archived
  day-ahead forecast, and the baseline.
- actuals: the actual value of each half hour.
- daily_scores: MAE per target day and method, on the half hours every method covers.
- unscored_days: target days still missing actual values two days after they ended.
- drift: one row per monitoring run.
- decisions: what each monitoring run decided about retraining, and why.
"""

import os

import pandas as pd
import sqlalchemy as sa

from carbon_intensity_forecast.config import env_value

METADATA = sa.MetaData()
FORECASTS = sa.Table(
    "forecasts", METADATA,
    sa.Column("issue_day", sa.Date, primary_key=True),
    sa.Column("target_time", sa.DateTime(timezone=True), primary_key=True),
    sa.Column("method", sa.Text, primary_key=True),   # model | neso | baseline
    sa.Column("value", sa.Float, nullable=False),
    sa.Column("source", sa.Text),   # model version, or when NESO's forecast was read
)  # fmt: skip
ACTUALS = sa.Table(
    "actuals", METADATA,
    sa.Column("period_start", sa.DateTime(timezone=True), primary_key=True),
    sa.Column("actual", sa.Float),
)  # fmt: skip
DAILY_SCORES = sa.Table(
    "daily_scores", METADATA,
    sa.Column("target_day", sa.Date, primary_key=True),
    sa.Column("method", sa.Text, primary_key=True),
    sa.Column("mae", sa.Float, nullable=False),
    sa.Column("half_hours", sa.Integer, nullable=False),
)  # fmt: skip
UNSCORED = sa.Table(
    "unscored_days", METADATA,
    sa.Column("target_day", sa.Date, primary_key=True),
    sa.Column("missing_half_hours", sa.Integer, nullable=False),
)  # fmt: skip
DRIFT = sa.Table(
    "drift", METADATA,
    sa.Column("run_day", sa.Date, primary_key=True),
    sa.Column("share", sa.Float, nullable=False),
    sa.Column("detected", sa.Boolean, nullable=False),
    sa.Column("features", sa.Integer, nullable=False),
    sa.Column("drifted", sa.Text),
)  # fmt: skip
DECISIONS = sa.Table(
    "decisions", METADATA,
    sa.Column("run_at", sa.DateTime(timezone=True), primary_key=True),
    sa.Column("trigger", sa.Text),   # null: no retraining
    sa.Column("rolling_mae", sa.Float),
    sa.Column("scored_days", sa.Integer),
    sa.Column("model_version", sa.Text),
    sa.Column("model_age_days", sa.Integer),
    sa.Column("reason", sa.Text, nullable=False),
)  # fmt: skip


def engine(cfg: dict) -> sa.Engine:
    """MONITORING_DATABASE_URL if set (compose, tests), else the configured URL with the
    password from .env (POSTGRES_PASSWORD)."""
    url = os.environ.get("MONITORING_DATABASE_URL")
    if not url:
        url = sa.make_url(cfg["monitoring"]["database_url"]).set(
            password=env_value("POSTGRES_PASSWORD")
        )
    e = sa.create_engine(url)
    METADATA.create_all(e)
    return e


def replace(e: sa.Engine, table: sa.Table, frame: pd.DataFrame) -> None:
    """Write `frame` into `table`, replacing rows with the same primary key."""
    if frame.empty:
        return
    keys = [c.name for c in table.primary_key.columns]
    # Missing values as SQL NULL: Postgres would otherwise store a float NaN, which
    # "is not null" lets through.
    frame = frame[[c.name for c in table.columns]].astype(object)
    rows = frame.where(frame.notna(), None).to_dict("records")
    with e.begin() as conn:
        for row in rows:
            conn.execute(sa.delete(table).where(*[table.c[k] == row[k] for k in keys]))
        conn.execute(sa.insert(table), rows)


def read(e: sa.Engine, query: str, **params) -> pd.DataFrame:
    with e.connect() as conn:
        return pd.read_sql(sa.text(query), conn, params=params)

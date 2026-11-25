"""[Role B] SQLite storage for the daily table and the prediction log.

Matches docs/CONTRACTS.md sections 2 and 3 (A's daily columns, the handbook's log columns).
Standard library sqlite3 only.

Guarantees:
- every timestamp is written in config.TIMEZONE (Asia/Tashkent), ISO format;
- running a job twice never creates a duplicate row (unique keys + upserts);
- a day that was stored with more valid hours is never replaced by a less complete fetch
  (an API gap cannot erase good data), and a row is always replaced as a whole, so the
  flags (is_reliable, target_ok) always agree with the values next to them.
"""
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo  # on Windows this needs the "tzdata" package

import pandas as pd

import config

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INT_COLUMNS = ("hours_in_window", "hours_full", "is_reliable", "target_ok")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_data (
    date            TEXT PRIMARY KEY,   -- Tashkent calendar day, YYYY-MM-DD
    pm25_today      REAL,               -- features: only hours before CUTOFF_HOUR
    pm25_max_today  REAL,
    pm25_last3h     REAL,
    temp_today      REAL,
    humidity_today  REAL,
    wind_today      REAL,
    pressure_today  REAL,
    precip_today    REAL,
    pm25_full       REAL,               -- mean over the full 24 h (NULL unless target_ok)
    hours_in_window INTEGER,
    hours_full      INTEGER,
    is_reliable     INTEGER,            -- 1 = enough hours before the cutoff (the bot checks this)
    target_ok       INTEGER,            -- 1 = enough hours in the full day (training checks this)
    fetched_at      TEXT
    -- extra columns that A adds later (lags, rolling means ...) are created automatically
);

CREATE TABLE IF NOT EXISTS predictions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    time           TEXT NOT NULL,       -- when the forecast was made (Tashkent time)
    target_date    TEXT NOT NULL,       -- the day the forecast is for
    input          TEXT,                -- JSON of the inputs the model used
    prediction     REAL,                -- predicted daily mean PM2.5, ug/m3
    prob_unhealthy REAL,
    is_unhealthy   INTEGER,
    confidence     REAL,
    model_version  TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'ok',   -- 'ok' or 'no_forecast'
    reason         TEXT,
    posted_at      TEXT,                -- set once the message reached the channel
    label          REAL,                -- real PM2.5 of target_date, filled the next day
    label_time     TEXT,
    UNIQUE (target_date, model_version)
);
"""


def _db_path() -> Path:
    """config.DATABASE_PATH if A defines it, else the DATABASE_PATH environment variable,
    else data/smog.db next to config.py."""
    configured = getattr(config, "DATABASE_PATH", None) or os.getenv("DATABASE_PATH")
    if configured:
        return Path(configured)
    root = getattr(config, "ROOT", Path(__file__).resolve().parent)
    return Path(root) / "data" / "smog.db"


def _now_iso() -> str:
    return datetime.now(ZoneInfo(config.TIMEZONE)).isoformat(timespec="seconds")


def _date_str(value) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


@contextmanager
def _connect():
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _py(value):
    """Convert pandas / numpy values to plain Python values that sqlite3 accepts (NaN -> NULL)."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    return int(value) if isinstance(value, bool) else value


def _sql_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series):
        return "REAL"
    return "TEXT"


def init_db() -> None:
    """Create the tables if they do not exist. Safe to call any number of times."""
    with _connect():
        pass


# ----------------------------------------------------------------------------- daily table
def upsert_daily(df: pd.DataFrame) -> int:
    """Insert or update daily rows (key: date). Returns the number of rows sent to the database.

    Accepts exactly what data_sources.hourly_to_daily() returns. Columns the table does not
    have yet are created automatically. If a date is already stored with MORE valid hours
    (hours_full) than the new row, the stored row is kept.
    """
    if df is None or len(df) == 0:
        return 0
    if "date" not in df.columns:
        raise ValueError("daily dataframe needs a 'date' column")

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
    data["fetched_at"] = _now_iso()
    data = data.drop_duplicates(subset="date", keep="last")

    cols = list(data.columns)
    for c in cols:
        if not _IDENT.match(str(c)):
            raise ValueError(f"unsafe column name: {c!r}")

    with _connect() as conn:
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(daily_data)")}
        for c in cols:
            if c not in existing:
                conn.execute(f'ALTER TABLE daily_data ADD COLUMN "{c}" {_sql_type(data[c])}')

        col_list = ", ".join(f'"{c}"' for c in cols)
        placeholders = ", ".join("?" for _ in cols)
        updates = ", ".join(f'"{c}" = excluded."{c}"' for c in cols if c != "date")
        sql = (f"INSERT INTO daily_data ({col_list}) VALUES ({placeholders}) "
               f"ON CONFLICT(date) DO UPDATE SET {updates}")
        if "hours_full" in cols:   # the more complete day wins
            sql += ' WHERE COALESCE(excluded."hours_full", 0) >= COALESCE(daily_data."hours_full", 0)'
        rows = [tuple(_py(v) for v in row) for row in data.itertuples(index=False, name=None)]
        conn.executemany(sql, rows)
    return len(rows)


def get_daily(start=None, end=None, with_meta: bool = False) -> pd.DataFrame:
    """Return daily rows between start and end (inclusive), oldest first.

    Same columns as data_sources.hourly_to_daily(): 'date' is datetime64 (no time zone),
    flags and hour counts are integers 0/1 and counts, missing values are NaN.
    with_meta=True also returns the fetched_at column.
    """
    query, params, conds = "SELECT * FROM daily_data", [], []
    if start is not None:
        conds.append("date >= ?")
        params.append(_date_str(start))
    if end is not None:
        conds.append("date <= ?")
        params.append(_date_str(end))
    if conds:
        query += " WHERE " + " AND ".join(conds)
    query += " ORDER BY date"

    with _connect() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    df["date"] = pd.to_datetime(df["date"])
    for c in _INT_COLUMNS:
        if c in df.columns and len(df) and df[c].notna().all():
            df[c] = df[c].astype(int)
    if not with_meta and "fetched_at" in df.columns:
        df = df.drop(columns="fetched_at")
    return df


# ----------------------------------------------------------------------------- prediction log
def log_prediction(record: dict) -> bool:
    """Store one forecast. Returns True if a new row was written, False if one already exists
    for the same (target_date, model_version), so a second run never duplicates a forecast.

    record uses the keys returned by predict.predict_tomorrow():
    target_date, pred_pm25, prob_unhealthy, is_unhealthy, confidence, model_version,
    input_json, status, reason.
    """
    input_json = record.get("input_json")
    if isinstance(input_json, (dict, list)):
        input_json = json.dumps(input_json, ensure_ascii=False, default=str)
    is_unhealthy = record.get("is_unhealthy")

    values = (
        _now_iso(),
        _date_str(record["target_date"]),
        input_json,
        _py(record.get("pred_pm25")),
        _py(record.get("prob_unhealthy")),
        None if is_unhealthy is None else int(bool(is_unhealthy)),
        _py(record.get("confidence")),
        record.get("model_version") or "none",
        record.get("status") or "ok",
        record.get("reason"),
    )
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO predictions (time, target_date, input, prediction, prob_unhealthy, "
            "is_unhealthy, confidence, model_version, status, reason) "
            "VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(target_date, model_version) DO NOTHING",
            values,
        )
        return cur.rowcount == 1


def mark_posted(target_date, model_version) -> bool:
    """Remember that the forecast reached the channel (a restart must not post it again)."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE predictions SET posted_at = ? "
            "WHERE target_date = ? AND model_version = ? AND posted_at IS NULL",
            (_now_iso(), _date_str(target_date), model_version),
        )
        return cur.rowcount == 1


def fill_label(target_date, actual_pm25, overwrite: bool = False) -> int:
    """Write the real value for target_date into every forecast made for that day.

    Call it only for days with target_ok == 1 in daily_data. Returns how many rows were
    updated. A label already filled is kept unless overwrite=True.
    """
    sql = "UPDATE predictions SET label = ?, label_time = ? WHERE target_date = ?"
    if not overwrite:
        sql += " AND label IS NULL"
    with _connect() as conn:
        cur = conn.execute(sql, (_py(actual_pm25), _now_iso(), _date_str(target_date)))
        return cur.rowcount


def get_predictions(*, model_version=None, status=None, target_date=None,
                    start=None, end=None, labeled=None) -> pd.DataFrame:
    """Return logged forecasts, oldest target_date first.

    labeled=True keeps only rows with a real value, labeled=False only rows still waiting for it.
    start / end filter by target_date (inclusive).
    """
    conds, params = [], []
    if model_version is not None:
        conds.append("model_version = ?")
        params.append(model_version)
    if status is not None:
        conds.append("status = ?")
        params.append(status)
    if target_date is not None:
        conds.append("target_date = ?")
        params.append(_date_str(target_date))
    if start is not None:
        conds.append("target_date >= ?")
        params.append(_date_str(start))
    if end is not None:
        conds.append("target_date <= ?")
        params.append(_date_str(end))
    if labeled is True:
        conds.append("label IS NOT NULL")
    elif labeled is False:
        conds.append("label IS NULL")

    query = "SELECT * FROM predictions"
    if conds:
        query += " WHERE " + " AND ".join(conds)
    query += " ORDER BY target_date, model_version"
    with _connect() as conn:
        return pd.read_sql_query(query, conn, params=params)

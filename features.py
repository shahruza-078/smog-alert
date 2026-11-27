"""features.py  [A]  Turn the daily table into model inputs (X) and tomorrow's targets (y).

The rule that must never be broken: the row for day D uses only what is known on the
EVENING of day D.
  * *_today columns .... hours before CUTOFF_HOUR of day D (later hours are not known yet)
  * lags / rolling ..... FULL days D-1, D-2, ... (they are complete by the evening of D)
  * pm25_full of day D itself is NEVER used: it contains the hours after the cutoff
  * the label of row D is the full-day mean of D+1

Training and the bot both call build_features(), so they cannot disagree (no
training-serving skew).
"""
import numpy as np
import pandas as pd

import config

TODAY_COLUMNS = ["pm25_today", "pm25_max_today", "pm25_last3h",
                 "temp_today", "humidity_today", "wind_today", "pressure_today", "precip_today"]

FEATURE_COLUMNS = [
    "pm25_today", "pm25_max_today", "pm25_last3h",           # PM2.5 before the cutoff
    "pm25_lag1", "pm25_lag2", "pm25_lag3",                   # full-day PM2.5 of 1, 2, 3 days ago
    "pm25_mean3", "pm25_mean7",                              # rolling means of the previous full days
    "temp_today", "humidity_today", "wind_today", "pressure_today", "precip_today",
    "dow_sin", "dow_cos",                                    # day of week, as a circle
    "doy_sin", "doy_cos",                                    # season (day of year), as a circle
]


def _continuous(daily_df):
    """One row for EVERY calendar day, sorted. A missing day becomes an all-NaN row, so
    shift(1) always means exactly one day earlier, never 'the previous row we happen to have'."""
    need = set(TODAY_COLUMNS) | {"date", "pm25_full", "is_reliable"}
    missing = need - set(daily_df.columns)
    if missing:
        raise KeyError(f"daily table is missing columns: {sorted(missing)}")
    d = daily_df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d.drop_duplicates(subset="date", keep="last").set_index("date").sort_index()
    return d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))


def build_features(daily_df):
    """DataFrame indexed by date D with the columns FEATURE_COLUMNS (NaN where unknown)."""
    d = _continuous(daily_df)
    full = d["pm25_full"]                        # NaN unless the day was complete
    X = pd.DataFrame(index=d.index)
    for c in TODAY_COLUMNS:
        X[c] = d[c]
    for k in (1, 2, 3):
        X[f"pm25_lag{k}"] = full.shift(k)
    previous = full.shift(1)                     # only days BEFORE D
    X["pm25_mean3"] = previous.rolling(3, min_periods=2).mean()
    X["pm25_mean7"] = previous.rolling(7, min_periods=4).mean()

    dow = X.index.dayofweek.to_numpy()
    doy = X.index.dayofyear.to_numpy()
    X["dow_sin"], X["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    X.index.name = "date"
    return X[FEATURE_COLUMNS]


def make_targets(daily_df, threshold=None):
    """(y_reg, y_clf) for D+1, indexed by D.

    y_reg: full-day mean PM2.5 of the next day (NaN if that day is unknown or incomplete)
    y_clf: 1.0 if y_reg >= threshold else 0.0 (NaN where y_reg is NaN)
    """
    thr = config.UNHEALTHY_THRESHOLD if threshold is None else float(threshold)
    d = _continuous(daily_df)
    y_reg = d["pm25_full"].shift(-1)
    y_clf = (y_reg >= thr).astype(float).where(y_reg.notna())
    return y_reg.rename("y_reg"), y_clf.rename("y_clf")


def make_dataset(daily_df, threshold=None):
    """Usable training rows only: the evening data of D was reliable AND the label of D+1 is known."""
    d = _continuous(daily_df)
    X = build_features(daily_df)
    y_reg, y_clf = make_targets(daily_df, threshold)
    usable = (d["is_reliable"].fillna(0) == 1) & y_reg.notna()
    return X[usable], y_reg[usable], y_clf[usable].astype(int)


def feature_ranges(X):
    """[min, max] of every feature on the training data. B uses it to catch inputs that look
    nothing like the training data."""
    out = {}
    for c in X.columns:
        col = X[c].dropna()
        out[c] = [float(col.min()), float(col.max())] if len(col) else [None, None]
    return out

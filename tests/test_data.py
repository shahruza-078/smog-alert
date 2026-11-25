"""Data tests [A]. They need NO internet: the API is replaced by a fake.

Run:  pytest -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config
import data_sources as ds


def make_hourly(start="2026-11-01", days=6, pm=20.0):
    times = pd.date_range(start, periods=days * 24, freq="h")
    return pd.DataFrame({"time": times, "pm2_5": pm, "temperature_2m": 5.0,
                         "relative_humidity_2m": 60.0, "wind_speed_10m": 3.0,
                         "surface_pressure": 950.0, "precipitation": 0.0})


def on_day(h, day):
    return h["time"].dt.date == pd.Timestamp(day).date()


def row(d, day):
    return d[d["date"] == pd.Timestamp(day)].iloc[0]


# ---------------------------------------------------------------- daily table
def test_columns_and_one_row_per_day():
    d = ds.hourly_to_daily(make_hourly())
    assert list(d.columns) == ds.DAILY_COLUMNS
    assert len(d) == 6 and d["date"].is_unique
    assert (d["is_reliable"] == 1).all() and (d["target_ok"] == 1).all()


def test_features_use_only_hours_before_cutoff():
    h = make_hourly()
    base = ds.hourly_to_daily(h)
    h2 = h.copy()
    h2.loc[on_day(h2, "2026-11-03") & (h2["time"].dt.hour >= config.CUTOFF_HOUR), "pm2_5"] = 999.0
    alt = ds.hourly_to_daily(h2)
    feats = ["pm25_today", "pm25_max_today", "pm25_last3h", "temp_today"]
    pd.testing.assert_frame_equal(base[feats], alt[feats])       # features did not move
    assert row(alt, "2026-11-03")["pm25_full"] > 20               # the label did (full day)


def test_missing_hours_rules():
    h = make_hourly()
    # 6 missing before the cutoff -> 14 valid < 16 -> not reliable; 18 of 24 valid -> label ok
    h.loc[on_day(h, "2026-11-02") & (h["time"].dt.hour < 6), "pm2_5"] = np.nan
    # 7 missing in the day -> 17 valid < 18 -> label not ok
    h.loc[on_day(h, "2026-11-03") & (h["time"].dt.hour >= 17), "pm2_5"] = np.nan
    d = ds.hourly_to_daily(h)
    r2, r3 = row(d, "2026-11-02"), row(d, "2026-11-03")
    assert r2["is_reliable"] == 0 and np.isnan(r2["pm25_today"])
    assert r2["target_ok"] == 1 and not np.isnan(r2["pm25_full"])
    assert r3["target_ok"] == 0 and np.isnan(r3["pm25_full"])


def test_missing_day_still_gets_a_row():
    h = make_hourly()
    h = h[~on_day(h, "2026-11-04")]
    d = ds.hourly_to_daily(h)
    assert len(d) == 6
    r = row(d, "2026-11-04")
    assert r["hours_full"] == 0 and r["is_reliable"] == 0 and r["target_ok"] == 0


# ---------------------------------------------------------------- fetch (fake API)
NOW = pd.Timestamp("2026-11-20 14:30")


def fake_get_json(url, params):
    if "past_days" in params:
        first = NOW.normalize() - pd.Timedelta(days=params["past_days"])
        last = NOW.normalize() + pd.Timedelta(days=params["forecast_days"] - 1)
    else:
        first, last = pd.Timestamp(params["start_date"]), pd.Timestamp(params["end_date"])
    times = pd.date_range(first, last + pd.Timedelta(hours=23), freq="h")
    payload = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
    for k in params["hourly"].split(","):
        if k == "pm2_5":
            v = 3.0
        else:
            v = 2.0 if url == ds.FORECAST_URL else 1.0   # tells us which API the weather came from
        payload[k] = [v] * len(times)
    return {"hourly": payload}


def test_fetch_hourly_no_future_hours_and_sources(monkeypatch):
    monkeypatch.setattr(ds, "_now_local", lambda: NOW)
    monkeypatch.setattr(ds, "_get_json", fake_get_json)
    h = ds.fetch_hourly("2026-11-01", "2026-12-31")           # end is in the future
    assert list(h.columns) == ds.HOURLY_COLUMNS
    assert h["time"].max() == pd.Timestamp("2026-11-20 14:00")  # nothing after "now"
    assert h["time"].is_unique
    assert (h["pm2_5"] == 3.0).all()
    old = h[h["time"] < pd.Timestamp("2026-11-15")]
    new = h[h["time"] >= pd.Timestamp("2026-11-15")]
    assert (old["temperature_2m"] == 1.0).all()                # archive API for older days
    assert (new["temperature_2m"] == 2.0).all()                # forecast API for the last days


def test_fetch_hourly_long_history_in_chunks(monkeypatch):
    monkeypatch.setattr(ds, "_now_local", lambda: NOW)
    monkeypatch.setattr(ds, "_get_json", fake_get_json)
    h = ds.fetch_hourly("2025-01-01", "2026-11-20")
    expected = pd.date_range("2025-01-01", "2026-11-20 14:00", freq="h")
    assert len(h) == len(expected) and h["time"].is_unique


def test_today_is_partial_so_label_not_ok(monkeypatch):
    monkeypatch.setattr(ds, "_now_local", lambda: NOW)
    monkeypatch.setattr(ds, "_get_json", fake_get_json)
    d = ds.hourly_to_daily(ds.fetch_hourly("2026-11-15", "2026-11-20"))
    today = row(d, "2026-11-20")
    assert today["hours_full"] == 15 and today["target_ok"] == 0 and np.isnan(today["pm25_full"])
    assert row(d, "2026-11-19")["target_ok"] == 1


# ---------------------------------------------------------------- split
def test_test_window_is_60_days():
    n = (pd.Timestamp(config.TEST_END_DATE) - pd.Timestamp(config.TEST_START_DATE)).days + 1
    assert n == 60


def test_load_train_never_returns_test_rows():
    days = pd.date_range("2026-01-01", "2026-12-31")
    daily = pd.DataFrame({"date": days, "pm25_full": 1.0})
    t_start, t_end = pd.Timestamp(config.TEST_START_DATE), pd.Timestamp(config.TEST_END_DATE)
    gap = pd.Timedelta(days=config.GAP_DAYS)

    train = ds.load_train(daily)
    assert train["date"].max() < t_start - gap
    assert not ((train["date"] >= t_start) & (train["date"] <= t_end)).any()

    more = ds.load_train(daily, include_new_days=True)
    assert more["date"].max() == days.max()
    assert more[(more["date"] >= t_start - gap) & (more["date"] <= t_end + gap)].empty

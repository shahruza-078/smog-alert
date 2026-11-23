"""Pull hourly PM2.5 + weather for Tashkent from Open-Meteo and store DAILY rows in SQLite.

Run on your own machine / hosting (needs internet):
    python fetch_data.py                      # last 90 days up to today
    python fetch_data.py --start 2023-11-01   # longer history (check the API's earliest date)

Time-zone trap: we ask the API for timezone=Asia/Tashkent, so every hour is already in
local time and "day" means a Tashkent calendar day. Never mix UTC and local days.
Missing-hours trap: days with fewer than MIN_HOURS valid PM2.5 hours are stored with
pm25_mean = NaN and complete = 0, so they are never used as a training label or a feature.
Re-running is safe: rows are upserted by date (no duplicates).
"""
import argparse
import sqlite3
from datetime import date, timedelta

import pandas as pd
import requests

LAT, LON = 41.3111, 69.2797  # Tashkent
TZ = "Asia/Tashkent"
DB = "smog.db"
MIN_HOURS = 18  # a day needs at least 18 of 24 valid hourly PM2.5 values

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://archive-api.open-meteo.com/v1/archive"
WX_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
           "surface_pressure", "precipitation"]


def get_hourly(url, variables, start, end):
    params = {"latitude": LAT, "longitude": LON, "hourly": ",".join(variables),
              "start_date": start, "end_date": end, "timezone": TZ}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])  # already Tashkent local time
    return df


def to_daily(aq, wx):
    aq = aq.copy()
    aq["date"] = aq["time"].dt.date
    wx = wx.copy()
    wx["date"] = wx["time"].dt.date

    pm = aq.groupby("date")["pm2_5"].agg(pm25_mean="mean", pm25_max="max",
                                         pm25_hours="count")
    weather = wx.groupby("date").agg(
        temp=("temperature_2m", "mean"),
        humidity=("relative_humidity_2m", "mean"),
        wind=("wind_speed_10m", "mean"),
        pressure=("surface_pressure", "mean"),
        precip=("precipitation", "sum"),
    )
    daily = pm.join(weather, how="outer")
    daily["complete"] = (daily["pm25_hours"].fillna(0) >= MIN_HOURS).astype(int)
    daily.loc[daily["complete"] == 0, ["pm25_mean", "pm25_max"]] = float("nan")
    daily.index = daily.index.astype(str)
    daily.index.name = "date"
    return daily.reset_index()


def upsert(daily):
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS daily (
        date TEXT PRIMARY KEY, pm25_mean REAL, pm25_max REAL, pm25_hours INTEGER,
        temp REAL, humidity REAL, wind REAL, pressure REAL, precip REAL,
        complete INTEGER)""")
    cols = ["date", "pm25_mean", "pm25_max", "pm25_hours", "temp", "humidity",
            "wind", "pressure", "precip", "complete"]
    rows = [tuple(None if pd.isna(v) else v for v in r)
            for r in daily[cols].itertuples(index=False, name=None)]
    con.executemany(f"INSERT OR REPLACE INTO daily ({','.join(cols)}) "
                    f"VALUES ({','.join('?' * len(cols))})", rows)
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM daily").fetchone()[0]
    con.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=str(date.today() - timedelta(days=90)))
    ap.add_argument("--end", default=str(date.today()))
    a = ap.parse_args()

    aq = get_hourly(AQ_URL, ["pm2_5"], a.start, a.end)
    # Weather archive lags a few days behind "today"; missing recent weather stays NaN
    # and those days are dropped later (we never invent values).
    wx = get_hourly(WX_URL, WX_VARS, a.start, a.end)
    daily = to_daily(aq, wx)
    n = upsert(daily)
    print(f"Stored {len(daily)} days from {a.start} to {a.end}. Table now has {n} rows.")
    print(f"Incomplete days (missing PM2.5 hours): {(daily['complete'] == 0).sum()}")


if __name__ == "__main__":
    main()

"""data_sources.py  [A]  Open-Meteo -> hourly table -> one row per day.

Public functions (the contract with B, see docs/CONTRACTS.md):
    fetch_hourly(start_date, end_date) -> DataFrame   hourly, Tashkent local time
    hourly_to_daily(hourly_df)         -> DataFrame   one row per date
    load_train(daily_df)               -> DataFrame   NEVER returns test rows

Build the history (needs internet; run on your computer):
    python data_sources.py                      # from config.HISTORY_START until now
    python data_sources.py --start 2024-01-01   # if the API refuses the earlier date

Why it is built this way
  * Time zone: we ask the API for timezone=Asia/Tashkent, so "day" always means a
    Tashkent calendar day. Times are kept as naive local times (Uzbekistan has no DST).
  * Evening post: features of day D use ONLY hours before CUTOFF_HOUR. The label
    pm25_full is the mean of the full 24 hours of the same row's day; the "tomorrow"
    shift happens later in features.py (make_targets).
  * Missing hours: a day with too few valid hours is NOT averaged into a shaky number.
    It is flagged (is_reliable / target_ok) and its values become NaN.
  * No future: hours later than "now" are dropped (the API also returns forecast hours
    for today, and a forecast must never be used as an observation).
"""
import argparse
import os
import sys
import time
from datetime import timedelta

import pandas as pd
import requests

import config

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                "surface_pressure", "precipitation"]
HOURLY_COLUMNS = ["time", "pm2_5"] + WEATHER_VARS

# The archive weather API is a few days behind "today"; the newest days come from the
# forecast API (past_days). Check these numbers against the current Open-Meteo docs.
ARCHIVE_DELAY_DAYS = 6
CHUNK_DAYS = 180          # long histories are downloaded in pieces

DAILY_COLUMNS = [
    "date",
    "pm25_today", "pm25_max_today", "pm25_last3h",      # PM2.5 before the cutoff
    "temp_today", "humidity_today", "wind_today",        # weather before the cutoff
    "pressure_today", "precip_today",
    "pm25_full",                                         # mean of the full 24 h (label source)
    "hours_in_window", "hours_full",                     # how many valid PM2.5 hours we had
    "is_reliable",                                       # 1 = enough hours before the cutoff
    "target_ok",                                         # 1 = enough hours in the full day
]


# --------------------------------------------------------------------------- helpers
def _now_local():
    """Current Tashkent time as a naive Timestamp (tests replace this function)."""
    return pd.Timestamp.now(tz=config.TIMEZONE).tz_localize(None)


def _to_date(x):
    return pd.Timestamp(x).date()


def _get_json(url, params, retries=3):
    """GET with a few retries. On failure the error shows the API's own reason text."""
    last = "unknown error"
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
        except requests.RequestException as e:
            last = f"network error: {e}"
        else:
            if r.status_code == 200:
                return r.json()
            try:
                reason = r.json().get("reason", r.text[:300])
            except ValueError:
                reason = r.text[:300]
            last = f"HTTP {r.status_code}: {reason}"
            if r.status_code < 500 and r.status_code != 429:
                break  # our request is wrong, retrying will not help
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Open-Meteo request failed ({url}): {last}")


def _hourly_frame(payload):
    h = payload.get("hourly")
    if not h or "time" not in h:
        raise RuntimeError(f"Unexpected Open-Meteo answer (no 'hourly' block): {str(payload)[:300]}")
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])   # already Tashkent local time
    return df


def _base_params(variables):
    return {"latitude": config.LATITUDE, "longitude": config.LONGITUDE,
            "hourly": ",".join(variables), "timezone": config.TIMEZONE}


def _download(url, variables, start, end):
    """Download [start, end] (dates) in chunks and glue the pieces together."""
    frames, s = [], start
    while s <= end:
        e = min(s + timedelta(days=CHUNK_DAYS - 1), end)
        params = _base_params(variables)
        params.update({"start_date": s.isoformat(), "end_date": e.isoformat()})
        frames.append(_hourly_frame(_get_json(url, params)))
        s = e + timedelta(days=1)
    return pd.concat(frames, ignore_index=True)


def _download_recent_weather(start, today):
    params = _base_params(WEATHER_VARS)
    params.update({"past_days": (today - start).days, "forecast_days": 1})
    return _hourly_frame(_get_json(FORECAST_URL, params))


# --------------------------------------------------------------------------- contract
def fetch_hourly(start_date, end_date):
    """Hourly PM2.5 + weather for Tashkent, local time, no future hours.

    Columns: time, pm2_5, temperature_2m, relative_humidity_2m, wind_speed_10m,
             surface_pressure, precipitation
    """
    now = _now_local()
    today = now.date()
    start, end = _to_date(start_date), min(_to_date(end_date), today)
    if end < start:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}")

    aq = _download(AQ_URL, ["pm2_5"], start, end)

    parts = []
    archive_end = min(end, today - timedelta(days=ARCHIVE_DELAY_DAYS))
    if start <= archive_end:
        parts.append(_download(ARCHIVE_URL, WEATHER_VARS, start, archive_end))
    recent_start = max(start, archive_end + timedelta(days=1))
    if recent_start <= end:
        parts.append(_download_recent_weather(recent_start, today))
    wx = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["time"] + WEATHER_VARS)
    wx = wx.drop_duplicates(subset="time", keep="first")

    df = aq.drop_duplicates(subset="time", keep="last").merge(wx, on="time", how="outer")
    for c in HOURLY_COLUMNS:
        if c not in df.columns:
            df[c] = float("nan")
    for c in HOURLY_COLUMNS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) + pd.Timedelta(days=1)
    df = df[(df["time"] >= lo) & (df["time"] < hi) & (df["time"] <= now.floor("h"))]
    return df[HOURLY_COLUMNS].sort_values("time").reset_index(drop=True)


def hourly_to_daily(hourly_df, cutoff_hour=config.CUTOFF_HOUR):
    """One row per calendar day (days with no data at all still get a row).

    Features (*_today): only hours 0 .. cutoff_hour-1.
    pm25_full: mean of all 24 h (only if target_ok). It is the source of the label.
    is_reliable: enough valid hours BEFORE the cutoff (this is what the bot checks).
    target_ok:   enough valid hours in the FULL day (this is what training checks).
    """
    if hourly_df is None or len(hourly_df) == 0:
        return pd.DataFrame(columns=DAILY_COLUMNS)

    df = hourly_df.copy()
    df["time"] = pd.to_datetime(df["time"])
    for c in HOURLY_COLUMNS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = df["time"].dt.normalize()
    hour = df["time"].dt.hour

    win = df[hour < cutoff_hour].groupby("date")
    last3 = df[(hour >= cutoff_hour - 3) & (hour < cutoff_hour)].groupby("date")
    full = df.groupby("date")

    out = pd.DataFrame({
        "pm25_today": win["pm2_5"].mean(),
        "pm25_max_today": win["pm2_5"].max(),
        "pm25_last3h": last3["pm2_5"].mean(),
        "temp_today": win["temperature_2m"].mean(),
        "humidity_today": win["relative_humidity_2m"].mean(),
        "wind_today": win["wind_speed_10m"].mean(),
        "pressure_today": win["surface_pressure"].mean(),
        "precip_today": win["precipitation"].sum(min_count=1),
        "pm25_full_raw": full["pm2_5"].mean(),
        "hours_in_window": win["pm2_5"].count(),
        "hours_full": full["pm2_5"].count(),
    })
    out = out.reindex(pd.date_range(df["date"].min(), df["date"].max(), freq="D"))
    for c in ("hours_in_window", "hours_full"):
        out[c] = out[c].fillna(0).astype(int)

    reliable = out["hours_in_window"] >= config.MIN_HOURS_WINDOW
    target_ok = out["hours_full"] >= config.MIN_HOURS_FULL
    out["pm25_full"] = out["pm25_full_raw"].where(target_ok)
    for c in ("pm25_today", "pm25_max_today", "pm25_last3h"):
        out[c] = out[c].where(reliable)          # never keep a number built from too few hours
    out["is_reliable"] = reliable.astype(int)
    out["target_ok"] = target_ok.astype(int)
    out.index.name = "date"
    return out.reset_index()[DAILY_COLUMNS]


def load_train(daily_df, include_new_days=False):
    """Rows that are allowed to be used for training. TEST ROWS ARE NEVER RETURNED.

    Train = all dates before (test_start - gap). With include_new_days=True (only when
    training v2) the live days after (test_end + gap) are added as well.
    Only release_eval.py may open the test part, and only once.
    """
    d = daily_df.copy()
    d["date"] = pd.to_datetime(d["date"])
    test_start = pd.Timestamp(config.TEST_START_DATE)
    test_end = pd.Timestamp(config.TEST_END_DATE)
    gap = pd.Timedelta(days=config.GAP_DAYS)
    keep = d["date"] < test_start - gap
    if include_new_days:
        keep = keep | (d["date"] > test_end + gap)
    return d[keep].sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- report
def describe(daily):
    n = len(daily)
    ok = daily[(daily["is_reliable"] == 1) & (daily["target_ok"] == 1)]
    print("\n=== DATA SUMMARY (copy these numbers into docs/data_plan.md) ===")
    print(f"Daily rows:                 {n}")
    print(f"Date range:                 {daily['date'].min().date()} .. {daily['date'].max().date()}")
    if len(ok):
        print(f"First complete day:         {ok['date'].min().date()}")
    print(f"Complete days (both flags): {len(ok)}")
    print(f"Days with no PM2.5 at all:  {int((daily['hours_full'] == 0).sum())}")
    print(f"Unreliable before cutoff:   {int((daily['is_reliable'] == 0).sum())}")
    print(f"Label not usable:           {int((daily['target_ok'] == 0).sum())}")

    train = load_train(daily)
    y = train.loc[train["target_ok"] == 1, "pm25_full"].dropna()
    print(f"\nTRAIN part only: {len(train)} rows, "
          f"{train['date'].min().date() if len(train) else '-'} .. "
          f"{train['date'].max().date() if len(train) else '-'}   "
          f"(test starts {config.TEST_START_DATE}, test is not inspected)")
    if len(y):
        print(f"Daily mean PM2.5 in train: median {y.median():.1f}, mean {y.mean():.1f}, max {y.max():.1f} ug/m3")
        for thr in (15, 25, 35, 55):
            print(f"  share of train days >= {thr:>2} ug/m3: {(y >= thr).mean():.0%}")
        winter = train[(train["date"].dt.month.isin([12, 1, 2])) & (train["target_ok"] == 1)]
        print(f"Winter days (Dec-Feb) in train: {len(winter)}"
              f"{'   <-- WARNING: no winter in training data' if len(winter) == 0 else ''}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=config.HISTORY_START)
    ap.add_argument("--end", default=None, help="default: today (Tashkent)")
    ap.add_argument("--outdir", default=str(config.ROOT / "data"))
    a = ap.parse_args()

    end = a.end or str(_now_local().date())
    print(f"Downloading {a.start} .. {end} (Tashkent time). This can take a minute ...")
    hourly = fetch_hourly(a.start, end)
    if hourly.empty:
        sys.exit("The API returned no hours. Try a later --start date.")
    daily = hourly_to_daily(hourly)

    os.makedirs(a.outdir, exist_ok=True)
    hourly.to_csv(os.path.join(a.outdir, "hourly_raw.csv"), index=False)
    daily.to_csv(os.path.join(a.outdir, "daily_all.csv"), index=False)
    ok = daily[(daily["is_reliable"] == 1) & (daily["target_ok"] == 1)]
    ok.head(50).to_csv(os.path.join(a.outdir, "sample_50.csv"), index=False)
    print(f"Saved hourly_raw.csv, daily_all.csv and sample_50.csv in {a.outdir}")
    describe(daily)


if __name__ == "__main__":
    main()

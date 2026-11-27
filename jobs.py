"""[B] The two scheduled jobs. Both are safe to run twice and to restart at any point.

    python jobs.py evening     # forecast for tomorrow -> log -> post to the channel
    python jobs.py nextday     # fill in real values for yesterday's forecasts

How a forecast is never posted twice: the log has one row per (target_date, model_version),
and `posted_at` is set only after Telegram accepted the message. A rerun sees the row, sees
posted_at, and does nothing. If sending failed, posted_at stays empty and a rerun sends it.
"""
import logging
import sys

import pandas as pd

import config
import data_sources
import db
import telegram_client

log = logging.getLogger(__name__)
LOOKBACK_DAYS = 14   # recent days re-downloaded on every run (also picks up data revisions)


def _today(now=None) -> pd.Timestamp:
    now = now if now is not None else pd.Timestamp.now(tz=config.TIMEZONE).tz_localize(None)
    return pd.Timestamp(now).normalize()


# ----------------------------------------------------------------------------- messages
def format_forecast(rec: dict) -> str:
    day = pd.Timestamp(rec["target_date"]).strftime("%a %d %b %Y")
    threshold = config.UNHEALTHY_THRESHOLD
    lines = [f"PM2.5 forecast for Tashkent, {day}",
             f"Expected daily mean: {rec['pred_pm25']:.0f} \u00b5g/m\u00b3"]
    if rec.get("prob_unhealthy") is not None:
        lines.append(f"Chance of an unhealthy day: {rec['prob_unhealthy']:.0%}")
    if rec.get("is_unhealthy"):
        lines.append(f"\u26a0\ufe0f UNHEALTHY day expected (above {threshold:.0f} \u00b5g/m\u00b3). "
                     "Consider moving outdoor lessons indoors.")
    else:
        lines.append(f"\u2705 Below the warning level ({threshold:.0f} \u00b5g/m\u00b3).")
    lines.append(f"Model {rec.get('model_version', '?')}. Data: Open-Meteo.")
    return "\n".join(lines)


def format_no_forecast(rec: dict) -> str:
    day = pd.Timestamp(rec["target_date"]).strftime("%a %d %b %Y")
    return (f"No PM2.5 forecast for {day}: {rec.get('reason') or 'not enough recent data'}.\n"
            "Please do not rely on this channel for tomorrow.")


# ----------------------------------------------------------------------------- jobs
def evening_job(now=None, *, fetch=data_sources.fetch_hourly, to_daily=data_sources.hourly_to_daily,
                predict_fn=None, send=telegram_client.send_message) -> dict:
    """Update the data, forecast tomorrow, log it, post it. Returns a short summary dict."""
    today = _today(now)
    target = today + pd.Timedelta(days=1)

    try:
        hourly = fetch(today - pd.Timedelta(days=LOOKBACK_DAYS), today)
        db.upsert_daily(to_daily(hourly))
        window = db.get_daily(today - pd.Timedelta(days=LOOKBACK_DAYS), today)
        if predict_fn is None:                      # imported here: predict.py is written by A
            import predict
            bundle = predict.load_model()
            predict_fn = lambda daily: predict.predict_tomorrow(daily, bundle)
        rec = dict(predict_fn(window))
    except Exception as exc:                        # API down, model missing, ...
        log.error("evening job could not make a forecast: %s", type(exc).__name__)
        rec = {"status": "no_forecast", "model_version": "none",
               "reason": "the data or the model was not available"}
    rec.setdefault("target_date", target)

    db.log_prediction(rec)
    version = rec.get("model_version") or "none"
    row = db.get_predictions(target_date=rec["target_date"], model_version=version).iloc[0]
    if pd.notna(row["posted_at"]):
        return {"status": "already_posted", "target_date": str(rec["target_date"])[:10]}

    stored = row.to_dict()
    text = (format_forecast(rec) if stored["status"] == "ok" else format_no_forecast(rec))
    if send(text):
        db.mark_posted(rec["target_date"], version)
        return {"status": "posted", "forecast": stored["status"], "target_date": str(rec["target_date"])[:10]}
    return {"status": "send_failed", "target_date": str(rec["target_date"])[:10]}


def nextday_job(now=None, *, fetch=data_sources.fetch_hourly, to_daily=data_sources.hourly_to_daily) -> int:
    """Write the real value into every forecast whose day is now complete. Returns how many rows."""
    today = _today(now)
    try:
        hourly = fetch(today - pd.Timedelta(days=LOOKBACK_DAYS), today)
        db.upsert_daily(to_daily(hourly))
    except Exception as exc:
        log.error("nextday job could not download data: %s", type(exc).__name__)
        return 0

    daily = db.get_daily(today - pd.Timedelta(days=LOOKBACK_DAYS), today)
    complete = daily[(daily["target_ok"] == 1) & daily["pm25_full"].notna()]
    actual = dict(zip(complete["date"], complete["pm25_full"]))

    filled = 0
    waiting = db.get_predictions(labeled=False, end=today - pd.Timedelta(days=1))
    for day in waiting["target_date"].unique():
        value = actual.get(pd.Timestamp(day))
        if value is not None:
            filled += db.fill_label(day, value)
    return filled


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    job = sys.argv[1] if len(sys.argv) > 1 else ""
    if job == "evening":
        print(evening_job())
    elif job == "nextday":
        print(nextday_job())
    else:
        sys.exit("usage: python jobs.py evening | nextday")

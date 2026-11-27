"""Tests for jobs.py and telegram_client.py [B]. No internet: the API, the model and Telegram are fakes."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest
import requests

import config
import data_sources as ds
import db
import jobs
import telegram_client

NOW = pd.Timestamp("2026-11-25 20:30")          # a Wednesday evening in Tashkent


@pytest.fixture(autouse=True)
def temp_database(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATABASE_PATH", str(tmp_path / "test.db"), raising=False)
    db.init_db()


def fake_fetch(start, end):
    """Hourly data up to 20:00 of 'now' (like the real API in the evening)."""
    times = pd.date_range(pd.Timestamp(start), pd.Timestamp("2026-11-25 20:00"), freq="h")
    return pd.DataFrame({"time": times, "pm2_5": 30.0, "temperature_2m": 5.0,
                         "relative_humidity_2m": 60.0, "wind_speed_10m": 3.0,
                         "surface_pressure": 950.0, "precipitation": 0.0})


def broken_fetch(start, end):
    raise RuntimeError("API down")


def ok_predictor(daily):
    return {"target_date": "2026-11-26", "pred_pm25": 42.0, "prob_unhealthy": 0.8, "is_unhealthy": True,
            "confidence": 0.8, "model_version": "v1", "input_json": {"pm25_today": 30.0},
            "status": "ok", "reason": None}


def no_forecast_predictor(daily):
    return {"target_date": "2026-11-26", "model_version": "v1", "status": "no_forecast",
            "reason": "missing hours", "input_json": {}}


class Sender:
    def __init__(self, results=(True,)):
        self.messages, self.results = [], list(results)

    def __call__(self, text):
        self.messages.append(text)
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]


def run_evening(sender, predictor=ok_predictor, fetch=fake_fetch):
    return jobs.evening_job(NOW, fetch=fetch, predict_fn=predictor, send=sender)


# ------------------------------------------------------------------ evening job
def test_evening_posts_once_even_if_run_twice():
    sender = Sender()
    assert run_evening(sender)["status"] == "posted"
    assert run_evening(sender)["status"] == "already_posted"
    assert len(sender.messages) == 1
    assert len(db.get_predictions()) == 1


def test_failed_send_is_retried_on_the_next_run_without_a_duplicate_row():
    sender = Sender(results=[False, True])
    assert run_evening(sender)["status"] == "send_failed"
    assert db.get_predictions().iloc[0]["posted_at"] is None
    assert run_evening(sender)["status"] == "posted"
    assert len(db.get_predictions()) == 1 and len(sender.messages) == 2


def test_download_failure_posts_no_forecast_not_a_number():
    sender = Sender()
    assert run_evening(sender, fetch=broken_fetch)["forecast"] == "no_forecast"
    assert "No PM2.5 forecast" in sender.messages[0] and "\u00b5g" not in sender.messages[0]
    assert db.get_predictions().iloc[0]["status"] == "no_forecast"


def test_unposted_no_forecast_is_replaced_when_data_comes_back():
    sender = Sender(results=[False, True])
    run_evening(sender, predictor=no_forecast_predictor)         # could not send
    run_evening(sender, predictor=ok_predictor)                  # data is back
    row = db.get_predictions().iloc[0]
    assert row["status"] == "ok" and row["prediction"] == 42.0
    assert "UNHEALTHY" in sender.messages[-1]


def test_forecast_message_text():
    text = jobs.format_forecast(ok_predictor(None))
    assert "Thu 26 Nov 2026" in text and "42" in text and "UNHEALTHY" in text and "80%" in text
    calm = dict(ok_predictor(None), is_unhealthy=False, prob_unhealthy=None)
    assert "Below the warning level" in jobs.format_forecast(calm)


# ------------------------------------------------------------------ next-day job
def seed(day, version="v1"):
    db.log_prediction({"target_date": day, "pred_pm25": 40.0, "prob_unhealthy": 0.5, "is_unhealthy": False,
                       "confidence": 0.5, "model_version": version, "input_json": {}, "status": "ok"})


def test_nextday_fills_only_days_with_complete_data():
    seed("2026-11-23")
    seed("2026-11-24")
    daily = ds.hourly_to_daily(fake_fetch("2026-11-11", "2026-11-25"))
    daily.loc[daily["date"] == pd.Timestamp("2026-11-24"), ["target_ok", "pm25_full"]] = [0, float("nan")]

    def fetch(start, end):                      # the jobs call to_daily(fetch(...)), so hand back hourly data
        return fake_fetch(start, end)

    db.upsert_daily(daily)
    filled = jobs.nextday_job(NOW, fetch=fetch,
                              to_daily=lambda hourly: daily)    # day 24 stays incomplete
    assert filled == 1
    labels = db.get_predictions().set_index("target_date")["label"]
    assert labels["2026-11-23"] == 30.0 and pd.isna(labels["2026-11-24"])
    assert jobs.nextday_job(NOW, fetch=fetch, to_daily=lambda hourly: daily) == 0   # nothing to do twice


def test_nextday_survives_a_download_failure():
    seed("2026-11-23")
    assert jobs.nextday_job(NOW, fetch=broken_fetch) == 0
    assert pd.isna(db.get_predictions().iloc[0]["label"])


# ------------------------------------------------------------------ telegram client
class FakeResponse:
    def __init__(self, code):
        self.status_code = code


def test_telegram_success_and_hard_errors_do_not_retry(monkeypatch):
    calls = []
    monkeypatch.setattr(telegram_client.requests, "post",
                        lambda url, data, timeout: calls.append(1) or FakeResponse(200))
    assert telegram_client.send_message("hi", token="T", channel="@c", sleep=lambda s: None) is True
    calls.clear()
    monkeypatch.setattr(telegram_client.requests, "post",
                        lambda url, data, timeout: calls.append(1) or FakeResponse(400))
    assert telegram_client.send_message("hi", token="T", channel="@c", sleep=lambda s: None) is False
    assert len(calls) == 1


def test_telegram_server_errors_are_retried(monkeypatch):
    calls = []
    monkeypatch.setattr(telegram_client.requests, "post",
                        lambda url, data, timeout: calls.append(1) or FakeResponse(500))
    assert telegram_client.send_message("hi", token="T", channel="@c", sleep=lambda s: None) is False
    assert len(calls) == 3


def test_telegram_never_logs_the_token(monkeypatch):
    secret = "123456:SECRET-TOKEN"
    records = []

    class Catch(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Catch()
    telegram_client.log.addHandler(handler)

    def boom(url, data, timeout):
        raise requests.ConnectionError(f"cannot reach {url}")      # the real error text contains the URL

    monkeypatch.setattr(telegram_client.requests, "post", boom)
    try:
        assert telegram_client.send_message("hi", token=secret, channel="@c", sleep=lambda s: None) is False
    finally:
        telegram_client.log.removeHandler(handler)
    assert records and all("SECRET" not in r for r in records)


def test_telegram_without_settings_returns_false(monkeypatch):
    # make sure a real .env on this computer cannot leak in (and nothing is sent for real)
    monkeypatch.setattr(telegram_client.os, "getenv", lambda key, default=None: None)
    assert telegram_client.send_message("hi") is False


# ----------------------------------------------------------------------------- guards added in week 11
def test_too_early_does_nothing():
    sender = Sender()
    result = jobs.evening_job(pd.Timestamp("2026-11-25 14:00"), fetch=fake_fetch,
                              predict_fn=ok_predictor, send=sender)
    assert result["status"] == "too_early"
    assert sender.messages == []
    assert len(db.get_predictions()) == 0


def test_dry_run_logs_but_does_not_post():
    sender = Sender(results=(None,))             # send_or_dry_run returns None in dry-run mode
    result = run_evening(sender)
    assert result["status"] == "dry_run"
    row = db.get_predictions().iloc[0]
    assert pd.isna(row["posted_at"])


def test_dry_run_sender_respects_env(monkeypatch):
    monkeypatch.setenv("DRY_RUN", "1")
    assert telegram_client.send_or_dry_run("hello") is None


def test_impossible_input_gives_no_forecast():
    def bad_fetch(start, end):
        df = fake_fetch(start, end)
        df["relative_humidity_2m"] = 400.0
        return df
    sender = Sender()
    result = run_evening(sender, fetch=bad_fetch)
    assert result["forecast"] == "no_forecast"
    assert "No PM2.5 forecast" in sender.messages[0]

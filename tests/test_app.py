"""Tests for app.py [B]: the public page and the scheduler's "is it due?" rules."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

import app
import config
import db


@pytest.fixture(autouse=True)
def temp_database(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATABASE_PATH", str(tmp_path / "test.db"), raising=False)
    db.init_db()


def log_ok(day="2026-11-26", pred=40.0, unhealthy=True):
    db.log_prediction({"target_date": day, "pred_pm25": pred, "prob_unhealthy": 0.7,
                       "is_unhealthy": unhealthy, "confidence": 0.7, "model_version": "v1",
                       "input_json": {}, "status": "ok"})


def test_page_works_on_empty_database():
    page = app.render_page()
    assert "No forecast yet" in page and "What we store" in page


def test_page_shows_forecast_and_accuracy():
    log_ok()
    db.fill_label("2026-11-26", 36.0)
    page = app.render_page()
    assert "2026-11-26" in page and "Average error: 4.0" in page


def test_accuracy_counts_caught_and_false_alarms():
    log_ok("2026-11-20", 40, True); db.fill_label("2026-11-20", 30)    # caught
    log_ok("2026-11-21", 40, True); db.fill_label("2026-11-21", 5)     # false alarm
    acc = app.accuracy()
    assert (acc["n"], acc["bad_days"], acc["caught"], acc["false_alarms"]) == (2, 1, 1, 1)


def test_evening_due_window_and_after_posting():
    assert not app.evening_due(pd.Timestamp("2026-11-25 19:00"))
    assert app.evening_due(pd.Timestamp("2026-11-25 20:20"))
    log_ok("2026-11-26")
    assert app.evening_due(pd.Timestamp("2026-11-25 20:20"))           # logged but not yet posted
    db.mark_posted("2026-11-26", "v1")
    assert not app.evening_due(pd.Timestamp("2026-11-25 20:20"))       # posted: never again
    assert not app.evening_due(pd.Timestamp("2026-11-25 23:45"))


def test_nextday_due_only_when_labels_missing():
    assert not app.nextday_due(pd.Timestamp("2026-11-27 08:00"))
    log_ok("2026-11-26")
    assert app.nextday_due(pd.Timestamp("2026-11-27 08:00"))
    assert not app.nextday_due(pd.Timestamp("2026-11-27 06:00"))
    db.fill_label("2026-11-26", 30.0)
    assert not app.nextday_due(pd.Timestamp("2026-11-27 08:00"))

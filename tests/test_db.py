"""Tests for db.py [B]. They need no internet and use a temporary database. Run: pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

import config
import db

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def temp_database(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATABASE_PATH", str(tmp_path / "test.db"), raising=False)
    db.init_db()


def _daily(values, start="2026-01-01", hours_full=24):
    dates = pd.date_range(start, periods=len(values))
    return pd.DataFrame({"date": dates, "pm25_full": values, "hours_full": hours_full,
                         "is_reliable": 1, "target_ok": 1})


# ------------------------------------------------------------------ daily table
def test_upsert_daily_twice_keeps_one_row_per_date():
    db.upsert_daily(_daily([10.0, 20.0]))
    db.upsert_daily(_daily([10.0, 20.0]))
    assert len(db.get_daily()) == 2


def test_new_feature_column_is_added_automatically():
    db.upsert_daily(_daily([10.0, 20.0]))
    more = _daily([10.0, 20.0])
    more["pm25_lag1"] = [1.5, 2.5]                  # a column that did not exist before
    db.upsert_daily(more)
    assert db.get_daily().loc[1, "pm25_lag1"] == 2.5


def test_more_complete_day_wins_and_a_gap_cannot_erase_data():
    db.upsert_daily(_daily([10.0], hours_full=24))
    db.upsert_daily(_daily([99.0], hours_full=10))      # API gap: fewer hours -> ignored
    assert db.get_daily().loc[0, "pm25_full"] == 10.0
    db.upsert_daily(_daily([12.0], hours_full=24))      # same completeness -> updated
    assert db.get_daily().loc[0, "pm25_full"] == 12.0


def test_today_is_updated_as_more_hours_arrive():
    db.upsert_daily(_daily([np.nan], hours_full=15))
    db.upsert_daily(_daily([np.nan], hours_full=19))
    assert db.get_daily().loc[0, "hours_full"] == 19


def test_missing_values_are_stored_as_nan_and_flags_stay_integers():
    d = _daily([np.nan, 20.0])
    d.loc[0, ["is_reliable", "target_ok"]] = 0
    db.upsert_daily(d)
    out = db.get_daily()
    assert np.isnan(out.loc[0, "pm25_full"])
    assert list(out["is_reliable"]) == [0, 1]
    assert out["is_reliable"].dtype.kind == "i"


def test_get_daily_date_range_and_types():
    db.upsert_daily(_daily([1.0, 2.0, 3.0, 4.0]))
    out = db.get_daily("2026-01-02", "2026-01-03")
    assert list(out["pm25_full"]) == [2.0, 3.0]
    assert str(out["date"].dtype).startswith("datetime64")
    assert "fetched_at" not in out.columns
    assert "fetched_at" in db.get_daily(with_meta=True).columns


def test_unsafe_column_name_is_rejected():
    bad = _daily([1.0])
    bad["x; DROP TABLE daily_data"] = 1
    with pytest.raises(ValueError):
        db.upsert_daily(bad)


def test_integration_with_A_sample_50_twice_gives_50_rows_unchanged():
    path = ROOT / "data" / "sample_50.csv"
    if not path.exists():
        pytest.skip("data/sample_50.csv not found")
    sample = pd.read_csv(path)
    db.upsert_daily(sample)
    db.upsert_daily(sample)
    out = db.get_daily()
    assert len(out) == len(sample) == 50
    expected = sample.assign(date=pd.to_datetime(sample["date"]))
    pd.testing.assert_frame_equal(out[list(sample.columns)], expected,
                                  check_dtype=False, check_exact=False)


# ------------------------------------------------------------------ prediction log
def _record(**kw):
    base = dict(target_date="2026-02-01", pred_pm25=42.0, prob_unhealthy=0.8, is_unhealthy=True,
                confidence=0.8, model_version="v1", input_json={"pm25_today": 40.0},
                status="ok", reason=None)
    base.update(kw)
    return base


def test_log_prediction_twice_does_not_duplicate():
    assert db.log_prediction(_record()) is True
    assert db.log_prediction(_record()) is False
    assert len(db.get_predictions()) == 1


def test_two_model_versions_can_forecast_the_same_day():
    db.log_prediction(_record(model_version="v1"))
    db.log_prediction(_record(model_version="v2"))
    assert len(db.get_predictions(target_date="2026-02-01")) == 2


def test_no_forecast_is_logged_without_numbers():
    db.log_prediction(_record(status="no_forecast", reason="missing hours", pred_pm25=None,
                              prob_unhealthy=None, is_unhealthy=None, confidence=None))
    row = db.get_predictions().iloc[0]
    assert row["status"] == "no_forecast" and pd.isna(row["prediction"])


def test_fill_label_updates_all_versions_once():
    db.log_prediction(_record(model_version="v1"))
    db.log_prediction(_record(model_version="v2"))
    assert db.fill_label("2026-02-01", 37.5) == 2
    assert db.fill_label("2026-02-01", 99.0) == 0          # a filled label is not overwritten
    assert set(db.get_predictions()["label"]) == {37.5}
    assert db.fill_label("2026-02-01", 40.0, overwrite=True) == 2


def test_get_predictions_filters_labeled():
    db.log_prediction(_record(target_date="2026-02-01"))
    db.log_prediction(_record(target_date="2026-02-02"))
    db.fill_label("2026-02-01", 30.0)
    assert len(db.get_predictions(labeled=True)) == 1
    assert len(db.get_predictions(labeled=False)) == 1
    assert len(db.get_predictions(start="2026-02-02")) == 1


def test_mark_posted_only_once():
    db.log_prediction(_record())
    assert db.mark_posted("2026-02-01", "v1") is True
    assert db.mark_posted("2026-02-01", "v1") is False


def test_timestamps_are_in_tashkent_time():
    db.log_prediction(_record())
    assert db.get_predictions().iloc[0]["time"].endswith("+05:00")


def test_database_path_falls_back_when_config_has_none(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATABASE_PATH", None, raising=False)
    monkeypatch.setattr(config, "ROOT", tmp_path, raising=False)
    assert db._db_path() == tmp_path / "data" / "smog.db"

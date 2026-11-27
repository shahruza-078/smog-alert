"""ML tests [A]: features, baseline, train.py, predict.py. No internet needed.

Run:  python -m pytest -q
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

import config
import data_sources as ds
import db
import features
import predict
import train
from ml.baselines import PersistenceRegressor


def make_daily(start="2023-01-01", end="2026-11-24", seed=0):
    """Fake daily table (same columns as data_sources.hourly_to_daily)."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    n, doy = len(days), days.dayofyear.to_numpy()
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i - 1] + rng.normal(0, 4)
    full = np.clip(18 + 12 * np.cos(2 * np.pi * (doy - 15) / 365) + noise, 2, None)
    today = np.clip(full + rng.normal(0, 2, n), 1, None)
    d = pd.DataFrame({
        "date": days, "pm25_today": today, "pm25_max_today": today * 1.5,
        "pm25_last3h": today + rng.normal(0, 2, n), "temp_today": 10 + rng.normal(0, 3, n),
        "humidity_today": rng.uniform(30, 90, n), "wind_today": rng.uniform(1, 10, n),
        "pressure_today": rng.normal(950, 5, n), "precip_today": rng.exponential(0.5, n),
        "pm25_full": full, "hours_in_window": 20, "hours_full": 24, "is_reliable": 1, "target_ok": 1})
    return d[ds.DAILY_COLUMNS]


def quiet(*args, **kwargs):
    pass


def at(frame, day):
    return frame.loc[pd.Timestamp(day)]


# ------------------------------------------------------------------ features
def test_feature_table_shape_and_columns():
    X = features.build_features(make_daily())
    assert list(X.columns) == features.FEATURE_COLUMNS
    assert X.index.is_monotonic_increasing and X.index.is_unique
    assert len(X) == len(pd.date_range("2023-01-01", "2026-11-24"))


def test_features_of_day_D_do_not_use_the_future():
    daily = make_daily()
    D = pd.Timestamp("2025-03-10")
    base = at(features.build_features(daily), D)

    changed = daily.copy()
    later = pd.to_datetime(changed["date"]) > D
    changed.loc[later, ["pm25_today", "pm25_full", "temp_today"]] = 999.0      # everything after D
    changed.loc[pd.to_datetime(changed["date"]) == D, "pm25_full"] = 999.0      # D's own full-day mean
    alt = at(features.build_features(changed), D)
    pd.testing.assert_series_equal(base, alt)


def test_lags_mean_exactly_one_day_even_when_a_day_is_missing():
    daily = make_daily("2026-01-01", "2026-01-20")
    full = daily.set_index("date")["pm25_full"]
    gone = daily[pd.to_datetime(daily["date"]) != pd.Timestamp("2026-01-10")]
    X = features.build_features(gone)
    assert np.isnan(at(X, "2026-01-11")["pm25_lag1"])                             # 10th is missing
    assert at(X, "2026-01-12")["pm25_lag2"] != at(X, "2026-01-12")["pm25_lag1"]
    assert at(X, "2026-01-15")["pm25_lag1"] == pytest.approx(full[pd.Timestamp("2026-01-14")])


def test_targets_are_next_day_and_threshold_applied():
    daily = make_daily("2026-01-01", "2026-01-31")
    y_reg, y_clf = features.make_targets(daily, threshold=20)
    full = daily.set_index("date")["pm25_full"]
    assert y_reg[pd.Timestamp("2026-01-05")] == pytest.approx(full[pd.Timestamp("2026-01-06")])
    assert np.isnan(y_reg.iloc[-1]) and np.isnan(y_clf.iloc[-1])
    assert (y_clf.dropna() == (y_reg.dropna() >= 20).astype(float)).all()


def test_make_dataset_keeps_only_reliable_rows_with_a_known_label():
    daily = make_daily("2026-01-01", "2026-01-31")
    daily.loc[daily["date"] == "2026-01-10", "is_reliable"] = 0           # bot would say "no forecast"
    daily.loc[daily["date"] == "2026-01-21", ["pm25_full", "target_ok"]] = [np.nan, 0]  # label unknown
    X, y_reg, y_clf = features.make_dataset(daily)
    assert pd.Timestamp("2026-01-10") not in X.index
    assert pd.Timestamp("2026-01-20") not in X.index                      # its label (21st) is unknown
    assert not y_reg.isna().any() and set(y_clf.unique()) <= {0, 1}
    assert len(X) == len(y_reg) == len(y_clf)


# ------------------------------------------------------------------ baseline
def test_persistence_predicts_todays_value_and_survives_joblib(tmp_path):
    X, y_reg, _ = features.make_dataset(make_daily("2025-01-01", "2025-06-30"))
    pipe = Pipeline([("model", PersistenceRegressor())]).fit(X, y_reg)
    assert np.allclose(pipe.predict(X), X["pm25_today"])
    joblib.dump(pipe, tmp_path / "p.joblib")
    assert np.allclose(joblib.load(tmp_path / "p.joblib").predict(X), X["pm25_today"])


def test_persistence_falls_back_to_median_when_today_is_missing():
    X, y_reg, _ = features.make_dataset(make_daily("2025-01-01", "2025-03-31"))
    pipe = Pipeline([("model", PersistenceRegressor())]).fit(X, y_reg)
    X2 = X.iloc[:3].copy()
    X2["pm25_today"] = np.nan
    assert np.allclose(pipe.predict(X2), float(np.median(y_reg)))


# ------------------------------------------------------------------ train.py
def test_train_cv_only_logs_and_saves_nothing(tmp_path):
    out = train.run(make_daily(), models_dir=tmp_path / "m", logs_dir=tmp_path / "l", echo=quiet)
    assert set(out["results"]) == {"dummy_median", "persistence", "linear_regression"}
    assert not (tmp_path / "m").exists()
    log = pd.read_csv(tmp_path / "l" / "experiment_log.csv")
    assert list(log.columns) == train.LOG_COLUMNS and len(log) == 3
    train.run(make_daily(), models_dir=tmp_path / "m", logs_dir=tmp_path / "l", echo=quiet)
    assert len(pd.read_csv(tmp_path / "l" / "experiment_log.csv")) == 6   # runs are appended, never replaced
    assert (tmp_path / "l" / "error_worksheet_1_candidates.csv").exists()


def test_persistence_beats_the_dummy_on_synthetic_data(tmp_path):
    r = train.run(make_daily(), models_dir=tmp_path, logs_dir=tmp_path, echo=quiet)["results"]
    assert r["persistence"]["mae"] < r["dummy_median"]["mae"]


def test_train_never_uses_the_test_period_and_never_overwrites(tmp_path):
    train.run(make_daily(), models_dir=tmp_path / "m", logs_dir=tmp_path / "l", save=True, echo=quiet)
    meta = json.loads((tmp_path / "m" / "model_v0.json").read_text())
    last_train_day = pd.Timestamp(meta["train_date_range"][1])
    assert last_train_day < pd.Timestamp(config.TEST_START_DATE) - pd.Timedelta(days=config.GAP_DAYS)
    for key in ("version", "created_at", "git_commit", "train_date_range", "n_rows", "features", "cv_scores",
                "test_scores", "unhealthy_threshold", "decision_probability_threshold", "feature_ranges"):
        assert key in meta
    assert meta["test_scores"] is None
    assert set(meta["feature_ranges"]) == set(features.FEATURE_COLUMNS)
    with pytest.raises(SystemExit):                                       # second save of v0 is refused
        train.run(make_daily(), models_dir=tmp_path / "m", logs_dir=tmp_path / "l", save=True, echo=quiet)


# ------------------------------------------------------------------ predict.py
@pytest.fixture
def bundle(tmp_path):
    train.run(make_daily(), models_dir=tmp_path / "m", logs_dir=tmp_path / "l", save=True, echo=quiet)
    return predict.load_model(models_dir=tmp_path / "m")


def test_load_model_picks_the_newest_version(tmp_path):
    train.run(make_daily(), models_dir=tmp_path, logs_dir=tmp_path, version=0, save=True, echo=quiet)
    train.run(make_daily(), models_dir=tmp_path, logs_dir=tmp_path, version=2, save=True, echo=quiet)
    assert predict.load_model(models_dir=tmp_path)["version"] == "v2"
    assert predict.load_model("v0", models_dir=tmp_path)["version"] == "v0"
    with pytest.raises(FileNotFoundError):
        predict.load_model(models_dir=tmp_path / "empty")


def test_predict_tomorrow_ok(bundle):
    daily = make_daily()
    rec = predict.predict_tomorrow(daily, bundle, day="2026-11-20")
    assert rec["status"] == "ok" and rec["reason"] is None
    assert rec["target_date"] == "2026-11-21" and rec["model_version"] == "v0"
    assert rec["pred_pm25"] == pytest.approx(at(daily.set_index("date"), "2026-11-20")["pm25_today"], abs=0.01)
    assert rec["is_unhealthy"] == (rec["pred_pm25"] >= config.UNHEALTHY_THRESHOLD)
    assert rec["prob_unhealthy"] is None and rec["confidence"] is None       # v0 has no classifier
    assert rec["input_json"]["day"] == "2026-11-20"
    assert set(rec) == {"target_date", "pred_pm25", "prob_unhealthy", "is_unhealthy", "confidence",
                        "model_version", "input_json", "status", "reason"}


def test_predict_uses_today_in_tashkent_by_default(bundle, monkeypatch):
    monkeypatch.setattr(predict, "_now_local", lambda: pd.Timestamp("2026-11-20 21:00"))
    assert predict.predict_tomorrow(make_daily(), bundle)["target_date"] == "2026-11-21"


def test_no_forecast_when_data_is_missing_or_unreliable(bundle):
    daily = make_daily()
    bad = daily.copy()
    bad.loc[bad["date"] == "2026-11-20", ["is_reliable", "hours_in_window"]] = [0, 11]
    rec = predict.predict_tomorrow(bad, bundle, day="2026-11-20")
    assert rec["status"] == "no_forecast" and "11 of 20" in rec["reason"] and rec["pred_pm25"] is None

    gone = daily[daily["date"] != "2026-11-20"]
    assert predict.predict_tomorrow(gone, bundle, day="2026-11-20")["status"] == "no_forecast"
    assert predict.predict_tomorrow(daily.iloc[0:0], bundle, day="2026-11-20")["status"] == "no_forecast"


def test_same_answer_with_a_14_day_window_as_the_bot_uses(bundle):
    daily = make_daily()
    D = pd.Timestamp("2026-11-20")
    window = daily[(daily["date"] >= D - pd.Timedelta(days=14)) & (daily["date"] <= D)]
    full = predict.predict_tomorrow(daily, bundle, day=D)
    small = predict.predict_tomorrow(window, bundle, day=D)
    assert full["pred_pm25"] == small["pred_pm25"] and full["input_json"] == small["input_json"]


def test_result_can_be_logged_by_bs_database(bundle, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATABASE_PATH", str(tmp_path / "t.db"), raising=False)
    rec = predict.predict_tomorrow(make_daily(), bundle, day="2026-11-20")
    assert db.log_prediction(rec) is True
    assert db.log_prediction(rec) is False                                   # second run: no duplicate
    row = db.get_predictions().iloc[0]
    assert row["status"] == "ok" and row["model_version"] == "v0" and row["target_date"] == "2026-11-21"
    assert json.loads(row["input"])["day"] == "2026-11-20"

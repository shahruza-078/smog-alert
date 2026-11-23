"""train.py - rebuild the PM2.5 model from the database with ONE command.

    python train.py                      # compare models with cross-validation (test set untouched)
    python train.py --release            # ONLY when ready: fit, score the test set once, save v1
    python train.py --release --version 2   # later: a v2 candidate (promotion gate comes in week 14)

Task: on the evening of day D, predict the daily mean PM2.5 of day D+1 (regression)
and whether D+1 is an unhealthy day (classification).

Rules from the handbook that this script enforces:
  * Split by DATE, never at random. The cut-off is written once to split.json ("day one").
  * Models are chosen with time-ordered cross-validation on the TRAIN part only.
  * The test part is only read when --release is given, and a released version
    (model_vN.joblib / model_vN.json) is never overwritten.
  * Every run is appended to experiment_log.csv, including baselines and bad runs.
  * Features use only data known on the evening of day D (nothing from D+1).
"""
import argparse
import csv
import json
import os
import sqlite3
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Agree this with the client! WHO 24-hour guideline is 15 ug/m3, but most Tashkent
# winter days exceed it. A higher value (e.g. 35 or 55) may match "move PE indoors".
DEFAULT_THRESHOLD = 15.0
TEST_FRACTION = 0.15
MIN_ROWS = 60

FEATURES = ["pm25_today", "pm25_lag1", "pm25_lag2", "pm25_mean7",
            "temp", "humidity", "wind", "pressure", "precip",
            "dow_sin", "dow_cos", "month_sin", "month_cos"]


# ----------------------------------------------------------------- data
def load_daily(db):
    con = sqlite3.connect(db)
    df = pd.read_sql("SELECT * FROM daily", con, parse_dates=["date"])
    con.close()
    df = df.set_index("date").sort_index()
    # Reindex to every calendar day so shift(1) always means "exactly one day earlier".
    return df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))


def build_features(daily):
    """One row per evening D. Row index = D. Target y = PM2.5 mean of D+1.

    Note: daily weather means of day D are used as if known at posting time. At the real
    evening post, the last hours of D are not yet in; mention this in your limitations.
    """
    pm = daily["pm25_mean"]
    X = pd.DataFrame(index=daily.index)
    X["pm25_today"] = pm
    X["pm25_lag1"] = pm.shift(1)
    X["pm25_lag2"] = pm.shift(2)
    X["pm25_mean7"] = pm.rolling(7, min_periods=5).mean()
    for c in ["temp", "humidity", "wind", "pressure", "precip"]:
        X[c] = daily[c]
    dow, month = X.index.dayofweek, X.index.month
    X["dow_sin"], X["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    X["month_sin"], X["month_cos"] = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
    X["y"] = pm.shift(-1)  # tomorrow's value = the label
    return X.dropna()


def get_split(data, outdir):
    """Test cut-off is decided once and stored. Never recomputed."""
    path = os.path.join(outdir, "split.json")
    if os.path.exists(path):
        with open(path) as f:
            test_start = pd.Timestamp(json.load(f)["test_start"])
    else:
        test_start = data.index[int(len(data) * (1 - TEST_FRACTION))]
        with open(path, "w") as f:
            json.dump({"test_start": str(test_start.date()),
                       "created": datetime.now().isoformat(timespec="seconds"),
                       "note": "Rows on/after test_start are the test set. Do not edit."}, f, indent=2)
        print(f"Created {path}: test set = dates from {test_start.date()} (set aside).")
    # 1-day gap: a train row's label (D+1) must not be a test row's input (D).
    train = data[data.index < test_start - pd.Timedelta(days=1)]
    test = data[data.index >= test_start]
    return train, test


# ----------------------------------------------------------------- models
def make_pipeline(model):
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def regression_candidates():
    c = {"linear_regression": make_pipeline(LinearRegression())}
    for k in (5, 10, 20):
        c[f"knn_k{k}"] = make_pipeline(KNeighborsRegressor(n_neighbors=k))
    for d in (3, 5, 8):
        c[f"tree_depth{d}"] = make_pipeline(
            DecisionTreeRegressor(max_depth=d, min_samples_leaf=10, random_state=0))
    return c


def make_classifier():
    # class_weight balanced: the client says missing a bad day is worse than a false alarm.
    return make_pipeline(LogisticRegression(class_weight="balanced", max_iter=1000))


# ----------------------------------------------------------------- evaluation
def cv_mae(train, estimator=None, n_splits=5):
    """Time-ordered CV. estimator=None means persistence (tomorrow = today)."""
    errs = []
    for tr, va in TimeSeriesSplit(n_splits=n_splits).split(train):
        a, b = train.iloc[tr], train.iloc[va]
        if estimator is None:
            pred = b["pm25_today"]
        else:
            pred = clone(estimator).fit(a[FEATURES], a["y"]).predict(b[FEATURES])
        errs.append(mean_absolute_error(b["y"], pred))
    return float(np.mean(errs)), float(np.std(errs))


def cv_unhealthy(train, thr, estimator="persistence", n_splits=5):
    rec, prec = [], []
    for tr, va in TimeSeriesSplit(n_splits=n_splits).split(train):
        a, b = train.iloc[tr], train.iloc[va]
        ya, yb = (a["y"] >= thr).astype(int), (b["y"] >= thr).astype(int)
        if isinstance(estimator, str):
            pred = (b["pm25_today"] >= thr).astype(int)
        else:
            if ya.nunique() < 2:
                continue
            pred = clone(estimator).fit(a[FEATURES], ya).predict(b[FEATURES])
        if yb.sum() > 0:
            rec.append(recall_score(yb, pred, zero_division=0))
            prec.append(precision_score(yb, pred, zero_division=0))
    f = lambda v: float(np.mean(v)) if v else float("nan")
    return f(rec), f(prec)


# ----------------------------------------------------------------- experiment log
LOG_COLS = ["#", "Date", "What changed", "Why we tried it", "CV score", "Kept?"]


def log_run(path, what, why, score, kept):
    new = not os.path.exists(path)
    n = 1
    if not new:
        with open(path, newline="") as f:
            n = sum(1 for _ in csv.reader(f))  # header counts as 1 -> next row number
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LOG_COLS)
        w.writerow([n, datetime.now().strftime("%Y-%m-%d %H:%M"), what, why, score, kept])


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="smog.db")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--why", default="", help="why you are running this (goes in the log)")
    ap.add_argument("--release", action="store_true",
                    help="fit, score the TEST set once and save model_vN files")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    logpath = os.path.join(args.outdir, "experiment_log.csv")

    daily = load_daily(args.db)
    data = build_features(daily)
    if len(data) < MIN_ROWS:
        print(f"WARNING: only {len(data)} usable days. Pull more history with fetch_data.py --start ...")
    train, test = get_split(data, args.outdir)
    thr = args.threshold
    print(f"Usable days: {len(data)} | train {len(train)} "
          f"({train.index.min().date()}..{train.index.max().date()}) | test {len(test)} (not touched)")
    print(f"Share of unhealthy days (>= {thr} ug/m3) in train: {(train['y'] >= thr).mean():.0%}")

    # --- baselines
    why_base = "Reference point"
    dummy = make_pipeline(DummyRegressor(strategy="median"))
    d_mae, _ = cv_mae(train, dummy)
    p_mae, _ = cv_mae(train, None)
    log_run(logpath, "Dummy baseline (median)", why_base, f"MAE {d_mae:.2f}", "Yes (reference)")
    log_run(logpath, "Smart baseline: persistence (tomorrow = today)", "The rule to beat",
            f"MAE {p_mae:.2f}", "Yes (reference)")

    # --- course models
    results = {}
    for name, est in regression_candidates().items():
        m, s = cv_mae(train, est)
        results[name] = (m, s)
    best = min(results, key=lambda k: results[k][0])
    for name, (m, s) in results.items():
        log_run(logpath, name, args.why or "Course model comparison",
                f"MAE {m:.2f} +/- {s:.2f}", "Yes (best)" if name == best else "No")

    # --- classifier for the unhealthy flag
    clf = make_classifier()
    r_rec, r_prec = cv_unhealthy(train, thr, "persistence")
    c_rec, c_prec = cv_unhealthy(train, thr, clf)
    log_run(logpath, f"Unhealthy flag, persistence rule (thr {thr})", "Smart baseline for the flag",
            f"recall {r_rec:.2f}, precision {r_prec:.2f}", "Yes (reference)")
    log_run(logpath, f"Unhealthy flag, logistic regression (thr {thr})", args.why or "Course model",
            f"recall {c_rec:.2f}, precision {c_prec:.2f}", "Yes")

    print("\nCV results on TRAIN only (time-ordered folds), MAE in ug/m3 - lower is better")
    print(f"  dummy median          {d_mae:6.2f}")
    print(f"  persistence (smart)   {p_mae:6.2f}")
    for name, (m, s) in sorted(results.items(), key=lambda kv: kv[1][0]):
        print(f"  {name:<21} {m:6.2f} +/- {s:.2f}{'   <- best' if name == best else ''}")
    print(f"\nUnhealthy flag CV: persistence recall {r_rec:.2f} / precision {r_prec:.2f} | "
          f"logistic recall {c_rec:.2f} / precision {c_prec:.2f}")
    beats = results[best][0] < p_mae
    if not beats:
        print("NOTE: persistence beats every course model. That is allowed - report it honestly.")

    if not args.release:
        print("\nTest set NOT used. Run with --release when the decision note is ready.")
        return

    # --- release: fit on train, score test exactly once, save versioned files
    mpath = os.path.join(args.outdir, f"model_v{args.version}.joblib")
    jpath = os.path.join(args.outdir, f"model_v{args.version}.json")
    if os.path.exists(mpath) or os.path.exists(jpath):
        sys.exit(f"{mpath} already exists. A released version never changes - use --version {args.version + 1}.")
    reg = clone(regression_candidates()[best]).fit(train[FEATURES], train["y"])
    ytr = (train["y"] >= thr).astype(int)
    if ytr.nunique() < 2:
        sys.exit("Train data has only one class for the unhealthy flag - change --threshold or add data.")
    clf = make_classifier().fit(train[FEATURES], ytr)

    yt, yt_flag = test["y"], (test["y"] >= thr).astype(int)
    pred = reg.predict(test[FEATURES])
    flag = clf.predict(test[FEATURES])
    test_metrics = {
        "mae_model": float(mean_absolute_error(yt, pred)),
        "mae_dummy_median": float(mean_absolute_error(
            yt, make_pipeline(DummyRegressor(strategy="median")).fit(train[FEATURES], train["y"]).predict(test[FEATURES]))),
        "mae_persistence": float(mean_absolute_error(yt, test["pm25_today"])),
        "unhealthy_recall_model": float(recall_score(yt_flag, flag, zero_division=0)),
        "unhealthy_precision_model": float(precision_score(yt_flag, flag, zero_division=0)),
        "unhealthy_recall_persistence": float(recall_score(yt_flag, (test["pm25_today"] >= thr).astype(int), zero_division=0)),
        "n_test_days": int(len(test)),
        "n_test_unhealthy": int(yt_flag.sum()),
    }
    meta = {
        "version": f"v{args.version}",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "regressor": best,
        "classifier": "logistic_regression(class_weight=balanced)",
        "features": FEATURES,
        "unhealthy_threshold": thr,
        "train_range": [str(train.index.min().date()), str(train.index.max().date())],
        "test_range": [str(test.index.min().date()), str(test.index.max().date())],
        "n_train_days": int(len(train)),
        "cv_mae": {k: {"mean": v[0], "std": v[1]} for k, v in results.items()},
        "cv_mae_dummy": d_mae,
        "cv_mae_persistence": p_mae,
        "cv_unhealthy": {"persistence_recall": r_rec, "persistence_precision": r_prec,
                         "model_recall": c_rec, "model_precision": c_prec},
        "test": test_metrics,
        "model_beats_persistence_cv": bool(beats),
    }
    joblib.dump({"regressor": reg, "classifier": clf, "features": FEATURES,
                 "threshold": thr, "version": f"v{args.version}"}, mpath)
    with open(jpath, "w") as f:
        json.dump(meta, f, indent=2)
    log_run(logpath, f"RELEASE v{args.version}: {best}", "Test set scored once",
            f"test MAE {test_metrics['mae_model']:.2f} (persistence {test_metrics['mae_persistence']:.2f})", "Yes")
    print(f"\nSaved {mpath} and {jpath}")
    print(f"TEST (scored once): model MAE {test_metrics['mae_model']:.2f} | "
          f"persistence {test_metrics['mae_persistence']:.2f} | dummy {test_metrics['mae_dummy_median']:.2f}")
    print(f"Unhealthy flag on test: recall {test_metrics['unhealthy_recall_model']:.2f}, "
          f"precision {test_metrics['unhealthy_precision_model']:.2f}")
    print("Now tag it:  git tag v1 && git push --tags")


if __name__ == "__main__":
    main()

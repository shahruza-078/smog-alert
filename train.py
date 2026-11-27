"""train.py  [A]  Evaluate the baselines on TRAIN data only and (optionally) save model v0.

    python train.py                          # cross-validation, writes the experiment log
    python train.py --save                   # also saves models/model_v0.joblib + .json
    python train.py --why "tried X"          # your reason goes into logs/experiment_log.csv

Rules enforced here
  * Only data_sources.load_train() rows are used. The test part is never read in this file;
    only release_eval.py (week 13) opens it, once.
  * Cross-validation is time-ordered (TimeSeriesSplit with a 7-row gap), never random.
  * Every run is appended to logs/experiment_log.csv, good or bad. Never delete rows.
  * A saved model_vN.joblib / .json is never overwritten.

v0 = the baseline the app serves in week 11 (persistence: tomorrow = today). The course
models (kNN, decision tree, logistic regression) and promote.py come in weeks 12-14.
"""
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
import data_sources
import features
from ml.baselines import PersistenceRegressor

MODELS_DIR = config.ROOT / "models"
LOGS_DIR = config.ROOT / "logs"
N_SPLITS = 5
CV_GAP = 7          # rows skipped between a training fold and its validation fold
MIN_ROWS = 100
LOG_COLUMNS = ["#", "Date", "What changed", "Why we tried it", "CV score", "Kept?"]


# ------------------------------------------------------------------------- models
def make_pipeline(estimator):
    """The one helper every real model goes through: fill gaps, scale, then the model."""
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", estimator)])


def candidate_pipelines():
    return {
        # no imputer/scaler: these two must see the raw feature values
        "dummy_median": Pipeline([("model", DummyRegressor(strategy="median"))]),
        "persistence": Pipeline([("model", PersistenceRegressor())]),
        "linear_regression": make_pipeline(LinearRegression()),
    }


# ------------------------------------------------------------------------- evaluation
def _nan_to_none(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else x


def cv_evaluate(X, y_reg, y_clf, pipelines, threshold, n_splits=N_SPLITS, gap=CV_GAP):
    """Time-ordered CV. Returns (results per model, out-of-fold predictions)."""
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    oof = pd.DataFrame({"y_true": y_reg}, index=X.index)
    results = {}
    for name, pipe in pipelines.items():
        pred = pd.Series(np.nan, index=X.index)
        maes = []
        for tr, va in splitter.split(X):
            model = clone(pipe).fit(X.iloc[tr], y_reg.iloc[tr])
            p = model.predict(X.iloc[va])
            pred.iloc[va] = p
            maes.append(mean_absolute_error(y_reg.iloc[va], p))
        oof[f"pred_{name}"] = pred

        seen = pred.notna()
        flag = (pred[seen] >= threshold).astype(int)
        real = y_clf[seen]
        tp = int(((flag == 1) & (real == 1)).sum())
        results[name] = {
            "mae": float(np.mean(maes)), "mae_std": float(np.std(maes)),
            "recall": tp / int(real.sum()) if real.sum() else None,       # real unhealthy days caught
            "precision": tp / int(flag.sum()) if flag.sum() else None,    # alarms that were right
            "n_unhealthy": int(real.sum()), "n_flagged": int(flag.sum()),
        }
    return results, oof


def write_error_worksheet(path, X, oof, model="persistence", n=20):
    """The n worst out-of-fold forecasts, ready for error worksheet 1 (fill in `cause`)."""
    err = (oof["y_true"] - oof[f"pred_{model}"]).abs().dropna().sort_values(ascending=False).head(n)
    ws = pd.concat([oof.loc[err.index, ["y_true", f"pred_{model}"]], err.rename("abs_error"),
                    X.loc[err.index, ["pm25_today", "pm25_lag1", "pm25_mean7", "temp_today", "wind_today"]]],
                   axis=1)
    ws.insert(0, "date", ws.index.strftime("%Y-%m-%d"))
    ws["cause"] = ""   # wrong label | missing feature | rare event | typo/outlier | duplicate/leak | other
    ws["notes"] = ""
    ws.to_csv(path, index=False)


# ------------------------------------------------------------------------- logging
def log_run(path, what, why, score, kept):
    new = not path.exists()
    n = 1
    if not new:
        with open(path, newline="", encoding="utf-8") as f:
            n = sum(1 for _ in csv.reader(f))      # header is row 1 -> next number
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LOG_COLUMNS)
        w.writerow([n, datetime.now(ZoneInfo(config.TIMEZONE)).strftime("%Y-%m-%d %H:%M"),
                    what, why, score, kept])


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


# ------------------------------------------------------------------------- the run
def run(daily, models_dir=MODELS_DIR, logs_dir=LOGS_DIR, version=0, model_name="persistence",
        threshold=None, why="", save=False, echo=print):
    thr = config.UNHEALTHY_THRESHOLD if threshold is None else float(threshold)
    models_dir, logs_dir = Path(models_dir), Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    train = data_sources.load_train(daily)                 # TRAIN ROWS ONLY
    X, y_reg, y_clf = features.make_dataset(train, thr)
    if len(X) < MIN_ROWS:
        sys.exit(f"Only {len(X)} usable training days (need {MIN_ROWS}). Load more history first.")
    pipes = candidate_pipelines()
    if model_name not in pipes:
        sys.exit(f"Unknown model '{model_name}'. Choose one of: {', '.join(pipes)}")

    echo(f"Training rows: {len(X)} ({X.index.min().date()} .. {X.index.max().date()}); "
         f"the test part ({config.TEST_START_DATE} onwards) is not touched.")
    echo(f"Unhealthy threshold {thr:g} ug/m3: {y_clf.mean():.0%} of training days are unhealthy.")

    results, oof = cv_evaluate(X, y_reg, y_clf, pipes, thr)

    # ---- table
    echo(f"\nCross-validation ({N_SPLITS} time-ordered folds). MAE in ug/m3: lower is better. "
         "Recall = share of real unhealthy days we caught.")
    echo(f"{'model':<20}{'MAE':>8}{'+/-':>7}{'recall':>9}{'precision':>11}")
    fmt = lambda v: "   n/a" if v is None else f"{v:>9.2f}"
    for name, r in results.items():
        echo(f"{name:<20}{r['mae']:>8.2f}{r['mae_std']:>7.2f}{fmt(r['recall'])}{fmt(r['precision']):>11}")

    # ---- experiment log (every run, good or bad)
    mae_best_ref = results["persistence"]["mae"]
    notes = {"dummy_median": ("Reference point (always guess the median)", "Yes (reference)"),
             "persistence": ("The smart baseline to beat: tomorrow = today", "Yes (reference)"),
             "linear_regression": ("First course model, all features",
                                   "Yes" if results["linear_regression"]["mae"] < mae_best_ref
                                   else "No - persistence is better")}
    for name, r in results.items():
        rec = "n/a" if r["recall"] is None else f"{r['recall']:.2f}"
        pre = "n/a" if r["precision"] is None else f"{r['precision']:.2f}"
        reason, kept = notes[name]
        log_run(logs_dir / "experiment_log.csv", f"{name} ({N_SPLITS}-fold time CV, threshold {thr:g})",
                why or reason, f"MAE {r['mae']:.2f} +/- {r['mae_std']:.2f}; recall {rec}; precision {pre}", kept)

    oof.to_csv(logs_dir / f"oof_v{version}.csv", index_label="date")
    write_error_worksheet(logs_dir / "error_worksheet_1_candidates.csv", X, oof)
    echo(f"\nLogged to {logs_dir / 'experiment_log.csv'} and wrote the 20 worst persistence forecasts to "
         f"error_worksheet_1_candidates.csv (fill in the `cause` column).")
    if results["linear_regression"]["mae"] >= mae_best_ref:
        echo("NOTE: persistence is at least as good as linear regression. That is allowed - report it honestly.")

    summary = {"results": results, "n_rows": len(X), "threshold": thr}
    if not save:
        echo("\nNothing saved. Run with --save to write the model files.")
        return summary

    # ---- save the model (never overwrite a released version)
    models_dir.mkdir(parents=True, exist_ok=True)
    tag = f"model_v{version}"
    jpath, mpath = models_dir / f"{tag}.joblib", models_dir / f"{tag}.json"
    if jpath.exists() or mpath.exists():
        sys.exit(f"{jpath.name} already exists. A saved version never changes; use --version {version + 1}.")
    final = clone(pipes[model_name]).fit(X, y_reg)
    bundle = {"regressor": final, "classifier": None, "features": list(X.columns),
              "threshold": thr, "version": f"v{version}", "cutoff_hour": config.CUTOFF_HOUR}
    meta = {
        "version": f"v{version}",
        "created_at": datetime.now(ZoneInfo(config.TIMEZONE)).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "model": model_name,
        "train_date_range": [str(X.index.min().date()), str(X.index.max().date())],
        "n_rows": int(len(X)),
        "features": list(X.columns),
        "cv_scores": {k: {m: _nan_to_none(v) for m, v in r.items()} for k, r in results.items()},
        "test_scores": None,                          # filled by release_eval.py, once, in week 13
        "unhealthy_threshold": thr,
        "decision_probability_threshold": None,       # set when a classifier exists (week 12)
        "feature_ranges": features.feature_ranges(X),
        "cutoff_hour": config.CUTOFF_HOUR,
        "min_hours_window": config.MIN_HOURS_WINDOW,
    }
    joblib.dump(bundle, jpath)
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log_run(logs_dir / "experiment_log.csv", f"Saved {tag} ({model_name})",
            "Baseline served by the app in week 11", f"CV MAE {results[model_name]['mae']:.2f}", "Yes")
    echo(f"\nSaved {jpath} and {mpath}")
    summary["saved"] = [str(jpath), str(mpath)]
    return summary


def load_daily(source="db"):
    if source == "csv":
        return pd.read_csv(config.ROOT / "data" / "daily_all.csv", parse_dates=["date"])
    import db
    daily = db.get_daily()
    if daily.empty:
        sys.exit("The database has no daily rows yet. Fill it first:  python data_sources.py --to-db")
    return daily


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["db", "csv"], default="db", help="where the daily table comes from")
    ap.add_argument("--version", type=int, default=0)
    ap.add_argument("--model", default="persistence", help="which model to save as this version")
    ap.add_argument("--threshold", type=float, default=None, help="default: config.UNHEALTHY_THRESHOLD")
    ap.add_argument("--why", default="", help="why you are running this (goes into the experiment log)")
    ap.add_argument("--save", action="store_true", help="write models/model_vN.joblib and .json")
    a = ap.parse_args()
    run(load_daily(a.source), version=a.version, model_name=a.model,
        threshold=a.threshold, why=a.why, save=a.save)


if __name__ == "__main__":
    main()

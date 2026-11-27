"""predict.py  [A]  Load a saved model and forecast tomorrow. B's jobs.py calls exactly this:

    bundle = predict.load_model()                         # newest models/model_vN.joblib
    rec    = predict.predict_tomorrow(daily_window, bundle)   # daily_window = db.get_daily(...)

predict_tomorrow() never raises for "normal" bad situations (missing or unreliable data).
It returns status="no_forecast" with a human-readable reason, and the bot then posts
"no forecast today" instead of a guess.
"""
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config
import features

MODELS_DIR = config.ROOT / "models"


def _now_local():
    return pd.Timestamp.now(tz=config.TIMEZONE).tz_localize(None)


def load_model(version=None, models_dir=None):
    """version=None -> the newest model_vN. Accepts 1, "1" or "v1". Returns the bundle dict."""
    models_dir = MODELS_DIR if models_dir is None else Path(models_dir)
    found = {}
    for p in models_dir.glob("model_v*.joblib"):
        m = re.fullmatch(r"model_v(\d+)\.joblib", p.name)
        if m:
            found[int(m.group(1))] = p
    if not found:
        raise FileNotFoundError(f"No model_vN.joblib in {models_dir}. Run: python train.py --save")
    if version is None:
        number = max(found)
    else:
        number = int(str(version).lstrip("vV"))
        if number not in found:
            raise FileNotFoundError(f"model_v{number}.joblib not found in {models_dir}")
    bundle = joblib.load(found[number])
    meta_path = models_dir / f"model_v{number}.json"
    bundle["meta"] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return bundle


def _clean(v):
    if v is None:
        return None
    v = float(v)
    return None if np.isnan(v) else round(v, 4)


def predict_tomorrow(daily_df, bundle, day=None):
    """Forecast day+1 using only data known on the evening of `day` (default: today, Tashkent).

    Returns a dict with: target_date, pred_pm25, prob_unhealthy, is_unhealthy, confidence,
    model_version, input_json, status ("ok" | "no_forecast"), reason.
    """
    version = bundle.get("version", "none")
    D = pd.Timestamp(day).normalize() if day is not None else _now_local().normalize()
    target = D + pd.Timedelta(days=1)

    def no_forecast(reason):
        return {"target_date": target.strftime("%Y-%m-%d"), "pred_pm25": None, "prob_unhealthy": None,
                "is_unhealthy": None, "confidence": None, "model_version": version,
                "input_json": None, "status": "no_forecast", "reason": reason}

    if daily_df is None or len(daily_df) == 0:
        return no_forecast("there is no data")
    dates = pd.to_datetime(daily_df["date"]).dt.normalize()
    rows = daily_df[dates == D]
    if rows.empty:
        return no_forecast(f"there is no data for {D.date()}")
    row = rows.iloc[-1]
    if int(row.get("is_reliable") or 0) != 1:
        have = int(row.get("hours_in_window") or 0)
        return no_forecast(f"only {have} of {bundle.get('cutoff_hour', config.CUTOFF_HOUR)} "
                           f"hours before the cutoff had PM2.5 data")

    cols = bundle.get("features", features.FEATURE_COLUMNS)
    X = features.build_features(daily_df).loc[[D], cols]
    pred = max(0.0, float(bundle["regressor"].predict(X)[0]))
    threshold = bundle.get("threshold", config.UNHEALTHY_THRESHOLD)

    prob = None
    clf = bundle.get("classifier")
    if clf is not None:                       # exists from week 12; v0 has none
        prob = float(clf.predict_proba(X)[0][1])
        cut = (bundle.get("meta") or {}).get("decision_probability_threshold") or 0.5
        is_unhealthy = prob >= cut
        confidence = max(prob, 1 - prob)
    else:
        is_unhealthy = pred >= threshold
        confidence = None

    inputs = {c: _clean(X.iloc[0][c]) for c in cols}
    inputs["day"] = str(D.date())
    return {"target_date": target.strftime("%Y-%m-%d"), "pred_pm25": round(pred, 2),
            "prob_unhealthy": _clean(prob), "is_unhealthy": bool(is_unhealthy),
            "confidence": _clean(confidence), "model_version": version,
            "input_json": inputs, "status": "ok", "reason": None}

# Tashkent smog alert (PM2.5 forecast)

Predicts tomorrow's daily mean PM2.5 and an unhealthy-day flag for Tashkent, for a school that decides the evening before whether PE lessons move indoors.

Live link: _TODO after deployment_

Data: [Open-Meteo](https://open-meteo.com/) (air quality and weather). PM2.5 here is modelled data for the grid cell around Tashkent, not a ground sensor.

## How to run (Windows, from the project folder)

    py -3.12 -m venv venv
    venv\Scripts\Activate.ps1          (PowerShell)   or   venv\Scripts\activate   (Command Prompt)
    pip install -r requirements.txt
    python data_sources.py --to-db     # download history, write data/*.csv, fill the database
    python train.py                    # cross-validate the baselines (train data only), update the log
    python train.py --save             # also save models/model_v0.joblib + .json
    python -m pytest -q                # offline tests

## Files

| File | Owner | Purpose |
|---|---|---|
| `config.py` | A | all shared constants |
| `split_config.json` | A | test dates, decided once. Do not edit |
| `data_sources.py` | A | download, hourly to daily table, `load_train()` |
| `features.py` | A | model inputs (lags, rolling means, season) and next-day targets |
| `ml/baselines.py` | A | `PersistenceRegressor` (tomorrow = today) |
| `train.py` | A | time-ordered CV of the baselines, experiment log, saves model v0 |
| `predict.py` | A | `load_model()` and `predict_tomorrow()`, called by B's jobs |
| `logs/experiment_log.csv` | A | every training run, including failures. Never delete rows |
| `tests/test_data.py`, `tests/test_ml.py` | A | data and ML tests (no internet needed) |
| `docs/CONTRACTS.md` | A + B | what we agreed so our parts fit |
| `docs/data_plan.md` | A | week 9 data plan |
| `docs/WEEK9_B_TASKS.md` | B | week 9 tasks for the service side |

Coming later: `promote.py`, `release_eval.py` (A); `db.py`, `jobs.py`, `app.py`, `monitoring.py`, `Dockerfile` (B).

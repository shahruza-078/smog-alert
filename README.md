# Tashkent smog alert (PM2.5 forecast)

Predicts tomorrow's daily mean PM2.5 and an unhealthy-day flag for Tashkent, for a school that decides the evening before whether PE lessons move indoors.

Live link: _TODO after deployment_

Data: [Open-Meteo](https://open-meteo.com/) (air quality and weather). PM2.5 here is modelled data for the grid cell around Tashkent, not a ground sensor.

## How to run (Windows, from the project folder)

    python -m venv venv
    venv\Scripts\Activate.ps1          (PowerShell)   or   venv\Scripts\activate   (Command Prompt)
    pip install -r requirements.txt
    python data_sources.py             # download history, writes data/*.csv, prints a summary
    pytest -q                          # offline tests

## Files

| File | Owner | Purpose |
|---|---|---|
| `config.py` | A | all shared constants |
| `split_config.json` | A | test dates, decided once. Do not edit |
| `data_sources.py` | A | download, hourly to daily table, `load_train()` |
| `tests/test_data.py` | A | data tests (no internet needed) |
| `docs/CONTRACTS.md` | A + B | what we agreed so our parts fit |
| `docs/data_plan.md` | A | week 9 data plan |
| `docs/WEEK9_B_TASKS.md` | B | week 9 tasks for the service side |

Coming later: `features.py`, `ml/baselines.py`, `train.py`, `promote.py`, `release_eval.py`, `predict.py` (A); `db.py`, `jobs.py`, `app.py`, `monitoring.py`, `Dockerfile` (B).

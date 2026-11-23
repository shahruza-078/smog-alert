# Tashkent smog alert (PM2.5 forecast)

Predicts tomorrow's daily mean PM2.5 and an unhealthy-day flag for Tashkent, for a school director who decides the evening before whether PE lessons move indoors.

Live link: _TODO after deployment_

## How to run

    pip install -r requirements.txt
    python fetch_data.py --start 2023-11-01     # pull hourly data, store daily rows in smog.db
    python train.py --why "first comparison"    # CV on the training part only; test set untouched
    python train.py --release                   # only when ready: scores test ONCE, saves model_v1.*
    git tag v1

## Files

| File | Purpose |
|---|---|
| `fetch_data.py` | Open-Meteo hourly PM2.5 + weather, Tashkent time, daily rows, safe to re-run |
| `train.py` | one Pipeline, time-ordered CV, experiment log, versioned model + metrics |
| `split.json` | test cut-off date, written once on the first run. Do not edit |
| `experiment_log.csv` | every run, including failures. Never delete rows |
| `model_v1.joblib`, `model_v1.json` | released model and its metrics. Never overwritten |

## Decisions to make (write them in the decision note)

- Unhealthy threshold (`--threshold`, default 15 = WHO guideline). Agree it with the client.
- Missing hours: days with fewer than 18 valid hours are excluded. The bot must post "no forecast today" instead of a guess.

## Still to build (see handbook checklist)

Evening job + Telegram channel, next-day job that fills in the real value, prediction log, input checks, monitoring/accuracy page, hosting, tests, Dockerfile, model card.

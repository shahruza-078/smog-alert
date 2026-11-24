# Data plan: Tashkent smog alert channel

Week 9 checkpoint. Numbers in [brackets] must be copied from the DATA SUMMARY that `python data_sources.py` prints.

## 1. Goal
Every evening, predict tomorrow's daily mean PM2.5 for Tashkent and whether tomorrow is an unhealthy day, so a school can decide the evening before whether PE moves indoors.

## 2. Sources
| Source | Used for | Notes |
|---|---|---|
| Open-Meteo Air Quality API | hourly PM2.5 | Modelled air-quality data for the grid cell around Tashkent, **not** a ground sensor. This goes in the limitations section. |
| Open-Meteo Historical Weather (archive) API | hourly weather, older days | A few days behind today. |
| Open-Meteo Forecast API (`past_days`) | hourly weather, last few days | Fills the gap the archive API leaves. Archive and forecast values can differ slightly, so we note it as a limitation. |

Terms: free for non-commercial use, no API key, limit 10,000 requests per day. Our usage is a handful of requests for the history and a few per day for the bot, far below the limit. We credit Open-Meteo in the README and on the bot's start message.

## 3. Variables
- Target: mean PM2.5 (µg/m³) over the **full** next day (24 h). Second target: unhealthy yes/no, threshold decided with the client in week 10 (now a placeholder in `config.py`).
- Features of day D (only hours before 20:00, because the post is in the evening): PM2.5 mean, max and last-3-hours mean; temperature, humidity, wind speed, surface pressure, precipitation. Week 10 adds lags (1–3 days), rolling means (3 and 7 days), weekday and season (sin/cos).

## 4. Period and size
- Downloaded from 2023-01-01 to 2026-11-24 (today, incomplete; last complete day 2026-11-23); one row per Tashkent calendar day.
- 1424 daily rows, of which 1423 have both flags set.
- Train part: 1356 rows. Test part: 60 days (see section 6).

## 5. Cleaning rules
1. All times are Tashkent local time (`timezone=Asia/Tashkent` in every request). No UTC days.
2. A day needs at least **16 valid hours out of the 20 before the cutoff** to be a reliable feature day (`is_reliable`), and at least **18 out of 24** to be a usable label (`target_ok`). Otherwise the values become NaN and the bot posts "no forecast today".
3. Hours later than "now" are dropped. The API returns forecast hours for today, and a forecast is never used as an observation.
4. Days with no data at all still get a row, so "yesterday" is always exactly one row above.
5. Raw hourly data is kept in `data/hourly_raw.csv` (not committed); the sample of 50 complete days is in `data/sample_50.csv`.
6. Re-running the download is safe: rows are keyed by date.

## 6. Split
- By **date**, never at random.
- Test = the last 60 days at the time of decision: **2026-09-25 to 2026-11-23**. Chosen on 2026-11-24 before any modelling, stored in `split_config.json`.
- A **7-day gap** between train and test (features look back up to 7 days).
- Cross-validation on train only, with `TimeSeriesSplit`.
- `load_train()` can only return train rows. Only `release_eval.py` opens the test part, once.
- Live days after 2026-11-23 are new data; they are added only when training v2.
- Known weakness: the test part is autumn and early winter only, and 60 days contain few unhealthy days at a high threshold, so recall on test will be noisy. We report it honestly.

## 7. Ethics and logging
- Only public weather and air-quality data. No personal data in the training set.
- The service logs: forecast time, target date, model inputs, prediction, confidence, model version, and later the real value. Nothing about who reads the channel.
- Optional `/threshold` command (week 14): stores only a Telegram user ID and a number, only after telling the user, and only if the instructor approves.
- The bot's start message and the README say what is logged and why.

## 8. Open questions for the instructor
See section 8 of the project plan (questions 1–4 are the priority ones).

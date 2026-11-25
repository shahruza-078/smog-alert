# CONTRACTS (A and B). Change only together, in a pull request both approve.

A = Data & ML: ____________    B = Service & Deployment: ____________

## 1. Constants (`config.py`, owned by A)
| Name | Value | Meaning |
|---|---|---|
| TIMEZONE | Asia/Tashkent | All stored times |
| LATITUDE / LONGITUDE | 41.3111 / 69.2797 | Tashkent |
| CUTOFF_HOUR | 20 | Features of day D use hours 0..19 only. The evening job runs shortly after 20:00. |
| MIN_HOURS_WINDOW | 16 | Valid PM2.5 hours (of 20) needed before the cutoff, otherwise `no_forecast` |
| MIN_HOURS_FULL | 18 | Valid PM2.5 hours (of 24) needed for a day to be a label |
| UNHEALTHY_THRESHOLD | placeholder 15 | Decided with the client in week 10 |
| test dates | `split_config.json` | Read by `config.py`, never typed twice |

## 2. The daily table (`daily_data`) — what `hourly_to_daily` returns, what B stores
One row per Tashkent date. `date` is stored as text `YYYY-MM-DD` (primary key). Flags are 0/1.

| Column | Meaning |
|---|---|
| date | the day |
| pm25_today, pm25_max_today, pm25_last3h | PM2.5 mean / max / last-3-hours mean, hours before cutoff (NaN if not reliable) |
| temp_today, humidity_today, wind_today, pressure_today, precip_today | weather before the cutoff (precip is a sum) |
| pm25_full | mean over the full 24 h, NaN unless `target_ok`. Source of the label. |
| hours_in_window, hours_full | number of valid PM2.5 hours |
| is_reliable | 1 = enough hours before the cutoff. **The bot checks this** to decide `no_forecast`. |
| target_ok | 1 = enough hours in the full day. **Training checks this.** |

**Proposed change to the original plan:** two flags instead of one. At 20:00 the current day is incomplete, so "full-day ok" would always be 0 for today, while the bot only needs "enough hours before the cutoff". Agree this on Day 0.

## 3. Interfaces
**A:**
- `data_sources.fetch_hourly(start_date, end_date)` → hourly DataFrame (`time`, `pm2_5`, `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `surface_pressure`, `precipitation`), Tashkent time, no future hours.
- `data_sources.hourly_to_daily(hourly_df)` → the table in section 2.
- `data_sources.load_train(daily_df, include_new_days=False)` → only train rows.
- week 10: `features.FEATURE_COLUMNS`, `features.build_features(daily_df)`, `features.make_targets(daily_df)` → `(y_reg, y_clf)` for D+1.
- week 10+: `predict.load_model(version=None)`, `predict.predict_tomorrow(daily_df, bundle)` → dict with `target_date, pred_pm25, prob_unhealthy, is_unhealthy, confidence, model_version, input_json, status (ok | no_forecast), reason`.

**B:**
- `db.init_db()`, `db.upsert_daily(df)`, `db.get_daily(start, end)`.
- `db.log_prediction(dict)` unique on (`target_date`, `model_version`) so a double run never duplicates. `db.fill_label(target_date, actual)`, `db.get_predictions(...)`.
- `jobs.evening_job()`, `jobs.nextday_job()`.
- `predictions` table, minimum columns from the handbook: `time, target_date, input, prediction, confidence, model_version, label, label_time`.

## 4. Model files (week 12+)
`models/model_vN.joblib` is one scikit-learn Pipeline that `predict.py` can load. `models/model_vN.json` holds: version, created_at (Tashkent), git_commit, train_date_range, n_rows, features, cv_scores, test_scores (release only), unhealthy_threshold, decision_probability_threshold, feature_ranges (min and max per feature on training data; B uses them to catch inputs unlike the training data).

## 5. Test-set discipline
- `TEST_START_DATE` chosen by A on day one: 2026-09-25 (60 days), with a 7-day gap.
- Only `release_eval.py` opens the test part, once.
- Cross-validation is `TimeSeriesSplit`, never random KFold.

## 6. Git rules
Branches and pull requests only, never push to `main`. The partner reviews every PR. Each person commits under their own name, often and small. Secrets only in `.env` (listed in `.gitignore`). The `v1` tag is set only on release day, together.

## 7. Rhythm
Two short calls a week (15–20 min), one integration session at the end of the week (45–60 min): fresh clone, run from scratch, tick the checklist. Keep `docs/STATUS.md` current: who does what, what blocks.

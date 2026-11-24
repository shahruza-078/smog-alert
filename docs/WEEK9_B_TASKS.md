# Week 9 tasks for B (Service & Deployment)

Deliverables due this week (the group submits together): data plan (A writes it), first 50 rows (A produces them), **hosting plan, half a page (B)**.

A has pushed: `config.py`, `data_sources.py`, `split_config.json`, `tests/`, `docs/CONTRACTS.md` (draft), `docs/data_plan.md` (draft). Please read `docs/CONTRACTS.md` first and tell A what you disagree with.

## 1. Hosting comparison (about 1–2 hours)
Make a table of 3–4 hosting options. For each one answer:
- Does the service go to sleep when nobody visits it?
- Is the disk wiped on every deploy? (our SQLite database must survive)
- How many hours and how much memory do we get in weeks 10–15?
- HTTPS included?
- Does it restart by itself after a crash or reboot?
- Cost.

Choose one and write the half-page hosting plan: where it runs, how we deploy, how the database survives a restart, where the secrets live.

## 2. Telegram bot and channel (30 minutes)
1. In Telegram, message @BotFather, send `/newbot`, follow the steps. You get a token.
2. Put the token in a file named `.env` (it is already in `.gitignore`). **Never commit it.** Add a `.env.example` with a fake value.
3. Create a public channel, add the bot as an **administrator** so it can post.

## 3. Course starter repository
Clone the instructor's starter repo, run it locally, and write down where the app, database and training script live. Ask the instructor how the starter and our repo should be combined.

## 4. `db.py` with the schema
Tables `daily_data` and `predictions`, columns exactly as in `docs/CONTRACTS.md` sections 2 and 3. Functions: `init_db`, `upsert_daily`, `get_daily`, `log_prediction`, `fill_label`, `get_predictions`. Use a unique key so running a job twice never creates a duplicate.

## 5. Integration check with A (end of the week)
A gives you `data/sample_50.csv`. Load it with `db.upsert_daily` **without editing the file or the columns**. Run it twice: you must still have 50 rows. If a column does not match, we fix the contract together, not in secret.

## 6. Together
- Decide who the client is for the threshold (the instructor or a school teacher).
- Send the checkpoint.
- Do a 15-minute cross-lesson: A explains why the split is by date, B explains how a job avoids duplicate posts. (Both must be able to explain every file at the viva.)

## Git
Branch for every piece of work (`git checkout -b week9-db`), pull request, A reviews. Never push to `main`.

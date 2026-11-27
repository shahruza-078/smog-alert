# Deployment note [B]

**Where it runs:** Fly.io, one small machine (shared-cpu-1x, 512 MB), region fra, HTTPS included, restarts by itself after a crash (health check on `/health`).
**What runs:** `python app.py` = public page + scheduler (evening forecast after 20:15 Tashkent, next-day labels after 07:05).
**Database:** SQLite file `/data/smog.db` on a Fly volume. The volume survives restarts and deploys.
**Secrets:** `fly secrets set TELEGRAM_BOT_TOKEN=... TELEGRAM_CHANNEL_ID=...`. Never in git.

## Deploy
    fly launch --no-deploy --copy-config --name <app-name>
    fly volumes create smogdata --region fra --size 1
    fly secrets set TELEGRAM_BOT_TOKEN=... TELEGRAM_CHANNEL_ID=@...
    fly secrets set DRY_RUN=1          # rehearsal: logs forecasts, posts nothing
    fly deploy --ha=false
    fly secrets unset DRY_RUN          # go live

## Rollback
    fly releases
    fly deploy --image <image-of-previous-release> --ha=false
The model version can be pinned: `fly secrets set MODEL_VERSION=v1`. v1 is tagged and never changed.

## Restart test (do before the demo)
    fly machine restart <id>   # the page and /health must come back, no duplicate post, database rows unchanged

## What we log (told to users)
Only air-quality numbers, weather numbers, our forecasts and the real values. Nothing about subscribers. No access log is kept.

## After the course
    fly apps destroy <app-name>     # stops all charges; also deletes the volume

## Live app
https://smog-tashkent-b7k2.fly.dev/  (public page, `/health` for the platform check)

## Backup of the database (do weekly and before every deploy that changes the schema)
    fly ssh sftp get /data/smog.db smog-backup.db
Keep the copy outside git (it is in `.gitignore` and `.dockerignore` as `data/smog.db*`; name the copy `smog-backup-YYYY-MM-DD.db`). Fly also keeps automatic volume snapshots (5 days).
Restore: `fly ssh sftp shell`, then `put smog-backup.db /data/smog.db`, then `fly machine restart <id>`.

## Notice for channel subscribers (pin this message in the channel)
> This channel posts a daily PM2.5 air-quality forecast for Tashkent. We store only air-quality and weather numbers, our forecasts and the real values. We do not collect any information about subscribers (Telegram does not tell us who reads the channel). Questions or feedback: write to <contact>. Forecasts are not medical advice.

## Restart test (record the result in the report)
1. Open the public page, note the "Last forecasts" table.
2. `fly machine list`, then `fly machine restart <id>`.
3. After about 30 seconds: `/health` says `ok`, the table is unchanged, no second post in the channel.

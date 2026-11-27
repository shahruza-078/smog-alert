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

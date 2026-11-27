# Hosting plan [B]

## Comparison

| Question | **Fly.io (chosen)** | GitHub Actions + Pages | University computer | Free web hosts with sleep (e.g. Render free) |
|---|---|---|---|---|
| Sleeps when nobody visits? | No: `auto_stop_machines = off`, 1 machine always running | No server at all, jobs run on a timer | No, if the PC stays on | Yes, after ~15 min idle |
| Disk wiped on deploy? | No: the SQLite file lives on a Fly volume (`/data`) | Yes: no persistent disk, the DB would have to be committed to git | No, but depends on the PC | Yes, unless a paid disk is added |
| Hours / memory in weeks 10-15 | 24/7, shared-cpu-1x, 512 MB | Timed runs only, delays of several minutes | 24/7 only if nobody turns it off | Limited hours, 512 MB |
| HTTPS included? | Yes, automatic | Yes (Pages) | No, needs extra setup | Yes |
| Restarts itself after a crash / reboot? | Yes, plus health check on `/health` | Each run is a fresh start | No | Yes |
| Cost | Free plan covers $5 usage/month; expected ~$2-3/month for the machine and 1 GB volume | Free (repo must be public on the Free plan) | Free | Free, but sleep breaks the evening post |

## Decision
**Fly.io.** It is the only option that keeps a database safely, runs on time every evening, and restarts by itself without extra tools. The sleeping free hosts would miss the evening post, GitHub Actions has no persistent disk, and the university PC has no uptime guarantee.

## How it works
- **Where it runs:** one Fly.io machine in Frankfurt (`fra`), app `smog-tashkent-b7k2`, public page https://smog-tashkent-b7k2.fly.dev/ .
- **How we deploy:** `fly deploy --ha=false` from the repository (Dockerfile build on Fly's builder). Rollback: redeploy the previous release image.
- **How the database survives:** SQLite file `/data/smog.db` on a Fly volume (`smogdata`, 1 GB). Restarts and deploys do not touch the volume. Manual backup with `fly ssh sftp get`, plus Fly's automatic volume snapshots.
- **Where the secrets live:** Telegram token and channel id are Fly secrets (`fly secrets set`), never in git. Locally they are in `.env`, which is in `.gitignore`.
- **Rehearsal mode:** `DRY_RUN=1` logs forecasts without posting; removed only when the model and token are confirmed.
- **After the course:** `fly apps destroy` stops all charges.

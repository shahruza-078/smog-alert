"""[B] The service that runs on the server: one process, two jobs.

  * a tiny web server: "/" public page (latest forecast, accuracy, what we log), "/health" for Fly
  * a scheduler thread: evening forecast after 20:15 Tashkent time (retry every 30 min until 23:30),
    next-day labels after 07:05.

Everything is safe to restart at any moment: the database (on the persistent volume) remembers
what was already posted, so a restart never posts twice and never loses a forecast.
"""
import html
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd

import config
import db
import jobs

log = logging.getLogger("app")
STATE = {"last_evening": None, "last_nextday": None}      # what the scheduler did last (for the page)
EVENING_FROM, EVENING_UNTIL = (20, 15), (23, 30)
NEXTDAY_FROM = (7, 5)
RETRY_SECONDS = 30 * 60


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=config.TIMEZONE).tz_localize(None)


# ----------------------------------------------------------------------------- accuracy
def accuracy() -> dict:
    """Honest numbers computed only from forecasts whose real value is already known."""
    df = db.get_predictions(status="ok", labeled=True)
    if df.empty:
        return {"n": 0}
    err = (df["prediction"] - df["label"]).abs()
    out = {"n": int(len(df)), "mae": float(err.mean())}
    threshold = config.UNHEALTHY_THRESHOLD
    actual_bad = df["label"] >= threshold
    out["bad_days"] = int(actual_bad.sum())
    out["caught"] = int((actual_bad & (df["is_unhealthy"] == 1)).sum())
    out["false_alarms"] = int((~actual_bad & (df["is_unhealthy"] == 1)).sum())
    daily = db.get_daily()
    if "pm25_today" in daily.columns:                     # smart baseline: "tomorrow = today"
        today_val = dict(zip(daily["date"], daily["pm25_today"]))
        base = [(abs(today_val.get(pd.Timestamp(d) - pd.Timedelta(days=1)) - y))
                for d, y in zip(df["target_date"], df["label"])
                if pd.notna(today_val.get(pd.Timestamp(d) - pd.Timedelta(days=1)))]
        if base:
            out["baseline_mae"] = float(sum(base) / len(base))
    return out


# ----------------------------------------------------------------------------- page
def render_page() -> str:
    e = html.escape
    preds = db.get_predictions()
    rows = preds.sort_values("target_date", ascending=False).head(14)
    latest = rows[rows["status"] == "ok"].head(1)
    parts = ["<h1>Tashkent PM2.5 forecast</h1>"]
    if len(latest):
        r = latest.iloc[0]
        flag = "UNHEALTHY expected" if r["is_unhealthy"] == 1 else "below the warning level"
        parts.append(f"<p><b>{e(str(r['target_date']))}</b>: about {r['prediction']:.0f} &micro;g/m&sup3; "
                     f"({e(flag)}, warning level {config.UNHEALTHY_THRESHOLD:.0f}). "
                     f"Model {e(str(r['model_version']))}.</p>")
    else:
        parts.append("<p>No forecast yet.</p>")

    acc = accuracy()
    parts.append("<h2>How accurate are we? (public)</h2>")
    if acc["n"] == 0:
        parts.append("<p>Not enough finished days yet.</p>")
    else:
        line = f"<p>{acc['n']} forecasts checked. Average error: {acc['mae']:.1f} &micro;g/m&sup3;."
        if "baseline_mae" in acc:
            line += f" Simple rule &ldquo;tomorrow = today&rdquo;: {acc['baseline_mae']:.1f}."
        line += (f" Unhealthy days: {acc['bad_days']}, caught {acc['caught']}, "
                 f"false alarms {acc['false_alarms']}.</p>")
        parts.append(line)

    parts.append("<h2>Last forecasts</h2><table border=1 cellpadding=4><tr><th>day</th><th>forecast</th>"
                 "<th>real</th><th>status</th><th>model</th></tr>")
    for _, r in rows.iterrows():
        pred = "-" if pd.isna(r["prediction"]) else f"{r['prediction']:.0f}"
        real = "-" if pd.isna(r["label"]) else f"{r['label']:.0f}"
        parts.append(f"<tr><td>{e(str(r['target_date']))}</td><td>{pred}</td><td>{real}</td>"
                     f"<td>{e(str(r['status']))}</td><td>{e(str(r['model_version']))}</td></tr>")
    parts.append("</table>")

    parts.append("<h2>Service status</h2>")
    posted = preds[preds["posted_at"].notna()]
    parts.append(f"<p>Last post to the channel: {e(str(posted['posted_at'].max())) if len(posted) else 'none yet'}.<br>"
                 f"No-forecast days: {int((preds['status'] == 'no_forecast').sum())}.<br>"
                 f"Last evening job: {e(str(STATE['last_evening']))}.<br>"
                 f"Last next-day job: {e(str(STATE['last_nextday']))}.</p>")

    parts.append("<h2>What we store</h2><p>This service stores only air-quality and weather numbers, "
                 "our forecasts and the real values. We do not collect or store anything about "
                 "channel subscribers. Telegram does not tell us who reads the channel.</p>")
    return ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            "<title>Tashkent PM2.5 forecast</title><body style='font-family:sans-serif;max-width:40rem;"
            "margin:1rem auto;padding:0 1rem'>" + "".join(parts))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            if self.path.startswith("/health"):
                db.init_db()
                body, ctype = b"ok", "text/plain"
            elif self.path in ("/", "/index.html"):
                body, ctype = render_page().encode("utf-8"), "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
        except Exception as exc:
            log.error("page failed: %s", type(exc).__name__)
            self.send_error(500)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):          # no access log: nothing about visitors is stored
        pass


# ----------------------------------------------------------------------------- scheduler
def evening_due(now: pd.Timestamp) -> bool:
    """True if it is evening-post time and tomorrow's forecast has not been posted yet."""
    if not ((now.hour, now.minute) >= EVENING_FROM and (now.hour, now.minute) <= EVENING_UNTIL):
        return False
    target = now.normalize() + pd.Timedelta(days=1)
    done = db.get_predictions(target_date=target, status="ok")
    return not (len(done) and done["posted_at"].notna().any())


def nextday_due(now: pd.Timestamp) -> bool:
    if (now.hour, now.minute) < NEXTDAY_FROM or now.hour >= 12:
        return False
    return len(db.get_predictions(labeled=False, status="ok", end=now.normalize() - pd.Timedelta(days=1))) > 0


def scheduler_loop(stop: threading.Event, tick=60, clock=_now):
    last = {"evening": 0.0, "nextday": 0.0}
    while not stop.is_set():
        now = clock()
        try:
            for name, due, job in (("evening", evening_due, jobs.evening_job),
                                   ("nextday", nextday_due, jobs.nextday_job)):
                if due(now) and time.time() - last[name] >= RETRY_SECONDS:
                    last[name] = time.time()
                    result = job()
                    STATE["last_" + name] = f"{now:%Y-%m-%d %H:%M} -> {result}"
                    log.info("%s job: %s", name, result)
        except Exception as exc:               # the loop must never die
            log.error("scheduler error: %s", type(exc).__name__)
        stop.wait(tick)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    db.init_db()
    stop = threading.Event()
    threading.Thread(target=scheduler_loop, args=(stop,), daemon=True).start()
    port = int(os.getenv("PORT", "8080"))
    log.info("listening on %s", port)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()

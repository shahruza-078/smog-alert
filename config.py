"""config.py - every shared constant lives here ([A] owns it; change only together with B).

The test-set dates are NOT typed here. They live in split_config.json (decided once, on
day one) and are read from there, so there is a single source of truth.
"""
import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- place and clock -------------------------------------------------------
TIMEZONE = "Asia/Tashkent"      # everything is stored in this time zone, nothing else
LATITUDE = 41.3111              # Tashkent (check against the API docs)
LONGITUDE = 69.2797

# ---- the evening-post rule -------------------------------------------------
# The bot posts in the evening, so on day D it only knows the hours BEFORE CUTOFF_HOUR.
# Features of day D use hours 0..CUTOFF_HOUR-1. The label is the mean of the FULL next day.
# Training and the bot call the same function with the same value (no training-serving skew).
CUTOFF_HOUR = 20
MIN_HOURS_WINDOW = 16           # need >= 16 valid PM2.5 hours out of the 20 before the cutoff
MIN_HOURS_FULL = 18             # need >= 18 valid hours out of 24 for a day to be a label
HISTORY_START = "2023-01-01"    # first day we try to download (the API may allow a later one)

# ---- the "unhealthy day" threshold (ug/m3) ---------------------------------
# DRAFT: 25 (proposal). On the training data 51% of days are >= 15, 13% >= 25, 6% >= 35, 1% >= 55.
# The WHO value 15 flags half of all days, so it would be useless as a warning.
# Agree it together (A + B) and write the reason in docs/decision_note.md.
UNHEALTHY_THRESHOLD = 25.0
DATABASE_PATH = os.environ.get("DATABASE_PATH", str(ROOT / "data" / "smog.db"))
# ---- test split (read from split_config.json) ------------------------------
def _load_split():
    with open(ROOT / "split_config.json", encoding="utf-8") as f:
        return json.load(f)


_split = _load_split()
TEST_START_DATE = _split["test_start_date"]   # first day of the locked-away test part
TEST_END_DATE = _split["test_end_date"]       # last day of the locked-away test part
GAP_DAYS = _split["gap_days"]                 # buffer between train and test

assert 0 < MIN_HOURS_WINDOW <= CUTOFF_HOUR <= 24
assert 0 < MIN_HOURS_FULL <= 24

"""[B] Input check: refuse to forecast from impossible numbers.

If the API sends garbage (humidity 300 %, PM2.5 of -5), a forecast would still be produced and
posted as if it were fine. Better to say "no forecast" honestly. The limits are physical, wide
on purpose: they catch broken data, not unusual weather.
"""
import pandas as pd

LIMITS = {                       # column: (min, max)
    "pm25_today": (0, 1000), "pm25_max_today": (0, 1500), "pm25_last3h": (0, 1500),
    "temp_today": (-45, 60), "humidity_today": (0, 100), "wind_today": (0, 80),
    "pressure_today": (800, 1100), "precip_today": (0, 400),
}


def check_row(row, limits=None) -> list:
    """Return a list of problems (empty list = the row is fine). `row` is a dict or Series."""
    limits = limits or LIMITS
    problems = []
    for col, (lo, hi) in limits.items():
        if col not in row:
            continue
        value = row[col]
        if value is None or pd.isna(value):
            continue                                  # missing values are handled by the model's own rules
        if not (lo <= float(value) <= hi):
            problems.append(f"{col}={value} is outside {lo}..{hi}")
    return problems

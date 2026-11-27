import pandas as pd

import input_checks


def test_normal_row_passes():
    row = {"pm25_today": 35.0, "temp_today": 12.0, "humidity_today": 60, "wind_today": 3.2,
           "pressure_today": 960, "precip_today": 0.0}
    assert input_checks.check_row(row) == []


def test_impossible_values_are_reported():
    problems = input_checks.check_row({"humidity_today": 300, "pm25_today": -5})
    assert len(problems) == 2


def test_missing_values_are_not_an_error():
    assert input_checks.check_row(pd.Series({"pm25_today": float("nan"), "temp_today": None})) == []

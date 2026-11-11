"""Returns, delistings and capitalisation on made-up cases whose answers can be worked out by hand."""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from slt import constants as C  # noqa: E402
from slt import returns as R  # noqa: E402


def ulong():
    months = pd.PeriodIndex(["2000-02"] * 5, freq="M")
    return pd.DataFrame({"month": months, "permno": [1, 2, 3, 4, 5], "date_prev": pd.Timestamp("2000-01-31")})


def monthly():
    return pd.DataFrame({
        "permno": [1, 1, 2, 2, 3, 3, 4, 5],
        "date": pd.to_datetime(["2000-01-31", "2000-02-29", "2000-01-31", "2000-02-29", "2000-01-31", "2000-02-29", "2000-01-31", "2000-01-31"]),
        "prc": [-10.0, 11.0, 20.0, 19.0, 5.0, np.nan, 0.0, 8.0],
        "ret": [np.nan, 0.10, np.nan, -0.05, np.nan, np.nan, np.nan, np.nan],
        "retx": [np.nan, 0.08, np.nan, -0.05, np.nan, np.nan, np.nan, np.nan],
        "shrout": [100.0, 100.0, 50.0, 50.0, 10.0, 10.0, 10.0, 10.0]})


def delistings():
    return pd.DataFrame({"permno": [2, 3, 4, 5, 1], "dlstcd": [233, 552, 331, 100, np.nan],
                         "dlstdt": pd.to_datetime(["2000-02-15", "2000-02-10", "2000-02-20", "2024-12-31", "2000-02-01"]),
                         "dlret": [0.02, np.nan, np.nan, np.nan, np.nan], "dlretx": [0.01, np.nan, np.nan, np.nan, np.nan]})


def test_panel_applies_delistings_by_rule_d4_and_reads_the_cap_from_the_month_before():
    p = R.monthly_panel(ulong(), monthly(), delistings()).set_index("permno")
    # PERMNO 1: no delisting (its row has no code), cap |-10| x 100 = 1000 from January
    assert p.loc[1, "delisting"] == "none" and p.loc[1, "cap"] == 1000.0 and p.loc[1, "ret_total"] == pytest.approx(0.10)
    # PERMNO 2: a given delisting return compounded with the month's return
    assert p.loc[2, "delisting"] == "given" and p.loc[2, "ret_total"] == pytest.approx(0.95 * 1.02 - 1)
    assert p.loc[2, "retx_total"] == pytest.approx(0.95 * 1.01 - 1)
    # PERMNO 3: performance-related code 552, delisting return missing, no monthly return: minus 30 per cent
    assert p.loc[3, "delisting"] == "shumway" and p.loc[3, "ret_total"] == pytest.approx(C.DELIST_PERFORMANCE_MISSING_RETURN)
    # PERMNO 4: code 331, delisting return missing: zero; its January price is 0, so its cap is missing
    assert p.loc[4, "delisting"] == "zero" and p.loc[4, "ret_total"] == 0.0 and np.isnan(p.loc[4, "cap"])
    # PERMNO 5: code 100 is an active security and not a delisting; no return in February, so the month is counted
    assert p.loc[5, "delisting"] == "none" and bool(p.loc[5, "return_missing"]) and p.loc[5, "cap"] == 80.0


def test_delisting_counts_by_year():
    by_year = R.delisting_by_year(R.monthly_panel(ulong(), monthly(), delistings())).iloc[0]
    assert by_year["delisting_months"] == 3 and by_year["delisting_return_given"] == 1
    assert by_year["delisting_return_minus_30"] == 1 and by_year["delisting_return_zero"] == 1
    assert by_year["months_without_return"] == 1 and by_year["months_without_cap"] == 1


def test_g4_counts_names_by_their_share_of_months_with_a_cap():
    p = R.monthly_panel(ulong(), monthly(), delistings())
    g4 = R.cap_coverage(p, "2000-02")
    assert g4["names"] == 5 and g4["names_meeting_month_share"] == 4 and g4["share_of_names"] == pytest.approx(0.8)
    assert g4["g4_met"] is False


def test_removals_and_the_trading_day_window_of_g3():
    spells = pd.DataFrame({"permno": [1, 2, 3, 4], "start": pd.Timestamp("1980-01-01"),
                           "ending": pd.to_datetime(["2000-03-15", C.CRSP_END, "1985-06-28", "2000-03-15"])})
    removed = R.removals(spells)
    assert set(removed["permno"]) == {1, 3, 4}
    calendar = pd.bdate_range("2000-02-01", "2000-04-28").values
    w = R.removal_windows(removed, calendar, days=2).set_index("permno")
    assert 3 not in w.index
    assert w.loc[1, "window_start"] == pd.Timestamp("2000-03-13") and w.loc[1, "window_end"] == pd.Timestamp("2000-03-17")
    daily = pd.DataFrame({"permno": [1, 4, 4], "date": pd.to_datetime(["2000-03-17", "2000-03-10", "2000-03-20"]),
                          "prc": [5.0, 7.0, np.nan]})
    found = R.priced_near(w.reset_index(), daily)
    assert found == {(1, pd.Timestamp("2000-03-15"))}
    g3 = R.g3_verdict(w.reset_index(), found)
    assert g3["removals"] == 2 and g3["removals_with_price"] == 1 and g3["g3_met"] is False


def test_removals_before_the_daily_file_on_the_monthly_file():
    removed = pd.DataFrame({"permno": [3, 4], "ending": pd.to_datetime(["1985-06-12", "1985-06-12"])})
    monthly_file = pd.DataFrame({"permno": [3, 4], "date": pd.to_datetime(["1985-05-31", "1985-03-29"]), "prc": [4.0, 6.0]})
    out = R.removals_before_daily_file(removed, monthly_file)
    assert out["removals"] == 2 and out["removals_with_price"] == 1


def test_daily_coverage():
    long = pd.DataFrame({"month": pd.PeriodIndex(["2000-03", "2000-03"], freq="M"), "permno": [1, 2]})
    calendar = pd.bdate_range("2000-03-01", "2000-03-31")
    days = R.trading_days_by_month(calendar.values)
    assert days.iloc[0] == len(calendar)
    daily = pd.DataFrame({"permno": [1, 1, 2, 9], "date": calendar[:4], "ret": [0.01, np.nan, 0.02, 0.03]})
    present = {2000: R.member_days_with_return(daily, long)}
    out = R.daily_coverage_by_year(present, days, long).iloc[0]
    assert out["member_days_expected"] == 2 * len(calendar) and out["member_days_with_return"] == 2


def test_delisting_code_families():
    f = R.delisting_code_families(R.monthly_panel(ulong(), monthly(), delistings()))
    assert f.to_dict() == {200: 1, 300: 1, 500: 1}

"""The universe on made-up spells whose answers can be worked out by hand."""
import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from slt import universe as U  # noqa: E402

ENDS = pd.DataFrame({"caldt": pd.to_datetime(["2000-01-31", "2000-02-29", "2000-03-31", "2000-04-28", "2000-05-31"]),
                     "totcnt": [2, 2, 1, 2, 2]})


def spells():
    return pd.DataFrame({"permno": [1, 2, 2, 3],
                         "start": pd.to_datetime(["1999-12-01", "1999-12-01", "2000-04-28", "2000-02-15"]),
                         "ending": pd.to_datetime(["2000-05-31", "2000-02-29", "2000-05-31", "2000-03-30"])})


def test_month_ends_are_monthly_and_refuse_a_gap():
    ends = U.month_ends(ENDS)
    assert [str(m) for m in ends.index] == ["2000-01", "2000-02", "2000-03", "2000-04", "2000-05"]
    with pytest.raises(ValueError):
        U.month_ends(ENDS.drop(index=2))


def test_membership_uses_the_last_trading_day_of_the_month_before_both_ends_included():
    m = U.membership_matrix(spells(), U.month_ends(ENDS))
    assert [str(x) for x in m.index] == ["2000-02", "2000-03", "2000-04", "2000-05"]
    # PERMNO 1 covers every month-end. PERMNO 2: in on 31 Jan and 29 Feb (its last day), out on 31 Mar, back from 28 Apr.
    assert m[1].tolist() == [True, True, True, True]
    assert m[2].tolist() == [True, True, False, True]
    # PERMNO 3 starts on 15 Feb, so 29 Feb is the first month-end it covers, and it ends on 30 Mar, before 31 Mar.
    assert m[3].tolist() == [False, True, False, False]


def test_universe_long_carries_the_date_of_the_month_before():
    ends = U.month_ends(ENDS)
    long = U.universe_long(U.membership_matrix(spells(), ends), ends)
    row = long[(long["permno"] == 3)].iloc[0]
    assert str(row["month"]) == "2000-03" and row["date_prev"] == pd.Timestamp("2000-02-29")
    assert len(long) == 4 + 3 + 1


def test_counts_align_month_t_with_crsp_row_t_minus_1_and_find_the_usable_window():
    ends = U.month_ends(ENDS)
    m = U.membership_matrix(spells(), ends)
    counts = U.counts_against_crsp(m, ENDS)
    # universe counts: Feb 2, Mar 3, Apr 1, May 2; CRSP's rows for Jan, Feb, Mar, Apr: 2, 2, 1, 2
    assert counts["universe_count"].tolist() == [2, 3, 1, 2]
    assert counts["crsp_count"].tolist() == [2, 2, 1, 2]
    assert counts["difference"].tolist() == [0, 1, 0, 0]
    w = U.usable_window(counts, within=0)
    assert w["first_month"] == "2000-02" and w["months"] == 4 and w["g2_met"] is False
    by_year = U.count_check_by_year(counts, within=0)
    assert by_year.loc[0, "months_outside_tolerance"] == 1 and by_year.loc[0, "mean_count_difference"] == 0.25
    assert "months_difference_nonzero" not in by_year.columns


def test_compustat_list_and_agreement():
    ends = U.month_ends(ENDS)
    history = pd.DataFrame({"gvkey": ["A", "B", "Z"], "gvkeyx": ["000003", "000003", "000999"],
                            "from": pd.to_datetime(["2000-01-01", "2000-03-15", "2000-01-01"]), "thru": pd.NaT})
    comp = U.compustat_members_long(history, ends)
    assert set(comp.loc[comp["gvkey"] == "A", "month"].astype(str)) == {"2000-02", "2000-03", "2000-04", "2000-05"}
    assert set(comp.loc[comp["gvkey"] == "B", "month"].astype(str)) == {"2000-04", "2000-05"}
    assert "Z" not in set(comp["gvkey"])
    universe = pd.DataFrame({"month": pd.PeriodIndex(["2000-02", "2000-04", "2000-04"], freq="M"), "gvkey": ["A", "A", "C"]})
    a = U.list_agreement_by_year(universe, comp).iloc[0]
    # Compustat months: A x4, B x2 = 6; in the universe: A in Feb and Apr = 2; universe-only: C in Apr = 1
    assert a["compustat_months"] == 6 and a["compustat_months_in_universe"] == 2
    assert a["universe_months_not_in_compustat"] == 1 and a["compustat_months_not_in_universe"] == 4
    assert a["two_sided_share_counting_exits_compustat_omits"] == pytest.approx(5 / 7, abs=1e-6)
    assert a["union_months"] == 7


def test_share_codes():
    ends = U.month_ends(ENDS)
    long = U.universe_long(U.membership_matrix(spells(), ends), ends)
    names = pd.DataFrame({"permno": [1, 2, 2], "shrcd": [11, 31, 10],
                          "namedt": pd.to_datetime(["1990-01-01", "1990-01-01", "2000-03-01"]),
                          "nameendt": pd.to_datetime(["2024-12-31", "2000-02-29", "2024-12-31"])})
    coded, overlaps = U.share_codes(long, names)
    assert overlaps == 0
    by_year = U.share_codes_by_year(coded).iloc[0]
    # PERMNO 2 is code 31 in Feb and Mar (month-ends 31 Jan and 29 Feb) and 10 in May; PERMNO 3 has no record
    assert by_year["months_outside_common_codes"] == 2 and by_year["months_without_share_code"] == 1
    assert by_year["names_outside_common_codes"] == 1
    totals = dict(zip(U.share_code_totals(coded)["share_code"], U.share_code_totals(coded)["universe_months"]))
    assert totals[11] == 4 and totals[31] == 2 and totals["none"] == 1


def test_spell_checks():
    s = pd.DataFrame({"permno": [1, 1, 2, 3],
                      "start": pd.to_datetime(["1990-01-01", "1995-01-01", "2000-01-01", "2010-01-01"]),
                      "ending": pd.to_datetime(["1996-01-01", "2024-12-31", "1999-01-01", "2024-12-31"])})
    c = U.spell_checks(s)
    assert c["spells"] == 4 and c["permnos"] == 3 and c["permnos_with_more_than_one_spell"] == 1
    assert c["overlapping_spells"] == 1 and c["spells_ending_before_start"] == 1 and c["spells_ending_on_last_date"] == 2


def test_compustat_rows_summary_and_join_dates():
    rows = pd.DataFrame({"gvkey": ["A", "B", "C", "D", "A"],
                         "from": pd.to_datetime(["2000-03-15", "2000-05-10", "2010-01-04", "2025-06-01", "2014-04-03"])})
    links = pd.DataFrame({"gvkey": ["A", "B"], "permno": [1, 2], "linkdt": pd.to_datetime(["1990-01-01", "1990-01-01"]),
                          "linkenddt": pd.Timestamp("2099-12-31")})
    s = U.compustat_rows_summary(rows, links)
    assert s == {"rows": 5, "companies": 4, "rows_joined_after_last_date": 1, "rows_without_a_primary_link": 1}
    spells = pd.DataFrame({"permno": [1, 2, 2], "start": pd.to_datetime(["2000-03-15", "1995-01-03", "2000-07-01"]),
                           "ending": pd.to_datetime(["2024-12-31", "1998-01-01", "2024-12-31"])})
    j = U.join_dates(rows, links, spells).set_index("gvkey")
    # A: same day on both lists. B: CRSP's interval covering Compustat's months starts two months later.
    assert j.loc["A", "months_apart"].tolist() == [0.0, -169.0]   # the second row: March 2000 against April 2014
    assert j.loc["B", "months_apart"] == 2.0 and pd.isna(j.loc["C", "months_apart"])
    by = U.join_dates_by_year(j.reset_index())
    y2000 = by[by["year"] == 2000].iloc[0]
    assert y2000["members_joined"] == 2 and y2000["same_month"] == 1 and y2000["further_apart_crsp_later"] == 1
    assert by[by["year"] == 2010].iloc[0]["no_crsp_interval"] == 1
    assert by[by["year"] == 2014].iloc[0]["further_apart_crsp_earlier"] == 1

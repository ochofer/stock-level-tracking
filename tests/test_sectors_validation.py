"""Links, GICS by month, sector weights, the benchmark and the verdicts, on made-up cases."""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from slt import constants as C  # noqa: E402
from slt import sectors as S  # noqa: E402
from slt import validation as V  # noqa: E402


def ulong():
    return pd.DataFrame({"month": pd.PeriodIndex(["2000-02", "2000-02", "2000-02", "2000-02"], freq="M"),
                         "permno": [1, 2, 3, 4], "date_prev": pd.Timestamp("2000-01-31")})


def raw_links():
    return pd.DataFrame({
        "gvkey": ["A", "B", "C", "D", "E", "F"],
        "lpermno": [1.0, 2.0, 2.0, 3.0, 4.0, np.nan],
        "linktype": ["LC", "LU", "LC", "LS", "LC", "LC"],
        "linkprim": ["P", "C", "P", "P", "C", "P"],
        "linkdt": pd.to_datetime(["1990-01-01", "1999-01-01", "1980-01-01", "1990-01-01", "2000-02-01", "1990-01-01"]),
        "linkenddt": pd.to_datetime([None, None, "2000-01-31", None, None, None])})


def test_primary_links_and_the_tie_break():
    links = S.primary_links(raw_links())
    assert set(links["gvkey"]) == {"A", "B", "C", "E"}           # LS and a missing PERMNO are dropped
    linked, multiple = S.link_gvkeys(ulong(), links)
    g = dict(zip(linked["permno"], linked["gvkey"]))
    assert g[1] == "A"
    assert g[2] == "C" and multiple == 1                          # B (C flag) and C (P flag, ends 31 Jan, inclusive) both valid: P wins
    assert pd.isna(g[3])                                          # only an LS link
    assert pd.isna(g[4])                                          # its link starts after 31 Jan


def test_gics_in_force_on_the_month_before_and_unclassified():
    linked, _ = S.link_gvkeys(ulong(), S.primary_links(raw_links()))
    history = pd.DataFrame({"gvkey": ["A", "C", "C"], "gsector": ["10", "45", "55"], "ggroup": ["1010", "4510", "5510"],
                            "gind": ["101010", "451010", "551010"], "gsubind": ["10101010", "45101010", "55101010"],
                            "indfrom": pd.to_datetime(["1999-06-30", "1999-06-30", "2000-02-15"]),
                            "indthru": pd.to_datetime([None, "2000-02-14", None])})
    classified, overlapping = S.gics_by_month(linked, history)
    s = dict(zip(classified["permno"], classified["sector"]))
    assert s[1] == "10" and s[2] == "45" and overlapping == 0
    assert s[3] == C.UNCLASSIFIED and s[4] == C.UNCLASSIFIED
    cov = S.coverage_by_year(classified).iloc[0]
    assert cov["universe_months"] == 4 and cov["months_with_gvkey"] == 2 and cov["months_unclassified"] == 2


def test_sector_weights_sum_to_one_and_have_twelve_columns():
    panel = pd.DataFrame({"month": pd.PeriodIndex(["2000-02"] * 4, freq="M"), "cap": [100.0, 300.0, np.nan, 100.0],
                          "sector": ["10", "45", "10", C.UNCLASSIFIED]})
    w = S.sector_weights(panel).iloc[0]
    assert w["weight_10"] == pytest.approx(0.2) and w["weight_45"] == pytest.approx(0.6)
    assert w["weight_unclassified"] == pytest.approx(0.2)
    assert len(S.sector_columns()) == C.SECTOR_COUNT and sum(w[c] for c in S.sector_columns()) == pytest.approx(1.0)


def test_benchmark_weights_by_start_of_month_cap_and_leaves_out_missing():
    panel = pd.DataFrame({"month": pd.PeriodIndex(["2000-02"] * 4, freq="M"),
                          "cap": [100.0, 300.0, np.nan, 50.0], "ret_total": [0.10, -0.10, 0.50, np.nan],
                          "retx_total": [0.10, -0.10, 0.50, 0.0]})
    b = V.benchmark(panel).iloc[0]
    assert b["benchmark_return"] == pytest.approx((100 * 0.10 + 300 * -0.10) / 400)
    assert b["benchmark_return_price_only"] == pytest.approx((100 * 0.10 + 300 * -0.10 + 50 * 0.0) / 450)
    assert b["universe_count"] == 4 and b["universe_cap_sum"] == 450.0
    assert b["members_without_cap"] == 1 and b["members_without_return"] == 1


def test_compare_and_the_verdicts():
    idx = pd.period_range("2000-01", periods=4, freq="M")
    ours = pd.Series([0.01, 0.02, 0.00, 0.03], index=idx)
    theirs = pd.Series([0.01, 0.02, 0.01, 0.03], index=idx)
    c = V.compare(ours, theirs, tolerance=0.005)
    assert c["months"] == 4 and c["mean_difference"] == pytest.approx(-0.0025)
    assert c["tracking_error"] == pytest.approx(np.std([0, 0, -0.01, 0], ddof=1) * np.sqrt(12))
    assert c["months_outside_tolerance"] == 1
    v = V.judge_g1_e1(c, c)
    assert v["G1"]["met"] is False and v["E1"]["met"] is False
    same = V.compare(ours, ours)
    v = V.judge_g1_e1(same, same)
    assert v["G1"]["met"] is True and v["E1"]["met"] is True
    assert V.verdict_line("G1", False, "x").startswith("G1: not met.")


def test_e2_aligns_month_t_with_crsp_row_t_minus_1():
    idx = pd.period_range("2000-01", periods=3, freq="M")
    bench = pd.DataFrame({"universe_count": [500, 501, 499], "universe_cap_sum": [1000.0, 1010.0, 990.0]}, index=idx)
    crsp = pd.DataFrame({"crsp_count": [500, 501, 499], "crsp_total_value": [1000.0, 1010.0, 990.0]}, index=idx)
    e2 = V.judge_e2(bench, crsp, "2000-02")
    # February's universe (501) against January's row (500): within 10; values 1010 against 1000: 1 per cent
    assert e2["months"] == 2 and e2["months_count_outside"] == 0 and e2["months_value_within"] == 2 and e2["met"] is True


def test_unclassified_months_by_cause_and_the_later_record():
    linked, _ = S.link_gvkeys(ulong(), S.primary_links(raw_links()))
    history = pd.DataFrame({"gvkey": ["A", "C"], "gsector": ["10", "45"], "ggroup": ["1010", "4510"],
                            "gind": ["101010", "451010"], "gsubind": ["10101010", "45101010"],
                            "indfrom": pd.to_datetime(["1999-06-30", "2000-06-30"]), "indthru": pd.NaT})
    classified, _ = S.gics_by_month(linked, history)
    out = S.later_classification_by_year(classified, history).iloc[0]
    # PERMNO 1 (A) is classified; PERMNO 2 (C) has a GVKEY and a record that starts later; 3 and 4 have no GVKEY
    assert out["months_unclassified"] == 3 and out["unclassified_without_gvkey"] == 2
    assert out["unclassified_with_later_record"] == 1


def test_e2_value_by_year():
    idx = pd.period_range("2000-01", periods=4, freq="M")
    bench = pd.DataFrame({"universe_cap_sum": [1.0, 1030.0, 1000.0, 990.0]}, index=idx)
    crsp = pd.DataFrame({"crsp_total_value": [1000.0, 1000.0, 1000.0, 1000.0]}, index=idx)
    out = V.e2_value_by_year(bench, crsp, "2000-02").iloc[0]
    assert out["months"] == 3 and out["months_value_outside"] == 1
    assert out["mean_relative_value_difference"] == pytest.approx((0.03 + 0.0 - 0.01) / 3, abs=1e-6)


def test_coverage_by_month_and_the_evaluation_start():
    months = pd.PeriodIndex(["1999-06"] * 2 + ["1999-07"] * 2 + ["1999-08"] * 2 + ["1999-09"] * 2, freq="M")
    classified = pd.DataFrame({"month": months, "cap": [10.0, 90.0, 10.0, 90.0, 10.0, 90.0, 10.0, 90.0],
                               "sector": [C.UNCLASSIFIED, C.UNCLASSIFIED, C.UNCLASSIFIED, "45", "10", "45", "10", "45"]})
    by = S.coverage_by_month(classified)
    assert by.loc[pd.Period("1999-07", "M"), "share_members_with_sector"] == 0.5
    assert by.loc[pd.Period("1999-07", "M"), "share_cap_with_sector"] == pytest.approx(0.9)
    e = S.evaluation_start(by, start="1999-07", end="1999-09")
    # July misses both levels, August is the first month that meets them, and September holds
    assert e["start"] == "1999-08" and e["months"] == 2 and e["condition_met"] is True
    by.loc[pd.Period("1999-09", "M"), "share_cap_with_sector"] = 0.5
    e = S.evaluation_start(by, start="1999-07", end="1999-09")
    assert e["condition_met"] is False and e["first_month_below_after_start"] == "1999-09"
    none = S.evaluation_start(by.iloc[:2], start="1999-07", end="1999-09")
    assert none["start"] is None and none["condition_met"] is False
    y = S.coverage_by_year_after(by, start="1999-06", end="1999-09").iloc[0]
    assert y["months"] == 4 and y["months_below_a_level"] == 3


def test_sector_links_reach_a_second_share_class_through_a_j_link():
    raw = pd.concat([raw_links(), pd.DataFrame({"gvkey": ["E"], "lpermno": [3.0], "linktype": ["LC"], "linkprim": ["J"],
                                                "linkdt": pd.to_datetime(["1990-01-01"]), "linkenddt": pd.NaT})],
                    ignore_index=True)
    assert set(S.primary_links(raw)["gvkey"]) == {"A", "B", "C", "E"}
    linked, multiple = S.link_gvkeys(ulong(), S.sector_links(raw))
    g = dict(zip(linked["permno"], zip(linked["gvkey"], linked["linkprim"])))
    assert g[3] == ("E", "J") and g[2] == ("C", "P") and multiple == 1   # PERMNO 3 reaches E only through the J link
    history = pd.DataFrame({"gvkey": ["E"], "gsector": ["45"], "ggroup": ["4510"], "gind": ["451010"], "gsubind": ["45101010"],
                            "indfrom": pd.to_datetime(["1999-06-30"]), "indthru": pd.NaT})
    classified, _ = S.gics_by_month(linked, history)
    cov = S.coverage_by_year(classified).iloc[0]
    assert cov["months_with_sector"] == 1 and cov["months_with_sector_through_a_j_or_n_link"] == 1


def test_weights_of_the_classified_capitalisation_and_the_cell_rule():
    months = pd.PeriodIndex(["2000-02"] * 6, freq="M")
    panel = pd.DataFrame({"month": months, "cap": [10.0, 10.0, 10.0, 70.0, 20.0, np.nan],
                          "sector": ["10", "10", "10", "45", C.UNCLASSIFIED, "45"]})
    w = S.sector_weights(panel, classified_only=True).iloc[0]
    assert list(w.index[1:]) == S.gics_columns() and len(S.gics_columns()) == 11
    assert w["weight_10"] == pytest.approx(0.3) and w["weight_45"] == pytest.approx(0.7) and w["weight_15"] == 0.0
    cells = S.cells_by_month(panel)
    r = S.cell_rule(cells, C.GICS_SECTORS, "2000-02", "2000-02")
    # sector 10: three members, none above 80 per cent; sector 45: one member with a capitalisation; nine empty
    assert r["cells"] == 11 and r["empty_cells"] == 9 and r["cells_failing"] == 1 and r["rule_met"] is False
    assert r["fewest_members"] == 1 and r["largest_single_share"] == pytest.approx(1.0)
    ok = S.cell_rule(cells, ["10"], "2000-02", "2000-02")
    assert ok["rule_met"] is True and ok["fewest_members"] == 3 and ok["largest_single_share"] == pytest.approx(1 / 3)

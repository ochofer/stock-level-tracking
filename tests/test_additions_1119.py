"""The four additions to the paths and the attribution, and the three to the data layer, set on 19 November 2026
after every recorded run, each on made-up data: the benchmark's own turnover, the forecast by month and the interval on
the bias statistic, the allocation by group of sectors, the factor contributions with Newey-West standard errors,
the count of partial-month delisting returns, the return on CRSP's own definition of its index, and, where the
extraction is present, no return in it below minus one."""
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slt import attributions as A  # noqa: E402
from slt import brinson as B  # noqa: E402
from slt import constants as C  # noqa: E402
from slt import evaluation as V  # noqa: E402
from slt import mandate as M  # noqa: E402
from slt import returns as R  # noqa: E402
from slt import sorts as Q  # noqa: E402
from slt import validation as VA  # noqa: E402

RES = ROOT / C.RESULTS_DIR


def members_frame(months, permnos_by_month, caps, rets):
    rows = []
    for mo in months:
        for p in permnos_by_month[mo]:
            rows.append({"month": pd.Period(mo, "M"), "permno": p, "cap": caps[mo][p], "sector": "45", "gsubind": "",
                         "permco": p, "ret_total": rets[mo][p]})
    return pd.DataFrame(rows)


def test_the_benchmark_turnover_has_no_rebalancing_and_its_parts_sum_to_the_total():
    months = ["2001-01", "2001-02", "2001-03"]
    held = {"2001-01": [1, 2, 3, 4], "2001-02": [1, 2, 3, 5], "2001-03": [1, 2, 3, 5]}
    rng = np.random.default_rng(3)
    caps = {mo: {p: float(rng.uniform(5, 50)) for p in held[mo]} for mo in months}
    rets = {mo: {p: float(rng.normal(0, 0.05)) for p in held[mo]} for mo in months}
    out = V.benchmark_turnover({"members": members_frame(months, held, caps, rets)}, first="2001-01", last="2001-03")
    assert out["months"] == 2 and out["first_month"] == "2001-02"
    assert out["largest_rebalancing"] < 1e-12 and abs(out["rebalancing"]) < 1e-12
    assert np.isclose(out["membership"] + out["reweighting"] - out["overlap"], out["total"])
    assert out["membership"] > 0 and out["restore_last_weights"] > 0
    # the month with no change of membership: membership zero, and the total is the move from drifted to new weights
    b1 = pd.Series(caps["2001-02"]) / sum(caps["2001-02"].values())
    b2 = pd.Series(caps["2001-03"]) / sum(caps["2001-03"].values())
    drifted, _ = M.drift(b1, pd.Series(rets["2001-02"]))
    first = M.one_way(pd.Series({1: caps["2001-02"][1], 2: caps["2001-02"][2], 3: caps["2001-02"][3], 5: caps["2001-02"][5]}) / sum(caps["2001-02"].values()),
                      M.drift(pd.Series(caps["2001-01"]) / sum(caps["2001-01"].values()), pd.Series(rets["2001-01"]))[0])
    assert np.isclose(out["total"], 12 * (first + M.one_way(b2, drifted)) / 2)


def test_the_forecast_table_leaves_empty_the_cells_the_active_return_leaves_empty():
    records = {}
    idx = pd.Index(pd.period_range("2001-01", periods=6, freq="M").astype(str), name="month")
    for k, p in enumerate(C.MONTHLY_PATHS):
        records[C.PATH_LABELS[p]] = pd.DataFrame({"active_return": np.linspace(-0.001, 0.001, 6), "forecast_te": 0.006,
                                                  "cell_ok": [k % 6 != i for i in range(6)]}, index=idx)
    active, _ = V.active_returns_table(records)
    forecast, empty = V.forecast_table(records)
    assert list(forecast.columns) == ["month"] + list(C.MONTHLY_FORECAST_COLUMNS)
    assert (active.drop(columns="month").isna().to_numpy() == forecast.drop(columns="month").isna().to_numpy()).all()
    assert all(v == 1 for v in empty.values())


def test_the_interval_is_the_source_algorithm_and_brackets_the_statistic():
    rng = np.random.default_rng(5)
    n = 306
    f = rng.uniform(0.004, 0.008, n)
    a = rng.standard_t(5, n) * f / np.sqrt(12) * 1.5
    lo, hi = V.bias_band(a, f)
    bias = V.bias_statistic(a, f)
    assert lo < bias < hi
    # the source's block_bootstrap_band, written out again: the same samples give the same interval
    rr = np.random.default_rng(C.BIAS_BAND_SEED)
    k = int(np.ceil(n / 12))
    starts = rr.integers(0, n, size=(C.BIAS_BAND_DRAWS, k))
    idx = ((starts[:, :, None] + np.arange(12)[None, None, :]) % n).reshape(C.BIAS_BAND_DRAWS, k * 12)[:, :n]
    stats = np.std((a / (f / np.sqrt(12)))[idx], axis=1, ddof=1)
    assert np.allclose((lo, hi), np.percentile(stats, [2.5, 97.5]))
    assert V.bias_band(a, f) == (lo, hi)


def test_the_committed_bias_statistics_are_recomputed_from_the_two_tables_by_month():
    """Once the forecast by month is committed, a reader recomputes each bias statistic and its interval from the
    repository alone; the figures equal paths_summary.json's and bias_intervals.csv's."""
    if not (RES / "forecast_te_monthly.csv").exists():
        pytest.skip("the forecast by month is committed with the run of notebook 09")
    active = pd.read_csv(RES / "active_returns_monthly.csv")
    forecast = pd.read_csv(RES / "forecast_te_monthly.csv")
    bands = V.bias_bands(active, forecast)
    summary = json.loads((RES / "paths_summary.json").read_text())["paths"]
    for _, r in bands.iterrows():
        assert np.isclose(r["bias"], summary[r["path"]]["bias"], rtol=0, atol=1e-12), r["path"]
    committed = pd.read_csv(RES / "bias_intervals.csv")
    assert np.allclose(committed[["bias", "interval_low", "interval_high"]], bands[["bias", "interval_low", "interval_high"]], atol=1e-10)


def test_the_allocation_by_group_takes_the_absolute_value_after_the_sum():
    rows = []
    for mo, vals in (("2001-01", {"10": 0.002, "55": -0.001, "45": 0.003, "Unclassified": -0.004}),
                     ("2001-02", {"10": -0.002, "55": -0.001, "45": 0.001, "Unclassified": 0.0})):
        for s in C.ATTRIBUTION_SECTORS:
            rows.append({"month": mo, "sector": s, "allocation": vals.get(s, 0.0), "selection": 0.0, "interaction": 0.0})
    out = B.allocation_by_group(pd.DataFrame(rows))
    assert np.isclose(out["all"], (abs(0.000) + abs(-0.002)) / 2)
    assert np.isclose(out["tilted"], (abs(0.001) + abs(-0.003)) / 2)
    assert np.isclose(out["other"], (abs(-0.001) + abs(0.001)) / 2)
    assert set(C.ALLOCATION_GROUPS["tilted"]) | set(C.ALLOCATION_GROUPS["other"]) == set(C.ATTRIBUTION_SECTORS)


def test_the_factor_table_has_a_newey_west_error_on_every_term_and_counts_them():
    rng = np.random.default_rng(7)
    n = 120
    att = pd.DataFrame({f"contribution {f}": rng.normal(0.0, 0.001, n) for f in C.FACTOR_ATTRIBUTION_FACTORS})
    att[f"contribution {C.FACTOR_ATTRIBUTION_FACTORS[0]}"] += 0.002
    att["factor total"] = att[[f"contribution {f}" for f in C.FACTOR_ATTRIBUTION_FACTORS]].sum(axis=1)
    att["residual"] = rng.normal(0.0, 0.002, n)
    t = A.factor_newey_west(att, Q.newey_west_se)
    assert list(t["term"]) == list(C.FACTOR_ATTRIBUTION_FACTORS) + ["factor total", "residual"]
    x = att["residual"].to_numpy()
    assert np.isclose(t.set_index("term").loc["residual", "newey_west_se"], 12 * Q.newey_west_se(x, C.NEWEY_WEST_LAGS))
    assert bool(t.set_index("term").loc[C.FACTOR_ATTRIBUTION_FACTORS[0], "beyond"])
    c = A.factor_beyond_count(t)
    assert c["contributions"] == len(C.FACTOR_ATTRIBUTION_FACTORS) and np.isclose(c["by_chance"], 0.05 * c["contributions"])


def test_a_partial_month_return_is_counted_by_the_guides_condition():
    panel = pd.DataFrame({"permno": [1, 2, 3, 4], "month": pd.PeriodIndex(["2001-03"] * 4, freq="M"),
                          "dlstcd": [231, 552, 470, 560], "delisting": ["given", "given", "given", "shumway"]})
    d = pd.DataFrame({"permno": [1, 2, 3, 4], "dlstcd": [231, 552, 470, 560],
                      "dlstdt": pd.to_datetime(["2001-03-10", "2001-03-20", "2001-03-05", "2001-03-07"]),
                      "dlpdt": pd.to_datetime(["2001-04-30", "2001-03-20", "2001-03-01", pd.NaT]),
                      "dlamt": [25.0, 3.1, 0.0, np.nan]})
    out = R.partial_month_counts(panel, d).set_index("code_family")
    assert out.loc[200, "partial_month_return"] == 0 and out.loc[500, "partial_month_return"] == 1
    assert out.loc[400, "partial_month_return"] == 0 and int(out["delisting_return_given"].sum()) == 3
    assert (out["partial_month_return"] + out["after_delisting"] == out["delisting_return_given"]).all()


def test_the_return_on_crsps_definition_takes_month_end_members_with_two_prices_and_no_adr():
    ends = pd.Series(pd.to_datetime(["2001-01-31", "2001-02-28", "2001-03-30"]), index=pd.period_range("2001-01", periods=3, freq="M"))
    spells = pd.DataFrame({"permno": [1, 2, 3, 4, 5], "start": pd.to_datetime(["2000-01-03"] * 3 + ["2001-02-15", "2000-01-03"]),
                           "ending": pd.to_datetime(["2024-12-31", "2001-02-10", "2024-12-31", "2024-12-31", "2024-12-31"])})
    rows = []
    for mo, date in zip(ends.index, ends.values):
        for p, (prc, sh, r) in {1: (10.0, 100.0, 0.01), 2: (20.0, 50.0, 0.02), 3: (-5.0, 200.0, 0.03), 4: (8.0, 10.0, 0.04), 5: (30.0, 30.0, 0.05)}.items():
            rows.append({"permno": p, "date": date, "prc": prc, "shrout": sh, "ret": r, "retx": r - 0.005})
    monthly = pd.DataFrame(rows)
    names = pd.DataFrame({"permno": [1, 2, 3, 4, 5], "namedt": pd.to_datetime(["1990-01-01"] * 5),
                          "nameendt": pd.to_datetime(["2024-12-31"] * 5), "shrcd": [11, 11, 10, 11, 31]})
    out = VA.crsp_definition(spells, ends, monthly, names)
    feb = out.loc[pd.Period("2001-02", "M")]
    # at the end of February: 1, 3 and 4 (2 left on 10 February, 4 joined on 15 February), and 5, an ADR, left out
    w = np.array([10.0 * 100, 5.0 * 200, 8.0 * 10])
    assert feb["count_used"] == 3 and feb["adrs_left_out"] == 1
    assert np.isclose(feb["return"], (w @ np.array([0.01, 0.03, 0.04])) / w.sum())
    assert np.isclose(feb["return_price_only"], (w @ np.array([0.005, 0.025, 0.035])) / w.sum())
    assert pd.Period("2001-01", "M") not in out.index      # no price at the end of December 2000
    assert list(out.index) == list(pd.period_range("2001-02", "2001-03", freq="M"))   # the last month of the file is kept
    assert out.loc[pd.Period("2001-03", "M"), "count_used"] == 3


@pytest.mark.skipif(not os.environ.get(C.DATA_ROOT_ENV, "").strip(), reason="the data root is not set on this machine")
def test_no_monthly_return_and_no_delisting_return_in_the_extraction_is_below_minus_one():
    """CRSP codes a missing return as -66 to -99 and a missing delisting return as -55 to -99 (its guide, pp. 30, 75
    and 122). No value below minus one may be read as a return."""
    from slt import extraction as X
    root = X.data_root()
    m, _ = X.read_table(root, "crsp_msf", columns=["ret", "retx"])
    d, _ = X.read_table(root, "crsp_msedelist", columns=["dlret", "dlretx"])
    for frame, cols in ((m, ["ret", "retx"]), (d, ["dlret", "dlretx"])):
        for c in cols:
            x = pd.to_numeric(frame[c], errors="coerce")
            assert not (x < C.RETURN_FLOOR).any(), f"{c}: {int((x < C.RETURN_FLOOR).sum())} values below {C.RETURN_FLOOR}"


def test_the_reading_of_the_return_on_crsps_definition_is_fixed_by_two_levels():
    assert C.CRSP_DEFINITION_READING_LEVELS == (0.00023, 0.0010)
    assert VA.crsp_definition_reading(0.00023) == "first" and VA.crsp_definition_reading(0.0001) == "first"
    assert VA.crsp_definition_reading(0.00024) == "middle" and VA.crsp_definition_reading(0.00099) == "middle"
    assert VA.crsp_definition_reading(0.0010) == "third" and VA.crsp_definition_reading(0.0050) == "third"
    assert set(VA.CRSP_DEFINITION_FORMS) == {"first", "middle", "third"}


def test_the_figures_the_text_quotes_from_the_tables_of_19_november():
    """The README and the note quote these figures from the seven tables added on 19 November 2026."""
    bp = lambda x: round(1e4 * x, 1)   # noqa: E731
    d = json.loads((RES / "reconstruction_crsp_definition.json").read_text())
    assert (bp(d["with_dividends"]["tracking_error"]), bp(d["price_only"]["tracking_error"])) == (8.2, 8.2)
    assert (bp(d["benchmark_against_crsp_definition_with_dividends"]["tracking_error"]),
            bp(d["benchmark_against_crsp_definition_price_only"]["tracking_error"])) == (13.3, 13.4)
    assert bp(d["benchmark_with_dividends"]["tracking_error"]) == 11.5 and d["months"] == 1188
    assert d["reading"]["form"] == VA.crsp_definition_reading(d["with_dividends"]["tracking_error"]) == "middle"
    assert (d["months_count_used_equal"], d["months_count_used_with_adrs_equal"], d["months_with_an_adr_left_out"],
            d["most_adrs_left_out_in_one_month"]) == (606, 1129, 544, 1)
    p = pd.read_csv(RES / "delisting_partial_month.csv").set_index("code_family")
    assert p["delisting_return_given"].sum() == 798 and p["partial_month_return"].sum() == 6
    assert p.loc[[400, 500], "partial_month_return"].sum() == 4
    families = json.loads((RES / "returns_summary.json").read_text())["delisting_months_by_code_family"]
    assert families["400"] + families["500"] == 36
    t = pd.read_csv(RES / "benchmark_turnover.csv").iloc[0]
    assert [round(100 * t[k], 2) for k in ("total", "membership", "reweighting", "overlap")] == [5.82, 2.84, 3.73, 0.75]
    b = pd.read_csv(RES / "bias_intervals.csv").set_index("path")
    two = lambda path: tuple(round(b.loc[path, k], 2) for k in ("bias", "interval_low", "interval_high"))   # noqa: E731
    assert two("ledoit_wolf_C3_cap200bp") == (1.48, 1.29, 1.67) and two("factor_C3_cap200bp") == (1.85, 1.63, 2.08)
    assert two("pca_C3_cap200bp") == (1.73, 1.51, 1.91) and two("ledoit_wolf_C3_cap50bp") == (1.26, 1.08, 1.45)
    assert all(b.loc[p_, "interval_low"] > 1.25 for p_ in ("ledoit_wolf_C3_cap200bp", "factor_C3_cap200bp", "pca_C3_cap200bp"))
    g = pd.read_csv(RES / "allocation_by_group.csv").set_index("path")
    assert (bp(g.loc["ledoit_wolf_C3_cap200bp", "tilted"]), bp(g.loc["ledoit_wolf_C3_cap200bp", "other"]),
            bp(g.loc["ledoit_wolf_C3_cap200bp", "all"])) == (15.7, 4.4, 16.6)
    assert (g["tilted"] > g["other"]).all() and len(g) == 4
    n = pd.read_csv(RES / "factor_attribution_newey_west.csv")
    lw = n[n["path"] == "ledoit_wolf_C3_cap200bp"].set_index("term")
    assert (bp(lw.loc["factor total", "per_year"]), bp(lw.loc["factor total", "newey_west_se"])) == (-9.7, 7.7)
    assert not lw.loc["factor total", "beyond"] and lw.loc["Mkt-RF", "beyond"]
    assert (bp(lw.loc["Mkt-RF", "per_year"]), bp(lw.loc["Mkt-RF", "newey_west_se"])) == (-1.7, 0.6)
    count = A.factor_beyond_count(n)
    assert (count["contributions"], count["beyond"], round(count["by_chance"], 1)) == (24, 3, 1.2)
    assert set(n[n["beyond"] & n["term"].isin(C.FACTOR_ATTRIBUTION_FACTORS)]["term"]) == {"Mkt-RF"}

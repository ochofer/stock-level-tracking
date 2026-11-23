"""
Every figure, sentence number and table of the dashboard and the note, read from the committed results only.

build(repo) reads the files in results/<vintage>/ (constants.RESULTS_DIR) and returns two dictionaries: the numbers and
the tables as HTML, keyed by the names the templates use, and the data the dashboard's charts draw. The dashboard
(dashboard/build_dashboard.py) and the note (note/build_note.py) both build from it, so that a figure is computed once
and reads the same on both. One file it reads is by month, cap_relaxed.csv, the months in which the turnover cap gave
way (notebook 10, 19 rows, each under the cell rule). Everything else is over the window or by year, the paired tests
of notebook 13 (paired_tests.csv and paired_size_check.csv) included.
"""
from __future__ import annotations

import datetime
import html
import json
import pathlib
import re
import zoneinfo

import pandas as pd

from . import constants as C
from . import figures as F

VINTAGE = pathlib.PurePath(C.RESULTS_DIR).name
# The periods over which the gap to the index's own price return is pooled, set on 18 November 2026: the years to
# 1956, when the list holds about 90 names, then 1957 to 1989, 1990 to 1998, 1999 to 2004, the years of the
# mandate before the index moved to float, and 2005 to 2024. Each figure is the square root of the mean of the
# squared tracking errors of the period's years, each year of 12 months.
GAP_PERIODS = ((1926, 1956), (1957, 1989), (1990, 1998), (1999, 2004), (2005, 2024))
MONTHS_WORDS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                "November", "December"]


def read(name, res):
    path = pathlib.Path(res) / name
    if not path.exists():
        raise SystemExit(f"{C.RESULTS_DIR}/{name} is missing: run the notebook that writes it first")
    if name.endswith(".json"):
        return json.loads(path.read_text())
    return pd.read_csv(path)


def num(x, d=1, sign=False):
    """A number for display: the true minus sign, a plus sign where asked (N3)."""
    s = f"{abs(x):,.{d}f}"
    if x < 0 and float(s.replace(",", "")) != 0:
        return "−" + s
    if sign and float(s.replace(",", "")) != 0:
        return "+" + s
    return s


def signed(x, d=1):
    """A signed number in words, for prose: plus 3.1, minus 3.6."""
    s = num(abs(x), d)
    return s if float(s.replace(",", "")) == 0 else ("minus " if x < 0 else "plus ") + s


def words(ym):
    y, m = ym.split("-")
    return f"{MONTHS_WORDS[int(m) - 1]} {y}"


def table(cols, rows, cls=""):
    """A table as HTML, the first column as row headers (N1, N6)."""
    longest = max([len(re.sub(r"<[^>]*>", "", str(r[0]))) for r in rows] + [0])
    classes = " ".join(c for c in [cls, "wrapfirst" if longest > 28 else ""] if c)
    h = [f'<div class="tscroll"><table{" class=" + chr(34) + classes + chr(34) if classes else ""}><thead><tr>']
    h += [f"<th>{c}</th>" for c in cols]
    h.append("</tr></thead><tbody>")
    for r in rows:
        h.append("<tr>" + f'<th scope="row">{r[0]}</th>' + "".join(f"<td>{c}</td>" for c in r[1:]) + "</tr>")
    h.append("</tbody></table></div>")
    return "".join(h)


def lohi(xs, d=1):
    xs = sorted(xs)
    return num(xs[0], d), num(xs[-1], d)



def build(repo, build_date: datetime.date | None = None) -> tuple:
    """The numbers and tables (P) and the charts' data (D), from the results committed under repo."""
    res = F.results_dir(repo)
    # --- inputs, one line per file, with the notebook that writes it ---
    v1, v2, v3, v4 = (read(f"verdicts_milestone{k}.json", res) for k in (1, 2, 3, 4))   # notebooks 04 and 02, 06 and 07, 09 and 10, 11
    paths = read("paths_summary.json", res)["paths"]                                    # notebook 09
    intervals = read("bias_intervals.csv", res).set_index("path")                      # notebook 09, added 19 November 2026
    brinson = read("brinson_summary.json", res)["paths"]                                # notebook 11
    factor = read("factor_attribution_summary.json", res)["paths"]                      # notebook 11
    recon = read("reconstruction_summary.json", res)                                    # notebook 04
    answer = read("answer_key_summary.json", res)["comparisons"]                        # notebook 06
    cov_year = read("covariance_by_year.csv", res)                                      # notebook 07
    cov = read("covariance_summary.json", res)                                          # notebook 07
    counts = read("attribution_counts_by_year.csv", res)                               # notebook 11
    universe = read("universe_summary.json", res)                                       # notebook 02

    EST = F.ORDER
    LBL = F.LABELS
    SHORT = {"ledoit_wolf": "Ledoit-Wolf", "sample": "Sample", "factor": "Factor", "pca": "PCA"}
    MAIN = {e: C.PATH_LABELS[(e, C.MANDATE, C.MAIN_CAP)] for e in EST}
    BP = 1e4
    P = {}

    first, last = C.EVAL_START, C.EVAL_END
    P["first_w"], P["last_w"] = words(first), words(last)
    P["months"] = str(paths[MAIN["ledoit_wolf"]]["months"])
    P["vintage"] = VINTAGE
    P["results"] = C.RESULTS_DIR
    day = build_date or datetime.datetime.now(zoneinfo.ZoneInfo("Europe/Amsterdam")).date()
    P["build_date"] = f"{day.day} {MONTHS_WORDS[day.month - 1]} {day.year}"

    # --- the key figures and Key points ---
    t1 = F.tracking_error_by_estimator(res)
    P["te_lo"], P["te_hi"] = lohi(t1["realised"])
    P["bias_lo"], P["bias_hi"] = lohi(t1["bias"], 2)
    # Added 18 November 2026: the bias statistic is not the ratio of the realised to the mean forecast, and the
    # pages say so beside the two columns. The understatement in per cent is the bias statistic less one.
    P["ratio_lo"], P["ratio_hi"] = lohi(t1["realised"] / t1["forecast"], 2)
    P["under_lo"], P["under_hi"] = lohi(100 * (t1["bias"] - 1), 0)
    P["recon_te"] = num(BP * recon["with_dividends"]["tracking_error"])
    P["recon_first_w"], P["recon_last_w"] = words(recon["usable_window"]["first_month"]), words(recon["usable_window"]["last_month"])
    P["recon_years_first"], P["recon_years_last"] = recon["usable_window"]["first_month"][:4], recon["usable_window"]["last_month"][:4]
    P["mom_corr"] = f"{answer['momentum']['quintiles']['correlation']:.3f}"
    P["var_corr"] = f"{answer['variance']['quintiles']['correlation']:.3f}"
    e8 = v4["E8"]["second_clause"]["by_estimator"]
    P["kpi_alloc"], P["kpi_sel"], P["kpi_int"] = (num(BP * e8["ledoit_wolf"][k]) for k in ("allocation", "selection", "interaction"))
    S3 = list(C.STRUCTURED_ESTIMATORS)
    P["alloc_lo"], P["alloc_hi"] = lohi([BP * e8[e]["allocation"] for e in S3])
    P["sel_lo"], P["sel_hi"] = lohi([BP * e8[e]["selection"] for e in S3])
    P["int_lo"], P["int_hi"] = lohi([BP * e8[e]["interaction"] for e in S3])
    P["share_lo"], P["share_hi"] = lohi([100 * factor[MAIN[e]]["factor share"] for e in EST], 0)
    lw = paths[MAIN["ledoit_wolf"]]
    P["forced"] = num(100 * lw["turnover_membership_per_year"], 2)
    P["total_turn"] = num(100 * lw["turnover_total_per_year"], 2)
    P["forced_share"] = num(100 * v3["E7"]["membership_share_of_total"])
    fr = F.frontier(res)
    fl = fr[fr["estimator"] == "ledoit_wolf"]
    P["chosen_lo"], P["chosen_hi"] = num(fl["chosen_turnover"].min()), num(fl["chosen_turnover"].max())
    P["fr_te_lo"], P["fr_te_hi"] = lohi(fl["realised"])
    ar = [BP * paths[MAIN[e]]["active_return_gross"] for e in EST]
    se = [BP * paths[MAIN[e]]["active_return_gross_se"] for e in EST]
    P["ar_lo"], P["ar_hi"] = lohi(ar)
    P["ar_se_lo"], P["ar_se_hi"] = lohi(se)
    P["ar_se12_lo"], P["ar_se12_hi"] = lohi([BP * paths[C.PATH_LABELS[p]]["active_return_gross_se"] for p in C.MONTHLY_PATHS])
    # Added 21 November 2026: the mean active return a path would need to be statistically significant at the 5 per cent
    # level, 1.96 Newey-West standard errors, at the mean standard error of the four paths and of the twelve.
    need4 = 1.96 * sum(se) / len(se)
    need12 = 1.96 * sum(BP * paths[C.PATH_LABELS[p]]["active_return_gross_se"] for p in C.MONTHLY_PATHS) / len(C.MONTHLY_PATHS)
    assert round(need4) == round(need12), "the text gives one figure for the four paths and the twelve"
    P["ar_need"] = f"{need4:.0f}"

    # --- the months in which the turnover cap gave way (notebook 10), by month ---
    relaxed = read("cap_relaxed.csv", res)
    capped = [C.PATH_LABELS[p] for p in C.MONTHLY_PATHS if p[2] is not None]
    first_rows = relaxed[relaxed["month"] == str(pd.Period(first, "M"))]
    assert sorted(first_rows["path"]) == sorted(capped), "the text says the cap gives way in the first month of every capped path"
    assert first_rows["turnover_chosen"].max() - first_rows["turnover_chosen"].min() < 1e-6, "one figure for the first month"
    later = relaxed[relaxed["month"] != str(pd.Period(first, "M"))]
    tight = min(p[2] for p in C.MONTHLY_PATHS if p[2] is not None)
    assert (later["cap"] == tight).all(), "the text says the later months are at the tightest cap"
    P["cr_first_w"] = words(first)
    P["cr_first"] = num(100 * first_rows["turnover_chosen"].iloc[0], 2)
    P["cr_tight"] = num(100 * tight, 1)
    P["cr_lw_more"], P["cr_fa_more"] = (str(int((later["path"] == C.PATH_LABELS[(e, C.MANDATE, tight)]).sum())) for e in C.FRONTIER_ESTIMATORS)
    assert set(later["path"]) <= {C.PATH_LABELS[(e, C.MANDATE, tight)] for e in C.FRONTIER_ESTIMATORS}
    bind_all = [p for p in C.MONTHLY_PATHS if p[0] == "ledoit_wolf" and p[2] is not None and paths[C.PATH_LABELS[p]]["cap_binding_share"] == 1.0]
    assert sorted(p[2] for p in bind_all) == [0.005, 0.01, 0.02], "the text names the caps at which the cap binds in every month on Ledoit-Wolf"
    for p in bind_all:
        assert paths[C.PATH_LABELS[p]]["turnover_chosen_per_year"] > 12 * p[2], "the text says chosen turnover lies above twelve times the cap"
    CAP_RELAXED = ("In a month where the other constraints need more chosen turnover than the cap allows, the cap gives way to the "
                   "least turnover they need. This happens in the first month of every capped path under the mandate, {{cr_first_w}}, "
                   "when the portfolio leaves the benchmark for the mandate and needs {{cr_first}} per cent, and in {{cr_lw_more}} further "
                   "months at the {{cr_tight}} per cent cap on the Ledoit-Wolf estimator and {{cr_fa_more}} on the factor estimator. So on "
                   "the Ledoit-Wolf estimator at caps of 0.5, 1 and 2 per cent, where the cap binds in every month, chosen turnover a "
                   "year lies a little above twelve times the cap.")
    P["cap_relaxed"] = re.sub(r"\{\{(\w+)\}\}", lambda m: P[m.group(1)], CAP_RELAXED)

    # --- Figure 2: by year, the Ledoit-Wolf path ---
    t2 = F.tracking_error_by_year(res)
    peak = t2.loc[t2["realised"].idxmax()]
    low = t2.loc[t2["realised"].idxmin()]
    P["peak_year"], P["peak_re"], P["peak_fc"] = str(int(peak.year)), num(peak.realised, 0), num(peak.forecast, 0)
    P["low_year"], P["low_re"] = str(int(low.year)), num(low.realised, 0)
    above = t2[t2["forecast"] > t2["realised"]]
    P["below_years"] = str(len(t2) - len(above))
    P["years"] = str(len(t2))
    P["above_list"] = ", ".join(str(int(y)) for y in above["year"][:-1]) + " and " + str(int(above["year"].iloc[-1]))

    # --- E5 ---
    P["lw_fc_tight"], P["lw_fc_none"] = num(fl.iloc[0]["forecast"]), num(fl.iloc[-1]["forecast"])
    ff = fr[fr["estimator"] == "factor"]
    P["fa_fc_tight"], P["fa_fc_none"] = num(ff.iloc[0]["forecast"]), num(ff.iloc[-1]["forecast"])
    P["lw_re_tight"], P["lw_re_4"] = num(fl.iloc[0]["realised"]), num(fl[fl["cap"] == "4"]["realised"].iloc[0])
    P["fa_re_tight"], P["fa_re_4"] = num(ff.iloc[0]["realised"]), num(ff[ff["cap"] == "4"]["realised"].iloc[0])
    P["fa_chosen_hi"] = num(ff["chosen_turnover"].max())
    P["fa_chosen_lo"] = num(ff["chosen_turnover"].min())
    P["fa_re_none"] = num(ff[ff["cap"] == "none"]["realised"].iloc[0])
    P["fa_bind"] = num(100 * paths[MAIN["factor"]]["cap_binding_share"])
    assert paths[MAIN["ledoit_wolf"]]["cap_binding_share"] == 1.0, "the text says the cap binds in every month on Ledoit-Wolf"

    # --- the attribution ---
    P["lw_alloc"], P["lw_sel"], P["lw_int"] = (num(BP * brinson[MAIN["ledoit_wolf"]]["total"][f"{k}_per_year"], 1) for k in C.ATTRIBUTION_TERMS)
    P["lw_alloc_se"], P["lw_sel_se"], P["lw_int_se"] = (num(BP * brinson[MAIN["ledoit_wolf"]]["total"][f"{k}_per_year_se"], 1) for k in C.ATTRIBUTION_TERMS)
    P["lw_ar"] = num(BP * lw["active_return_gross"])
    t4 = F.sector_terms(res)
    beyond = F.terms_beyond_two_se(res)
    P["beyond_lw"] = str(beyond["ledoit_wolf"]["beyond"])
    others = [beyond[e]["beyond"] for e in EST if e != "ledoit_wolf"]
    P["beyond_lo"], P["beyond_hi"] = str(min(others)), str(max(others))
    P["chance"] = num(0.05 * beyond["ledoit_wolf"]["terms"], 1)
    en, ut = t4.set_index("sector").loc["10"], t4.set_index("sector").loc["55"]
    P["en_alloc"], P["en_alloc_se"], P["ut_alloc"], P["ut_alloc_se"] = num(en.allocation), num(en.allocation_se), num(ut.allocation), num(ut.allocation_se)
    P["other_alloc_se_max"] = num(t4[t4["sector"] != "10"]["allocation_se"].max())
    P["fs_lw_total"], P["fs_lw_total_se"] = num(BP * factor[MAIN["ledoit_wolf"]]["factor total (per year)"]), num(BP * factor[MAIN["ledoit_wolf"]]["factor total (standard error)"])
    P["fs_lw_resid"], P["fs_lw_resid_se"] = num(BP * factor[MAIN["ledoit_wolf"]]["residual (per year)"]), num(BP * factor[MAIN["ledoit_wolf"]]["residual (standard error)"])
    P["mkt_exp_lo"], P["mkt_exp_hi"] = (num(x, 4, True) for x in sorted(factor[MAIN[e]]["exposure Mkt-RF"] for e in EST)[::3])
    tilt = read("tilt_cap.csv", res).set_index("path")                                 # notebook 11
    P["cap_lw_at"], P["cap_tilted"] = f"{int(tilt.loc[MAIN['ledoit_wolf'], 'at_cap']):,}", f"{int(tilt.loc[MAIN['ledoit_wolf'], 'tilted_stock_months']):,}"
    not_held = {e: int(brinson[MAIN[e]]["sector_months_not_held"]) for e in EST}
    P["nh_sample"], P["nh_lw"] = str(not_held["sample"]), str(not_held["ledoit_wolf"])
    P["no_return"] = str(int(counts["name_months_without_return"].sum()))

    # --- the data layer ---
    P["recon_te_px"] = num(BP * recon["price_only"]["tracking_error"])
    P["recon_eval_te"] = num(BP * recon["evaluation_window"]["with_dividends"]["tracking_error"])
    P["idx_to_2004"] = num(BP * recon["against_index_price_return"]["to_december_2004"]["tracking_error"])
    P["idx_from_2005"] = num(BP * recon["against_index_price_return"]["from_january_2005"]["tracking_error"])
    readme = (pathlib.Path(repo) / "README.md").read_text(encoding="utf-8")
    P["e4_months"] = str(cov["e4_first_clause"]["months_at_or_above_level"])
    P["e4_one_class"] = str(cov["share_class_check"]["months_one_class_at_or_above_level"])
    P["member_months"] = f"{universe['member_months']:,}"

    # --- the paired tests (notebook 13, added 18 November 2026) ---
    pt = read("paired_tests.csv", res).set_index("series 1")                           # notebook 13
    size = read("paired_size_check.csv", res)                                          # notebook 13
    e4 = pt.loc[MAIN["sample"]]
    assert e4["series 2"] == MAIN["ledoit_wolf"] and e4["lower"] > 0, "the text says the difference is distinguishable from zero"
    P["e4_corr"] = f"{e4['correlation']:.3f}"
    P["e4_d"], P["e4_lo"], P["e4_hi"] = num(e4["difference"]), num(e4["lower"]), num(e4["upper"])
    P["e4_p"], P["e4_holm"] = f"{e4['p-value']:.3f}", f"{e4['Holm p-value']:.3f}"
    for est, key in (("ledoit_wolf", "lw"), ("factor", "fa")):
        for cap, ck in ((0.04, "4"), (None, "n")):
            r = pt.loc[C.PATH_LABELS[(est, C.MANDATE, cap)]]
            assert r["series 2"] == C.PATH_LABELS[(est, C.MANDATE, 0.005)]
            assert r["lower"] < 0 < r["upper"], "the text says no change along the frontier is distinguishable from zero"
            P[f"{key}_d{ck}"] = signed(r["difference"])
            P[f"{key}_d{ck}_lo"], P[f"{key}_d{ck}_hi"] = signed(r["lower"]), signed(r["upper"])
            P[f"{key}_d{ck}_p"] = f"{r['p-value']:.2f}"
            P[f"{key}_d{ck}_det"] = num(r["detectable difference"])
            P[f"{key}_d{ck}_crit"] = num(r["critical value"])
            P[f"{key}_d{ck}_corr"] = f"{r['correlation']:.2f}"
    P["b_corr_lo"], P["b_corr_hi"] = lohi(pt[pt["comparison"] == "B"]["correlation"], 2)
    # Added 19 November 2026: the factor estimator's two rows of family B without July and December 2008.
    lo = read("paired_tests_without_two_months.csv", res).set_index("series 1")        # notebook 13
    share = "share of the difference between the two variances in those months"
    for cap, ck in ((0.04, "4"), (None, "n")):
        r = lo.loc[C.PATH_LABELS[("factor", C.MANDATE, cap)]]
        assert r["months left out"] == "2008-07 and 2008-12" and r["lower"] < 0 < r["upper"]
        assert r["critical value"] < pt.loc[C.PATH_LABELS[("factor", C.MANDATE, cap)], "critical value"] / 2, \
            "the text says the two months raise the critical value"
        P[f"fa_x{ck}_share"] = num(100 * r[share], 0)
        P[f"fa_x{ck}"] = signed(r["difference"])
        P[f"fa_x{ck}_lo"], P[f"fa_x{ck}_hi"] = signed(r["lower"]), signed(r["upper"])
        P[f"fa_x{ck}_crit"] = num(r["critical value"])
    assert (size["P1"] == "met").all() and (size["share rejected"] < C.PAIRED_SIZE_CONSERVATIVE_BELOW).all(), \
        "the text says P1 is met and the test rejects less often than its level at both settings"
    # Added 19 November 2026: the seven comparisons at the other block lengths, one sentence for the README, the note
    # and notebook 13, built from the table and asserted here.
    ob = read("paired_tests_other_blocks.csv", res)                                     # notebook 13
    assert sorted(ob["block"].unique()) == sorted(C.PAIRED_OTHER_BLOCKS) and C.PAIRED_BLOCK_MONTHS == 12
    first = ob[ob["series 1"] == MAIN["sample"]].set_index("block").loc[list(C.PAIRED_OTHER_BLOCKS)]
    rest = ob[ob["series 1"] != MAIN["sample"]]
    assert (first["series 2"] == MAIN["ledoit_wolf"]).all() and (first["lower"] > 0).all(), \
        "the text says the sample estimator's interval excludes zero at every length"
    assert len(rest) == 6 * len(C.PAIRED_OTHER_BLOCKS) and ((rest["lower"] < 0) & (rest["upper"] > 0)).all(), \
        "the text says the other six intervals include zero at every length"
    h24, p24, n24 = first.loc[24, "Holm p-value"], first.loc[24, "p-value"], int(first.loc[24, "draws"])
    se24 = h24 / p24 * (p24 * (1 - p24) / n24) ** 0.5
    assert abs(h24 - 0.05) < 1.96 * se24, "the text says the samples cannot place the value on either side of 0.05"

    def four(x):
        return f"{x:.4f}".rstrip("0")
    blocks = ", ".join(str(b) for b in C.PAIRED_OTHER_BLOCKS[:-1]) + f" and {C.PAIRED_OTHER_BLOCKS[-1]}"
    ivs = [f"{r.lower:.2f} to {r.upper:.2f}" for r in first.itertuples()]
    holms = [four(h) for h in first["Holm p-value"]]
    det = [f"{x:.1f}" for x in (rest["detectable difference"].min(), rest["detectable difference"].max())]
    P["blocks"] = (f"The block is fixed at {C.PAIRED_BLOCK_MONTHS} months, where Ledoit and Wolf (2008, Algorithm 3.1) "
                   f"choose its length from the data. At blocks of {blocks} months, added on 19 November 2026, after the "
                   f"study's results were known, with no verdict, the sample estimator's difference from Ledoit-Wolf has "
                   f"intervals of {ivs[0]}, {ivs[1]} and {ivs[2]}, with Holm p-values of {holms[0]}, {holms[1]} and "
                   f"{holms[2]}, and none of the other six differences is statistically significant at the 5 per cent level at "
                   f"any length, where they could detect differences of {det[0]} to {det[1]} basis points a year four times "
                   f"in five. The sample estimator's difference is therefore statistically significant at the 5 per cent "
                   f"level at all four lengths, tested on its own. At {C.PAIRED_OTHER_BLOCKS[-1]} months its Holm p-value is "
                   f"{holms[2]}, and {n24:,} samples cannot place it on either side of 0.05.")
    nb13 = json.loads((pathlib.Path(repo) / "notebooks" / "13_paired_tests.ipynb").read_text(encoding="utf-8"))
    nb13_text = "".join("".join(c["source"]) for c in nb13["cells"] if c["cell_type"] == "markdown")
    assert P["blocks"] in nb13_text, "notebook 13 carries the sentence built here"
    P["size_hi_rho"], P["size_lo_rho"] = (num(100 * size.set_index("correlation").loc[r, "share rejected"]) for r in (0.99, 0.80))
    rows8 = []
    for lab, r in pt.iterrows():
        rows8.append([r["label"][0].upper() + r["label"][1:] + (" (A)" if r["comparison"] == "A" else " (B)"),
                      num(r["difference"], 1, True), f"{num(r['lower'], 1, True)} to {num(r['upper'], 1, True)}",
                      f"{r['p-value']:.3f}", "" if pd.isna(r["Holm p-value"]) else f"{r['Holm p-value']:.3f}",
                      num(r["detectable difference"]), num(r["critical value"], 2), f"{r['correlation']:.3f}"])
    P["table8"] = table(["Comparison", "Difference, bp a year", "95 per cent interval", "p-value", "Holm p-value",
                         "Detectable difference", "Critical value", "Correlation"], rows8)

    # --- the gap to the index's own price return by period (reconstruction_by_year.csv, notebook 04) ---
    ry = read("reconstruction_by_year.csv", res)                                       # notebook 04
    assert (ry["months"] == 12).all() and ry["year"].min() == 1926 and ry["year"].max() == 2024
    for k, (a, b) in enumerate(GAP_PERIODS, 1):
        s = ry[(ry["year"] >= a) & (ry["year"] <= b)]
        P[f"gap{k}"] = num(BP * float((s["tracking_error_against_index_price"] ** 2).mean()) ** 0.5)
    s = ry[(ry["year"] >= 1957) & (ry["year"] <= 2004)]
    P["gap_57_04"] = num(BP * float((s["tracking_error_against_index_price"] ** 2).mean()) ** 0.5)
    # Added 19 November 2026: the months to December 2004 pooled over the years, beside the figure measured in one piece.
    s = ry[ry["year"] <= 2004]
    P["gap_to_2004"] = num(BP * float((s["tracking_error_against_index_price"] ** 2).mean()) ** 0.5)
    P["gap_1926"], P["gap_1927"] = (num(BP * ry.set_index("year").loc[y, "tracking_error_against_index_price"], 0) for y in (1926, 1927))

    # --- Table 1: the gates and expectations, each with its figure and verdict ---
    def verdict(met):
        return "Met" if met else "Not met"


    rows1 = [
        ["G1", "Tracking error of the benchmark against CRSP's own return of the same list, at most 50 bp a year", f"{num(BP * v1['G1']['tracking_error'])} bp a year", verdict(v1["G1"]["met"])],
        ["G2", "Usable window at least 15 years", f"{v1['G2']['years']:.0f} years", verdict(v1["G2"]["met"])],
        ["G3", "At least 90 per cent of removals with a CRSP price within 10 trading days", f"{v1['G3']['share'] * 100:.0f} per cent of {v1['G3']['removals']:,}", verdict(v1["G3"]["met"])],
        ["G4", "Capitalisation in at least 90 per cent of membership months for at least 90 per cent of names", f"{v1['G4']['share_of_names'] * 100:.1f} per cent of {v1['G4']['names']:,} names", verdict(v1["G4"]["met"])],
        ["E1", "That tracking error at most 20 bp a year and the mean difference within 2 bp, with dividends and price-only", f"{num(BP * v1['E1']['with_dividends']['tracking_error'])} and {num(BP * v1['E1']['price_only']['tracking_error'])} bp a year, {num(BP * v1['E1']['with_dividends']['mean_difference'], 2)} and {num(BP * v1['E1']['price_only']['mean_difference'], 2)} bp", verdict(v1["E1"]["met"])],
        ["E2", "Count within 10 names of CRSP's in every month, value within 2 per cent in at least 95 per cent of months", f"{v1['E2']['months'] - v1['E2']['months_count_outside']:,} and {v1['E2']['months_value_within']:,} of {v1['E2']['months']:,} months", verdict(v1["E2"]["met"])],
        ["E3", "Momentum and variance spreads: correlation at least 0.80 with French's, mean difference within two standard errors", f"{P['mom_corr']} and {P['var_corr']}, {num(abs(answer['momentum']['quintiles']['difference_in_standard_errors']), 2)} and {num(abs(answer['variance']['quintiles']['mean_difference'] / answer['variance']['quintiles']['standard_error']), 2)} standard errors", verdict(v2["E3_momentum"]["met"] and v2["E3_variance"]["met"])],
        ["E4, first clause", "Sample condition number at least 10 times the Ledoit-Wolf one in every month", f"{v2['E4_first_clause']['months_at_or_above_level']} of {v2['E4_first_clause']['months']} months", verdict(v2["E4_first_clause"]["met"])],
        ["E4, second clause", "Under the mandate, the sample estimator's realised tracking error above the Ledoit-Wolf one", f"{num(BP * v3['E4_second_clause']['sample_realised_te'])} against {num(BP * v3['E4_second_clause']['ledoit_wolf_realised_te'])} bp a year", verdict(v3["E4_second_clause"]["met"])],
        ["E5", "On the frontier, the mean forecast does not rise as the cap loosens, and the realised tracking error is lower at 4 per cent than at 0.5", f"holds for the factor estimator, fails for Ledoit-Wolf ({P['lw_re_4']} against {P['lw_re_tight']} bp a year)", verdict(v3["E5"]["met"])],
        ["E6", "Realised tracking error 50 to 150 bp a year and bias statistic 0.8 to 1.25, each structured estimator", f"tracking error {num(BP * v3['E6_ledoit_wolf']['realised_te'])} to {num(BP * v3['E6_pca']['realised_te'])}, bias {v3['E6_ledoit_wolf']['bias']:.2f} to {v3['E6_factor']['bias']:.2f}", verdict(v3["E6"]["met"])],
        ["E7", "Forced turnover less than a third of the total, Ledoit-Wolf path", f"{P['forced']} of {P['total_turn']} per cent a year, {P['forced_share']} per cent", verdict(v3["E7"].get("met", v3["E7"]["membership_share_of_total"] < 1 / 3))],
        ["E8", "The three terms sum to the active return, and allocation is the largest in absolute value and interaction the smallest, each structured estimator", f"largest gap {v4['E8']['identity']['largest_error']:.1e}, allocation {P['alloc_lo']} to {P['alloc_hi']}, selection {P['sel_lo']} to {P['sel_hi']}, interaction {P['int_lo']} to {P['int_hi']} bp a month", verdict(v4["E8"]["met"])],
        ["E9", "Contributions and residual sum to the active return, and the mean exposure to the market within 0.05 of zero on every path", f"largest gap {v4['E9']['identity']['largest_error']:.1e}, exposure {P['mkt_exp_lo']} to {P['mkt_exp_hi']}", verdict(v4["E9"]["met"])],
    ]
    P["table1"] = table(["", "Fixed before the run", "What the run found", "Verdict"], rows1, "verdicts")

    # --- Table 2: the twelve paths under the mandate ---
    rows2 = []
    for p in C.MONTHLY_PATHS:
        s = paths[C.PATH_LABELS[p]]
        cap = F.CAP_LABELS[p[2]]
        rows2.append([f"{LBL[p[0]]}, cap {cap}{'' if cap == 'none' else ' per cent'}",
                      num(BP * s["realised_te"]), num(BP * s["forecast_te_mean"]),
                      "{:.2f} ({:.2f} to {:.2f})".format(s["bias"], *intervals.loc[C.PATH_LABELS[p], ["interval_low", "interval_high"]]),
                      num(BP * s["active_return_gross"], 1, True), num(BP * s["active_return_gross_se"]),
                      num(BP * s["active_return_net_at_10bp"], 1, True), num(100 * s["turnover_total_per_year"]),
                      num(100 * s["turnover_chosen_per_year"]), f"{s['held_mean']:.0f}"])
    # Added 19 November 2026: the paths whose interval reaches into the band, listed from the table of twelve.
    band_hi = C.E6_BIAS_RANGE[1]
    reach = [p for p in C.MONTHLY_PATHS if intervals.loc[C.PATH_LABELS[p], "interval_low"] <= band_hi]
    assert sorted(reach, key=lambda p: p[2]) == [("ledoit_wolf", C.MANDATE, 0.005), ("ledoit_wolf", C.MANDATE, 0.01)], \
        "the text names the two paths whose interval reaches into the band"
    assert len(C.MONTHLY_PATHS) == 12 and all(intervals.loc[C.PATH_LABELS[p], "interval_low"] > band_hi
                                              for p in C.MONTHLY_PATHS if p not in reach)
    a_, b_ = (intervals.loc[C.PATH_LABELS[p]] for p in sorted(reach, key=lambda p: p[2]))
    P["band_reach"] = (f"Of the twelve paths under the mandate, two have an interval that reaches into the band, both on "
                       f"the Ledoit-Wolf estimator: at the 0.5 per cent cap the bias statistic is {a_['bias']:.2f}, with an "
                       f"interval of {a_['interval_low']:.2f} to {a_['interval_high']:.2f}, and at the 1 per cent cap "
                       f"{b_['bias']:.2f}, from {b_['interval_low']:.2f} to {b_['interval_high']:.2f}, so on those two paths "
                       f"the excess over {band_hi} is not statistically significant at the 5 per cent level, and the data do "
                       f"not rule out a forecast inside the band. On the other ten it is statistically significant at that "
                       f"level, each interval lying wholly above {band_hi}.")
    P["table2"] = table(["Path", "Realised tracking error", "Mean forecast", "Bias statistic (95 per cent interval)", "Active return, gross",
                         "Newey-West standard error", "Active return, net at 10 bp", "Total turnover, % a year",
                         "Chosen turnover, % a year", "Names held, mean"], rows2)

    # --- Table 3: the turnover's parts ---
    held = paths[C.PATH_LABELS[("ledoit_wolf", "C0", None)]]
    rows3 = [["The benchmark held (Ledoit-Wolf under C0)"] + [num(100 * held[f"turnover_{k}_per_year"], 2) for k in ("total", "membership", "reweighting", "rebalancing", "overlap")]]
    for e in EST:
        s = paths[MAIN[e]]
        rows3.append([f"{LBL[e]}, mandate at 2 per cent"] + [num(100 * s[f"turnover_{k}_per_year"], 2) for k in ("total", "membership", "reweighting", "rebalancing", "overlap")])
    P["table3"] = table(["Path", "Total", "Forced", "Reweighting", "Rebalancing", "Overlap"], rows3)
    P["held_total"], P["held_forced"], P["held_rew"] = (num(100 * held[f"turnover_{k}_per_year"], 2) for k in ("total", "membership", "reweighting"))

    # --- Table 4: the three terms by path ---
    rows4 = []
    for e in EST:
        t = brinson[MAIN[e]]["total"]
        cells = []
        for k in C.ATTRIBUTION_TERMS:
            cells += [num(BP * t[f"{k}_per_year"], 1, True), num(BP * t[f"{k}_per_year_se"])]
        cells += [num(BP * t[f"{k}_mean_abs_monthly"], 2) for k in C.ATTRIBUTION_TERMS]
        rows4.append([LBL[e] + ("" if e in S3 else ", no verdict")] + cells + [str(not_held[e])])
    P["table4"] = table(["Path", "Allocation, bp a year", "Standard error", "Selection, bp a year", "Standard error",
                         "Interaction, bp a year", "Standard error", "Allocation, mean absolute, bp a month",
                         "Selection, mean absolute, bp a month", "Interaction, mean absolute, bp a month",
                         "Sector-months not held"], rows4)

    # --- Table 5: the factor attribution by path ---
    FACT = list(C.FACTOR_ATTRIBUTION_FACTORS)
    FNAME = {"Mkt-RF": "Market", "SMB": "Size", "HML": "Value", "RMW": "Profitability", "CMA": "Investment", "Mom": "Momentum"}
    rows5 = []
    for f in FACT:
        rows5.append([f"{FNAME[f]}, contribution"] + [f"{num(BP * factor[MAIN[e]][f'contribution {f} (per year)'], 1, True)} ({num(BP * factor[MAIN[e]][f'contribution {f} (standard error)'])})" for e in EST])
    rows5.append(["The six together"] + [f"{num(BP * factor[MAIN[e]]['factor total (per year)'], 1, True)} ({num(BP * factor[MAIN[e]]['factor total (standard error)'])})" for e in EST])
    rows5.append(["Residual"] + [f"{num(BP * factor[MAIN[e]]['residual (per year)'], 1, True)} ({num(BP * factor[MAIN[e]]['residual (standard error)'])})" for e in EST])
    rows5.append(["Active return"] + [f"{num(BP * factor[MAIN[e]]['active return (per year)'], 1, True)} ({num(BP * factor[MAIN[e]]['active return (standard error)'])})" for e in EST])
    rows5.append(["Factor share, per cent"] + [num(100 * factor[MAIN[e]]["factor share"], 1) for e in EST])
    for f in FACT:
        rows5.append([f"{FNAME[f]}, mean active exposure"] + [num(factor[MAIN[e]][f"exposure {f}"], 4, True) for e in EST])
    P["table5"] = table(["", *[LBL[e] for e in EST]], rows5)

    # --- Table 6: the data layer ---
    mq, vq = answer["momentum"]["quintiles"], answer["variance"]["quintiles"]
    rows6 = [
        ["Benchmark against CRSP's value-weighted return, with dividends", f"{P['recon_te']} bp a year", f"{P['recon_first_w']} to {P['recon_last_w']}"],
        ["The same, price-only", f"{P['recon_te_px']} bp a year", f"{P['recon_first_w']} to {P['recon_last_w']}"],
        ["The same, with dividends, over the mandate's window", f"{P['recon_eval_te']} bp a year", f"{P['first_w']} to {P['last_w']}"],
        ["Benchmark's price return against the index's own, no verdict, years before 1957", f"{P['gap1']} bp a year", "1926 to 1956"],
        ["The same, 1957 to 1989", f"{P['gap2']} bp a year", "1957 to 1989"],
        ["The same, 1990 to 1998", f"{P['gap3']} bp a year", "1990 to 1998"],
        ["The same, 1999 to 2004", f"{P['gap4']} bp a year", "1999 to 2004"],
        ["The same, from 2005, when the index weights by float", f"{P['gap5']} bp a year", "2005 to 2024"],
        ["Member-months in the universe", P["member_months"], f"{P['recon_first_w']} to {P['recon_last_w']}"],
        ["Momentum spread against French's: correlation", f"{mq['correlation']:.3f}", f"{mq['months']:,} months from {words(mq['first_month'])}"],
        ["Momentum: mean monthly difference (standard error)", f"{num(BP * mq['mean_difference'])} ({num(BP * mq['standard_error'])}) bp", ""],
        ["Variance spread against French's: correlation", f"{vq['correlation']:.3f}", f"{vq['months']:,} months from {words(vq['first_month'])}"],
        ["Variance: mean monthly difference (standard error)", f"{num(BP * vq['mean_difference'])} ({num(BP * vq['standard_error'])}) bp", ""],
        ["Months with the sample condition number at least 10 times the Ledoit-Wolf one", f"{P['e4_months']} of {P['months']}", "all from 2017"],
        ["The same with one share class for each company, no verdict", f"{P['e4_one_class']} of {P['months']}", "all from 2020 to 2022"],
    ]
    P["table6"] = table(["Measure", "Figure", "Period"], rows6)

    # --- Table 7: conditioning by year ---
    rows7 = []
    for r in cov_year.itertuples():
        rows7.append([str(r.year), f"{r.condition_sample_median:,.0f}", f"{r.condition_ledoit_wolf_median:,.0f}",
                      f"{r.condition_factor_median:,.0f}", f"{r.condition_pca_median:,.0f}",
                      num(r.condition_sample_median / r.condition_ledoit_wolf_median, 1)])
    P["table7"] = table(["Year", "Sample", "Ledoit-Wolf", "Factor", "Principal-component", "Sample over Ledoit-Wolf"], rows7)

    # --- the figures' own tables (C11) ---
    P["t_f1"] = table(["Estimator", "Mean forecast, bp a year", "Realised, bp a year", "Bias statistic", "Months"],
                      [[r.label, num(r.forecast), num(r.realised), f"{r.bias:.2f}", str(r.bias_months)] for r in t1.itertuples()])
    P["t_f2"] = table(["Year", "Months", "Realised, bp a year", "Mean forecast, bp a year"],
                      [[str(int(r.year)), str(int(r.months)), num(r.realised), num(r.forecast)] for r in t2.itertuples()])
    P["t_f3"] = table(["Estimator and cap", "Chosen turnover, % a year", "Total turnover, % a year", "Realised, bp a year", "Mean forecast, bp a year"],
                      [[f"{r.label}, {r.cap}", num(r.chosen_turnover), num(r.total_turnover), num(r.realised), num(r.forecast)] for r in fr.itertuples()])
    P["t_f4"] = table(["Sector", "Allocation", "Standard error", "Selection", "Standard error", "Interaction", "Standard error"],
                      [[r.label, num(r.allocation, 2, True), num(r.allocation_se, 2), num(r.selection, 2, True), num(r.selection_se, 2),
                        num(r.interaction, 2, True), num(r.interaction_se, 2)] for r in t4.itertuples()])

    # --- Other projects (C14): the README's block, word for word, its links on the short words ---
    block = readme.split("## Other projects", 1)[1].strip().split("\n## ", 1)[0]
    paras = []
    for para in [x.strip() for x in block.split("\n\n") if x.strip()]:
        para = html.escape(para, quote=False)
        para = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", para)
        para = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', para)
        paras.append(f"<p>{para}</p>")
    P["other_projects"] = "\n".join(paras)

    # --- the data the charts draw, with two decimals more than the page shows ---
    D = {
        "est": list(EST), "label": LBL, "short": SHORT, "colour": F.COLOURS,
        "f1": {r.estimator: {"forecast": round(r.forecast, 3), "realised": round(r.realised, 3), "bias": round(r.bias, 4)} for r in t1.itertuples()},
        "f2": {"year": [int(y) for y in t2["year"]], "months": [int(m) for m in t2["months"]],
               "realised": [round(x, 3) for x in t2["realised"]], "forecast": [round(x, 3) for x in t2["forecast"]]},
        "f3": {e: {"cap": list(g["cap"]), "chosen": [round(x, 3) for x in g["chosen_turnover"]],
                   "realised": [round(x, 3) for x in g["realised"]], "forecast": [round(x, 3) for x in g["forecast"]]}
               for e, g in fr.groupby("estimator", sort=False)},
        "f4": {"sector": list(t4["label"]), **{k: [round(x, 4) for x in t4[k]] for k in ("allocation", "allocation_se", "selection", "selection_se")}},
    }

    # --- the four Key points of the dashboard; the first three as written on 22 November 2026 ---
    KEY_POINTS = [
        "Under the mandate the four covariance estimators realise 78.9 to 83.6 basis points a year of tracking error, and every estimator's forecast underestimates it. The bias statistic is 1.48 to 1.85, above the band of 0.8 to 1.25 fixed in advance, i.e., the realised active return is 48 to 85 per cent more volatile than forecast.",
        "Index changes require 2.99 per cent a year of the Ledoit-Wolf path's 26.26 per cent of one-way turnover, i.e., 11.4 per cent. Loosening the cap on chosen turnover from 0.5 per cent a month to none raises chosen turnover from 6.3 to 59.4 per cent a year on Ledoit-Wolf and from 6.2 to 14.9 on the factor estimator, and lowers the mean forecast on both. Neither estimator's realised tracking error changes by an amount that is statistically significant at the 5 per cent level on a paired test: +3.1 basis points a year on Ledoit-Wolf, with a 95 per cent interval of −3.6 to +9.8, and −3.7 on the factor estimator, from −18.9 to +11.4, where the comparisons could detect changes of 9.1 and 17.3 four times in five.",
        'Month by month, allocation is the largest of the three Brinson-Fachler terms on the three structured estimators, i.e., Ledoit-Wolf, factor and principal-component: 16.4 to 16.7 basis points a month in absolute value on average, against 10.6 to 13.8 for selection and 1.5 to 2.2 for interaction. Over the 306 months the mean active return is +0.2 to +13.7 basis points a year before costs, and on no path is it, or any of its three terms, statistically significant at the 5 per cent level: a mean of about 35 basis points a year in either direction would be needed. The six factors explain 2 to 13 per cent of its variance.',
        "The benchmark rebuilt from CRSP's list of index members reproduces CRSP's own value-weighted return of the same list with a tracking error of {{recon_te}} basis points a year from {{recon_years_first}} to {{recon_years_last}}. Its momentum and variance spreads have correlations of {{mom_corr}} and {{var_corr}} with Kenneth R. French's large-cap spreads.",
    ]
    for k, text in enumerate(KEY_POINTS, 1):
        P[f"kp{k}"] = re.sub(r"\{\{(\w+)\}\}", lambda m: P[m.group(1)], text)

    return P, D

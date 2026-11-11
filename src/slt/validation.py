"""
The data layer's checks against CRSP's own figures: the benchmark rebuilt from the universe, its comparison
with CRSP's return for the same index, and the counts and total values. The gates G1 to G4 and the
expectations E1 and E2 are judged here on the levels in constants.py.

The benchmark's return in month t is the start-of-month capitalisation-weighted return of the universe
members with a capitalisation and a return, delisting returns applied; members without either are left out
and the other weights rescaled, and both counts are reported. Comparisons with CRSP's series are summarised
(tracking error, mean difference, correlation, months outside a tolerance); no CRSP series by month and no
monthly difference from one is returned for committing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import constants as C


def benchmark(panel: pd.DataFrame) -> pd.DataFrame:
    """By month: the benchmark's return with dividends and price-only, the universe's count and summed
    capitalisation, and the members left out for a missing capitalisation or a missing return."""
    p = panel.copy()
    p["has_cap"] = p["cap"].notna()
    p["w_ret"] = np.where(p["has_cap"] & p["ret_total"].notna(), p["cap"], 0.0)
    p["w_retx"] = np.where(p["has_cap"] & p["retx_total"].notna(), p["cap"], 0.0)
    p["cap_ret"] = p["w_ret"] * p["ret_total"].fillna(0.0)
    p["cap_retx"] = p["w_retx"] * p["retx_total"].fillna(0.0)
    g = p.groupby("month")
    out = pd.DataFrame({
        "benchmark_return": g["cap_ret"].sum() / g["w_ret"].sum(),
        "benchmark_return_price_only": g["cap_retx"].sum() / g["w_retx"].sum(),
        "universe_count": g.size().astype("int64"),
        "universe_cap_sum": g["cap"].sum(min_count=1),
        "members_without_cap": (~p["has_cap"]).groupby(p["month"]).sum().astype("int64"),
        "members_without_return": p["ret_total"].isna().groupby(p["month"]).sum().astype("int64"),
    })
    out.index = pd.PeriodIndex(out.index, freq="M")
    out.index.name = "month"
    return out


# CRSP's index fields under the names the study uses for them. The frame stays in memory and is never committed.
CRSP_NAMES = {"vwretd": "crsp_return", "vwretx": "crsp_return_price_only", "sprtrn": "index_price_return",
              "totcnt": "crsp_count", "totval": "crsp_total_value"}


def crsp_by_month(index_file: pd.DataFrame) -> pd.DataFrame:
    """CRSP's monthly index file indexed by month, its fields renamed by CRSP_NAMES. It stays in memory and
    is never committed."""
    out = index_file.rename(columns=CRSP_NAMES)
    out.index = pd.PeriodIndex(pd.to_datetime(out["caldt"]), freq="M")
    out.index.name = "month"
    return out


def compare(ours: pd.Series, theirs: pd.Series, tolerance=None, relative: bool = False) -> dict:
    """Summary of one comparison over the months both series have: months, tracking error (standard deviation
    of the monthly difference, n minus 1, times the square root of 12), mean monthly difference, correlation,
    and, given a tolerance, the months outside it (on the ratio minus one when relative)."""
    both = pd.concat([ours.rename("a"), theirs.rename("b")], axis=1).dropna()
    diff = both["a"] - both["b"]
    out = {"months": int(len(both)),
           "tracking_error": float(diff.std(ddof=1) * np.sqrt(12)) if len(both) > 1 else None,
           "mean_difference": float(diff.mean()) if len(both) else None,
           "correlation": float(both["a"].corr(both["b"])) if len(both) > 1 else None}
    if tolerance is not None:
        gap = (both["a"] / both["b"] - 1).abs() if relative else diff.abs()
        out["tolerance"] = tolerance
        out["months_outside_tolerance"] = int((gap > tolerance).sum())
    return out


def compare_by_year(ours: pd.Series, theirs: pd.Series) -> pd.DataFrame:
    """Tracking error and mean difference by calendar year, for years with at least two months."""
    both = pd.concat([ours.rename("a"), theirs.rename("b")], axis=1).dropna()
    diff = (both["a"] - both["b"])
    g = diff.groupby(pd.PeriodIndex(both.index).year)
    out = pd.DataFrame({"months": g.size(), "tracking_error": g.std(ddof=1) * np.sqrt(12), "mean_difference": g.mean()})
    return out[out["months"] >= 2].rename_axis("year")


def judge_g1_e1(with_dividends: dict, price_only: dict) -> dict:
    """G1 on the tracking error against CRSP's return with dividends; E1 on both comparisons' tracking error
    and mean monthly difference."""
    def e1_part(c):
        return (c["tracking_error"] is not None and c["tracking_error"] <= C.E1_TE_MAX
                and abs(c["mean_difference"]) <= C.E1_MEAN_DIFF_MAX)
    return {
        "G1": {"tracking_error": with_dividends["tracking_error"], "level": C.G1_TE_MAX,
               "met": bool(with_dividends["tracking_error"] is not None and with_dividends["tracking_error"] <= C.G1_TE_MAX)},
        "E1": {"with_dividends": {"tracking_error": with_dividends["tracking_error"], "mean_difference": with_dividends["mean_difference"]},
               "price_only": {"tracking_error": price_only["tracking_error"], "mean_difference": price_only["mean_difference"]},
               "levels": {"tracking_error": C.E1_TE_MAX, "mean_difference": C.E1_MEAN_DIFF_MAX},
               "met": bool(e1_part(with_dividends) and e1_part(price_only))},
    }


def judge_e2(bench: pd.DataFrame, crsp: pd.DataFrame, first_month: str) -> dict:
    """E2 over the usable window: the universe at the start of month t against CRSP's count and total value
    on its row for month t-1, the same day."""
    b = bench[bench.index >= pd.Period(first_month, "M")]
    prev = crsp.reindex(b.index - 1)
    count_gap = (b["universe_count"] - prev["crsp_count"].values).abs()
    value_ratio = (b["universe_cap_sum"] / prev["crsp_total_value"].values - 1).abs()
    months = int(len(b))
    count_out = int((count_gap > C.E2_COUNT_WITHIN).sum())
    value_in = int((value_ratio <= C.E2_VALUE_REL_MAX).sum())
    value_share = value_in / months if months else 0.0
    return {"months": months, "months_count_outside": count_out, "count_level": C.E2_COUNT_WITHIN,
            "months_value_within": value_in, "share_value_within": round(value_share, 6),
            "value_level": C.E2_VALUE_REL_MAX, "share_level": C.E2_VALUE_MONTH_SHARE_MIN,
            "met": bool(months and count_out == 0 and value_share >= C.E2_VALUE_MONTH_SHARE_MIN)}


def e2_value_by_year(bench: pd.DataFrame, crsp: pd.DataFrame, first_month: str) -> pd.DataFrame:
    """E2's value comparison by year over the usable window: the mean relative difference of the summed
    capitalisation from CRSP's total value on the same day, and the months outside the tolerance."""
    b = bench[bench.index >= pd.Period(first_month, "M")]
    prev = crsp.reindex(b.index - 1)
    gap = pd.Series(b["universe_cap_sum"].values / prev["crsp_total_value"].values - 1, index=b.index)
    year = b.index.year
    out = pd.DataFrame({
        "months": gap.groupby(year).size(),
        "mean_relative_value_difference": gap.groupby(year).mean().round(6),
        "months_value_outside": (~(gap.abs() <= C.E2_VALUE_REL_MAX)).groupby(year).sum().astype("int64"),
    })
    return out.rename_axis("year").reset_index()


def verdict_line(name: str, met: bool, figures: str) -> str:
    """One verdict, in the same words whether met or not."""
    return f"{name}: {'met' if met else 'not met'}. {figures}"

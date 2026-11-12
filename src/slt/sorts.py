"""
French's answer key (expectation E3): value-weighted sorts of the universe on momentum and on variance, and their
top-minus-bottom spreads against the same spreads within French's largest size quintile.

Each month t the members with a signal known at the end of month t-1 and a start-of-month capitalisation are
ranked on the signal and cut into quantiles with equal numbers of names, to within one, ties broken by PERMNO.
A quantile's return in month t is the capitalisation-weighted return of its members with a return, delisting
returns applied, as the benchmark's is. The spread is the highest quantile's return minus the lowest's.

Momentum is the compounded return from month t-12 to month t-2 (French's prior 2-12 return). Variance is the
variance of daily returns over the 60 trading days ending on the last trading day of t-1. The conditions for
entering each sort are French's, as constants.py records them. Nothing here reads or writes a file, and nothing
by security leaves the functions that return values for committing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import constants as C


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def monthly_wide(monthly: pd.DataFrame, columns=("prc", "ret")) -> dict:
    """The monthly stock file as one matrix per column, months by PERMNO, the price in absolute value."""
    s = monthly.assign(month=pd.PeriodIndex(monthly["date"], freq="M")).drop_duplicates(["permno", "month"], keep="last")
    months = pd.period_range(s["month"].min(), s["month"].max(), freq="M")
    out = {}
    for c in columns:
        w = s.pivot(index="month", columns="permno", values=c).reindex(months)
        out[c] = w.abs() if c == "prc" else w
    return out


def momentum(ulong: pd.DataFrame, wide: dict, first: str = C.MOMENTUM_FIRST_MONTH) -> pd.DataFrame:
    """For each member-month from `first`: the momentum signal and the conditions it meets.

    Columns: month, permno, signal, has_price_t13, has_return_t2, skipped (the number of months from t-12 to
    t-3 with no return, skipped in the compounding). The signal is missing where a condition fails.
    """
    t0, t1 = C.MOMENTUM_MONTHS                      # 12, 2
    prc, ret = wide["prc"], wide["ret"]
    log = np.log1p(ret.fillna(0.0).to_numpy())       # a return of -100% gives minus infinity, and a signal of -1
    missing = ret.isna().to_numpy()
    months = ret.index
    u = ulong[ulong["month"] >= pd.Period(first, "M")]
    row = months.get_indexer(pd.PeriodIndex(u["month"], freq="M"))
    col = ret.columns.get_indexer(u["permno"])
    inside = (row - (t0 + 1) >= 0) & (col >= 0)
    r, c = np.where(inside, row, 0), np.where(inside, col, 0)
    total = np.zeros(len(u))
    skipped = np.zeros(len(u), dtype=int)
    for k in range(t1, t0 + 1):                       # months t-2 to t-12
        total = total + log[r - k, c]
        if k > t1:
            skipped = skipped + missing[r - k, c]
    has13 = inside & (np.nan_to_num(prc.to_numpy()[r - (t0 + 1), c]) > 0)
    has2 = inside & ~missing[r - t1, c]
    signal = np.where(has13 & has2, np.expm1(total), np.nan)
    return pd.DataFrame({"month": u["month"].values, "permno": u["permno"].values, "signal": signal,
                         "has_price_t13": has13, "has_return_t2": has2, "skipped": np.where(inside, skipped, 0)})


def variance(ulong: pd.DataFrame, daily: pd.DataFrame, ends: pd.Series, first: str = C.VARIANCE_SORT_START,
             days: int = C.VARIANCE_DAYS, minimum: int = C.VARIANCE_MIN_DAYS) -> pd.DataFrame:
    """For each member-month from `first`: the variance of its daily returns over the `days` trading days ending
    on the last trading day of month t-1, and the number of returns it rests on. The signal is missing below
    `minimum` returns.

    `daily` is a matrix of daily returns, trading days by PERMNO, held in memory only; `ends` gives the last
    trading day of each month (universe.month_ends).
    """
    u = ulong[ulong["month"] >= pd.Period(first, "M")]
    cal = daily.index.values
    X = daily.to_numpy()
    out = []
    for month, g in u.groupby("month"):
        last = ends.loc[pd.Period(month, "M") - 1]
        e = int(np.searchsorted(cal, np.datetime64(last), side="right")) - 1
        col = daily.columns.get_indexer(g["permno"])
        block = X[max(e - days + 1, 0): e + 1][:, np.where(col >= 0, col, 0)]
        block = np.where(col >= 0, block, np.nan)
        n = np.isfinite(block).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.nansum(block, axis=0) / n
            v = np.nansum((block - mean) ** 2, axis=0) / (n - 1)
        v = np.where(n >= minimum, v, np.nan)
        out.append(pd.DataFrame({"month": month, "permno": g["permno"].values, "signal": v, "returns": n}))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Sorts and spreads
# ---------------------------------------------------------------------------

def assign(frame: pd.DataFrame, q: int) -> np.ndarray:
    """Quantile numbers 0 to q-1 within each month, equal numbers of names to within one, ties by PERMNO.

    `frame` holds month, permno and signal for the members that enter the sort, and nothing else matters."""
    f = frame.reset_index(drop=True)
    order = np.lexsort((f["permno"].to_numpy(), f["signal"].to_numpy(), f["month"].astype(str).to_numpy()))
    out = np.empty(len(f), dtype=int)
    months = f["month"].to_numpy()[order]
    starts = np.flatnonzero(np.r_[True, months[1:] != months[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for a, b in zip(starts, ends):
        n = b - a
        out[order[a:b]] = (np.arange(n) * q) // n
    return out


def entering(signals: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """The member-months that enter a sort: a signal and a start-of-month capitalisation. With the month's
    return (delisting applied) and the capitalisation from the benchmark's panel."""
    m = signals.merge(panel[["month", "permno", "cap", "ret_total"]], on=["month", "permno"], how="left")
    return m[m["signal"].notna() & m["cap"].notna()].reset_index(drop=True)


def spread(sorted_frame: pd.DataFrame, q: int) -> pd.Series:
    """The top-minus-bottom spread by month: the capitalisation-weighted return of the highest quantile minus the
    lowest's, each over its members with a return."""
    f = sorted_frame.assign(quantile=assign(sorted_frame, q))
    f = f[f["quantile"].isin([0, q - 1]) & f["ret_total"].notna()]
    f = f.assign(wr=f["cap"] * f["ret_total"])
    g = f.groupby(["month", "quantile"])[["wr", "cap"]].sum()
    r = (g["wr"] / g["cap"]).unstack("quantile")
    out = (r[q - 1] - r[0]).rename("spread")
    out.index = pd.PeriodIndex(out.index, freq="M")
    return out


def names_by_quantile(sorted_frame: pd.DataFrame, q: int) -> pd.Series:
    """The mean number of names in a quantile by year, for the comparison with French's counts."""
    n = sorted_frame.groupby("month").size() / q
    return n.groupby(pd.PeriodIndex(n.index, freq="M").year).mean()


# ---------------------------------------------------------------------------
# The comparison with French
# ---------------------------------------------------------------------------

def newey_west_se(x, lags: int = C.E3_NEWEY_WEST_LAGS_REPORTED) -> float:
    """The Newey-West standard error of the mean of x, Bartlett weights over `lags` lags."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float("nan")
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(np.sqrt(s / n))


def compare(ours: pd.Series, french: pd.Series) -> dict:
    """Over the months both spreads exist: the months, the correlation, the mean of each, the mean difference
    (ours minus French's) and its standard error (constants.E3_STANDARD_ERROR)."""
    both = pd.concat([ours.rename("ours"), french.rename("french")], axis=1).dropna()
    n = len(both)
    d = both["ours"] - both["french"]
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"months": int(n),
            "first_month": str(both.index.min()) if n else None, "last_month": str(both.index.max()) if n else None,
            "correlation": float(both["ours"].corr(both["french"])) if n > 1 else float("nan"),
            "mean_universe_spread": float(both["ours"].mean()), "mean_french_spread": float(both["french"].mean()),
            "mean_difference": float(d.mean()), "standard_error": se,
            "standard_error_newey_west": newey_west_se(d.to_numpy()),
            "difference_in_standard_errors": float(d.mean() / se) if se and np.isfinite(se) else float("nan")}


def judge_e3(result: dict) -> dict:
    """E3 for one sort: correlation at least E3_CORRELATION_MIN and the mean difference within E3_MEAN_DIFF_SE
    standard errors of zero."""
    corr_met = bool(result["correlation"] >= C.E3_CORRELATION_MIN)
    diff_met = bool(abs(result["mean_difference"]) <= C.E3_MEAN_DIFF_SE * result["standard_error"])
    return {"correlation_met": corr_met, "mean_difference_met": diff_met, "met": corr_met and diff_met}


def compare_by_year(ours: pd.Series, french: pd.Series) -> pd.DataFrame:
    """compare() for each calendar year."""
    both = pd.concat([ours.rename("ours"), french.rename("french")], axis=1).dropna()
    rows = []
    for year, g in both.groupby(both.index.year):
        r = compare(g["ours"], g["french"])
        rows.append({"year": int(year), "months": r["months"], "correlation": r["correlation"],
                     "mean_universe_spread": r["mean_universe_spread"], "mean_french_spread": r["mean_french_spread"],
                     "mean_difference": r["mean_difference"], "standard_error": r["standard_error"]})
    return pd.DataFrame(rows)


def counts_by_year(signals: pd.DataFrame, entered: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Member-months by year: in the universe, left out by reason, and sorted, for one sort."""
    s = signals.assign(year=pd.PeriodIndex(signals["month"], freq="M").year)
    e = entered.assign(year=pd.PeriodIndex(entered["month"], freq="M").year)
    out = pd.DataFrame({"member_months": s.groupby("year").size()})
    if kind == "momentum":
        out["left_out_no_price_t13"] = (~s["has_price_t13"]).groupby(s["year"]).sum()
        out["left_out_no_return_t2"] = (s["has_price_t13"] & ~s["has_return_t2"]).groupby(s["year"]).sum()
        out["sorted_with_a_skipped_month"] = (e["skipped"] > 0).groupby(e["year"]).sum()
    else:
        out["left_out_too_few_daily_returns"] = s["signal"].isna().groupby(s["year"]).sum()
        out["sorted_with_fewer_than_60_returns"] = (e["returns"] < C.VARIANCE_DAYS).groupby(e["year"]).sum()
    out["left_out_no_capitalisation"] = out["member_months"] - out.filter(like="left_out").sum(axis=1) - e.groupby("year").size()
    out["sorted"] = e.groupby("year").size()
    out["months"] = s.groupby("year")["month"].nunique()
    return out.fillna(0).astype("int64").rename_axis("year").reset_index()

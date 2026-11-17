"""
The Brinson-Fachler attribution of a month's active return by sector (Brinson and Fachler 1985).

For one month: w and b the portfolio's and the benchmark's weights by stock, r the stocks' returns. Both sides are
taken over the names with a return, each rescaled to one over them, as the committed active return is
(constants.ATTRIBUTION_WEIGHTS). For each of the twelve sectors s, w_s and b_s are the sector weights, R_b,s the
benchmark sector's return and R_b the benchmark's, and the three terms are

    allocation   (w_s - b_s)(R_b,s - R_b)
    selection    b_s (R_p,s - R_b,s)
    interaction  (w_s - b_s)(R_p,s - R_b,s)

with R_p,s the portfolio sector's return. A sector that the portfolio does not hold (none of its stocks with a
return has a weight above constants.SECTOR_HELD_MIN_WEIGHT) has its allocation as defined, a selection of zero, and
as interaction what is left of its effect: the sum over its stocks of weight times return, less w_s R_b,s. A sector
with no member with a return contributes zero to all three. Summed over the sectors, the three terms equal the
portfolio's return less the benchmark's in every month; the identity error is recorded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import constants as C

PERIODS_PER_YEAR = 12


def month_terms(w: pd.Series, b: pd.Series, r: pd.Series, sector: pd.Series,
                sectors: tuple = C.ATTRIBUTION_SECTORS) -> tuple:
    """The month's terms by sector and the month's totals.

    w, b, r and sector are indexed by PERMNO over the month's members. Returns (frame, totals): frame has one row
    per sector with allocation, selection, interaction and held; totals holds the three sums over sectors, the
    portfolio's and the benchmark's returns, the identity error and the count of members with no return."""
    r = r.reindex(w.index)
    has = r.notna().to_numpy()
    raw = w.to_numpy(dtype=float)[has]
    wr = raw / raw.sum()
    br = b.reindex(w.index).to_numpy(dtype=float)[has]
    br = br / br.sum()
    rr = r.to_numpy(dtype=float)[has]
    sec = sector.reindex(w.index).astype(str).to_numpy()[has]
    Rp, Rb = float(wr @ rr), float(br @ rr)
    rows = []
    for s in sectors:
        idx = sec == s
        if not idx.any():
            rows.append({"sector": s, "allocation": 0.0, "selection": 0.0, "interaction": 0.0, "held": False, "members": 0})
            continue
        ws, bs = float(wr[idx].sum()), float(br[idx].sum())
        Rbs = float(br[idx] @ rr[idx]) / bs
        allocation = (ws - bs) * (Rbs - Rb)
        held = bool((raw[idx] > C.SECTOR_HELD_MIN_WEIGHT).any())
        if held:
            Rps = float(wr[idx] @ rr[idx]) / ws
            selection = bs * (Rps - Rbs)
            interaction = (ws - bs) * (Rps - Rbs)
        else:
            selection = 0.0
            interaction = float(wr[idx] @ rr[idx]) - ws * Rbs
        rows.append({"sector": s, "allocation": allocation, "selection": selection, "interaction": interaction,
                     "held": held, "members": int(idx.sum())})
    frame = pd.DataFrame(rows).set_index("sector")
    totals = {t: float(frame[t].sum()) for t in C.ATTRIBUTION_TERMS}
    totals.update({"portfolio_return": Rp, "benchmark_return": Rb, "active_return": Rp - Rb,
                   "identity_error": sum(totals[t] for t in C.ATTRIBUTION_TERMS) - (Rp - Rb),
                   "members_without_return": int((~has).sum()),
                   "sectors_not_held": int(((~frame["held"]) & (frame["members"] > 0)).sum())})
    return frame, totals


def by_year(terms: pd.DataFrame) -> pd.DataFrame:
    """The terms by sector and calendar year, each summed over the year's months, with a row 'total' per year for the
    sums over sectors. `terms` has one row per month and sector, columns month, sector and the three terms."""
    t = terms.assign(year=pd.PeriodIndex(terms["month"], freq="M").year)
    per_sector = t.groupby(["year", "sector"])[list(C.ATTRIBUTION_TERMS)].sum().reset_index()
    total = t.groupby("year")[list(C.ATTRIBUTION_TERMS)].sum().reset_index().assign(sector="total")
    months = t.groupby("year")["month"].nunique().rename("months").reset_index()
    out = pd.concat([per_sector, total], ignore_index=True).merge(months, on="year")
    order = {s: i for i, s in enumerate(C.ATTRIBUTION_SECTORS + ("total",))}
    return out.sort_values(["year", "sector"], key=lambda c: c.map(order) if c.name == "sector" else c).reset_index(drop=True)


def summary(terms: pd.DataFrame, totals: pd.DataFrame, newey_west_se) -> dict:
    """Over the window: for each sector and for the totals, each term's annual average (12 times the monthly mean)
    and its Newey-West standard error; the mean absolute monthly effect of each total (E8's second clause); the
    largest identity error; the counts."""
    out = {"months": int(len(totals)), "sectors": {}, "total": {}}
    for s, g in terms.groupby("sector", sort=False):
        out["sectors"][s] = {}
        for t in C.ATTRIBUTION_TERMS:
            x = g.sort_values("month")[t].to_numpy(dtype=float)
            out["sectors"][s][f"{t}_per_year"] = PERIODS_PER_YEAR * float(x.mean())
            out["sectors"][s][f"{t}_per_year_se"] = PERIODS_PER_YEAR * newey_west_se(x, C.NEWEY_WEST_LAGS)
    for t in C.ATTRIBUTION_TERMS:
        x = totals[t].to_numpy(dtype=float)
        out["total"][f"{t}_per_year"] = PERIODS_PER_YEAR * float(x.mean())
        out["total"][f"{t}_per_year_se"] = PERIODS_PER_YEAR * newey_west_se(x, C.NEWEY_WEST_LAGS)
        out["total"][f"{t}_mean_abs_monthly"] = float(np.abs(x).mean())
    out["largest_identity_error"] = float(totals["identity_error"].abs().max())
    out["sector_months_not_held"] = int(totals["sectors_not_held"].sum())
    return out


def judge_e8(summaries: dict) -> dict:
    """E8: the identity within E8_IDENTITY_MAX_ABS_ERROR on every path; and, as one verdict over E8_JUDGED_ON, the
    allocation's mean absolute monthly effect above the selection's with the interaction's the smallest, on each."""
    largest = max(s["largest_identity_error"] for s in summaries.values())
    order = {}
    for est, s in summaries.items():
        a, se, i = (s["total"][f"{t}_mean_abs_monthly"] for t in C.ATTRIBUTION_TERMS)
        order[est] = {"allocation": a, "selection": se, "interaction": i, "holds": bool(a > se and i < se and i < a)}
    holds = [e for e in C.E8_JUDGED_ON if order[e]["holds"]]
    return {"identity": {"largest_error": largest, "met": bool(largest <= C.E8_IDENTITY_MAX_ABS_ERROR)},
            "second_clause": {"met": len(holds) == len(C.E8_JUDGED_ON), "holds_for": holds,
                              "fails_for": [e for e in C.E8_JUDGED_ON if e not in holds],
                              "by_estimator": order, "no_verdict": [e for e in summaries if e not in C.E8_JUDGED_ON]},
            "met": bool(largest <= C.E8_IDENTITY_MAX_ABS_ERROR and len(holds) == len(C.E8_JUDGED_ON))}

"""
Both attributions of the mandate's four paths at the 2% cap (constants.ATTRIBUTION_PATHS), for notebook 11.

The weights by security are read from the script's records in outputs/evaluation/, on the machine that holds the
extraction. Each month, for each path: the Brinson-Fachler terms by sector (brinson.month_terms), and the inputs of
the source repository's factor attribution, copied unchanged in source/attribution.py: the weights over the names
with a return, each side rescaled to one (constants.ATTRIBUTION_WEIGHTS), and each member's six loadings, the factor
estimator's of the month (estimators.month_estimates, model rows included). French's monthly factor returns come
from french.monthly_factors. Everything here is held in memory; the notebook commits aggregates by year and over
the window only (constants.ATTRIBUTION_COMMITTED).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import brinson as B
from . import constants as C
from . import estimators as E
from . import evaluation as V
from .source import attribution as SA


def load_weights(out_dir, label: str) -> dict:
    """A path's target weights by month, as the script wrote them: {month: Series by PERMNO}."""
    frame = pd.read_parquet(os.path.join(out_dir, label + "_weights.parquet"))
    return {m: g.set_index("permno")["weight"] for m, g in frame.groupby("month")}


def rescaled(w: pd.Series, r: pd.Series) -> pd.Series:
    """The weights over the names with a return, rescaled to one; zero for a name without one."""
    has = r.reindex(w.index).notna()
    out = w.where(has, 0.0)
    return out / out.sum()


def run(inp: dict, out_dir, months, factors: pd.DataFrame, paths: tuple = C.ATTRIBUTION_PATHS,
        progress: bool = False) -> dict:
    """Both attributions, month by month, for every path. Returns, in memory: the Brinson-Fachler terms by month and
    sector and their monthly totals for each path; the factor attribution of each path (the copy's attribute()); the
    largest gap between the active return recomputed here and the one the script recorded; the counts."""
    labels = {p: C.PATH_LABELS[p] for p in paths}
    weights = {p: load_weights(out_dir, labels[p]) for p in paths}
    records = V.load_records(out_dir)
    terms = {p: [] for p in paths}
    totals = {p: [] for p in paths}
    w_rows = {p: {} for p in paths}
    b_rows, betas = {}, {}
    gap = 0.0
    for k, month in enumerate(months):
        key = str(pd.Period(month, "M"))
        m, r = V.month_view(inp["members"], month)
        b = pd.Series(m.b, index=m.permnos)
        sector = pd.Series(m.sector, index=m.permnos)
        frame = inp["members"]
        frame = frame[(frame["month"] == pd.Period(month, "M")) & frame["cap"].notna()].drop(columns="permco")
        est = E.month_estimates(month, inp["excess"], inp["factors"], inp["bench"], frame)
        order = est["matrices"]["order"]
        betas[key] = pd.DataFrame(est["matrices"]["loadings"], index=order, columns=list(C.FACTOR_ATTRIBUTION_FACTORS)).reindex(m.permnos)
        b_rows[key] = rescaled(b, r)
        for p in paths:
            w = weights[p][key].reindex(m.permnos).fillna(0.0)
            by_sector, tot = B.month_terms(w, b, r, sector)
            terms[p].append(by_sector.reset_index().assign(month=key))
            totals[p].append(dict(tot, month=key))
            w_rows[p][key] = rescaled(w, r)
            gap = max(gap, abs(tot["active_return"] - float(records[labels[p]].loc[key, "active_return"])))
        if progress and (k + 1) % 12 == 0:
            print(f"{key}: done", flush=True)
    out = {"labels": labels, "terms": {}, "totals": {}, "factor": {}, "active_return_gap": gap}
    keys = [str(pd.Period(mm, "M")) for mm in months]
    columns = sorted(set().union(*[set(s.index) for s in b_rows.values()]))
    bench = pd.DataFrame({k_: b_rows[k_] for k_ in keys}).T.reindex(columns=columns).fillna(0.0)
    beta = {k_: betas[k_].reindex(columns).fillna(0.0).to_numpy() for k_ in keys}
    f = factors.copy()
    f.index = [str(i) for i in f.index]
    for p in paths:
        out["terms"][p] = pd.concat(terms[p], ignore_index=True)
        out["totals"][p] = pd.DataFrame(totals[p]).set_index("month")
        wp = pd.DataFrame({k_: w_rows[p][k_] for k_ in keys}).T.reindex(columns=columns).fillna(0.0)
        active = records[labels[p]].loc[keys, "active_return"]
        out["factor"][p] = SA.attribute(wp, bench, beta, f.loc[keys], active)
    return out


def tilt_cap(inp: dict, out_dir, months, paths: tuple = C.ATTRIBUTION_PATHS) -> pd.DataFrame:
    """For each path, over the months: the stock-months of the tilted sectors (constants.TILT_SECTORS) among the
    month's members with a capitalisation, and how many of them sit at the tilt's cap, half the benchmark weight
    (constants.TILT_MAX_SHARE_OF_BENCHMARK), within constants.OPT_TOL, below it and above it, from the recorded
    weights. The benchmark weight is the capitalisation share among those members (evaluation.month_view)."""
    labels = {p: C.PATH_LABELS[p] for p in paths}
    weights = {p: load_weights(out_dir, labels[p]) for p in paths}
    counts = {p: {"tilted_stock_months": 0, "at_cap": 0, "below": 0, "above": 0} for p in paths}
    for month in months:
        key = str(pd.Period(month, "M"))
        m, _ = V.month_view(inp["members"], month)
        tilted = np.isin(m.sector, C.TILT_SECTORS)
        cap = C.TILT_MAX_SHARE_OF_BENCHMARK * m.b
        for p in paths:
            w = weights[p][key].reindex(m.permnos).fillna(0.0).to_numpy()
            c = counts[p]
            c["tilted_stock_months"] += int(tilted.sum())
            c["at_cap"] += int((tilted & (np.abs(w - cap) <= C.OPT_TOL)).sum())
            c["below"] += int((tilted & (w < cap - C.OPT_TOL)).sum())
            c["above"] += int((tilted & (w > cap + C.OPT_TOL)).sum())
    return pd.DataFrame([{"path": labels[p], "months": len(months), **counts[p]} for p in paths])


def factor_by_year(att: pd.DataFrame) -> pd.DataFrame:
    """The factor attribution by calendar year: the mean active exposures and the contributions per year (the copy's
    by_period), one row per year."""
    years = pd.Series(pd.PeriodIndex(att.index, freq="M").year, index=att.index, name="year")
    contrib = SA.by_period(att, years)
    expo = att[[f"exposure {n}" for n in C.FACTOR_ATTRIBUTION_FACTORS]].groupby(years).mean()
    largest = att["identity error"].abs().groupby(years).max().rename("largest identity error")
    return pd.concat([expo, contrib, largest], axis=1).reset_index()


def factor_newey_west(att: pd.DataFrame, newey_west_se, names: tuple = C.FACTOR_ATTRIBUTION_FACTORS) -> pd.DataFrame:
    """Added 19 November 2026: each factor's mean contribution a year, the six-factor total and the residual, each
    with its plain standard error, as the copy's summarise() gives it, and its Newey-West standard error at
    constants.NEWEY_WEST_LAGS, both times 12, and the mean in Newey-West standard errors. One row per term."""
    rows = []
    for col in [f"contribution {n}" for n in names] + ["factor total", "residual"]:
        x = att[col].to_numpy(dtype=float)
        mean = 12 * float(x.mean())
        plain = 12 * float(x.std(ddof=1)) / np.sqrt(len(x))
        nw = 12 * newey_west_se(x, C.NEWEY_WEST_LAGS)
        rows.append({"term": col.replace("contribution ", ""), "months": int(len(x)), "per_year": mean,
                     "standard_error": plain, "newey_west_se": nw, "in_newey_west_se": mean / nw if nw > 0 else float("nan"),
                     "beyond": bool(abs(mean) > C.FACTOR_BEYOND_SE * nw)})
    return pd.DataFrame(rows)


def factor_beyond_count(table: pd.DataFrame, names: tuple = C.FACTOR_ATTRIBUTION_FACTORS) -> dict:
    """Added 19 November 2026: of the factor contributions in `table` (one or more paths), how many lie beyond
    constants.FACTOR_BEYOND_SE Newey-West standard errors from zero, against the count constants.FACTOR_CHANCE_LEVEL
    would give on average if the tests were independent."""
    f = table[table["term"].isin(names)]
    return {"contributions": int(len(f)), "beyond": int(f["beyond"].sum()),
            "by_chance": C.FACTOR_CHANCE_LEVEL * len(f), "level": C.FACTOR_CHANCE_LEVEL, "standard_errors": C.FACTOR_BEYOND_SE}


def judge_e9(summaries: dict) -> dict:
    """E9: contributions plus residual equal the active return within E9_IDENTITY_MAX_ABS_ERROR in every month of
    every path; and, as one verdict over E9_JUDGED_ON, the mean active exposure to the market factor within
    E9_MARKET_EXPOSURE_MAX of zero on each."""
    market = C.FACTOR_ATTRIBUTION_FACTORS[0]
    largest = max(s["largest identity error"] for s in summaries.values())
    expo = {e: summaries[e][f"exposure {market}"] for e in C.E9_JUDGED_ON}
    holds = [e for e in C.E9_JUDGED_ON if abs(expo[e]) <= C.E9_MARKET_EXPOSURE_MAX]
    return {"identity": {"largest_error": largest, "met": bool(largest <= C.E9_IDENTITY_MAX_ABS_ERROR)},
            "market_exposure": {"by_estimator": expo, "met": len(holds) == len(C.E9_JUDGED_ON), "holds_for": holds,
                                "fails_for": [e for e in C.E9_JUDGED_ON if e not in holds]},
            "met": bool(largest <= C.E9_IDENTITY_MAX_ABS_ERROR and len(holds) == len(C.E9_JUDGED_ON))}

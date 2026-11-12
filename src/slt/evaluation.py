"""
The rolling evaluation: one portfolio path per specification, month by month over the evaluation window.

Adapted from src/bp/evaluation.py of the source repository at commit 5414f14 (constants.SOURCE_ADAPTED_FILES).
What differs from the source: the universe changes from month to month, so a portfolio is a series indexed by
PERMNO and not a vector over fixed industries; the previous portfolio the cap is measured from is the reference
portfolio of constants.py, section 7 (mandate.reference_portfolio), and the month's turnover is split into
membership, reweighting and rebalancing (constants.TURNOVER_PARTS); a portfolio's return is taken over the names with a return, as the benchmark's
is; and each month's record carries the counts the committed tables need and nothing by security. The measures
(realised tracking error, the bias statistic, the worst cumulative fall) are the source's.

A path starts as the benchmark. In every month the covariance matrix of the path's estimator is the assembled
matrix of notebook 07 (estimators.month_estimates), scaled to a month; the optimiser solves for the weights under
the path's constraint set; the portfolio and the benchmark earn the month's returns.
"""
from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np
import pandas as pd

from . import constants as C
from . import estimators as E
from . import extraction as X
from . import layer as L
from . import mandate as M
from . import sorts as Q
from . import universe as U

PERIODS_PER_YEAR = 12


# ---------------------------------------------------------------------------
# Inputs, held in memory
# ---------------------------------------------------------------------------

def inputs(root, progress: bool = False) -> dict:
    """Everything the paths read, from July 1996 (the first window) to December 2024: the members with their
    capitalisation, return, sector, sub-industry and company; the daily excess returns; French's daily factors
    and risk-free rate; the benchmark's daily excess return."""
    from . import french as FR
    universe = L.load_universe(root)
    data = L.load_panel(root, universe)
    panel = data["panel"]
    first = pd.Period(C.EVAL_START, "M") - 12 * C.COV_DAILY_WINDOW_YEARS
    p = panel[panel["month"] >= first]
    sectors = L.load_sectors(root, p)
    names, _ = X.read_table(root, "crsp_msenames")
    company = U.companies(universe["ulong"][universe["ulong"]["month"] >= first], names)
    members = (sectors["classified"][["month", "permno", "sector", "gsubind"]]
               .merge(p[["month", "permno", "cap", "ret_total"]], on=["month", "permno"])
               .merge(company[["month", "permno", "permco"]], on=["month", "permno"], how="left"))
    factors, rf, records = FR.daily_factors()
    years = [y for y in L.daily_years() if y >= first.year]
    daily, _ = L.daily_matrix(root, members["permno"].unique(), years, progress=progress)
    excess = daily.sub(rf.reindex(daily.index), axis=0)
    bench = E.benchmark_daily_excess(daily, rf, members)
    del daily
    return {"members": members, "excess": excess, "factors": factors.reindex(excess.index), "bench": bench,
            "french": records}


def month_view(members: pd.DataFrame, month) -> tuple:
    """The month's members with a capitalisation, in PERMNO order, as the optimiser's Month, and their returns."""
    g = members[(members["month"] == pd.Period(month, "M")) & members["cap"].notna()].sort_values("permno")
    cap = g["cap"].to_numpy(dtype=float)
    company = np.where(g["permco"].notna(), g["permco"].fillna(0).to_numpy(), -g["permno"].to_numpy()).astype(float)
    m = M.Month(permnos=g["permno"].to_numpy(), b=cap / cap.sum(), sector=g["sector"].astype(str).to_numpy(),
                subindustry=g["gsubind"].fillna("").astype(str).to_numpy(), company=company)
    r = pd.Series(g["ret_total"].to_numpy(dtype=float), index=g["permno"].to_numpy())
    return m, r


def month_matrices(month, inp: dict, m: M.Month) -> dict:
    """The four assembled matrices of the month (notebook 07's), in the month's PERMNO order, scaled to a month."""
    frame = inp["members"]
    frame = frame[(frame["month"] == pd.Period(month, "M")) & frame["cap"].notna()].drop(columns="permco")
    res = E.month_estimates(month, inp["excess"], inp["factors"], inp["bench"], frame)   # without permco: no share-class check here
    E.check_positive_semidefinite(res)
    order = np.asarray(res["matrices"]["order"])
    pos = pd.Series(np.arange(len(order)), index=order).reindex(m.permnos).to_numpy()
    return {name: M_[np.ix_(pos, pos)] * C.DAILY_TO_MONTHLY_SCALE for name, M_ in res["matrices"]["assembled"].items()}


# ---------------------------------------------------------------------------
# One path, one month at a time
# ---------------------------------------------------------------------------

class PathState:
    """A path's state between months: last month's target weights and returns, and its records so far."""

    def __init__(self, path: tuple):
        self.path = path
        self.label = C.PATH_LABELS[path]
        self.w_prev = None            # pd.Series by PERMNO
        self.r_prev = None
        self.b_prev = None            # last month's benchmark weights, pd.Series by PERMNO
        self.records = []
        self.weights = {}             # month -> pd.Series, kept on the machine that holds the extraction

    @property
    def months_done(self) -> int:
        return len(self.records)


def step(state: PathState, month, Sigma: np.ndarray, m: M.Month, r: pd.Series) -> dict:
    """Advance the path by one month and return the month's record."""
    estimator, cs, cap = state.path
    constraints = tuple(c for c in C.CONSTRAINT_SETS[cs] if c != "turnover_cap" or cap is not None)   # a cap of None: no cap
    b = pd.Series(m.b, index=m.permnos)
    if state.w_prev is None:
        w_drift, missing_drift = b.copy(), 0
        w_ref = b.copy()
        restored = b.copy()
    else:
        w_drift, missing_drift = M.drift(state.w_prev, state.r_prev)
        w_ref = M.reference_portfolio(w_drift, m.permnos, m.b)
        restored = M.restored_portfolio(state.w_prev, state.b_prev, m.permnos, m.b)
    sol = M.solve(Sigma, m, constraints, w_ref.reindex(m.permnos).to_numpy(), cap=cap)
    w = pd.Series(sol.weights, index=m.permnos)
    parts = M.decompose(w_drift, w_ref, restored, w)
    active = w - b
    rp, rb = M.portfolio_return(w, r), M.portfolio_return(b, r)
    rec = {"month": str(pd.Period(month, "M")), "members": int(len(b)), "forecast_te": sol.forecast_te,
           "active_return": rp - rb, "cell_ok": M.signed_cell(active, r),
           "turnover_total": parts["total"], "turnover_membership": parts["membership"],
           "turnover_reweighting": parts["reweighting"], "turnover_rebalancing": parts["rebalancing"],
           "turnover_overlap": parts["overlap"], "turnover_chosen": parts["chosen"],
           "cap_used": np.nan if sol.cap_used is None else sol.cap_used, "relaxed": bool(sol.relaxed),
           "cap_binds": bool(sol.binding.get("turnover_cap", 0)), "status": sol.status,
           "held": int((sol.weights > C.OPT_TOL).sum()), "max_abs_active": float(active.abs().max()),
           "names_off_benchmark": int((active.abs() > C.OPT_TOL).sum()),
           "missing_return_in_drift": missing_drift, "missing_return_held": int((r.reindex(w.index).isna() & (w > C.OPT_TOL)).sum()),
           "company_scope_mismatches": M.company_scope_mismatches(m, constraints)}
    for k, v in sol.binding.items():
        rec["binding_" + k] = v
    state.records.append(rec)
    state.weights[rec["month"]] = w
    state.w_prev, state.r_prev, state.b_prev = w, r, b
    return rec


def run_group(estimator: str, inp: dict, out_dir, months=None, progress: bool = True) -> list:
    """Every path of one estimator, in lockstep month by month, so that each month's matrices are built once.
    After each month every path's state is written to out_dir, and a run that stops resumes from it."""
    out_dir = os.fspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    paths = [p for p in C.PATHS if p[0] == estimator]
    states = []
    for p in paths:
        f = os.path.join(out_dir, C.PATH_LABELS[p] + ".pkl")
        if os.path.exists(f):
            with open(f, "rb") as fh:
                states.append(pickle.load(fh))
        else:
            states.append(PathState(p))
    months = list(pd.period_range(C.EVAL_START, C.EVAL_END, freq="M")) if months is None else list(months)
    t0 = time.time()
    for k, month in enumerate(months):
        due = [s for s in states if s.months_done == k]
        if not due:
            continue
        m, r = month_view(inp["members"], month)
        Sigma = month_matrices(month, inp, m)[estimator]
        for s in due:
            step(s, month, Sigma, m, r)
            tmp = os.path.join(out_dir, s.label + ".pkl.tmp")
            with open(tmp, "wb") as fh:
                pickle.dump(s, fh)
            os.replace(tmp, os.path.join(out_dir, s.label + ".pkl"))
        if progress and (month.month == 12 or k == len(months) - 1):
            print(f"{estimator} {month} {time.time() - t0:.0f} s", flush=True)
    for s in states:
        pd.DataFrame(s.records).to_csv(os.path.join(out_dir, s.label + "_records.csv"), index=False)
        pd.concat({k: v for k, v in s.weights.items()}, names=["month", "permno"]).rename("weight").reset_index() \
          .to_parquet(os.path.join(out_dir, s.label + "_weights.parquet"), index=False)
    return states


def load_records(out_dir) -> dict:
    """Every path's records by month, from the script's outputs, keyed by the path's label."""
    out = {}
    for p in C.PATHS:
        f = os.path.join(os.fspath(out_dir), C.PATH_LABELS[p] + "_records.csv")
        if os.path.exists(f):
            out[C.PATH_LABELS[p]] = pd.read_csv(f).set_index("month")
    return out


# ---------------------------------------------------------------------------
# Measures (the source's, with the stock-level additions)
# ---------------------------------------------------------------------------

def realised_tracking_error(active) -> float:
    """The standard deviation of the monthly active returns, times the square root of 12."""
    return float(np.std(np.asarray(active, dtype=float), ddof=1) * np.sqrt(PERIODS_PER_YEAR))


def bias_statistic(active, forecast_te) -> float:
    """sd(active return / forecast monthly standard deviation) over the months whose forecast tracking error is
    above constants.BIAS_FORECAST_MIN; nan when fewer than two are, as under the sets that hold the benchmark."""
    f = np.asarray(forecast_te, dtype=float)
    a = np.asarray(active, dtype=float)
    keep = f > C.BIAS_FORECAST_MIN
    if keep.sum() < 2:
        return float("nan")
    return float(np.std(a[keep] / (f[keep] / np.sqrt(PERIODS_PER_YEAR)), ddof=1))


def jsonable(x):
    """A record ready for JSON: NumPy scalars as Python numbers, nan and infinities as None."""
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def worst_cumulative_fall(active: pd.Series) -> dict:
    """The largest fall of the cumulative active return (the running sum of the monthly active returns, from zero
    before the first month) from an earlier high: its size, the months of the high and the low, and the months between."""
    c = active.cumsum()
    high = np.maximum.accumulate(np.concatenate([[0.0], c.to_numpy()]))[1:]
    fall = pd.Series(high - c.to_numpy(), index=c.index)
    if fall.max() <= 0:
        return {"fall": 0.0, "peak": None, "trough": None, "months": 0}
    trough = fall.idxmax()
    before = c.loc[:trough]
    peak = before.idxmax() if before.max() > 0 else "start"
    months = int(c.index.get_loc(trough) - (c.index.get_loc(peak) if peak != "start" else -1))
    return {"fall": float(fall.max()), "peak": peak, "trough": trough, "months": months}


def summarise(rec: pd.DataFrame) -> dict:
    """A path's measures over the window, in return units per year where they are returns or risks."""
    a = rec["active_return"]
    te = realised_tracking_error(a)
    gross = PERIODS_PER_YEAR * float(a.mean())
    total = rec["turnover_total"]
    out = {"months": int(len(rec)), "realised_te": te, "forecast_te_mean": float(rec["forecast_te"].mean()),
           "bias": bias_statistic(a, rec["forecast_te"]),
           "bias_months": int((rec["forecast_te"] > C.BIAS_FORECAST_MIN).sum()),
           "bias_months_left_out": int((rec["forecast_te"] <= C.BIAS_FORECAST_MIN).sum()),
           "active_return_gross": gross, "active_return_gross_se": PERIODS_PER_YEAR * Q.newey_west_se(a.to_numpy(), C.NEWEY_WEST_LAGS),
           "turnover_total_per_year": PERIODS_PER_YEAR * float(total.mean()),
           **{f"turnover_{k}_per_year": PERIODS_PER_YEAR * float(rec[f"turnover_{k}"].mean()) for k in C.TURNOVER_FIGURES[1:]},
           "turnover_chosen_per_year": PERIODS_PER_YEAR * float(rec["turnover_chosen"].mean()),
           "cap_binding_share": float(rec["cap_binds"].mean()), "relaxed_months": int(rec["relaxed"].sum()),
           "held_mean": float(rec["held"].mean()), "held_least": int(rec["held"].min()),
           "months_off_benchmark": int((rec["max_abs_active"] > C.OPT_TOL).sum()),
           "largest_abs_active_weight": float(rec["max_abs_active"].max()),
           "missing_return_in_drift": int(rec["missing_return_in_drift"].sum()),
           "missing_return_held": int(rec["missing_return_held"].sum()),
           "company_scope_mismatches": int(rec["company_scope_mismatches"].sum()),
           "statuses_not_optimal": int((rec["status"] != "optimal").sum()),
           "worst_cumulative_fall": {k: (str(v) if v is not None else None) if k in ("peak", "trough") else v
                                     for k, v in worst_cumulative_fall(a).items()}}
    for cost in (C.COST_PER_UNIT_TURNOVER, C.COST_SENSITIVITY):
        bp = int(round(cost * 1e4))
        net = a - cost * total
        out[f"cost_per_year_at_{bp}bp"] = PERIODS_PER_YEAR * cost * float(total.mean())
        out[f"active_return_net_at_{bp}bp"] = PERIODS_PER_YEAR * float(net.mean())
        out[f"active_return_net_at_{bp}bp_se"] = PERIODS_PER_YEAR * Q.newey_west_se(net.to_numpy(), C.NEWEY_WEST_LAGS)
    return out


def by_year(rec: pd.DataFrame) -> pd.DataFrame:
    """A path's figures by calendar year: no weight, no name."""
    r = rec.assign(year=pd.PeriodIndex(rec.index, freq="M").year)
    g = r.groupby("year")
    out = pd.DataFrame({
        "months": g.size(),
        "realised_te": g["active_return"].apply(realised_tracking_error),
        "forecast_te_mean": g["forecast_te"].mean(),
        "active_return_mean": g["active_return"].mean(),
        "turnover_total": g["turnover_total"].sum(),
        "turnover_chosen": g["turnover_chosen"].sum(),
        "cap_binding_months": g["cap_binds"].sum().astype("int64"),
        "relaxed_months": g["relaxed"].sum().astype("int64"),
        "held_mean": g["held"].mean(),
        "months_off_benchmark": g["max_abs_active"].apply(lambda x: int((x > C.OPT_TOL).sum())),
        "largest_abs_active_weight": g["max_abs_active"].max(),
    })
    return out.reset_index()


def turnover_by_year(rec: pd.DataFrame) -> pd.DataFrame:
    """The turnover decomposition by calendar year: the total, the three parts and the overlap
    (constants.TURNOVER_FIGURES), each one-way and summed over the year's months."""
    r = rec.assign(year=pd.PeriodIndex(rec.index, freq="M").year)
    g = r.groupby("year")
    out = pd.DataFrame({"months": g.size()})
    for k in C.TURNOVER_FIGURES:
        out[f"turnover_{k}"] = g[f"turnover_{k}"].sum()
    return out.reset_index()


def active_returns_table(records: dict) -> tuple:
    """The table by month: the active return of each path under the mandate, a cell left empty where the rule for
    a signed sum fails. Returns the table and the count of empty cells by column."""
    cols, empty = {}, {}
    for p in C.MONTHLY_PATHS:
        rec = records[C.PATH_LABELS[p]]
        col = "active_return_" + C.PATH_LABELS[p]
        cols[col] = rec["active_return"].where(rec["cell_ok"].astype(bool))
        empty[col] = int((~rec["cell_ok"].astype(bool)).sum())
    table = pd.DataFrame(cols)
    table.index.name = "month"
    return table.reset_index(), empty


# ---------------------------------------------------------------------------
# The expectations of milestone 3
# ---------------------------------------------------------------------------

def label(est, cs=C.MANDATE, cap=C.MAIN_CAP):
    return C.PATH_LABELS[(est, cs, cap)]


def judge(summaries: dict) -> dict:
    """E4's second clause, E5, E6 and E7 from the paths' summaries."""
    s = summaries
    out = {}
    te_s, te_lw = s[label("sample")]["realised_te"], s[label("ledoit_wolf")]["realised_te"]
    out["E4_second_clause"] = {"sample_realised_te": te_s, "ledoit_wolf_realised_te": te_lw, "met": bool(te_s > te_lw)}
    for est in C.FRONTIER_ESTIMATORS:
        fc = [s[label(est, cap=c)]["forecast_te_mean"] for c in C.FRONTIER_CAPS]
        nonincreasing = all(fc[i + 1] <= fc[i] + C.OPT_TOL for i in range(len(fc) - 1))
        loose, tight = C.E5_REALISED_CAPS_COMPARED
        rl, rt = s[label(est, cap=loose)]["realised_te"], s[label(est, cap=tight)]["realised_te"]
        out[f"E5_{est}"] = {"caps": [None if c is None else c for c in C.FRONTIER_CAPS], "forecast_te_mean": fc,
                            "forecast_nonincreasing": bool(nonincreasing), "realised_te_looser": rl, "realised_te_tighter": rt,
                            "met": bool(nonincreasing and rl < rt)}
    lo, hi = C.E6_TE_RANGE
    blo, bhi = C.E6_BIAS_RANGE
    for est in ("ledoit_wolf", "factor", "pca"):
        x = s[label(est)]
        te_ok = lo <= x["realised_te"] <= hi
        bias_ok = blo <= x["bias"] <= bhi
        out[f"E6_{est}"] = {"realised_te": x["realised_te"], "bias": x["bias"], "bias_months": x["bias_months"], "te_in_range": bool(te_ok),
                            "bias_in_range": bool(bias_ok), "met": bool(te_ok and bias_ok)}
    x = s[label("ledoit_wolf")]
    share = x["turnover_membership_per_year"] / x["turnover_total_per_year"] if x["turnover_total_per_year"] > 0 else float("nan")
    out["E7"] = {"membership_share_of_total": share, "max": C.E7_FORCED_SHARE_MAX, "met": bool(share < C.E7_FORCED_SHARE_MAX)}
    return out


def provenance(threads: int) -> dict:
    import platform
    import cvxpy
    import clarabel
    return {"cvxpy": cvxpy.__version__, "clarabel": clarabel.__version__, **X.versions(),
            "python": platform.python_version(), "machine": platform.machine(), "linear_algebra_threads": threads}

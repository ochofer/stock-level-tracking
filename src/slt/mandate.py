"""
The mandate at the stock level: one month's constrained tracking-error minimiser, and the reference portfolio
the turnover cap is measured from.

The problem is the copied optimiser's (src/slt/source/optimiser.py, never edited): minimise the forecast tracking
error sqrt(12 a' Sigma a), a = w - b, under per-weight bounds, beta neutrality and the turnover cap, with the
smallest feasible turnover found first so that a capped problem with no solution is never posed. This module
builds that problem with the copy's own pieces and adds the constraints the copy does not take, all of them sums
over groups of weights (constants.py, section 7):

- the bound per company: the sum of the active weights of a company's share classes covered by the bound per
  stock lies within the bound;
- the sector band: each banded sector's active weight lies within SECTOR_BAND; in a set with the tilt the band
  covers the untilted sectors only, Unclassified included.

The turnover cap is measured from the reference portfolio (reference_portfolio below): last month's portfolio
after its returns, with the names that left sold, the entrants bought at their benchmark weight, and the
continuing names rescaled. Every returned portfolio is checked against every constraint, the group ones included,
whatever the solver reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd

from . import constants as C
from .source import optimiser as O

# The study's names for the constraints the copy knows, and the copy's names for them.
COPY_NAMES = {"long_only": "long_only", "stock_bound": "active_weight_bound", "turnover_cap": "turnover_cap",
              "tilt": "tilt", "exclusion": "exclusion", "beta_neutral": "beta_neutral"}
GROUP_CONSTRAINTS = ("company_bound", "sector_band")


@dataclass
class Month:
    """What the optimiser needs to know about one month's members, in one order."""
    permnos: np.ndarray
    b: np.ndarray                     # benchmark weights, summing to one
    sector: np.ndarray                # GICS sector code, or constants.UNCLASSIFIED
    subindustry: np.ndarray           # GICS sub-industry code, or "" when there is none
    company: np.ndarray               # CRSP's PERMCO, or minus the PERMNO when there is none

    @property
    def tilt_mask(self) -> np.ndarray:
        return np.isin(self.sector, C.TILT_SECTORS)

    @property
    def exclude_mask(self) -> np.ndarray:
        return np.isin(self.subindustry, list(C.EXCLUDED_SUBINDUSTRIES))


def covered(m: Month, constraints: tuple) -> np.ndarray:
    """The stocks the bound per stock covers in this set: all of them, less the tilted ones in a set with the tilt
    and the excluded ones in a set with the exclusion (the copy's bounds give way to both)."""
    out = np.ones(len(m.b), bool)
    if "tilt" in constraints:
        out &= ~m.tilt_mask
    if "exclusion" in constraints:
        out &= ~m.exclude_mask
    return out


def groups(m: Month, constraints: tuple, sector_band: float = C.SECTOR_BAND,
           stock_bound: float = C.STOCK_ACTIVE_BOUND) -> list:
    """The group constraints of a set: a list of (indices, limit, kind), each meaning |sum of the active weights
    over the indices| <= limit."""
    out = []
    if "company_bound" in constraints:
        cov = covered(m, constraints)
        comp = pd.Series(m.company[cov], index=np.flatnonzero(cov))
        for _, idx in comp.groupby(comp.values).groups.items():
            if len(idx) > 1:
                out.append((np.asarray(idx), stock_bound, "company_bound"))
    if "sector_band" in constraints:
        tilted = set(C.TILT_SECTORS) if "tilt" in constraints else set()
        for s in pd.unique(m.sector):
            if s in tilted:
                continue
            out.append((np.flatnonzero(m.sector == s), sector_band, "sector_band"))
    return out


def company_scope_mismatches(m: Month, constraints: tuple) -> int:
    """Companies whose classes differ in sector or in the bound per stock's scope this month (none is expected)."""
    cov = covered(m, constraints)
    frame = pd.DataFrame({"company": m.company, "sector": m.sector, "covered": cov})
    g = frame.groupby("company")
    multi = g.size() > 1
    differ = (g["sector"].nunique() > 1) | (g["covered"].nunique() > 1)
    return int((multi & differ).sum())


def _with_groups(prob: cp.Problem, w: cp.Variable, b: np.ndarray, grp: list) -> cp.Problem:
    extra = []
    for idx, limit, _ in grp:
        s = cp.sum(w[idx]) - float(b[idx].sum())
        extra += [s <= limit, s >= -limit]
    return cp.Problem(prob.objective, list(prob.constraints) + extra) if extra else prob


def check_groups(x: np.ndarray, b: np.ndarray, grp: list, tol: float = C.OPT_TOL) -> None:
    for idx, limit, kind in grp:
        a = float(x[idx].sum() - b[idx].sum())
        if abs(a) > limit + tol:
            raise O.OptimiserError(f"{kind}: an active weight of {a:.6f} against a limit of {limit}")


@dataclass
class Result:
    weights: np.ndarray
    forecast_te: float
    turnover_chosen: float            # one-way, from the reference portfolio
    cap_used: float | None
    relaxed: bool
    status: str
    binding: dict = field(default_factory=dict)


def solve(Sigma: np.ndarray, m: Month, constraints: tuple, w_ref: np.ndarray, cap: float | None = None,
          stock_bound: float = C.STOCK_ACTIVE_BOUND, sector_band: float = C.SECTOR_BAND,
          tilt_share: float = C.TILT_MAX_SHARE_OF_BENCHMARK, solver: str = C.OPT_SOLVER, tol: float = C.OPT_TOL) -> Result:
    """The month's portfolio under the named set, the copy's problem with the group constraints added.

    `w_ref` is the reference portfolio over the month's members; the cap applies to the one-way turnover from it
    when "turnover_cap" is in the set, at `cap` (constants.MAIN_CAP when None is not meant: a set with the cap
    needs a number). If the cap makes the month infeasible it is raised to the smallest feasible turnover and the
    month is marked relaxed, as the copy does.
    """
    b = np.asarray(m.b, dtype=float)
    N = len(b)
    copy_set = tuple(COPY_NAMES[c] for c in constraints if c in COPY_NAMES)
    tilt_mask, exclude_mask = m.tilt_mask, m.exclude_mask
    if "exclusion" in constraints and not exclude_mask.any():
        exclude_mask = np.zeros(N, bool)          # a month with no member in an excluded sub-industry: nothing to exclude
    lower, upper = O.bounds(b, copy_set, tilt_mask, exclude_mask, stock_bound, tilt_share)
    if (lower > upper + tol).any():
        raise O.OptimiserError("a weight's lower bound exceeds its upper bound")
    grp = groups(m, constraints, sector_band, stock_bound)
    Ls = [O.cholesky_factor(np.asarray(Sigma, dtype=float))]
    betas = [O.benchmark_betas(Sigma, b)] if "beta_neutral" in constraints else None
    w0 = np.asarray(w_ref, dtype=float)
    use_cap = cap if "turnover_cap" in constraints else None
    if "turnover_cap" in constraints and cap is None:
        raise O.OptimiserError("a set with the turnover cap needs a cap")
    relaxed = False
    options = C.OPT_SOLVER_OPTIONS
    if use_cap is not None:
        w_t, prob_t = O._build(Ls, b, lower, upper, w0, None, objective="turnover", betas=betas)
        prob_t = _with_groups(prob_t, w_t, b, grp)
        O._run(prob_t, solver, options)
        if prob_t.status not in ("optimal", "optimal_inaccurate"):
            raise O.OptimiserError(f"infeasible without the turnover cap: {prob_t.status}")
        xt = O._clean(w_t.value)
        O.check(xt, b, lower, upper, w0, None, tol)
        check_groups(xt, b, grp, tol)
        if float(prob_t.value) > use_cap + 10 * tol:
            use_cap = float(prob_t.value) * (1.0 + 1e-6) + tol
            relaxed = True
    w, prob = O._build(Ls, b, lower, upper, w0 if use_cap is not None else None, use_cap, betas=betas)
    prob = _with_groups(prob, w, b, grp)
    O._run(prob, solver, options)
    x = O._clean(w.value)
    if prob.status != "optimal":
        raise O.OptimiserError(f"solver status {prob.status} on a problem known to be feasible")
    O.check(x, b, lower, upper, w0 if use_cap is not None else None, use_cap, tol)
    check_groups(x, b, grp, tol)
    a = x - b
    for beta in betas or []:
        if abs(beta @ a) > 10 * tol:
            raise O.OptimiserError(f"the portfolio's beta differs from the benchmark's by {beta @ a:.2e}")
    cov = covered(m, constraints)
    binding = {}
    if "long_only" in constraints:
        binding["long_only"] = int(((x <= tol) & ~(exclude_mask if "exclusion" in constraints else np.zeros(N, bool))).sum())
    if "stock_bound" in constraints:
        binding["stock_bound"] = int((cov & (np.abs(np.abs(a) - stock_bound) <= tol)).sum())
    if "tilt" in constraints:
        binding["tilt"] = int((tilt_mask & (np.abs(x - tilt_share * b) <= tol)).sum())
    for kind in GROUP_CONSTRAINTS:
        if kind in constraints:
            binding[kind] = int(sum(1 for idx, limit, k in grp if k == kind and abs(abs(a[idx].sum()) - limit) <= tol))
    chosen = O.one_way_turnover(x, w0)
    if use_cap is not None:
        binding["turnover_cap"] = int(abs(chosen - use_cap) <= 10 * tol)
    return Result(weights=x, forecast_te=O.forecast_tracking_error(x, b, Sigma), turnover_chosen=chosen,
                  cap_used=use_cap, relaxed=relaxed, status=prob.status, binding=binding)


# ---------------------------------------------------------------------------
# The reference portfolio and the turnover decomposition
# ---------------------------------------------------------------------------

def drift(w: pd.Series, r: pd.Series) -> tuple:
    """Last month's weights after last month's returns, delisting returns applied; a held name with no return
    drifts at constants.MISSING_RETURN_IN_DRIFT. Returns the drifted weights and the count of held names with no return."""
    rr = r.reindex(w.index)
    missing = int((rr.isna() & (w.abs() > C.OPT_TOL)).sum())
    grown = w * (1.0 + rr.fillna(C.MISSING_RETURN_IN_DRIFT))
    return grown / grown.sum(), missing


def reference_portfolio(w_prev: pd.Series, permnos: np.ndarray, b: np.ndarray) -> pd.Series:
    """Over this month's members: the names in `w_prev` that are still members keep their weights, rescaled; every
    entrant is bought at its benchmark weight; names that left are not in the result (sold to zero)."""
    b = pd.Series(np.asarray(b, dtype=float), index=permnos)
    cont = w_prev.reindex(permnos).dropna()
    entrants = b.index.difference(cont.index)
    out = pd.Series(0.0, index=permnos)
    out[entrants] = b[entrants]
    room = 1.0 - float(b[entrants].sum())
    if cont.sum() > 0:
        out[cont.index] = cont / cont.sum() * room
    else:
        out = b.copy()
    return out


def restored_portfolio(w_prev: pd.Series, b_prev: pd.Series, permnos: np.ndarray, b: np.ndarray) -> pd.Series:
    """Over this month's members: each continuing name at last month's target weight times the ratio of its benchmark
    weights this month and last, so that its weight relative to its benchmark weight is kept; every entrant at its
    benchmark weight; rescaled to one. The end of the reweighting part of constants.TURNOVER_PARTS."""
    b = pd.Series(np.asarray(b, dtype=float), index=permnos)
    cont = w_prev.reindex(permnos).dropna().index
    out = pd.Series(0.0, index=permnos)
    entrants = b.index.difference(cont)
    out[entrants] = b[entrants]
    ratio = b[cont] / b_prev.reindex(cont)
    kept = w_prev[cont] * ratio
    room = 1.0 - float(b[entrants].sum())
    if kept.sum() > 0:
        out[cont] = kept / kept.sum() * room
    else:
        out = b.copy()
    return out


def one_way(a: pd.Series, c: pd.Series) -> float:
    """One-way turnover between two portfolios over the union of their names."""
    idx = a.index.union(c.index)
    return float(0.5 * (a.reindex(idx, fill_value=0.0) - c.reindex(idx, fill_value=0.0)).abs().sum())


def decompose(w_drift: pd.Series, w_ref: pd.Series, w_restored: pd.Series, w_new: pd.Series) -> dict:
    """The month's one-way turnover and its three measured parts (constants.TURNOVER_PARTS): membership from the
    drifted portfolio to the reference, reweighting from the reference to the restored portfolio, rebalancing from
    the restored portfolio to the new weights. Their sum is the total plus the overlap, which the triangle inequality
    keeps at zero or above. Also the chosen turnover, from the reference to the new weights, which the cap bounds."""
    total = one_way(w_new, w_drift)
    membership = one_way(w_ref, w_drift)
    reweighting = one_way(w_restored, w_ref)
    rebalancing = one_way(w_new, w_restored)
    return {"total": total, "membership": membership, "reweighting": reweighting, "rebalancing": rebalancing,
            "overlap": membership + reweighting + rebalancing - total, "chosen": one_way(w_new, w_ref)}


def portfolio_return(w: pd.Series, r: pd.Series) -> float:
    """The month's return over the names with a return, the weights rescaled, as the benchmark's is."""
    rr = r.reindex(w.index)
    has = rr.notna()
    return float((w[has] * rr[has]).sum() / w[has].sum())


def signed_cell(active: pd.Series, r: pd.Series) -> bool:
    """The cell rule on a signed sum: at least SIGNED_CELL_MIN_STOCKS stocks with an active weight above OPT_TOL,
    and no stock's absolute contribution |w - b| |r| more than SIGNED_CELL_MAX_SHARE of their sum."""
    a = active[active.abs() > C.OPT_TOL]
    if len(a) < C.SIGNED_CELL_MIN_STOCKS:
        return False
    contrib = (a * r.reindex(a.index).fillna(0.0)).abs()
    total = contrib.sum()
    return bool(total == 0 or contrib.max() <= C.SIGNED_CELL_MAX_SHARE * total)

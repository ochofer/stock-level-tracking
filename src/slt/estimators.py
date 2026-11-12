"""
The four covariance estimators on the universe, month by month (notebook 07, expectation E4's first clause).

For month t the window is the trading days of the 36 calendar months before it, and the data are the daily
excess returns of the members at the start of month t over French's daily risk-free rate. A member is complete
when it lacks a return on at most COMPLETE_WINDOW_MISSING_MAX days of the window (the bar falls if more than
COMPLETE_DAYS_LEFT_OUT_MAX days would be left out). The copied estimators of src/slt/source/covariance.py run on
the complete members, on the days on which all of them have a return. Every other member takes a row from a
model fitted on its own days, or, below FACTOR_ROW_OWN_MIN returns, from its sector's complete members. Each of
the four matrices has one construction, the same in every month, positive semi-definite by construction, and its
block for the complete members is what the copied function returns for them (constants.py, section 7).

Only scalars by month leave this module: condition numbers, the shrinkage intensity, the smallest eigenvalue's
check and the counts. No matrix and no member's row is kept or written.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import constants as C
from .source import covariance as K


# ---------------------------------------------------------------------------
# Inputs held in memory
# ---------------------------------------------------------------------------

def benchmark_daily_excess(daily: pd.DataFrame, rf: pd.Series, members: pd.DataFrame) -> pd.Series:
    """The benchmark's daily excess return: each day, the returns of that month's members weighted by their
    start-of-month capitalisations, rescaled over the members with a return that day, less the risk-free rate.

    `members` holds month, permno and cap (the benchmark's panel); members without a capitalisation have no weight.
    """
    out = pd.Series(np.nan, index=daily.index)
    day_month = pd.PeriodIndex(daily.index, freq="M")
    m = members[members["cap"].notna()]
    values = daily.to_numpy()
    for month, g in m.groupby("month"):
        rows = np.flatnonzero(day_month == pd.Period(month, "M"))
        if len(rows) == 0:
            continue
        cols = daily.columns.get_indexer(g["permno"])
        keep = cols >= 0
        R = values[rows][:, cols[keep]]
        w = g["cap"].to_numpy()[keep]
        has = np.isfinite(R)
        out.iloc[rows] = np.nansum(R * w, axis=1) / np.where(has, w, 0.0).sum(axis=1)
    return out - rf.reindex(daily.index)


def window_days(calendar: pd.DatetimeIndex, month) -> pd.DatetimeIndex:
    """The trading days of the 36 calendar months before `month`, the copied daily_window's bounds."""
    month = pd.Period(month, "M")
    start = (month - 12 * C.COV_DAILY_WINDOW_YEARS).asfreq("D", "start").to_timestamp()
    end = (month - 1).asfreq("D", "end").to_timestamp()
    return calendar[(calendar >= start) & (calendar <= end)]


# ---------------------------------------------------------------------------
# Which members are complete
# ---------------------------------------------------------------------------

def complete_members(X: pd.DataFrame, bar: int = C.COMPLETE_WINDOW_MISSING_MAX,
                     ceiling: int = C.COMPLETE_DAYS_LEFT_OUT_MAX) -> dict:
    """The complete members of a window and the days left out.

    A member is complete when it lacks a return on at most `bar` days. The days left out are those on which a
    complete member lacks a return. If more than `ceiling` days would be left out, the bar falls by one day at a
    time, down to none, until they are within it. Returns the complete members, the days kept, the days left out
    and the bar used.
    """
    missing = X.isna()
    lacking = missing.sum(axis=0)
    for b in range(bar, -1, -1):
        complete = lacking.index[lacking <= b]
        out_days = missing[complete].any(axis=1)
        if int(out_days.sum()) <= ceiling:
            break
    return {"complete": list(complete), "kept": X.index[~out_days.to_numpy()], "left_out": int(out_days.sum()), "bar": b}


# ---------------------------------------------------------------------------
# Rows from a model, for the members that are not complete
# ---------------------------------------------------------------------------

def _regress(y: np.ndarray, Z: np.ndarray) -> tuple:
    """OLS of y on an intercept and Z: the slopes and the residuals."""
    A = np.column_stack([np.ones(len(y)), Z])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef[1:], y - A @ coef


def _group(member, sector: pd.Series, complete: list) -> list:
    """The complete members whose rows stand in for a member with too few returns: its sector's, when the
    sector has at least SECTOR_MIN_COMPLETE of them and is a GICS sector, the whole universe's otherwise."""
    s = sector.get(member, C.UNCLASSIFIED)
    if s != C.UNCLASSIFIED:
        same = [c for c in complete if sector.get(c) == s]
        if len(same) >= C.SECTOR_MIN_COMPLETE:
            return same
    return complete


def _stand_in(values: np.ndarray, sum_sq: np.ndarray, cap: np.ndarray, idx: np.ndarray) -> tuple:
    """Capitalisation-weighted mean of the rows of `values`, and the median of `sum_sq`, over `idx`."""
    w = np.nan_to_num(cap[idx])
    w = w / w.sum() if w.sum() > 0 else np.full(len(idx), 1.0 / len(idx))
    return w @ values[idx], np.median(sum_sq[idx])


# ---------------------------------------------------------------------------
# One month
# ---------------------------------------------------------------------------

def month_estimates(month, excess: pd.DataFrame, factors: pd.DataFrame, bench: pd.Series, members: pd.DataFrame,
                    scale: float = C.DAILY_TO_MONTHLY_SCALE) -> dict:
    """The four assembled matrices for `month`, their checks and the counts. Returns scalars only, and the
    matrices under the key "matrices" for the tests; the notebook drops that key at once.

    `excess` is daily excess returns, trading days by PERMNO; `factors` the six daily factors in the order of
    FACTOR_MODEL_FACTORS; `bench` the benchmark's daily excess return; `members` the month's members with permno,
    cap and sector.
    """
    days = window_days(excess.index, month)
    permnos = list(members["permno"])
    X = excess.loc[days].reindex(columns=permnos)
    # days without the factors or the benchmark are not usable by any estimator
    usable = factors.loc[days].notna().all(axis=1) & bench.loc[days].notna()
    X = X[usable.to_numpy()]
    sel = complete_members(X)
    A = sel["complete"]
    others = [p for p in permnos if p not in set(A)]
    kept = sel["kept"]
    XA = X.loc[kept, A].to_numpy()
    FA = factors.loc[kept].to_numpy()
    mA = bench.loc[kept].to_numpy()
    T, NA = XA.shape
    Kf = FA.shape[1]

    # the copied estimators on the complete members
    S_A = K.sample_cov(XA)
    LW_A, k = K.lw_scaled_identity(XA)
    FM_A, B_A = K.factor_cov(XA, FA)
    PC_A, _ = K.pca_cov(XA, C.PCA_N_COMPONENTS, market=mA)

    # the pieces the copies compute inside, recomputed in the same way for the assembly
    Fc = FA - FA.mean(axis=0)
    SigF1 = Fc.T @ Fc / (T - 1)                            # np.cov(F, ddof=1), as factor_cov
    B_chk, resA = _regress(XA, FA)
    beta_A, residM = _regress(XA, mA[:, None])
    beta_A = beta_A[0]
    var_m = float(np.var(mA, ddof=1))
    rss_A = (resA ** 2).sum(axis=0)                        # residual sums of squares on the six factors
    rssM_A = (residM ** 2).sum(axis=0)                     # on the benchmark alone

    cap = members.set_index("permno")["cap"]
    sector = members.set_index("permno")["sector"]
    capA = cap.reindex(A).to_numpy(dtype=float)
    nB = len(others)
    B_B = np.zeros((nB, Kf))
    beta_B = np.zeros(nB)
    rss_B, n_B, rssM_B = np.zeros(nB), np.zeros(nB), np.zeros(nB)
    stood_in = 0
    position = {p: i for i, p in enumerate(A)}
    for j, p in enumerate(others):
        own = X[p].notna().to_numpy()
        y = X[p].to_numpy()[own]
        n = len(y)
        n_B[j] = n
        if n >= C.FACTOR_ROW_OWN_MIN:
            B_B[j], e = _regress(y, factors.loc[X.index[own]].to_numpy())
            rss_B[j] = e @ e
            b, eM = _regress(y, bench.loc[X.index[own]].to_numpy()[:, None])
            beta_B[j] = b[0]
            rssM_B[j] = eM @ eM
        else:
            stood_in += 1
            idx = np.array([position[c] for c in _group(p, sector, A)])
            # the stand-in's residual sum of squares is the median of the complete members', over their T days,
            # so that dividing by T less the estimator's degrees of freedom gives the median residual variance
            B_B[j], rss_B[j] = _stand_in(B_chk.T, rss_A, capA, idx)
            b, rssM_B[j] = _stand_in(beta_A[:, None], rssM_A, capA, idx)
            beta_B[j] = float(b[0])
            n_B[j] = T

    def resid_var(rss, n, ddof):
        return rss / (n - ddof)

    # sample: denominator T - 1 throughout
    M_S = _assemble(S_A, B_chk.T, B_B, SigF1, resid_var(rss_B, n_B, 1))
    # Ledoit-Wolf: the sample assembly with denominator T, shrunk with the complete members' intensity and mean variance
    S_A_T = S_A * (T - 1) / T
    M_ST = _assemble(S_A_T, B_chk.T, B_B, SigF1 * (T - 1) / T, resid_var(rss_B, n_B, 0))
    m_mean = np.trace(S_A_T) / NA
    M_LW = (1 - k) * M_ST + k * m_mean * np.eye(NA + nB)
    # factor model: every member's loadings and residual variance on its own days, the copied denominator
    M_F = _assemble(FM_A, B_chk.T, B_B, SigF1, resid_var(rss_B, n_B, Kf + 1), sample_block=False)
    # PCA: a beta on the benchmark for every member, the four components on the complete members only
    M_P = _assemble_pca(PC_A, beta_A, beta_B, var_m, resid_var(rssM_B, n_B, 2))

    out = {"month": str(pd.Period(month, "M")), "members": len(permnos), "complete": NA, "model_rows": nB,
           "model_rows_under_252": int(sum(1 for p in others if X[p].notna().sum() < C.COV_MIN_DAILY_RETURNS)),
           "model_rows_under_60": stood_in, "window_days": int(len(days)), "days_without_factors": int((~usable).sum()),
           "days_left_out": sel["left_out"], "bar": sel["bar"], "bar_fell": sel["bar"] < C.COMPLETE_WINDOW_MISSING_MAX,
           "days_used": T, "lw_intensity": float(k)}
    mats = {"sample": M_S, "ledoit_wolf": M_LW, "factor": M_F, "pca": M_P}
    blocks = {"sample": S_A, "ledoit_wolf": LW_A, "factor": FM_A, "pca": PC_A}
    for name in C.COV_ESTIMATORS:
        M = K.scale_daily_to_monthly(mats[name], scale)
        vals = np.linalg.eigvalsh(M)
        out[f"smallest_eigenvalue_ratio_{name}"] = float(vals.min() / vals.max())
        out[f"condition_{name}"] = float(vals.max() / max(vals.min(), 1e-300))
        out[f"condition_complete_{name}"] = K.condition_number(blocks[name])
    out["ratio_sample_to_ledoit_wolf"] = out["condition_sample"] / out["condition_ledoit_wolf"]
    out["matrices"] = {"assembled": mats, "complete": blocks, "order": A + others,
                       "checks": {"factor_loadings_equal": float(np.abs(B_chk.T - B_A).max())}}
    return out


def _assemble(block: np.ndarray, B_A: np.ndarray, B_B: np.ndarray, SigF: np.ndarray, d_B: np.ndarray,
              sample_block: bool = True) -> np.ndarray:
    """The complete members' block, and for every other member its loadings times the factors' covariance times
    the others' loadings, with its residual variance on the diagonal."""
    nA, nB = B_A.shape[0], B_B.shape[0]
    M = np.empty((nA + nB, nA + nB))
    M[:nA, :nA] = block
    cross = B_A @ SigF @ B_B.T
    M[:nA, nA:] = cross
    M[nA:, :nA] = cross.T
    M[nA:, nA:] = B_B @ SigF @ B_B.T + np.diag(d_B)
    return M


def _assemble_pca(block: np.ndarray, beta_A: np.ndarray, beta_B: np.ndarray, var_m: float, own_B: np.ndarray) -> np.ndarray:
    """beta beta' var(benchmark) for every member, the complete members' components and own variances as the copied
    function gives them, and every other member's own variance on the diagonal."""
    nA, nB = len(beta_A), len(beta_B)
    beta = np.concatenate([beta_A, beta_B])
    M = np.outer(beta, beta) * var_m
    M[:nA, :nA] = block
    M[nA:, nA:] += np.diag(own_B)
    return M


def check_positive_semidefinite(result: dict, tolerance: float = C.PSD_FAULT_TOLERANCE) -> None:
    """Stop the run if an assembled matrix has an eigenvalue below minus `tolerance` times its largest: by
    construction none can, so one would be a fault in the code."""
    for name in C.COV_ESTIMATORS:
        r = result[f"smallest_eigenvalue_ratio_{name}"]
        if r < -tolerance:
            raise RuntimeError(f"{result['month']}: the {name} matrix has an eigenvalue of {r:.3g} times its largest, "
                               "below the fault tolerance; this is a fault in the code, and nothing from this run is reported.")


# ---------------------------------------------------------------------------
# By year, for committing
# ---------------------------------------------------------------------------

def by_year(rows: pd.DataFrame) -> pd.DataFrame:
    """The monthly scalars summarised by year: condition numbers (median, least, largest), the smallest ratio of
    the sample's to Ledoit-Wolf's and the months below E4's level, the intensity, and the counts."""
    r = rows.assign(year=pd.PeriodIndex(rows["month"], freq="M").year)
    g = r.groupby("year")
    out = pd.DataFrame({"months": g.size()})
    for name in C.COV_ESTIMATORS:
        for stat, fn in (("median", "median"), ("least", "min"), ("largest", "max")):
            out[f"condition_{name}_{stat}"] = g[f"condition_{name}"].agg(fn)
        out[f"condition_complete_{name}_median"] = g[f"condition_complete_{name}"].median()
    out["ratio_sample_to_ledoit_wolf_least"] = g["ratio_sample_to_ledoit_wolf"].min()
    out["months_ratio_below_level"] = (r["ratio_sample_to_ledoit_wolf"] < C.E4_CONDITION_RATIO_MIN).groupby(r["year"]).sum()
    out["lw_intensity_mean"] = g["lw_intensity"].mean()
    out["lw_intensity_least"] = g["lw_intensity"].min()
    out["lw_intensity_largest"] = g["lw_intensity"].max()
    out["complete_members_mean"] = g["complete"].mean()
    out["complete_members_least"] = g["complete"].min()
    out["model_rows_mean"] = g["model_rows"].mean()
    out["model_rows_most"] = g["model_rows"].max()
    out["model_rows_under_252_mean"] = g["model_rows_under_252"].mean()
    out["model_rows_under_252_most"] = g["model_rows_under_252"].max()
    out["model_rows_under_60_member_months"] = g["model_rows_under_60"].sum()
    out["days_left_out_mean"] = g["days_left_out"].mean()
    out["days_left_out_most"] = g["days_left_out"].max()
    out["months_bar_fell"] = g["bar_fell"].sum()
    out["days_without_factors"] = g["days_without_factors"].sum()
    out["window_days_least"] = g["window_days"].min()
    out["window_days_most"] = g["window_days"].max()
    ints = [c for c in out.columns if c.startswith(("months", "model_rows_most", "model_rows_under_252_most",
                                                     "model_rows_under_60", "days_left_out_most", "days_without",
                                                     "window_days", "complete_members_least"))]
    out[ints] = out[ints].astype("int64")
    return out.rename_axis("year").reset_index()


def judge_e4_first_clause(rows: pd.DataFrame) -> dict:
    """E4's first clause: in every month of the evaluation window, the sample's condition number at least
    E4_CONDITION_RATIO_MIN times Ledoit-Wolf's, on the assembled matrices."""
    ratio = rows["ratio_sample_to_ledoit_wolf"]
    below = rows.loc[ratio < C.E4_CONDITION_RATIO_MIN, "month"].tolist()
    worst = rows.loc[ratio.idxmin()]
    return {"months": int(len(rows)), "months_expected": C.EVAL_MONTHS, "months_at_or_above_level": int((ratio >= C.E4_CONDITION_RATIO_MIN).sum()),
            "smallest_ratio": float(ratio.min()), "month_of_smallest_ratio": str(worst["month"]),
            "median_ratio": float(ratio.median()), "months_below": len(below),
            "met": bool(len(below) == 0 and len(rows) == C.EVAL_MONTHS)}

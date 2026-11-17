# Copied verbatim from src/bp/attribution.py at commit 5414f14 of github.com/ochofer/optimal-vs-naive-diversification; every line below this one is the source file unchanged.
"""Factor attribution of a path's active return (notebook 13).

For each month t of a path: the active weights a_t = w_t - b_t (portfolio minus
benchmark, one number per industry); the industries' betas B_t to the factors
(49 by K), estimated on the data before t by the same regression as the factor
covariance estimator of notebook 10; the active exposures x_t = B_t' a_t, K
numbers that say how much more or less of each factor the portfolio holds than
the benchmark; the factor contributions x_{t,k} F_{t,k}, with F_t the factors'
returns in month t; and the residual, the active return minus the sum of the
contributions, the part of the active return the factors do not explain (the
industries' own returns). Contributions plus residual equal the active return
by construction; the identity is checked within
constants.ATTRIBUTION_IDENTITY_MAX_ABS_ERROR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import constants as C

PERIODS_PER_YEAR = 12


def active_exposures(B: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """x = B' (w - b): the portfolio's factor exposures minus the benchmark's, one per factor."""
    return np.asarray(B, dtype=float).T @ (np.asarray(w, dtype=float) - np.asarray(b, dtype=float))


def attribute(weights: pd.DataFrame, bench_weights: pd.DataFrame, betas: dict, factors: pd.DataFrame,
              active_return: pd.Series, names: tuple = C.ATTRIBUTION_FACTORS) -> pd.DataFrame:
    """One row per month: the active exposure to each factor, each factor's contribution (exposure times the
    factor's return in the month), their total, the residual (active return minus the total) and the identity
    error (total plus residual minus active return, zero up to floating-point arithmetic).

    `weights` holds the path's weights by month (months by N), `bench_weights` the benchmark's for the same months,
    `betas` maps a month to its B (N by K), `factors` holds the K factors' monthly returns, `active_return` the
    path's recorded active return by month."""
    rows = []
    for month in weights.index:
        w = weights.loc[month].to_numpy()
        b = bench_weights.loc[month].to_numpy()
        x = active_exposures(betas[month], w, b)
        f = factors.loc[month, list(names)].to_numpy(dtype=float)
        contrib = x * f
        a = float(active_return.loc[month])
        total = float(contrib.sum())
        row = {"month": month}
        row.update({f"exposure {n}": float(v) for n, v in zip(names, x)})
        row.update({f"contribution {n}": float(v) for n, v in zip(names, contrib)})
        row["factor total"] = total
        row["residual"] = a - total
        row["active return"] = a
        row["identity error"] = total + row["residual"] - a
        rows.append(row)
    return pd.DataFrame(rows).set_index("month")


def summarise(att: pd.DataFrame, names: tuple = C.ATTRIBUTION_FACTORS) -> dict:
    """The attribution of one path over its months: the mean active exposure to each factor; each factor's mean
    contribution per year and its standard error (the standard deviation of the monthly contributions divided by
    the square root of the number of months, times 12); the same for the factor total, the residual and the
    active return; and the factor share, the share of the variance of the active return that the factor total
    explains, 1 minus the variance of the residual over the variance of the active return."""
    T = len(att)
    out = {"months": T}
    for n in names:
        out[f"exposure {n}"] = float(att[f"exposure {n}"].mean())
    for col in [f"contribution {n}" for n in names] + ["factor total", "residual", "active return"]:
        x = att[col].to_numpy(dtype=float)
        out[f"{col} (per year)"] = PERIODS_PER_YEAR * float(x.mean())
        out[f"{col} (standard error)"] = PERIODS_PER_YEAR * float(x.std(ddof=1)) / np.sqrt(T)
    var_a = float(att["active return"].var(ddof=1))
    out["factor share"] = 1.0 - float(att["residual"].var(ddof=1)) / var_a if var_a > 0 else float("nan")
    out["largest identity error"] = float(att["identity error"].abs().max())
    return out


def by_period(att: pd.DataFrame, labels: pd.Series, names: tuple = C.ATTRIBUTION_FACTORS) -> pd.DataFrame:
    """Mean contributions per year by a grouping of the months (for example decades), one row per group:
    each factor, the factor total, the residual and the active return."""
    cols = [f"contribution {n}" for n in names] + ["factor total", "residual", "active return"]
    return PERIODS_PER_YEAR * att[cols].groupby(labels.loc[att.index]).mean()

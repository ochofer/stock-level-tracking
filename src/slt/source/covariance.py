# Copied verbatim from src/bp/covariance.py at commit 5414f14 of github.com/ochofer/optimal-vs-naive-diversification; every line below this one is the source file unchanged.
"""Four ways to estimate a covariance matrix from a window of excess returns, and the tests that judge them (notebook 10).

Every estimator takes a window of excess returns (rows are periods, columns are
assets) and returns an N by N covariance matrix for the period that follows. The
sample covariance uses all N(N+1)/2 numbers the window offers. Ledoit-Wolf
shrinkage pulls it towards a simple target. The factor model and the
principal-component model describe it with a small number of common sources of movement
plus each asset's own variance. The tests measure how well each estimate
forecasts the variance of test portfolios in the next month (the bias
statistic) and how well the minimum-variance portfolio built from it does
(the tests of Dom, Howard, Jansen and Lohre, 2024).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import constants as C
from . import rules as S

PERIODS_PER_YEAR = 12


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def sample_cov(X: np.ndarray) -> np.ndarray:
    """The sample covariance, denominator T - 1."""
    return np.cov(np.asarray(X, dtype=float), rowvar=False, ddof=1)


def ewma_cov(X: np.ndarray, halflife: float) -> np.ndarray:
    """Exponentially weighted covariance: the weight of an observation halves every `halflife` periods, the most recent weighing most.

    The mean is weighted the same way; weights are normalised to sum to one.
    """
    X = np.asarray(X, dtype=float)
    T = X.shape[0]
    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(T - 1, -1, -1)
    w = w / w.sum()
    mu = w @ X
    D = X - mu
    return (D * w[:, None]).T @ D / (1.0 - (w ** 2).sum())   # the unbiased normalisation for weighted samples


def lw_scaled_identity(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit and Wolf (2004b): (1 - k) S + k m I, with S the sample covariance (denominator T), m the average
    sample variance and k the intensity that minimises the expected squared error.

    Returns the estimate and k. The intensity is b^2 / d^2 clipped to [0, 1], where d^2 is the squared
    distance between S and m I and b^2 measures how much S moves from one observation to the next.
    """
    X = np.asarray(X, dtype=float)
    T, N = X.shape
    D = X - X.mean(axis=0)
    S = D.T @ D / T
    m = np.trace(S) / N
    d2 = ((S - m * np.eye(N)) ** 2).sum() / N
    b2_bar = sum(((np.outer(x, x) - S) ** 2).sum() for x in D) / (N * T * T)
    b2 = min(b2_bar, d2)
    k = float(b2 / d2) if d2 > 0 else 0.0
    return (1.0 - k) * S + k * m * np.eye(N), k


def lw_constant_correlation(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit and Wolf (2004a): shrink S towards the matrix with the same variances and one common correlation,
    the average sample correlation; the intensity follows the paper's appendix. Returns the estimate and k."""
    X = np.asarray(X, dtype=float)
    T, N = X.shape
    D = X - X.mean(axis=0)
    S = D.T @ D / T
    var = np.diag(S)
    sd = np.sqrt(var)
    corr = S / np.outer(sd, sd)
    rbar = (corr.sum() - N) / (N * (N - 1))
    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, var)
    # pi: the sum over i, j of the estimated asymptotic variances of the entries of sqrt(T) S
    Y = D ** 2
    pi_mat = (Y.T @ Y) / T - S ** 2
    pi = pi_mat.sum()
    # rho: the diagonal part plus the covariance between the entries of S and the target's correlation structure
    term1 = ((D ** 3).T @ D) / T          # theta_{ii,ij}: average over t of x_it^3 x_jt
    help_ = term1 - var[:, None] * S      # (1/T) sum_t (x_it^2 - S_ii)(x_it x_jt - S_ij)
    theta_ii = help_
    theta_jj = help_.T
    rho_off = (rbar / 2.0) * (np.sqrt(var[None, :] / var[:, None]) * theta_ii + np.sqrt(var[:, None] / var[None, :]) * theta_jj)
    np.fill_diagonal(rho_off, 0.0)
    rho = np.trace(pi_mat) + rho_off.sum()
    gamma = ((F - S) ** 2).sum()
    k = (pi - rho) / gamma / T if gamma > 0 else 0.0
    k = float(min(1.0, max(0.0, k)))
    return (1.0 - k) * S + k * F, k


def factor_cov(X: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The exact factor model: X regressed on the factors F (with an intercept) over the window;
    the estimate is B Sigma_F B' plus the diagonal of residual variances. Returns the estimate and B (N by K)."""
    X = np.asarray(X, dtype=float)
    F = np.asarray(F, dtype=float)
    T, K = F.shape
    A = np.column_stack([np.ones(T), F])
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)      # (K + 1) by N
    B = coef[1:].T                                     # N by K
    resid = X - A @ coef
    Sigma_F = np.cov(F, rowvar=False, ddof=1).reshape(K, K)
    resid_var = resid.var(axis=0, ddof=K + 1)
    return B @ Sigma_F @ B.T + np.diag(resid_var), B


def pca_cov(X: np.ndarray, n_components: int, market: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """The principal-component model.

    With `market` given (constants.PCA_FIRST_COMPONENT == "market"): each asset is regressed on the market
    series, and the covariance of the residuals is described by its first n_components - 1 principal
    components plus the diagonal of what they leave; the estimate is beta beta' var(market) plus that.
    Without a market series, the first n_components of X's own covariance plus the diagonal remainder.
    Returns the estimate and the number of components taken from the data.
    """
    X = np.asarray(X, dtype=float)
    if market is not None:
        m = np.asarray(market, dtype=float)
        A = np.column_stack([np.ones(len(m)), m])
        coef, *_ = np.linalg.lstsq(A, X, rcond=None)
        beta = coef[1]
        resid = X - A @ coef
        base = np.outer(beta, beta) * m.var(ddof=1)
        k = n_components - 1
        target = np.cov(resid, rowvar=False, ddof=2)
    else:
        base = 0.0
        k = n_components
        target = sample_cov(X)
    if k <= 0:
        return base + np.diag(np.diag(target)), 0
    vals, vecs = np.linalg.eigh(target)
    order = np.argsort(vals)[::-1][:k]
    V = vecs[:, order]
    L = np.clip(vals[order], 0.0, None)
    common = (V * L) @ V.T
    own = np.diag(np.clip(np.diag(target) - np.diag(common), 1e-12, None))
    return base + common + own, k


def scale_daily_to_monthly(Sigma_daily: np.ndarray, scale: float = C.DAILY_TO_MONTHLY_SCALE) -> np.ndarray:
    """A daily covariance times the number of trading days in a month; exact when daily returns do not predict the next day's."""
    return Sigma_daily * scale


def variance_ratio(x: np.ndarray, q: int = C.COV_SCALE_VR_HORIZON_DAYS) -> float:
    """The variance ratio of Lo and MacKinlay (1988): the variance of the overlapping sums of q consecutive
    values divided by q times the variance of the single values. It is 1 when the values do not predict each
    other, above 1 when a high value tends to be followed by a high one (positive autocorrelation) and below 1
    when a high value tends to be followed by a low one. Multiplying a daily covariance by q times this ratio,
    in place of q alone, gives the covariance of q-day sums that the autocorrelation implies."""
    v = np.asarray(x, dtype=float)
    if len(v) < 2 * q:
        return float("nan")
    sums = np.convolve(v, np.ones(q), mode="valid")
    return float(sums.var(ddof=1) / (q * v.var(ddof=1)))


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def monthly_window(excess_m: pd.DataFrame, month: pd.Period, months: int = C.EXT_ESTIMATION_WINDOW) -> pd.DataFrame:
    """The `months` monthly rows before `month`."""
    end = month - 1
    start = month - months
    return excess_m.loc[start:end]


def daily_window(excess_d: pd.DataFrame, month: pd.Period, years: int) -> pd.DataFrame:
    """The trading days of the 12 * years calendar months before `month`, days with any missing value left out."""
    end = (month - 1).asfreq("D", "end")
    start = (month - 12 * years).asfreq("D", "start")
    return excess_d.loc[start:end].dropna()


def daily_benchmark_excess(returns_d: pd.DataFrame, rf_d: pd.Series, weights_m: pd.DataFrame) -> pd.Series:
    """The benchmark's daily excess return: each day's industry returns weighted by that month's benchmark weights, minus the daily risk-free rate."""
    month_of_day = returns_d.index.asfreq("M")
    w = weights_m.reindex(month_of_day)
    w.index = returns_d.index
    return ((w * returns_d).sum(axis=1, min_count=returns_d.shape[1]) - rf_d.reindex(returns_d.index)).rename("benchmark_excess")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def bias_statistic(realised: np.ndarray, predicted_var: np.ndarray) -> float:
    """The standard deviation of realised / sqrt(predicted variance): 1 when the forecasts are calibrated, above 1 when risk is under-forecast."""
    z = np.asarray(realised, dtype=float) / np.sqrt(np.asarray(predicted_var, dtype=float))
    return float(z.std(ddof=1))


def random_active_weights(n_assets: int, draws: int = C.COV_TEST_ACTIVE_DRAWS, bound: float = C.ACTIVE_WEIGHT_BOUND,
                          seed: int = C.COV_TEST_SEED) -> np.ndarray:
    """`draws` active-weight vectors: weights sum to zero and the largest absolute weight equals `bound`."""
    rng = np.random.default_rng(seed)
    U = rng.uniform(-1.0, 1.0, size=(draws, n_assets))
    U = U - U.mean(axis=1, keepdims=True)
    return bound * U / np.abs(U).max(axis=1, keepdims=True)


def min_variance_unconstrained(Sigma: np.ndarray) -> np.ndarray:
    """w proportional to Sigma^-1 1, scaled to sum to one; negative weights allowed."""
    x = np.linalg.solve(Sigma, np.ones(Sigma.shape[0]))
    return x / x.sum()


def min_variance_long_only(Sigma: np.ndarray) -> np.ndarray:
    """The long-only minimum-variance portfolio, solved exactly by non-negative least squares (rules.constrained_min_variance)."""
    return S.constrained_min_variance(Sigma, 0.0)


def condition_number(Sigma: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(Sigma)
    return float(vals.max() / max(vals.min(), 1e-300))


def annualised_vol(x: pd.Series | np.ndarray) -> float:
    return float(np.std(np.asarray(x, dtype=float), ddof=1) * np.sqrt(PERIODS_PER_YEAR))


# ---------------------------------------------------------------------------
# The panels and the estimator variants, as notebooks 10 to 12 use them
# ---------------------------------------------------------------------------

class Panels:
    """Everything the estimators read: monthly and daily excess returns of the 49 industries, the six factors at both
    frequencies, the benchmark's weights and its excess return at both frequencies, and the plain monthly returns."""

    def __init__(self, start: str = C.EXT_SAMPLE_START):
        from . import french_loader as fl
        from . import benchmark as BM
        inputs = BM.load_inputs()
        f3 = fl.load_monthly("factors3", 0)
        f5 = fl.load_monthly("factors5", 0)
        mom = fl.load_monthly("momentum", 0)
        start_p = pd.Period(start, "M")
        self.returns_m = inputs["returns"].loc[start_p:]
        self.last = self.returns_m.index[-1]
        self.rf_m = f3["RF"].loc[start_p:self.last]
        self.excess_m = self.returns_m.sub(self.rf_m, axis=0)
        self.weights = BM.cap_weights(inputs["firms"].loc[start_p:], inputs["size"].loc[start_p:], C.CAPW_WEIGHT_TIMING)
        self.bench_m = (BM.combination_return(self.returns_m, self.weights) - self.rf_m).rename("benchmark_excess")
        self.factors_m = pd.concat([f5[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]], mom["Mom"]], axis=1).loc[start_p:self.last]
        returns_d = fl.load_monthly("ind49_daily", r"Value Weighted Returns -- Daily").loc[start_p.asfreq("D", "start"):]
        f3d = fl.load_monthly("factors3_daily", 0)
        f5d = fl.load_monthly("factors5_daily", 0)
        momd = fl.load_monthly("momentum_daily", 0)
        self.rf_d = f3d["RF"].reindex(returns_d.index)
        self.returns_d = returns_d
        self.excess_d = returns_d.sub(self.rf_d, axis=0)
        self.factors_d = pd.concat([f5d[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]], momd["Mom"]], axis=1).reindex(returns_d.index)
        self.bench_d = daily_benchmark_excess(returns_d, self.rf_d, self.weights)
        self.files = ("ind49", "factors3", "factors5", "momentum", "ind49_daily", "factors3_daily", "factors5_daily", "momentum_daily")


DAILY_YEARS = {"daily_3y": C.EXT_COV_DAILY_WINDOW_YEARS, "daily_1y": C.EXT_COV_DAILY_WINDOW_YEARS_SENSITIVITY[0],
               "daily_5y": C.EXT_COV_DAILY_WINDOW_YEARS_SENSITIVITY[1], "daily_5y_ewma": C.EXT_COV_DAILY_WINDOW_YEARS_SENSITIVITY[1]}


def estimate(name: str, freq: str, month: pd.Period, P: Panels) -> tuple[np.ndarray, float, int]:
    """One estimator variant for `month`, built on the data before it: the monthly covariance (daily ones scaled by
    DAILY_TO_MONTHLY_SCALE), the shrinkage intensity (nan for the others) and the number of observations in the window.
    The same dispatch as notebook 10's evaluation loop."""
    if freq == "monthly_120":
        X = monthly_window(P.excess_m, month); F = P.factors_m.loc[X.index]; mk = P.bench_m.loc[X.index]; scale = 1.0
    else:
        X = daily_window(P.excess_d, month, DAILY_YEARS[freq]); F = P.factors_d.loc[X.index]; mk = P.bench_d.loc[X.index]; scale = C.DAILY_TO_MONTHLY_SCALE
    Xv = X.to_numpy(); intensity = np.nan
    if name == "sample":
        S = ewma_cov(Xv, C.EXT_COV_EWMA_HALFLIFE_DAYS) if freq.endswith("ewma") else sample_cov(Xv)
    elif name == "ledoit_wolf":
        S, intensity = lw_scaled_identity(Xv)
    elif name == "ledoit_wolf_cc":
        S, intensity = lw_constant_correlation(Xv)
    elif name == "factor":
        S, _ = factor_cov(Xv, F.to_numpy())
    elif name == "pca":
        S, _ = pca_cov(Xv, C.PCA_N_COMPONENTS, market=mk.to_numpy())
    elif name == "pca_2":
        S, _ = pca_cov(Xv, C.PCA_N_COMPONENTS_SENSITIVITY, market=mk.to_numpy())
    else:
        raise KeyError(name)
    return S * scale, intensity, len(X)


# The five estimator variants of the rolling evaluation (notebook 12): the four estimators on three years of daily
# returns and the paper's convention, the sample covariance of 120 monthly returns.
EVALUATION_VARIANTS = tuple((e, "daily_3y") for e in C.COV_ESTIMATORS) + (("sample", "monthly_120"),)
# The robust portfolio (constants.ROBUST_VARIANT) is built from all five covariances at once, and RiskMetrics
# (constants.EWMA_VARIANT) is the recency-weighted sample covariance; the seven together are the variants the rolling evaluation reports.
ALL_VARIANTS = EVALUATION_VARIANTS + (C.ROBUST_VARIANT, C.EWMA_VARIANT)

# Copied verbatim from src/bp/paired.py at commit 0e651a0 of github.com/ochofer/optimal-vs-naive-diversification; every line below this one is the source file unchanged.
"""
Ledoit and Wolf (2011): inference on the difference of two variances measured
on the same months, the test Dom, Howard, Jansen and Lohre (2025) apply to
their portfolios' realised volatilities.

    Ledoit, O. and Wolf, M. (2011). Robust performance hypothesis testing with
    the variance. Wilmott, 2011(55), 86 to 89.

Added on 18 November 2026, after every recorded run of the study, by the design
of that day, and labelled so wherever it is used: notebook 12 had judged one
path against another by the standard error of a single tracking error, while
two paths that share their months and whose active returns move together have
a difference far less noisy than either figure. The module repeats the
structure of Signal Evaluation's test of two Sharpe ratios (Ledoit and Wolf
2008) with the variance in place of the Sharpe ratio.

The statistic of the paper is the difference of the log variances,
Delta = log(sigma_1^2) - log(sigma_2^2), a smooth function of four means, those
of r_1, r_2, r_1^2 and r_2^2 (sigma^2 = E[r^2] - mu^2). Its standard error
comes from the delta method: the gradient of the function times the long-run
covariance matrix Psi of the four series, which allows the months to be
correlated with one another and the returns to have fat tails.

  On the sample itself, Psi is the prewhitened Parzen-kernel estimate with
  Andrews' (1991) automatic bandwidth: a VAR(1) fitted to the four centred
  series, its singular values held within 0.97, the kernel estimate on the
  residuals, then recoloured. Ledoit and Wolf (2008, section 3.2.2 and its
  footnote 8) report that the prewhitened Parzen kernel performs like the
  prewhitened QS kernel they propose and is more convenient to implement; the
  2011 paper keeps the same machinery.

  The circular block bootstrap resamples blocks of consecutive months, wrapping
  from the last month to the first, so that the dependence within a block is
  kept. On each bootstrap sample, Psi is the average outer product of the
  blocks' scaled means, the estimate that matches the resampling.

  The studentised statistic |Delta* - Delta_hat| / se* is computed on every
  bootstrap sample. The share of bootstrap statistics at least as large as
  |Delta_hat| / se(Delta_hat) is the p-value of equal variances, in the form
  the paper reports.

From the same resamples a second statistic is read, the difference of the two
annualised standard deviations, sqrt(12) (sigma_1 - sigma_2), scaled to the
n minus 1 convention of the study's tables (times sqrt(T / (T - 1))) so that
it is the difference of two realised tracking errors or volatilities as the
tables print them. Its 95th percentile z* of the studentised statistic gives
the symmetric interval Delta_hat +/- z* se(Delta_hat), in the tables' units.
The log-variance difference is unchanged by that scaling.

Holm (1979) adjusts the p-values of a family of comparisons, as the paper does
across its seven: the smallest p-value is multiplied by the family's size, the
next by one less, and each adjusted value is at least the one before it.

A row whose interval includes zero says what the comparison could have seen
only beside the difference at which the interval would exclude zero about four
times in five: (z* + the standard normal quantile at the power) times the
standard error, the detectable difference, with the power in constants.py
(PAIRED_DETECTABLE_POWER, added 18 November 2026). The chance that the interval
excludes zero at a given true difference is approximated by the normal
probability of that difference over the standard error, less z*: a figure from
the row's own standard error and critical value under a normal approximation,
which refers to the interval.

The block size, the number of draws, the level and the seed are the study's
(EVAL_BOOTSTRAP_BLOCK_MONTHS, EVAL_BOOTSTRAP_DRAWS, EVAL_BOOTSTRAP_LEVEL,
EVAL_BOOTSTRAP_SEED in constants.py); every caller passes them. The module
depends on numpy and the standard library alone, so that another study can
import it unchanged.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["log_variance_difference", "sd_difference", "standard_errors", "circular_block_indices", "test", "holm",
           "simulate_null_pair", "size_check", "normal_cdf", "normal_quantile", "detectable_difference",
           "chance_to_exclude_zero"]


def _moments(r1: np.ndarray, r2: np.ndarray):
    return r1.mean(), r2.mean(), (r1 ** 2).mean(), (r2 ** 2).mean()


def _scale(months: int, periods: int, ddof: int) -> float:
    """sqrt(periods) for the annualisation times sqrt(T / (T - ddof)) for the tables' n minus ddof convention."""
    return float(np.sqrt(periods * months / (months - ddof)))


def _delta_log(mu1, mu2, g1, g2):
    return np.log(g1 - mu1 ** 2) - np.log(g2 - mu2 ** 2)


def _delta_sd(mu1, mu2, g1, g2, scale):
    return scale * (np.sqrt(g1 - mu1 ** 2) - np.sqrt(g2 - mu2 ** 2))


def _gradient_log(mu1, mu2, g1, g2) -> np.ndarray:
    """The gradient of the log-variance difference with respect to (mu1, mu2, g1, g2); works on arrays of draws too."""
    v1, v2 = g1 - mu1 ** 2, g2 - mu2 ** 2
    return np.stack([-2 * mu1 / v1, 2 * mu2 / v2, 1 / v1, -1 / v2], axis=-1)


def _gradient_sd(mu1, mu2, g1, g2, scale) -> np.ndarray:
    """The gradient of the scaled standard-deviation difference with respect to (mu1, mu2, g1, g2)."""
    s1, s2 = np.sqrt(g1 - mu1 ** 2), np.sqrt(g2 - mu2 ** 2)
    return scale * np.stack([-mu1 / s1, mu2 / s2, 0.5 / s1, -0.5 / s2], axis=-1)


def log_variance_difference(r1, r2) -> float:
    """log of the variance of r1 minus log of the variance of r2, population variances."""
    r1, r2 = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
    return float(_delta_log(*_moments(r1, r2)))


def sd_difference(r1, r2, periods: int = 12, ddof: int = 1) -> float:
    """The annualised standard deviation of r1 minus that of r2, in the n minus ddof convention."""
    r1, r2 = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
    return float(_delta_sd(*_moments(r1, r2), _scale(len(r1), periods, ddof)))


def _parzen(x: float) -> float:
    x = abs(x)
    if x <= 0.5:
        return 1 - 6 * x ** 2 + 6 * x ** 3
    if x <= 1:
        return 2 * (1 - x) ** 3
    return 0.0


def _gamma(v: np.ndarray, j: int) -> np.ndarray:
    """The sample autocovariance matrix at lag j, divided by T."""
    t = len(v)
    return v[j:].T @ v[:t - j] / t


def _ar1(x: np.ndarray):
    """Least-squares AR(1) with an intercept: the coefficient and the residual variance."""
    y, z = x[1:], x[:-1]
    zc, yc = z - z.mean(), y - y.mean()
    rho = float(zc @ yc / (zc @ zc))
    resid = yc - rho * zc
    return rho, float(resid @ resid / len(resid))


def _andrews_alpha(v: np.ndarray) -> float:
    """Andrews' (1991) plug-in quantity for the Parzen kernel, from univariate AR(1) fits, each series weighted the same."""
    num = den = 0.0
    for i in range(v.shape[1]):
        rho, sig2 = _ar1(v[:, i])
        num += 4 * rho ** 2 * sig2 ** 2 / (1 - rho) ** 8
        den += sig2 ** 2 / (1 - rho) ** 4
    return num / den


def _psi_parzen(v: np.ndarray) -> np.ndarray:
    """The Parzen-kernel long-run covariance with Andrews' bandwidth 2.6614 (alpha T)^(1/5)."""
    t, p = v.shape
    bandwidth = 2.6614 * (_andrews_alpha(v) * t) ** 0.2
    psi = _gamma(v, 0)
    j = 1
    while j < bandwidth and j < t:
        g = _gamma(v, j)
        psi = psi + _parzen(j / bandwidth) * (g + g.T)
        j += 1
    return psi * t / (t - p)


def _psi_prewhitened(v: np.ndarray) -> np.ndarray:
    """The prewhitened Parzen-kernel estimate of the long-run covariance of the four centred series."""
    x, y = v[:-1], v[1:]
    a = np.linalg.lstsq(x, y, rcond=None)[0].T                  # y_t = A y_{t-1} + e_t, no intercept
    u, d, vt = np.linalg.svd(a)
    a = u @ np.diag(np.clip(d, -0.97, 0.97)) @ vt
    resid = y - x @ a.T
    inv = np.linalg.inv(np.eye(4) - a)
    return inv @ _psi_parzen(resid) @ inv.T


def standard_errors(r1, r2, periods: int = 12, ddof: int = 1) -> dict:
    """The delta-method standard errors of the two statistics on the sample: the log-variance difference
    and the scaled standard-deviation difference, both from the prewhitened Parzen-kernel estimate of Psi."""
    r1, r2 = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
    mu1, mu2, g1, g2 = _moments(r1, r2)
    v = np.column_stack([r1 - mu1, r2 - mu2, r1 ** 2 - g1, r2 ** 2 - g2])
    t = len(v)
    psi = _psi_prewhitened(v)
    scale = _scale(t, periods, ddof)
    gl, gs = _gradient_log(mu1, mu2, g1, g2), _gradient_sd(mu1, mu2, g1, g2, scale)
    return {"se_log": float(np.sqrt(gl @ psi @ gl / t)), "se_sd": float(np.sqrt(gs @ psi @ gs / t))}


def circular_block_indices(months: int, block: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of `draws` circular block bootstrap samples: floor(months / block) blocks of `block`
    consecutive months each, every block starting at a month drawn uniformly, wrapping past the last
    month to the first. Shape (draws, blocks * block)."""
    n_blocks = months // block
    starts = rng.integers(0, months, size=(draws, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % months
    return idx.reshape(draws, n_blocks * block)


def _bootstrap_statistics(r1, r2, delta_log_hat, delta_sd_hat, scale, block, draws, rng):
    """The studentised statistics of both differences on the same resamples: shape (draws,) each."""
    t = len(r1)
    idx = circular_block_indices(t, block, draws, rng)
    a, b = r1[idx], r2[idx]
    n = idx.shape[1]
    n_blocks = n // block
    mu1, mu2 = a.mean(axis=1), b.mean(axis=1)
    g1, g2 = (a ** 2).mean(axis=1), (b ** 2).mean(axis=1)
    y = np.stack([a - mu1[:, None], b - mu2[:, None], a ** 2 - g1[:, None], b ** 2 - g2[:, None]], axis=-1)
    zeta = np.sqrt(block) * y.reshape(draws, n_blocks, block, 4).mean(axis=2)
    psi = np.einsum("dki,dkj->dij", zeta, zeta) / n_blocks
    gl, gs = _gradient_log(mu1, mu2, g1, g2), _gradient_sd(mu1, mu2, g1, g2, scale)
    se_log = np.sqrt(np.einsum("di,dij,dj->d", gl, psi, gl) / n)
    se_sd = np.sqrt(np.einsum("di,dij,dj->d", gs, psi, gs) / n)
    stat_log = np.abs(_delta_log(mu1, mu2, g1, g2) - delta_log_hat) / se_log
    stat_sd = np.abs(_delta_sd(mu1, mu2, g1, g2, scale) - delta_sd_hat) / se_sd
    return stat_log, stat_sd


def test(r1, r2, block: int = 12, draws: int = 2000, level: float = 0.95, seed: int | np.random.Generator = 0,
         periods: int = 12, ddof: int = 1) -> dict:
    """Ledoit and Wolf's test of equal variances, r1 against r2, on paired monthly series.

    Returns the two annualised standard deviations and their difference (r1 minus r2, in the n minus ddof
    convention), the correlation of the two series, the log-variance difference with its standard error,
    the p-value of equal variances from the studentised bootstrap of the log-variance difference (the
    paper's form), the symmetric studentised interval of the standard-deviation difference at `level`
    with its half-width and critical value, whether that interval excludes zero, and the settings. The
    same seed gives the same result."""
    r1, r2 = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
    if r1.shape != r2.shape or r1.ndim != 1:
        raise ValueError("two one-dimensional series over the same months are needed")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    t = len(r1)
    scale = _scale(t, periods, ddof)
    mu1, mu2, g1, g2 = _moments(r1, r2)
    delta_log_hat = float(_delta_log(mu1, mu2, g1, g2))
    delta_sd_hat = float(_delta_sd(mu1, mu2, g1, g2, scale))
    se = standard_errors(r1, r2, periods, ddof)
    stat_log, stat_sd = _bootstrap_statistics(r1, r2, delta_log_hat, delta_sd_hat, scale, block, draws, rng)
    d_log = abs(delta_log_hat) / se["se_log"]
    p_log = (int((stat_log >= d_log).sum()) + 1) / (draws + 1)
    d_sd = abs(delta_sd_hat) / se["se_sd"]
    p_sd = (int((stat_sd >= d_sd).sum()) + 1) / (draws + 1)
    z_sd = float(np.quantile(stat_sd, level))
    half = z_sd * se["se_sd"]
    sd1, sd2 = scale * float(np.sqrt(g1 - mu1 ** 2)), scale * float(np.sqrt(g2 - mu2 ** 2))
    return {"months": t, "sd1": sd1, "sd2": sd2, "difference": delta_sd_hat, "se": se["se_sd"],
            "correlation": float(np.corrcoef(r1, r2)[0, 1]),
            "log_difference": delta_log_hat, "se_log": se["se_log"], "p_value": p_log, "p_value_sd": p_sd,
            "lower": delta_sd_hat - half, "upper": delta_sd_hat + half, "half_width": half, "critical": z_sd,
            "excludes_zero": bool(abs(delta_sd_hat) > half),
            "block": block, "draws": draws, "level": level, "periods": periods, "ddof": ddof}


def holm(p_values) -> np.ndarray:
    """Holm's (1979) step-down adjustment of a family of p-values, in the family's order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def simulate_null_pair(months: int, sd: float, rho: float, dof: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two monthly series with mean zero and the same standard deviation, so equal variances, and
    correlation rho: a common Student-t shock and one own shock each, every shock scaled to variance 1."""
    scale = np.sqrt(dof / (dof - 2))
    common, own1, own2 = (rng.standard_t(dof, size=(3, months)) / scale)
    e1 = np.sqrt(rho) * common + np.sqrt(1 - rho) * own1
    e2 = np.sqrt(rho) * common + np.sqrt(1 - rho) * own2
    return sd * e1, sd * e2


def size_check(pairs: int, months: int, sd: float, rho: float, dof: int, seed: int,
               block: int = 12, draws: int = 2000, level: float = 0.95, periods: int = 12, ddof: int = 1) -> dict:
    """The share of simulated pairs with equal variances whose p-value falls below 1 - level: the test's
    rejection rate under the null, which should be close to 1 - level. Returns that rate, the share of
    intervals of the standard-deviation difference that exclude zero, the counts, and each pair's p-value
    and difference."""
    root = np.random.SeedSequence(seed)
    data_seed, boot_seed = root.spawn(2)
    data_rng = np.random.default_rng(data_seed)
    boot_rngs = [np.random.default_rng(s) for s in boot_seed.spawn(pairs)]
    p_values, excludes, differences = [], [], []
    for k in range(pairs):
        r1, r2 = simulate_null_pair(months, sd, rho, dof, data_rng)
        out = test(r1, r2, block=block, draws=draws, level=level, seed=boot_rngs[k], periods=periods, ddof=ddof)
        p_values.append(out["p_value"])
        excludes.append(out["excludes_zero"])
        differences.append(out["difference"])
    p_values, excludes = np.array(p_values), np.array(excludes)
    rejected = p_values < 1 - level
    return {"rate": float(rejected.mean()), "rejected": int(rejected.sum()),
            "rate_interval": float(excludes.mean()), "rejected_interval": int(excludes.sum()), "pairs": pairs,
            "p_values": p_values, "differences": np.array(differences)}


def normal_cdf(x: float) -> float:
    """The standard normal distribution function, from the error function."""
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def normal_quantile(p: float) -> float:
    """The standard normal quantile at p, by bisection on normal_cdf to 1e-12."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between 0 and 1")
    lo, hi = -40.0, 40.0
    while hi - lo > 1e-12:
        mid = 0.5 * (lo + hi)
        if normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def detectable_difference(se: float, critical: float, power: float) -> float:
    """The difference at which the interval (difference plus and minus critical times se) would exclude
    zero with probability `power`, under a normal approximation: (critical + the normal quantile at
    power) times the standard error, in the units of the standard error."""
    return (float(critical) + normal_quantile(power)) * float(se)


def chance_to_exclude_zero(gain: float, se: float, critical: float) -> float:
    """The approximate probability that the interval excludes zero when the true difference is `gain`:
    the normal probability of |gain| / se less the critical value."""
    return normal_cdf(abs(float(gain)) / float(se) - float(critical))


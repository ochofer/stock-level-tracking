"""
The paired tests of notebook 13: two realised tracking errors measured on the same months, compared with the variance
test of Ledoit and Wolf (2011) as the source repository's paired.py computes it (src/slt/source/paired.py, copied
unchanged).

Added on 18 November 2026, after every recorded run of the study, by the design of that day, and in no verdict
(constants.PAIRED_). Everything here reads the active returns as committed, results/<vintage>/active_returns_monthly.csv,
and runs no path, so anyone can rerun it from the repository alone. size_table() is the expectation P1, the test's
rejection rate on simulated pairs with equal variances, and runs before paired_table() touches the study's returns.
"""
from __future__ import annotations

import hashlib
import pathlib

import numpy as np
import pandas as pd

from . import constants as C
from .source import paired as PT

ACTIVE_FILE = "active_returns_monthly.csv"
LABELS = {"sample": "sample", "ledoit_wolf": "Ledoit-Wolf", "factor": "factor", "pca": "principal-component"}
DECIMALS = 8          # the tables are written to eight decimals, as the source's are


def settings() -> dict:
    """The bootstrap's settings as every call passes them."""
    return dict(block=C.PAIRED_BLOCK_MONTHS, draws=C.PAIRED_DRAWS, level=C.PAIRED_LEVEL, periods=C.PAIRED_PERIODS)


def read_active(repo) -> tuple:
    """The committed active returns by month, one column per path under the mandate, and the file's SHA-256."""
    path = pathlib.Path(repo) / C.RESULTS_DIR / ACTIVE_FILE
    data = path.read_bytes()
    frame = pd.read_csv(path, index_col="month")
    return frame, hashlib.sha256(data).hexdigest()


def cap_words(cap) -> str:
    return "no cap" if cap is None else f"cap {100 * cap:g} per cent"


def label(path_1, path_2) -> str:
    if path_1[0] == path_2[0]:
        return f"{LABELS[path_1[0]]}: {cap_words(path_1[2])} against {cap_words(path_2[2])}"
    return f"{LABELS[path_1[0]]} against {LABELS[path_2[0]]}, {cap_words(path_1[2])}"


def size_table() -> pd.DataFrame:
    """P1: for each correlation of constants.PAIRED_SIZE_CASES, the share of simulated pairs with equal variances that
    the test rejects at 5 per cent by its p-value, with the share of intervals that exclude zero beside it."""
    rows = []
    for rho, seed in C.PAIRED_SIZE_CASES:
        out = PT.size_check(pairs=C.PAIRED_SIZE_PAIRS, months=C.PAIRED_SIZE_MONTHS, sd=C.PAIRED_SIZE_SD, rho=rho,
                            dof=C.PAIRED_SIZE_T_DOF, seed=seed, **settings())
        rows.append({"pairs": out["pairs"], "months": C.PAIRED_SIZE_MONTHS, "monthly sd": C.PAIRED_SIZE_SD,
                     "correlation": rho, "t degrees of freedom": C.PAIRED_SIZE_T_DOF, "seed": seed,
                     "block": C.PAIRED_BLOCK_MONTHS, "draws": C.PAIRED_DRAWS, "level": C.PAIRED_LEVEL,
                     "rejected by p-value": out["rejected"], "share rejected": out["rate"],
                     "intervals excluding zero": out["rejected_interval"],
                     "share of intervals excluding zero": out["rate_interval"],
                     "maximum": C.PAIRED_SIZE_MAX, "conservative below": C.PAIRED_SIZE_CONSERVATIVE_BELOW,
                     "P1": "met" if out["rate"] <= C.PAIRED_SIZE_MAX else "not met"})
    return pd.DataFrame(rows).round(DECIMALS)


def _row(family, path_1, path_2, active) -> dict:
    s1 = active["active_return_" + C.PATH_LABELS[path_1]].to_numpy(dtype=float)
    s2 = active["active_return_" + C.PATH_LABELS[path_2]].to_numpy(dtype=float)
    if np.isnan(s1).any() or np.isnan(s2).any():
        raise ValueError(f"{C.PATH_LABELS[path_1]} or {C.PATH_LABELS[path_2]} has an empty month")
    out = PT.test(s1, s2, seed=C.PAIRED_SEED, **settings())
    detectable = PT.detectable_difference(out["se"], out["critical"], C.PAIRED_DETECTABLE_POWER)
    bp = 1e4
    return {"comparison": family, "label": label(path_1, path_2), "series 1": C.PATH_LABELS[path_1],
            "series 2": C.PATH_LABELS[path_2], "constraint set": C.MANDATE, "months": out["months"], "unit": "bp a year",
            "sd 1": bp * out["sd1"], "sd 2": bp * out["sd2"], "difference": bp * out["difference"],
            "standard error": bp * out["se"], "critical value": out["critical"], "lower": bp * out["lower"],
            "upper": bp * out["upper"], "detectable difference": bp * detectable,
            "detectable difference, share of sd 2": detectable / out["sd2"], "correlation": out["correlation"],
            "p-value": out["p_value"], "Holm p-value": np.nan, "log-variance difference": out["log_difference"],
            "its standard error": out["se_log"], "block": out["block"], "draws": out["draws"], "level": out["level"],
            "seed": C.PAIRED_SEED}


def share_in_months(s1: pd.Series, s2: pd.Series, months) -> float:
    """The share of the difference between the two variances that falls in the given months: the sum over those
    months of each path's squared deviation from its own mean over all months, the first less the second, over the
    same sum over all months."""
    d = (s1 - s1.mean()) ** 2 - (s2 - s2.mean()) ** 2
    return float(d.loc[list(months)].sum() / d.sum())


def left_out_table(active: pd.DataFrame) -> pd.DataFrame:
    """The factor estimator's two rows of family B again, with the months of constants.PAIRED_LEFT_OUT_MONTHS left
    out, beside the share of the difference between the two variances that those months carry. No verdict."""
    months = list(C.PAIRED_LEFT_OUT_MONTHS)
    keep = ~active.index.isin(months)
    rows = []
    for a, b in C.PAIRED_LEFT_OUT_ROWS:
        s1 = active["active_return_" + C.PATH_LABELS[a]]
        s2 = active["active_return_" + C.PATH_LABELS[b]]
        row = _row("B", a, b, active[keep])
        row = {"months left out": " and ".join(months),
               "share of the difference between the two variances in those months": share_in_months(s1, s2, months),
               **row}
        rows.append(row)
    return pd.DataFrame(rows).round(DECIMALS)


def paired_table(active: pd.DataFrame) -> pd.DataFrame:
    """The seven rows: family A with Holm's correction across its three, family B with none."""
    rows = [_row("A", a, b, active) for a, b in C.PAIRED_FAMILY_A]
    for r, h in zip(rows, PT.holm([r["p-value"] for r in rows])):
        r["Holm p-value"] = float(h)
    rows += [_row("B", a, b, active) for a, b in C.PAIRED_FAMILY_B]
    return pd.DataFrame(rows).round(DECIMALS)

"""The Brinson-Fachler attribution on made-up months: the identity against the active return as the script records
it, the rule for a sector the portfolio does not hold, a textbook month by hand, the sums by year and the two
verdicts' arithmetic (constants.py, the decisions of 17 November 2026)."""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from slt import attributions as A  # noqa: E402
from slt import brinson as B  # noqa: E402
from slt import constants as C  # noqa: E402
from slt import mandate as M  # noqa: E402
from slt import sorts as Q  # noqa: E402


def made_up_month(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n) + 10001
    sectors = np.array(list(C.ATTRIBUTION_SECTORS))[rng.integers(0, 11, n)]
    sectors[sectors == "60"] = "20"                     # sector 60 has no member this month
    sectors[:2] = C.UNCLASSIFIED
    cap = rng.lognormal(0, 1, n)
    b = pd.Series(cap / cap.sum(), index=idx)
    w = (b * rng.uniform(0.5, 1.5, n)).clip(lower=0)
    w[sectors == C.UNCLASSIFIED] = 5e-9                 # held at remnants below OPT_TOL only
    w = w / w.sum()
    r = pd.Series(rng.normal(0.01, 0.08, n), index=idx)
    r.iloc[[5, 17, 40]] = np.nan                        # three members without a return
    return w, b, r, pd.Series(sectors, index=idx)


def test_the_terms_sum_to_the_active_return_as_recorded():
    for seed in range(5):
        w, b, r, sector = made_up_month(seed=seed)
        frame, tot = B.month_terms(w, b, r, sector)
        recorded = M.portfolio_return(w, r) - M.portfolio_return(b, r)
        assert abs(tot["active_return"] - recorded) < 1e-15
        assert abs(tot["identity_error"]) < 1e-13
        assert tot["members_without_return"] == 3


def test_a_sector_not_held_has_no_selection_and_keeps_the_identity():
    w, b, r, sector = made_up_month()
    frame, tot = B.month_terms(w, b, r, sector)
    u = frame.loc[C.UNCLASSIFIED]
    assert not u["held"] and u["selection"] == 0.0
    has = r.notna()
    wr = w[has] / w[has].sum()
    br = b[has] / b[has].sum()
    s = sector[has] == C.UNCLASSIFIED
    Rbs = (br[s] * r[has][s]).sum() / br[s].sum()
    assert np.isclose(u["interaction"], (wr[s] * r[has][s]).sum() - wr[s].sum() * Rbs, atol=1e-18)
    assert frame.loc["60"].abs()[["allocation", "selection", "interaction"]].sum() == 0.0
    assert tot["sectors_not_held"] == 1


def test_a_textbook_month_by_hand():
    idx = [1, 2, 3, 4]
    b = pd.Series([0.3, 0.2, 0.3, 0.2], index=idx)
    w = pd.Series([0.4, 0.2, 0.1, 0.3], index=idx)
    r = pd.Series([0.05, 0.01, -0.02, 0.00], index=idx)
    sector = pd.Series(["10", "10", "20", "20"], index=idx)
    frame, tot = B.month_terms(w, b, r, sector, sectors=("10", "20"))
    # sector 10: b 0.5, w 0.6, Rb 0.034, Rp (0.02+0.002)/0.6; sector 20: b 0.5, w 0.4, Rb -0.012, Rp -0.002/0.4
    Rb = 0.3 * 0.05 + 0.2 * 0.01 - 0.3 * 0.02
    assert np.isclose(frame.loc["10", "allocation"], 0.1 * (0.034 - Rb))
    assert np.isclose(frame.loc["10", "selection"], 0.5 * (0.022 / 0.6 - 0.034))
    assert np.isclose(frame.loc["20", "interaction"], -0.1 * (-0.002 / 0.4 + 0.012))
    assert abs(tot["identity_error"]) < 1e-15


def test_the_sums_by_year_and_the_summary():
    rows, tots = [], []
    for k, month in enumerate(pd.period_range("2001-01", periods=24, freq="M").astype(str)):
        w, b, r, sector = made_up_month(seed=k)
        frame, tot = B.month_terms(w, b, r, sector)
        rows.append(frame.reset_index().assign(month=month))
        tots.append(dict(tot, month=month))
    terms = pd.concat(rows, ignore_index=True)
    totals = pd.DataFrame(tots).set_index("month")
    y = B.by_year(terms)
    t = y[y["sector"] == "total"].set_index("year")
    assert np.allclose(t["allocation"].to_numpy(), totals["allocation"].groupby(totals.index.str[:4]).sum().to_numpy())
    assert set(y["sector"]) == set(C.ATTRIBUTION_SECTORS) | {"total"}
    s = B.summary(terms, totals, Q.newey_west_se)
    assert np.isclose(s["total"]["selection_mean_abs_monthly"], totals["selection"].abs().mean())
    assert s["months"] == 24 and s["largest_identity_error"] < 1e-13


def test_the_two_verdicts_are_one_each():
    base = {"largest_identity_error": 1e-15}
    good = dict(base, total={"allocation_mean_abs_monthly": 3.0, "selection_mean_abs_monthly": 2.0, "interaction_mean_abs_monthly": 1.0})
    bad = dict(base, total={"allocation_mean_abs_monthly": 1.0, "selection_mean_abs_monthly": 2.0, "interaction_mean_abs_monthly": 0.5})
    v = B.judge_e8({"sample": bad, "ledoit_wolf": good, "factor": good, "pca": bad})
    assert not v["met"] and v["second_clause"]["fails_for"] == ["pca"] and v["second_clause"]["no_verdict"] == ["sample"]
    s = {e: {"largest identity error": 1e-16, "exposure Mkt-RF": x} for e, x in zip(C.COV_ESTIMATORS, (0.01, -0.02, 0.06, 0.0))}
    v9 = A.judge_e9(s)
    assert not v9["met"] and v9["market_exposure"]["fails_for"] == ["factor"] and v9["identity"]["met"]


def test_the_rescaled_weights_sum_to_one_over_the_names_with_a_return():
    w, b, r, sector = made_up_month()
    x = A.rescaled(w, r)
    assert np.isclose(x.sum(), 1.0) and (x[r.isna()] == 0).all()

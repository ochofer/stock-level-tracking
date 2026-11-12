"""The mandate's optimiser on made-up months: the same weights as the copied optimiser on the source's own
constraint sets, the cap-raising rule included, the group constraints held, and the reference portfolio and the
turnover decomposition as defined in constants.py, section 7."""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from slt import constants as C  # noqa: E402
from slt import mandate as M  # noqa: E402
from slt.source import optimiser as O  # noqa: E402

RNG = np.random.default_rng(C.SEED)


def made_up_month(n=60, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.normal(1.0, 0.3, (n, 1))
    Sigma = (B @ B.T) * 0.002 + np.diag(rng.uniform(0.002, 0.01, n))
    cap = rng.lognormal(0, 1, n)
    b = cap / cap.sum()
    sectors = np.array(["10", "15", "20", "25", "30", "35", "40", "45", "55", C.UNCLASSIFIED])[np.arange(n) % 10]
    sub = np.where(np.arange(n) % 17 == 3, "30203010", "")
    company = np.arange(n) + 1000
    company[[7, 8]] = 5000                     # one company with two classes, same sector
    sectors[8] = sectors[7]
    sub[8] = sub[7]
    m = M.Month(permnos=np.arange(n) + 10001, b=b, sector=sectors, subindustry=sub, company=company)
    return Sigma, m


SOURCE_SETS = [("long_only",), ("long_only", "stock_bound"), ("long_only", "stock_bound", "turnover_cap"),
               ("long_only", "stock_bound", "turnover_cap", "tilt", "exclusion", "beta_neutral")]


@pytest.mark.parametrize("constraints", SOURCE_SETS, ids=lambda c: "+".join(c))
def test_same_weights_as_the_copy_on_the_sources_own_sets(constraints):
    Sigma, m = made_up_month()
    w_prev = m.b + RNG.normal(0, 0.002, len(m.b)); w_prev = np.clip(w_prev, 0, None); w_prev /= w_prev.sum()
    bound = 0.01
    ours = M.solve(Sigma, m, constraints, w_prev, cap=0.02, stock_bound=bound)
    copy_set = tuple(M.COPY_NAMES[c] for c in constraints)
    theirs = O.solve(Sigma, m.b, copy_set, w_prev=w_prev, tilt_mask=m.tilt_mask, exclude_mask=m.exclude_mask,
                     turnover_cap=0.02, active_bound=bound)
    assert np.abs(ours.weights - theirs.weights).max() <= C.OPT_TOL
    assert ours.relaxed == theirs.relaxed


def test_same_weights_as_the_copy_when_the_cap_has_to_be_raised():
    Sigma, m = made_up_month()
    w_prev = np.zeros(len(m.b)); w_prev[:5] = 0.2          # far from anything the bounds allow
    constraints = ("long_only", "stock_bound", "turnover_cap")
    ours = M.solve(Sigma, m, constraints, w_prev, cap=0.02, stock_bound=0.01)
    theirs = O.solve(Sigma, m.b, ("long_only", "active_weight_bound", "turnover_cap"), w_prev=w_prev,
                     turnover_cap=0.02, active_bound=0.01)
    assert ours.relaxed and theirs.relaxed
    assert abs(ours.cap_used - theirs.turnover_cap_used) <= C.OPT_TOL
    assert np.abs(ours.weights - theirs.weights).max() <= C.OPT_TOL


def test_the_mandate_holds_every_group_constraint():
    Sigma, m = made_up_month()
    w_prev = m.b.copy()
    for cs in ("C1", "C2", "C3"):
        constraints = C.CONSTRAINT_SETS[cs]
        res = M.solve(Sigma, m, constraints, w_prev, cap=0.02)
        a = res.weights - m.b
        for idx, limit, kind in M.groups(m, constraints):
            assert abs(a[idx].sum()) <= limit + C.OPT_TOL, (cs, kind)
        cov = M.covered(m, constraints)
        assert (np.abs(a[cov]) <= C.STOCK_ACTIVE_BOUND + C.OPT_TOL).all()
        if "tilt" in constraints:
            assert (res.weights[m.tilt_mask] <= 0.5 * m.b[m.tilt_mask] + C.OPT_TOL).all()
            assert (np.abs(res.weights[m.exclude_mask]) <= C.OPT_TOL).all()


def test_the_band_leaves_the_tilted_sectors_to_the_tilt():
    Sigma, m = made_up_month()
    kinds = {(k, tuple(idx)) for idx, _, k in M.groups(m, C.CONSTRAINT_SETS["C3"])}
    banded = {m.sector[idx[0]] for k, idx in kinds if k == "sector_band"}
    assert banded == {"15", "20", "25", "30", "35", "40", "45", C.UNCLASSIFIED}
    banded_c2 = {m.sector[idx[0]] for idx, _, k in M.groups(m, C.CONSTRAINT_SETS["C2"]) if k == "sector_band"}
    assert banded_c2 == {"10", "15", "20", "25", "30", "35", "40", "45", "55", C.UNCLASSIFIED}


def test_the_company_bound_covers_the_classes_the_stock_bound_covers():
    Sigma, m = made_up_month()
    comp = [idx for idx, _, k in M.groups(m, C.CONSTRAINT_SETS["C2"]) if k == "company_bound"]
    assert len(comp) == 1 and set(comp[0]) == {7, 8}
    assert M.company_scope_mismatches(m, C.CONSTRAINT_SETS["C3"]) == 0


def test_under_c0_the_portfolio_is_the_benchmark():
    Sigma, m = made_up_month()
    res = M.solve(Sigma, m, C.CONSTRAINT_SETS["C0"], m.b)
    assert np.abs(res.weights - m.b).max() <= 1e-6 and res.forecast_te < 1e-6


def test_reference_portfolio_and_decomposition():
    prev = pd.Series({1: 0.5, 2: 0.3, 3: 0.2})                 # target weights last month
    r = pd.Series({1: 0.10, 2: -0.10, 3: np.nan})               # name 3 has no return: drifts at zero
    w_d, missing = M.drift(prev, r)
    assert missing == 1 and np.isclose(w_d.sum(), 1.0) and np.isclose(w_d[1], 0.55 / 1.02)
    permnos = np.array([1, 2, 4])                               # 3 left, 4 entered
    b = np.array([0.5, 0.4, 0.1])
    ref = M.reference_portfolio(w_d, permnos, b)
    assert np.isclose(ref.sum(), 1.0) and np.isclose(ref[4], 0.1) and 3 not in ref.index
    assert np.isclose(ref[1] / ref[2], w_d[1] / w_d[2])
    b_prev = pd.Series({1: 0.45, 2: 0.35, 3: 0.2})
    restored = M.restored_portfolio(prev, b_prev, permnos, b)
    new = pd.Series({1: 0.45, 2: 0.43, 4: 0.12})
    parts = M.decompose(w_d, ref, restored, new)
    assert np.isclose(parts["membership"], M.one_way(ref, w_d))
    assert np.isclose(parts["reweighting"], M.one_way(restored, ref))
    assert np.isclose(parts["rebalancing"], M.one_way(new, restored))
    assert parts["overlap"] >= -1e-15
    assert np.isclose(parts["membership"] + parts["reweighting"] + parts["rebalancing"] - parts["overlap"], parts["total"])


def test_signed_cell_rule():
    r = pd.Series(0.01, index=range(10))
    assert M.signed_cell(pd.Series(0.001, index=range(10)), r)
    assert not M.signed_cell(pd.Series([0.001, -0.001] + [0.0] * 8, index=range(10)), r)      # two stocks
    big = pd.Series([0.05] + [1e-5] * 9, index=range(10))
    assert not M.signed_cell(big, r)                                                            # one stock over 80%


def test_a_path_that_holds_the_benchmark_has_no_rebalancing():
    b_prev = pd.Series({1: 0.5, 2: 0.3, 3: 0.2})
    permnos = np.array([1, 2, 4])                               # 3 left, 4 entered
    b = np.array([0.55, 0.3, 0.15])
    out = M.restored_portfolio(b_prev, b_prev, permnos, b)
    assert np.allclose(out.to_numpy(), b)
    new = pd.Series(b, index=permnos)
    assert M.decompose(new, new, out, new)["rebalancing"] < 1e-15
    w_prev = pd.Series({1: 0.25, 2: 0.55, 3: 0.2})              # name 1 at half its benchmark weight
    out = M.restored_portfolio(w_prev, b_prev, permnos, b)
    assert np.isclose(out.sum(), 1.0) and np.isclose(out[4], 0.15)
    assert np.isclose(out[1] / out[2], (0.25 * 0.55 / 0.5) / (0.55 * 0.3 / 0.3))

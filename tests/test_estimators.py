"""The assembled covariance matrices of notebook 07 on made-up data: the complete members' block is what the copied
functions return, every matrix is positive semi-definite, and the rules for incomplete members hold."""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from slt import constants as C  # noqa: E402
from slt import estimators as E  # noqa: E402
from slt.source import covariance as K  # noqa: E402

RNG = np.random.default_rng(C.SEED)


def made_up(n_members=60, start="2005-01-03", n_days=1000):
    days = pd.bdate_range(start, periods=n_days)
    F = pd.DataFrame(RNG.normal(0, 0.01, (n_days, 6)), index=days, columns=list(C.FACTOR_MODEL_FACTORS))
    B = RNG.normal(0, 0.6, (n_members, 6)); B[:, 0] += 1.0
    R = F.to_numpy() @ B.T + RNG.normal(0, 0.015, (n_days, n_members))
    permnos = list(range(10001, 10001 + n_members))
    X = pd.DataFrame(R, index=days, columns=permnos)
    bench = pd.Series(X.mean(axis=1), index=days)
    sectors = ((["45"] * 20 + ["40"] * 20 + ["10"] * 15 + [C.UNCLASSIFIED] * 5) * (n_members // 60 + 1))[:n_members]
    members = pd.DataFrame({"permno": permnos, "cap": RNG.uniform(1, 10, n_members), "sector": sectors})
    return X, F, bench, members


def test_complete_members_and_the_falling_bar():
    X, *_ = made_up(n_members=10, n_days=100)
    X.iloc[:3, 0] = np.nan            # 3 days missing: complete
    X.iloc[:40, 1] = np.nan           # 40 days missing: not complete
    sel = E.complete_members(X, bar=5, ceiling=21)
    assert 10001 in sel["complete"] and 10002 not in sel["complete"]
    assert sel["left_out"] == 3 and sel["bar"] == 5 and len(sel["kept"]) == 97
    for j in range(2, 10):            # eight members each missing 4 different days: 32 days would go
        X.iloc[50 + 4 * j: 54 + 4 * j, j] = np.nan
    sel = E.complete_members(X, bar=5, ceiling=21)
    assert sel["bar"] < 5 and sel["left_out"] <= 21


@pytest.fixture(scope="module")
def one_month():
    X, F, bench, members = made_up()
    X.iloc[:700, 0] = np.nan          # listed 300 days before the month: a model row on its own days
    X.iloc[:-30, 1] = np.nan          # 30 returns: a stand-in from its sector
    X.iloc[:-10, 59] = np.nan         # 10 returns and no sector: a stand-in from the universe
    X.iloc[[5, 9], 2] = np.nan        # two gaps: complete, two days left out
    month = pd.Period(X.index[-1], "M") + 1
    res = E.month_estimates(month, X, F, bench, members)
    return res, X, F, bench, members, month


def test_counts(one_month):
    res, *_ = one_month
    assert res["model_rows"] == 3 and res["model_rows_under_60"] == 2 and res["model_rows_under_252"] == 2
    assert res["complete"] == 57 and res["bar"] == 5 and not res["bar_fell"]


def test_the_complete_block_is_what_the_copied_functions_return(one_month):
    res, X, F, bench, members, month = one_month
    days = E.window_days(X.index, month)
    order = res["matrices"]["order"]
    A = order[:res["complete"]]
    XA = X.loc[days, A].dropna()
    nA = len(A)
    blocks = {"sample": K.sample_cov(XA.to_numpy()), "ledoit_wolf": K.lw_scaled_identity(XA.to_numpy())[0],
              "factor": K.factor_cov(XA.to_numpy(), F.loc[XA.index].to_numpy())[0],
              "pca": K.pca_cov(XA.to_numpy(), C.PCA_N_COMPONENTS, market=bench.loc[XA.index].to_numpy())[0]}
    for name, M in res["matrices"]["assembled"].items():
        assert np.allclose(M[:nA, :nA], blocks[name], rtol=1e-12, atol=1e-16), name
    assert res["matrices"]["checks"]["factor_loadings_equal"] < 1e-10


def test_every_assembled_matrix_is_positive_semidefinite(one_month):
    res, *_ = one_month
    for name, M in res["matrices"]["assembled"].items():
        assert np.allclose(M, M.T), name
        vals = np.linalg.eigvalsh(M)
        assert vals.min() >= -C.PSD_FAULT_TOLERANCE * vals.max(), name
    E.check_positive_semidefinite(res)


def test_a_model_row_follows_the_factor_model_on_its_own_days(one_month):
    res, X, F, bench, members, month = one_month
    order = res["matrices"]["order"]
    j = order.index(10001)
    days = E.window_days(X.index, month)
    y = X.loc[days, 10001].dropna()
    A = np.column_stack([np.ones(len(y)), F.loc[y.index].to_numpy()])
    coef, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
    e = y.to_numpy() - A @ coef
    M = res["matrices"]["assembled"]["factor"]
    nA = res["complete"]
    XA = X.loc[days, order[:nA]].dropna()
    SigF = np.cov(F.loc[XA.index].to_numpy(), rowvar=False, ddof=1)
    b = coef[1:]
    assert np.isclose(M[j, j], b @ SigF @ b + e @ e / (len(y) - 7))


def test_a_stand_in_takes_its_sectors_weighted_loadings(one_month):
    res, X, F, bench, members, month = one_month
    order = res["matrices"]["order"]
    nA = res["complete"]
    j = order.index(10002)               # sector 45, with 19 complete members
    M = res["matrices"]["assembled"]["factor"]
    days = E.window_days(X.index, month)
    XA = X.loc[days, order[:nA]].dropna()
    _, B_A = K.factor_cov(XA.to_numpy(), F.loc[XA.index].to_numpy())
    cap = members.set_index("permno")["cap"]
    same = [i for i, p in enumerate(order[:nA]) if members.set_index("permno").loc[p, "sector"] == "45"]
    w = cap.loc[[order[i] for i in same]].to_numpy(); w = w / w.sum()
    b = w @ B_A[same]
    SigF = np.cov(F.loc[XA.index].to_numpy(), rowvar=False, ddof=1)
    assert np.allclose(M[j, :nA], B_A @ SigF @ b)


def test_benchmark_daily_excess_rescales_over_members_with_a_return():
    days = pd.bdate_range("2010-02-01", periods=3)
    daily = pd.DataFrame({1: [0.01, 0.02, np.nan], 2: [0.03, 0.04, 0.05]}, index=days)
    members = pd.DataFrame({"month": pd.Period("2010-02", "M"), "permno": [1, 2], "cap": [1.0, 3.0]})
    rf = pd.Series(0.0001, index=days)
    out = E.benchmark_daily_excess(daily, rf, members)
    assert np.allclose(out.to_numpy(), [(0.01 + 0.09) / 4 - 0.0001, (0.02 + 0.12) / 4 - 0.0001, 0.05 - 0.0001])


def test_the_fault_check_stops_the_run():
    with pytest.raises(RuntimeError):
        E.check_positive_semidefinite({"month": "2000-01", **{f"smallest_eigenvalue_ratio_{n}": (-1e-6 if n == "pca" else 1e-3)
                                                              for n in C.COV_ESTIMATORS}})


def test_one_class_per_company_keeps_the_larger_capitalisation():
    members = pd.DataFrame({"permno": [1, 2, 3, 4, 5], "cap": [5.0, 9.0, 3.0, 3.0, 1.0], "permco": [10, 10, 20, 20, np.nan]})
    oc = E.one_class_per_company(members)
    assert oc["kept"] == [2, 3, 5]                 # 2 is larger than 1; 3 and 4 tie, so the lower PERMNO
    assert oc["companies_with_more_than_one_class"] == 2 and oc["classes_left_out"] == 2


def test_share_class_pairs_and_the_one_class_check():
    X, F, bench, members = made_up()
    twin = X[10003] + RNG.normal(0, 0.001, len(X))           # a second class of the company of 10003
    X = X.assign(**{"99999": twin}).rename(columns={"99999": 99999})
    members = pd.concat([members, pd.DataFrame({"permno": [99999], "cap": [1.0], "sector": ["45"]})], ignore_index=True)
    members["permco"] = members["permno"]
    members.loc[members["permno"] == 99999, "permco"] = 10003
    month = pd.Period(X.index[-1], "M") + 1
    res = E.month_estimates(month, X, F, bench, members)
    assert res["share_class_pairs"] == 1 and res["pair_correlations"][0] > 0.95
    assert res["companies_with_more_than_one_class"] == 1 and res["classes_left_out"] == 1
    assert res["complete_one_class"] == res["complete"] - 1
    assert res["condition_sample_one_class"] < res["condition_complete_sample"]
    E.check_positive_semidefinite(res)

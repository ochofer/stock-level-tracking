"""The sorts of notebook 06 on made-up data: the signals, the quantiles, the spread and the comparison."""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from slt import constants as C  # noqa: E402
from slt import sorts as S  # noqa: E402

RNG = np.random.default_rng(C.SEED)


def test_quantiles_have_equal_numbers_of_names_to_within_one():
    for n in (7, 10, 23, 500, 503):
        f = pd.DataFrame({"month": pd.Period("2000-01", "M"), "permno": np.arange(n), "signal": RNG.normal(size=n)})
        for q in (5, 10):
            sizes = np.bincount(S.assign(f, q), minlength=q)
            assert sizes.max() - sizes.min() <= 1 and sizes.sum() == n


def test_quantiles_follow_the_signal_and_break_ties_by_permno():
    f = pd.DataFrame({"month": pd.Period("2000-01", "M"), "permno": [5, 3, 9, 1, 7], "signal": [1.0, 1.0, 0.0, 2.0, 1.0]})
    q = S.assign(f, 5)
    order = f.assign(q=q).sort_values("q")["permno"].tolist()
    assert order == [9, 3, 5, 7, 1]


def test_quantiles_are_formed_within_each_month():
    f = pd.DataFrame({"month": [pd.Period("2000-01", "M")] * 5 + [pd.Period("2000-02", "M")] * 5,
                      "permno": list(range(5)) * 2, "signal": [1, 2, 3, 4, 5, 50, 40, 30, 20, 10]})
    q = S.assign(f, 5)
    assert list(q[:5]) == [0, 1, 2, 3, 4] and list(q[5:]) == [4, 3, 2, 1, 0]


def _wide(n_months=30, permnos=(1, 2)):
    months = pd.period_range("2000-01", periods=n_months, freq="M")
    ret = pd.DataFrame(RNG.normal(0.01, 0.05, (n_months, len(permnos))), index=months, columns=list(permnos))
    prc = pd.DataFrame(20.0, index=months, columns=list(permnos))
    return {"prc": prc, "ret": ret}


def test_momentum_compounds_t_minus_12_to_t_minus_2():
    w = _wide()
    t = pd.Period("2001-06", "M")
    ul = pd.DataFrame({"month": [t, t], "permno": [1, 2]})
    out = S.momentum(ul, w, first="2000-01")
    r = w["ret"][1]
    expected = np.prod(1 + r.loc[t - 12:t - 2].to_numpy()) - 1
    assert np.isclose(out.loc[out["permno"] == 1, "signal"].iloc[0], expected)


def test_momentum_conditions_and_skipped_months():
    w = _wide()
    t = pd.Period("2001-06", "M")
    w["prc"].loc[t - 13, 1] = np.nan          # no price at t-13: left out
    w["ret"].loc[t - 2, 2] = np.nan           # no return at t-2: left out
    ul = pd.DataFrame({"month": [t, t], "permno": [1, 2]})
    out = S.momentum(ul, w, first="2000-01").set_index("permno")
    assert np.isnan(out.loc[1, "signal"]) and not out.loc[1, "has_price_t13"]
    assert np.isnan(out.loc[2, "signal"]) and out.loc[2, "has_price_t13"] and not out.loc[2, "has_return_t2"]
    w = _wide()
    w["ret"].loc[t - 5, 1] = np.nan           # a gap inside the window: skipped and counted
    out = S.momentum(ul, w, first="2000-01").set_index("permno")
    r = w["ret"][1].loc[t - 12:t - 2].dropna()
    assert out.loc[1, "skipped"] == 1 and np.isclose(out.loc[1, "signal"], np.prod(1 + r.to_numpy()) - 1)


def test_momentum_needs_thirteen_months_of_history():
    w = _wide()
    ul = pd.DataFrame({"month": [pd.Period("2001-01", "M")], "permno": [1]})   # t-13 is 1999-12, before the file
    assert np.isnan(S.momentum(ul, w, first="2000-01")["signal"].iloc[0])


def test_variance_uses_the_sixty_days_before_the_month_with_a_minimum():
    days = pd.bdate_range("2000-01-03", periods=200)
    daily = pd.DataFrame(RNG.normal(0, 0.02, (200, 3)), index=days, columns=[1, 2, 3])
    daily[3] = np.nan
    daily.iloc[100:115, 2] = 0.01                      # permno 3: 15 returns in the window, under the minimum of 20
    ends = pd.Series(days).groupby(pd.PeriodIndex(days, freq="M")).max()
    t = pd.Period("2000-07", "M")
    ul = pd.DataFrame({"month": [t] * 3, "permno": [1, 2, 3]})
    out = S.variance(ul, daily, ends, first="2000-01").set_index("permno")
    last = ends.loc[t - 1]
    window = daily.loc[:last].iloc[-60:]
    assert np.isclose(out.loc[1, "signal"], window[1].var(ddof=1)) and out.loc[1, "returns"] == 60
    assert np.isnan(out.loc[3, "signal"])


def test_spread_is_the_value_weighted_top_minus_bottom():
    m = pd.Period("2000-01", "M")
    f = pd.DataFrame({"month": m, "permno": range(10), "signal": range(10),
                      "cap": [1, 3, 1, 1, 1, 1, 1, 1, 2, 2], "ret_total": [0.1, 0.2, 0, 0, 0, 0, 0, 0, 0.3, 0.5]})
    s = S.spread(f, 5)
    assert np.isclose(s.loc[m], (2 * 0.3 + 2 * 0.5) / 4 - (0.1 + 3 * 0.2) / 4)


def test_spread_rescales_over_members_with_a_return():
    m = pd.Period("2000-01", "M")
    f = pd.DataFrame({"month": m, "permno": range(10), "signal": range(10), "cap": 1.0,
                      "ret_total": [0.1, np.nan, 0, 0, 0, 0, 0, 0, 0.3, 0.5]})
    assert np.isclose(S.spread(f, 5).loc[m], 0.4 - 0.1)


def test_comparison_and_verdict():
    idx = pd.period_range("2000-01", periods=240, freq="M")
    french = pd.Series(RNG.normal(0.005, 0.04, 240), index=idx)
    ours = french + RNG.normal(0, 0.01, 240)
    r = S.compare(ours, french)
    assert r["months"] == 240 and r["correlation"] > 0.9
    d = ours - french
    assert np.isclose(r["standard_error"], d.std(ddof=1) / np.sqrt(240))
    assert S.judge_e3(r)["met"] == (abs(r["mean_difference"]) <= 2 * r["standard_error"])
    shifted = S.compare(ours + 0.01, french)
    assert not S.judge_e3(shifted)["mean_difference_met"] and S.judge_e3(shifted)["correlation_met"]
    noise = S.compare(pd.Series(RNG.normal(0, 0.04, 240), index=idx), french)
    assert not S.judge_e3(noise)["correlation_met"]


def test_newey_west_with_no_lags_is_the_plain_standard_error_with_denominator_n():
    x = RNG.normal(size=500)
    assert np.isclose(S.newey_west_se(x, 0), x.std(ddof=0) / np.sqrt(500))

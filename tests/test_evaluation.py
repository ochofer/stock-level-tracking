"""The measures of notebooks 09 and 10 on made-up records: the bias statistic over the months with a forecast,
the turnover parts summing to the total under either restoration, the table by month under the cell rule, and
records written as JSON with no nan."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from slt import constants as C  # noqa: E402
from slt import evaluation as V  # noqa: E402


def made_up_records(n=36, seed=1, zero_forecast=0):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2001-01", periods=n, freq="M").astype(str)
    total = rng.uniform(0.01, 0.04, n)
    membership = total * rng.uniform(0.1, 0.4, n)
    rec = pd.DataFrame({
        "active_return": rng.normal(0, 0.002, n), "forecast_te": rng.uniform(0.005, 0.01, n), "cell_ok": True,
        "turnover_total": total, "turnover_membership": membership,
        "turnover_reweighting": total * rng.uniform(0.0, 0.2, n), "turnover_rebalancing": total * rng.uniform(0.3, 0.9, n),
        "turnover_chosen": total - membership * 0.5, "cap_binds": False, "relaxed": False, "status": "optimal",
        "held": 400, "max_abs_active": 0.004, "missing_return_in_drift": 0, "missing_return_held": 0,
        "company_scope_mismatches": 0}, index=pd.Index(idx, name="month"))
    rec["turnover_overlap"] = rec["turnover_membership"] + rec["turnover_reweighting"] + rec["turnover_rebalancing"] - total
    rec.iloc[:zero_forecast, rec.columns.get_loc("forecast_te")] = 0.0
    return rec


def test_the_bias_statistic_leaves_out_the_months_without_a_forecast():
    rec = made_up_records(zero_forecast=6)
    s = V.summarise(rec)
    keep = rec.iloc[6:]
    z = keep["active_return"] / (keep["forecast_te"] / np.sqrt(12))
    assert np.isclose(s["bias"], z.std(ddof=1)) and s["bias_months_left_out"] == 6 and s["bias_months"] == 30
    rec = made_up_records(zero_forecast=35)
    assert np.isnan(V.summarise(rec)["bias"])


def test_the_turnover_parts_less_the_overlap_sum_to_the_total_by_year():
    rec = made_up_records()
    t = V.turnover_by_year(rec)
    assert list(t.columns) == ["year", "months"] + [f"turnover_{k}" for k in C.TURNOVER_FIGURES]
    parts = t["turnover_membership"] + t["turnover_reweighting"] + t["turnover_rebalancing"] - t["turnover_overlap"]
    assert np.allclose(parts, t["turnover_total"])
    assert t["months"].sum() == len(rec)


def test_the_table_by_month_leaves_a_failing_cell_empty_and_counts_it():
    records = {}
    for k, p in enumerate(C.MONTHLY_PATHS):
        rec = made_up_records(seed=k)
        rec.iloc[k % 5, rec.columns.get_loc("cell_ok")] = False
        records[C.PATH_LABELS[p]] = rec
    table, empty = V.active_returns_table(records)
    assert list(table.columns) == ["month"] + list(C.MONTHLY_ACTIVE_COLUMNS)
    assert all(v == 1 for v in empty.values())
    assert int(table.drop(columns="month").isna().sum().sum()) == len(C.MONTHLY_PATHS)


def test_a_summary_is_written_as_json_with_no_nan():
    rec = made_up_records(zero_forecast=36)
    text = json.dumps(V.jsonable(V.summarise(rec)), allow_nan=False)
    assert '"bias": null' in text


def test_an_expectation_for_two_or_more_estimators_is_one_verdict():
    out = {"E5_ledoit_wolf": {"met": False}, "E5_factor": {"met": True}}
    v = V.one_verdict(out, "E5", ("ledoit_wolf", "factor"))
    assert v == {"met": False, "holds_for": ["factor"], "fails_for": ["ledoit_wolf"]}
    out["E5_ledoit_wolf"]["met"] = True
    assert V.one_verdict(out, "E5", ("ledoit_wolf", "factor"))["met"]


def test_the_months_the_cap_gave_way_are_recomputed_from_the_weights(tmp_path):
    """Two months, five members, one capped path under the mandate on which the cap gave way in both. In the first
    month the chosen turnover runs from the benchmark, in the second from the reference portfolio built from the
    first month's weights after its returns; both must equal the record's, and both pass the cell rule. A record
    that disagrees with the weights stops the count."""
    from slt import mandate as M
    months = ["2001-01", "2001-02"]
    cap = np.array([40.0, 20.0, 20.0, 10.0, 10.0])
    ret = np.array([0.02, -0.01, 0.03, 0.0, 0.01])
    rows = [{"month": pd.Period(mo, "M"), "permno": 10 + k, "cap": cap[k], "sector": "45", "gsubind": "", "permco": 100 + k,
             "ret_total": ret[k]} for mo in months for k in range(5)]
    members = pd.DataFrame(rows)
    b = pd.Series(cap / cap.sum(), index=10 + np.arange(5))
    w1 = b + np.array([-0.02, -0.01, 0.01, 0.01, 0.01])
    w2 = b + np.array([-0.03, 0.01, 0.0, 0.01, 0.01])
    ref2 = M.reference_portfolio(M.drift(w1, pd.Series(ret, index=b.index))[0], b.index.to_numpy(), b.to_numpy())
    chosen = [M.one_way(w1, b), M.one_way(w2, ref2)]
    path = next(p for p in C.MONTHLY_PATHS if p[2] is not None)
    label = C.PATH_LABELS[path]
    pd.DataFrame([{"month": mo, "permno": int(k), "weight": float(w[k])} for mo, w in zip(months, [w1, w2]) for k in b.index]) \
      .to_parquet(tmp_path / (label + "_weights.parquet"))
    rec = pd.DataFrame({"relaxed": [True, True], "turnover_chosen": chosen}, index=pd.Index(months, name="month"))
    out = V.cap_relaxed({"members": members}, tmp_path, {label: rec})
    assert list(out["month"]) == months and (out["path"] == label).all() and (out["cap"] == path[2]).all()
    assert np.allclose(out["turnover_chosen"], chosen, atol=1e-15) and out["cell_ok"].all()
    rec.loc["2001-02", "turnover_chosen"] += 1e-6
    import pytest
    with pytest.raises(ValueError):
        V.cap_relaxed({"members": members}, tmp_path, {label: rec})


def test_the_committed_months_the_cap_gave_way_match_the_counts_of_each_path():
    """cap_relaxed.csv holds, for each capped path under the mandate, as many rows as its relaxed_months in
    paths_summary.json, under the columns constants.CAP_RELAXED_COMMITTED names (constants.CAP_RELAXED_COLUMNS)."""
    res = pathlib.Path(__file__).resolve().parents[1] / C.RESULTS_DIR
    table = pd.read_csv(res / "cap_relaxed.csv")
    assert list(table.columns) == ["path", "month", "cap", "turnover_chosen"]
    assert set(table.columns) - {"month"} == set(C.CAP_RELAXED_COLUMNS)
    summary = json.loads((res / "paths_summary.json").read_text())["paths"]
    for p in C.MONTHLY_PATHS:
        label = C.PATH_LABELS[p]
        assert int((table["path"] == label).sum()) == summary[label]["relaxed_months"], label
    assert table["path"].isin([C.PATH_LABELS[p] for p in C.MONTHLY_PATHS if p[2] is not None]).all()
    assert (table["turnover_chosen"] > table["cap"]).all()

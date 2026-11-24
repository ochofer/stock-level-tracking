"""The two additions of 24 November 2026, set after the study's results were known, each on made-up data and, once
committed, against the repository's own tables: the studentised interval on the bias statistic of notebook 09, and
the monthly allocation and selection of notebook 11 with the interval on the difference of their mean absolute
values."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slt import brinson as B  # noqa: E402
from slt import constants as C  # noqa: E402
from slt import evaluation as V  # noqa: E402
from slt.source import paired as PT  # noqa: E402

RES = ROOT / C.RESULTS_DIR


def made_up(n=306, seed=11):
    rng = np.random.default_rng(seed)
    f = rng.uniform(0.004, 0.008, n)
    a = rng.standard_t(5, n) * f / np.sqrt(12) * 1.5
    return a, f


def test_the_studentised_interval_brackets_the_statistic_and_reads_back_as_a_bias_statistic():
    a, f = made_up()
    for block in C.BIAS_STUDENTISED_BLOCKS:
        s = V.bias_band_studentised(a, f, block)
        assert np.isclose(s["bias"], V.bias_statistic(a, f), rtol=0, atol=1e-12)
        assert s["interval_low"] < s["bias"] < s["interval_high"]
        assert s["whole_blocks"] == 306 // block and s["critical_value"] > 0
        # the interval is symmetric on the log variance and read back with the n minus 1 convention
        k = s["months"] / (s["months"] - 1)
        assert np.isclose(np.log(s["interval_low"] ** 2 / k) + np.log(s["interval_high"] ** 2 / k), 2 * s["log_variance"])
        assert V.bias_band_studentised(a, f, block) == s


def test_the_standard_error_is_the_delta_method_on_the_means_of_whole_blocks():
    a, f = made_up(n=50)
    z = a / (f / np.sqrt(12))
    block = 6
    s = V.bias_band_studentised(a, f, block, draws=50)
    mu, g = z.mean(), (z ** 2).mean()
    v = g - mu ** 2
    blocks = np.stack([z - mu, z ** 2 - g], axis=1)[:48].reshape(8, 6, 2).mean(axis=1) * np.sqrt(6)
    psi = blocks.T @ blocks / 8
    grad = np.array([-2 * mu / v, 1 / v])
    assert np.isclose(s["standard_error"], np.sqrt(grad @ psi @ grad / 50))
    assert np.isclose(s["log_variance"], np.log(v))


def test_the_resamples_are_the_paired_tests():
    a, f = made_up(n=60)
    z = a / (f / np.sqrt(12))
    s = V.bias_band_studentised(a, f, 12, draws=200, seed=3)
    idx = PT.circular_block_indices(60, 12, 200, np.random.default_rng(3))
    zs = z[idx]
    mu, g = zs.mean(axis=1), (zs ** 2).mean(axis=1)
    v = g - mu ** 2
    y = np.stack([zs - mu[:, None], zs ** 2 - g[:, None]], axis=-1)
    zeta = np.sqrt(12) * y.reshape(200, 5, 12, 2).mean(axis=2)
    psi = np.einsum("dki,dkj->dij", zeta, zeta) / 5
    grad = np.stack([-2 * mu / v, 1 / v], axis=-1)
    se = np.sqrt(np.einsum("di,dij,dj->d", grad, psi, grad) / 60)
    critical = np.quantile(np.abs(np.log(v) - s["log_variance"]) / se, 0.95)
    assert np.isclose(s["critical_value"], critical)


def test_the_committed_studentised_intervals_are_recomputed_from_the_two_tables_by_month():
    path = RES / "bias_intervals_studentised.csv"
    if not path.exists():
        pytest.skip("the studentised intervals are committed with the run of notebook 09")
    active = pd.read_csv(RES / "active_returns_monthly.csv")
    forecast = pd.read_csv(RES / "forecast_te_monthly.csv")
    again = V.bias_bands_studentised(active, forecast)
    committed = pd.read_csv(path)
    assert list(committed.columns) == list(again.columns)
    assert committed["path"].tolist() == again["path"].tolist()
    num = [c for c in committed.columns if c != "path"]
    assert np.allclose(committed[num].to_numpy(float), again[num].to_numpy(float), rtol=0, atol=2e-8)
    bands = pd.read_csv(RES / "bias_intervals.csv").set_index("path")
    for _, r in committed.iterrows():
        assert np.isclose(r["bias"], bands.loc[r["path"], "bias"], atol=1e-8)
        assert np.isclose(r["percentile_low"], bands.loc[r["path"], "interval_low"], atol=1e-8)
    assert set(committed["block"]) == set(C.BIAS_STUDENTISED_BLOCKS)


def test_the_monthly_table_leaves_empty_the_cells_the_active_return_leaves_empty():
    months = pd.period_range("2001-01", periods=6, freq="M").astype(str)
    labels = {p: C.PATH_LABELS[p] for p in C.ATTRIBUTION_PATHS}
    totals = {p: pd.DataFrame({"allocation": np.linspace(-1e-3, 1e-3, 6), "selection": 2e-4, "interaction": 1e-5},
                              index=pd.Index(months, name="month")) for p in C.ATTRIBUTION_PATHS}
    active = pd.DataFrame({"month": months, **{"active_return_" + C.PATH_LABELS[p]: 1e-3 for p in C.MONTHLY_PATHS}})
    active.loc[2, "active_return_" + labels[C.ATTRIBUTION_PATHS[1]]] = np.nan
    table, empty = B.monthly_table(totals, labels, active)
    assert list(table.columns) == ["month"] + list(C.MONTHLY_ATTRIBUTION_COLUMNS)
    lab = labels[C.ATTRIBUTION_PATHS[1]]
    assert table[f"allocation_{lab}"].isna().tolist() == [False, False, True, False, False, False]
    assert empty[f"selection_{lab}"] == 1 and sum(empty.values()) == 2
    assert "interaction" not in "".join(table.columns)


def test_the_difference_interval_is_bias_bands_resampling_of_the_two_terms_together():
    rng = np.random.default_rng(9)
    n = 306
    x = rng.normal(0, 0.002, n)
    y = rng.normal(0, 0.0015, n)
    out = B.mean_abs_difference_band(x, y)
    assert np.isclose(out["difference"], np.abs(x).mean() - np.abs(y).mean())
    assert out["interval_low"] < out["difference"] < out["interval_high"]
    rr = np.random.default_rng(C.ALLOCATION_SELECTION_SEED)
    k = int(np.ceil(n / 12))
    starts = rr.integers(0, n, size=(C.ALLOCATION_SELECTION_DRAWS, k))
    idx = ((starts[:, :, None] + np.arange(12)[None, None, :]) % n).reshape(C.ALLOCATION_SELECTION_DRAWS, k * 12)[:, :n]
    stats = np.abs(x)[idx].mean(axis=1) - np.abs(y)[idx].mean(axis=1)
    assert np.allclose((out["interval_low"], out["interval_high"]), np.percentile(stats, [2.5, 97.5]))
    assert B.mean_abs_difference_band(x, y) == out


def test_the_committed_difference_is_recomputed_from_the_table_by_month_and_matches_e8():
    path = RES / "allocation_selection_difference.csv"
    if not path.exists():
        pytest.skip("the monthly allocation and selection are committed with the run of notebook 11")
    monthly = pd.read_csv(RES / "allocation_selection_monthly.csv")
    assert list(monthly.columns) == ["month"] + list(C.MONTHLY_ATTRIBUTION_COLUMNS)
    labels = {p: C.PATH_LABELS[p] for p in C.ATTRIBUTION_PATHS}
    again = B.mean_abs_difference_table(monthly, labels)
    committed = pd.read_csv(path)
    num = [c for c in committed.columns if c != "path"]
    assert np.allclose(committed[num].to_numpy(float), again[num].to_numpy(float), rtol=0, atol=2e-8)
    summary = json.loads((RES / "brinson_summary.json").read_text())["paths"]
    for _, r in committed.iterrows():
        total = summary[r["path"]]["total"]
        assert np.isclose(r["allocation_mean_abs"], total["allocation_mean_abs_monthly"], atol=1e-8)
        assert np.isclose(r["selection_mean_abs"], total["selection_mean_abs_monthly"], atol=1e-8)

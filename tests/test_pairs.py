"""The paired tests of notebook 13: the settings and the comparisons as the constants fix them, the copied test on
known answers, and the committed tables against a rerun from the committed active returns."""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slt import constants as C  # noqa: E402
from slt import pairs as PR  # noqa: E402
from slt.source import paired as PT  # noqa: E402

RES = ROOT / C.RESULTS_DIR


def test_the_settings_are_the_ones_set_on_18_november():
    assert C.PAIRED_TEST_DATE == "2026-11-18"
    assert (C.PAIRED_BLOCK_MONTHS, C.PAIRED_DRAWS, C.PAIRED_LEVEL, C.PAIRED_SEED, C.PAIRED_DETECTABLE_POWER) == (12, 2000, 0.95, 20261009, 0.80)
    assert (C.PAIRED_SIZE_PAIRS, C.PAIRED_SIZE_MONTHS, C.PAIRED_SIZE_SD, C.PAIRED_SIZE_T_DOF) == (1000, 306, 0.0023, 5)
    assert C.PAIRED_SIZE_CASES == ((0.99, 20261120), (0.80, 20261121))
    assert (C.PAIRED_SIZE_MAX, C.PAIRED_SIZE_CONSERVATIVE_BELOW) == (0.065, 0.035)


def test_the_seven_comparisons():
    a, b = C.PAIRED_FAMILY_A, C.PAIRED_FAMILY_B
    assert [x[0] for x, _ in a] == ["sample", "factor", "pca"] and all(y == ("ledoit_wolf", "C3", 0.02) for _, y in a)
    assert all(x[1] == "C3" and x[2] == 0.02 for x, _ in a)
    assert [(x[0], x[2]) for x, _ in b] == [("ledoit_wolf", 0.04), ("ledoit_wolf", None), ("factor", 0.04), ("factor", None)]
    assert all(y == (x[0], "C3", 0.005) for x, y in b)
    assert all(p in C.MONTHLY_PATHS for pair in a + b for p in pair)


def test_the_copied_test_finds_a_large_difference_and_keeps_the_tables_conventions():
    rng = np.random.default_rng(6)
    r1, r2 = PT.simulate_null_pair(960, 0.0023, 0.99, 5, rng)
    out = PT.test(r1, r2 * 1.08, draws=499, seed=2)
    assert out["excludes_zero"] and out["upper"] < 0 and out["p_value"] < 0.01
    te = lambda r: float(np.std(r, ddof=1) * np.sqrt(12))   # noqa: E731
    assert out["difference"] == pytest.approx(te(r1) - te(r2 * 1.08))


def test_holm_by_hand():
    assert np.allclose(PT.holm([0.01, 0.04, 0.03, 0.20]), [0.04, 0.09, 0.09, 0.20])


def test_the_test_rejects_a_true_null_at_about_its_level_on_306_months():
    out = PT.size_check(pairs=150, months=306, sd=0.0023, rho=0.99, dof=5, seed=11, block=12, draws=299)
    assert out["rate"] <= 0.05 + 0.036 + 0.02


def test_the_committed_paired_table_is_the_rerun_from_the_committed_active_returns():
    committed = pd.read_csv(RES / "paired_tests.csv")
    active, _ = PR.read_active(ROOT)
    rerun = PR.paired_table(active)
    assert list(committed.columns) == list(rerun.columns) and len(committed) == 7
    assert (committed["comparison"] == ["A"] * 3 + ["B"] * 4).all()
    for col in rerun.columns:
        if rerun[col].dtype.kind in "fi":
            assert np.allclose(committed[col].to_numpy(float), rerun[col].to_numpy(float), atol=1e-7, equal_nan=True), col
        else:
            assert (committed[col].astype(str) == rerun[col].astype(str)).all(), col
    a = committed[committed["comparison"] == "A"]
    assert np.allclose(a["Holm p-value"], PT.holm(a["p-value"]), atol=1e-8)
    assert committed[committed["comparison"] == "B"]["Holm p-value"].isna().all()


def test_the_committed_figures_are_those_the_text_quotes():
    t = pd.read_csv(RES / "paired_tests.csv").set_index("series 1")
    s = t.loc["sample_C3_cap200bp"]
    assert (round(s["difference"], 2), round(s["lower"], 2), round(s["upper"], 2)) == (1.69, 0.78, 2.61)
    f = t.loc["factor_C3_cap400bp"]
    assert (round(f["difference"], 2), round(f["lower"], 2), round(f["upper"], 2)) == (-3.72, -18.81, 11.36)
    lw = t.loc["ledoit_wolf_C3_cap400bp"]
    assert (round(lw["difference"], 2), round(lw["lower"], 2), round(lw["upper"], 2)) == (2.68, -3.20, 8.57)


def test_the_committed_size_check():
    s = pd.read_csv(RES / "paired_size_check.csv")
    assert list(zip(s["correlation"], s["seed"])) == list(C.PAIRED_SIZE_CASES) and (s["pairs"] == 1000).all()
    assert (s["share rejected"] == s["rejected by p-value"] / s["pairs"]).all()
    assert ((s["share rejected"] <= C.PAIRED_SIZE_MAX) == (s["P1"] == "met")).all()


def test_the_factor_rows_without_july_and_december_2008():
    """Added 19 November 2026: the committed table is the rerun, and its figures are those the text quotes."""
    assert C.PAIRED_LEFT_OUT_MONTHS == ("2008-07", "2008-12")
    assert [x[0] for x, _ in C.PAIRED_LEFT_OUT_ROWS] == ["factor", "factor"]
    committed = pd.read_csv(RES / "paired_tests_without_two_months.csv")
    active, _ = PR.read_active(ROOT)
    rerun = PR.left_out_table(active)
    assert list(committed.columns) == list(rerun.columns) and len(committed) == 2 and (committed["months"] == 304).all()
    for col in rerun.columns:
        if rerun[col].dtype.kind in "fi":
            assert np.allclose(committed[col].to_numpy(float), rerun[col].to_numpy(float), atol=1e-7, equal_nan=True), col
    t = committed.set_index("series 1")
    share = "share of the difference between the two variances in those months"
    r4, rn = t.loc["factor_C3_cap400bp"], t.loc["factor_C3_nocap"]
    assert (round(100 * r4[share], 1), round(100 * rn[share], 1)) == (96.7, 98.8)
    assert (round(r4["difference"], 2), round(r4["standard error"], 2), round(r4["critical value"], 2),
            round(r4["lower"], 2), round(r4["upper"], 2)) == (-0.10, 1.03, 2.10, -2.27, 2.06)
    assert (round(rn["difference"], 2), round(rn["standard error"], 2), round(rn["critical value"], 2),
            round(rn["lower"], 2), round(rn["upper"], 2)) == (-0.02, 1.02, 2.06, -2.12, 2.08)

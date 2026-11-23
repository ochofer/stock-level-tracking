"""The figures read the committed results only and draw the same file twice from the same tables."""
import io
import json
import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from slt import constants as C  # noqa: E402
from slt import figures as F  # noqa: E402

RES = F.results_dir()


def test_figure_one_equals_the_committed_summary():
    t = F.tracking_error_by_estimator(RES)
    paths = json.loads((RES / "paths_summary.json").read_text())["paths"]
    assert list(t["estimator"]) == list(F.ORDER)
    for r in t.itertuples():
        s = paths[C.PATH_LABELS[(r.estimator, C.MANDATE, C.MAIN_CAP)]]
        assert r.realised == pytest.approx(1e4 * s["realised_te"], abs=1e-12)
        assert r.forecast == pytest.approx(1e4 * s["forecast_te_mean"], abs=1e-12)
        assert r.bias == s["bias"] and r.bias_months == s["bias_months"]


def test_figure_two_is_one_path_by_year():
    t = F.tracking_error_by_year(RES)
    assert t["year"].is_monotonic_increasing and t["months"].sum() == 306
    assert t["year"].min() == 1999 and t["year"].max() == 2024


def test_figure_three_has_five_caps_for_each_frontier_estimator():
    t = F.frontier(RES)
    assert len(t) == len(C.FRONTIER_ESTIMATORS) * len(C.FRONTIER_CAPS)
    assert set(t["cap"]) == {"0.5", "1", "2", "4", "none"}
    assert (t["total_turnover"] >= t["chosen_turnover"]).all()


def test_figure_four_has_twelve_sectors_and_sums_to_the_total():
    t = F.sector_terms(RES)
    assert list(t["sector"]) == list(C.ATTRIBUTION_SECTORS)
    total = json.loads((RES / "brinson_summary.json").read_text())["paths"][C.PATH_LABELS[("ledoit_wolf", C.MANDATE, C.MAIN_CAP)]]["total"]
    for term in C.ATTRIBUTION_TERMS:
        assert t[term].sum() == pytest.approx(1e4 * total[f"{term}_per_year"], abs=1e-9)


def test_terms_beyond_two_standard_errors_count_out_of_36():
    out = F.terms_beyond_two_se(RES)
    assert set(out) == set(F.ORDER)
    assert all(v["terms"] == 36 and 0 <= v["beyond"] <= 36 for v in out.values())


def test_cap_labels_merge_dots_closer_than_the_gap():
    g = pd.DataFrame({"chosen_turnover": [6.0, 10.0, 13.0, 14.5, 14.9], "realised": [83, 82, 80, 79.6, 79.5],
                      "cap": ["0.5", "1", "2", "4", "none"]})
    labels = F.cap_labels(g, gap=2.0)
    assert [x[2] for x in labels] == ["0.5", "1", "2, 4, none"]
    assert labels[2][0] == pytest.approx((13.0 + 14.5 + 14.9) / 3)


def test_the_same_tables_give_the_same_svg():
    pytest.importorskip("matplotlib")
    for draw, table in [(F.draw_tracking_error, F.tracking_error_by_estimator(RES)), (F.draw_frontier, F.frontier(RES))]:
        a, b = io.BytesIO(), io.BytesIO()
        F.save(draw(table), a)
        F.save(draw(table), b)
        assert a.getvalue() == b.getvalue()
        assert a.getvalue().startswith(b"<?xml")

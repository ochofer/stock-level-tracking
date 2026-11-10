"""Checks on the pre-committed numbers: the counts and formats the code relies on."""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from slt import constants as C  # noqa: E402


def months_between(start, end):
    """Number of months from start to end inclusive, both as YYYY-MM."""
    y0, m0 = map(int, start.split("-"))
    y1, m1 = map(int, end.split("-"))
    return (y1 - y0) * 12 + (m1 - m0) + 1


def test_path_count():
    assert len(C.PATHS) == C.PATH_COUNT == 24
    assert len(set(C.PATHS)) == len(C.PATHS)
    assert C.SPEC_COUNT == C.PATH_COUNT


def test_every_estimator_under_every_constraint_set():
    for cs in C.CONSTRAINT_SETS:
        for est in C.COV_ESTIMATORS:
            cap = C.MAIN_CAP if "turnover_cap" in C.CONSTRAINT_SETS[cs] else None
            assert (est, cs, cap) in C.PATHS


def test_frontier_has_five_caps_per_estimator():
    for est in C.FRONTIER_ESTIMATORS:
        caps = {p[2] for p in C.PATHS if p[0] == est and p[1] == C.MANDATE}
        assert caps == set(C.FRONTIER_CAPS)


def test_evaluation_window():
    assert months_between(C.EVAL_START, C.EVAL_END) == C.EVAL_MONTHS == 384
    first_estimate_needs = int(C.DAILY_FILE_START[:4]) + C.COV_DAILY_WINDOW_YEARS
    assert int(C.EVAL_START[:4]) >= first_estimate_needs
    assert C.EVAL_END == C.CRSP_END[:7]


def test_sectors():
    assert C.SECTOR_COUNT == 12
    assert set(C.TILT_SECTORS) <= set(C.GICS_SECTORS)
    for code in C.EXCLUDED_SUBINDUSTRIES:
        assert re.fullmatch(r"\d{8}", code)
        assert code[:2] in C.GICS_SECTORS


def test_gates_and_expectations_are_defined():
    names = dir(C)
    for g in C.GATES:
        assert any(n.startswith(g + "_") for n in names), g
    for e in C.EXPECTATIONS:
        assert any(n.startswith(e + "_") for n in names), e


def test_g3_window_is_the_daily_file():
    assert C.G3_WINDOW == (C.DAILY_FILE_START, C.CRSP_END)


def test_hashes_are_sha256():
    assert re.fullmatch(r"[0-9a-f]{64}", C.MANIFEST_SHA256)
    assert re.fullmatch(r"[0-9a-f]{64}", C.MANIFEST_SHA256_2026_09_28)
    assert C.MANIFEST_SHA256 != C.MANIFEST_SHA256_2026_09_28
    for _, (src, digest) in {**C.COPIED_FILES, **C.SOURCE_ADAPTED_FILES}.items():
        assert src.startswith("src/bp/")
        assert re.fullmatch(r"[0-9a-f]{64}", digest)

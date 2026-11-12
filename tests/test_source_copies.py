"""The code copied from the source repository: each copy is the source file unchanged below its first line, and
the constants the copies read are the source's.

The source repository is github.com/ochofer/optimal-vs-naive-diversification at constants.SOURCE_COMMIT. Its
files are hashed in constants.COPIED_FILES; a copy fails here if one byte below its first line differs.
src/slt/source/constants.py must hold exactly the names the copies reference, and no module outside
src/slt/source/ may read it. Its values, and the values this study takes from the source unchanged, are compared
with the source's constants file at the commit, read with git, when a clone of the source repository sits beside
this one (as ../B-P or ../optimal-vs-naive-diversification); those comparisons are skipped otherwise.
"""
import ast
import hashlib
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slt import constants as C  # noqa: E402
from slt.source import constants as SC  # noqa: E402

SOURCE = ROOT / C.COPIED_INTO
COPIES = sorted(p for p in SOURCE.glob("*.py") if p.name not in ("__init__.py", "constants.py"))
FIRST_LINE = re.compile(r"^# Copied verbatim from (src/bp/\w+\.py) at commit (\w+) of github\.com/ochofer/optimal-vs-naive-diversification;")

# Where this study takes a value from the source unchanged: this study's name, and the source's.
TAKEN_UNCHANGED = {
    "COV_ESTIMATORS": "COV_ESTIMATORS",
    "LW_TARGET": "LW_TARGET",
    "FACTOR_MODEL_FACTORS": "FACTOR_MODEL_FACTORS",
    "PCA_FIRST_COMPONENT": "PCA_FIRST_COMPONENT",
    "DAILY_TO_MONTHLY_SCALE": "DAILY_TO_MONTHLY_SCALE",
    "COV_DAILY_WINDOW_YEARS": "EXT_COV_DAILY_WINDOW_YEARS",
    "PCA_N_COMPONENTS": "PCA_N_COMPONENTS",
    "TURNOVER_CAP_MONTHLY_ONE_WAY": "TURNOVER_CAP_MONTHLY_ONE_WAY",
    "TILT_MAX_SHARE_OF_BENCHMARK": "TILT_MAX_SHARE_OF_BENCHMARK",
    "OPT_SOLVER": "OPT_SOLVER",
    "OPT_SOLVER_OPTIONS": "OPT_SOLVER_OPTIONS",
    "OPT_TOL": "OPT_TOL",
}


def test_the_copies_due_by_this_milestone_are_present():
    present = {p.name for p in COPIES}
    assert {n for n, m in C.COPIED_FOR_MILESTONE.items() if m <= 2} <= present
    assert present <= set(C.COPIED_FILES)


@pytest.mark.parametrize("path", COPIES, ids=lambda p: p.name)
def test_each_copy_is_the_source_file_below_its_first_line(path):
    data = path.read_bytes()
    first, rest = data.split(b"\n", 1)
    m = FIRST_LINE.match(first.decode())
    assert m, f"{path.name}: the first line does not name the source file and the commit"
    source_path, digest = C.COPIED_FILES[path.name]
    assert m.group(1) == source_path and m.group(2) == C.SOURCE_COMMIT
    assert hashlib.sha256(rest).hexdigest() == digest, f"{path.name} differs from the source file"


def test_the_source_constants_hold_exactly_the_names_the_copies_reference():
    referenced = set()
    for path in COPIES:
        referenced |= set(re.findall(r"\bC\.([A-Z_][A-Z0-9_]*)", path.read_text()))
    held = {n for n in dir(SC) if n.isupper()}
    assert held == referenced, f"held and not referenced: {sorted(held - referenced)}; referenced and not held: {sorted(referenced - held)}"


def test_no_module_outside_the_copies_reads_the_source_constants():
    pattern = re.compile(r"source\s+import\s+[^\n]*\bconstants\b|source\.constants|from\s+\.source\s+import\s+constants")
    readers = []
    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "notebooks").glob("*.ipynb")) + [ROOT / "build_notebooks.py"]:
        if path.is_file() and SOURCE not in path.parents and pattern.search(path.read_text()):
            readers.append(str(path.relative_to(ROOT)))
    assert not readers, f"modules outside {C.COPIED_INTO} that read its constants: {readers}"


def _source_constants_at_commit():
    for name in ("B-P", "optimal-vs-naive-diversification"):
        clone = ROOT.parent / name
        if (clone / ".git").exists():
            r = subprocess.run(["git", "show", f"{C.SOURCE_COMMIT}:src/bp/constants.py"], cwd=clone, capture_output=True)
            if r.returncode == 0:
                return r.stdout
    return None


@pytest.fixture(scope="module")
def source_constants():
    data = _source_constants_at_commit()
    if data is None:
        pytest.skip("no clone of the source repository beside this one")
    assert hashlib.sha256(data).hexdigest() == C.SOURCE_CONSTANTS_SHA256
    assert len(data.decode().splitlines()) == 752
    values = {}
    exec(compile(data.decode(), "constants.py", "exec"), values)
    return data.decode(), values


def test_the_source_constants_equal_the_source_file_at_the_commit(source_constants):
    text, source = source_constants
    ours = {n: getattr(SC, n) for n in dir(SC) if n.isupper()}
    differ = {n: (v, source.get(n)) for n, v in ours.items() if source.get(n) != v}
    assert not differ, f"values that differ from the source: {differ}"
    lines = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    lines.setdefault(t.id, node.lineno)
    held = (SOURCE / "constants.py").read_text()
    for n in ours:
        m = re.search(rf"^{n} = .*# line (\d+)$", held, flags=re.M)
        assert m and int(m.group(1)) == lines[n], n


def test_the_values_taken_from_the_source_unchanged_equal_the_source(source_constants):
    _, source = source_constants
    differ = {ours: (getattr(C, ours), source[theirs]) for ours, theirs in TAKEN_UNCHANGED.items()
              if getattr(C, ours) != source[theirs]}
    assert not differ, differ


def test_the_values_taken_from_the_source_that_the_copies_also_read_are_equal():
    for ours, theirs in TAKEN_UNCHANGED.items():
        if hasattr(SC, theirs):
            assert getattr(C, ours) == getattr(SC, theirs), ours

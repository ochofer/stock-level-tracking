"""Builds the dashboard, dashboard/index.html, from the tables committed in results/<vintage>/ and from template.html.

Usage, from the repository root, on any machine with a clone (no extraction is read):
    python3 dashboard/build_dashboard.py [--date YYYY-MM-DD]

Every number on the page is read from a committed file by src/slt/report.py and substituted into the template, so that
the page and the tables cannot differ. The source line under each exhibit is for the reader and names no file: the
Method view's table "Where each exhibit comes from" names each exhibit's committed files and the notebooks that write
them, read from the notebooks, each linked on GitHub at the commit that last changed it. The build numbers the figures and the tables in page order and writes the study map at the top of each part's view and under
the Summary's key figures. The four charts are drawn in the browser by Plotly from the data embedded in the page,
and every chart has its table under a toggle, written as HTML so that it reads without the charting library. One table
it reads is by month, cap_relaxed.csv, the months in which the turnover cap gave way, for the text of Figure 3:
everything else is summaries over the window and tables by year, and the page's data holds only the tables of its four
charts. The build date is the day the script runs, in Amsterdam, unless --date gives it, so that a rebuild on a given
date from the same files, each at the same last commit, is byte-identical.
"""
import argparse
import datetime
import glob
import html
import json
import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO, "src"))

from slt import constants as C  # noqa: E402
from slt import report as R  # noqa: E402

OUT = os.path.join(SCRIPT_DIR, "index.html")
REPO_URL = "https://github.com/ochofer/stock-level-tracking"

# The study map: the study's three parts in the order it builds them, each under its tab's
# name, with a subtitle in the reader's words. The names and subtitles are as cleared on 18 November 2026.
PARTS = [("data", "Benchmark", "A point-in-time S&P 500 universe, built and checked"),
         ("mandate", "Mandate", "Its tracking error and what its turnover is made of"),
         ("attribution", "Attribution", "The mandate's active return by sector and by factor")]

# Where each exhibit comes from: the committed files each exhibit is drawn from, by the exhibit's key in the
# template. The notebook that writes each file is read from the notebooks themselves.
SOURCES = {
    "f-estimators": ["paths_summary.json"],
    "t-verdicts": ["verdicts_milestone1.json", "verdicts_milestone2.json", "verdicts_milestone3.json", "verdicts_milestone4.json",
                   "src/slt/constants.py"],
    "t-checks": ["reconstruction_summary.json", "universe_summary.json", "answer_key_summary.json", "covariance_summary.json"],
    "t-condition": ["covariance_by_year.csv"],
    "f-years": ["paths_by_year.csv"],
    "f-frontier": ["paths_summary.json", "cap_relaxed.csv"],
    "t-paths": ["paths_summary.json"],
    "t-turnover": ["paths_summary.json", "verdicts_milestone3.json"],
    "f-sectors": ["brinson_summary.json", "tilt_cap.csv"],
    "t-terms": ["brinson_summary.json", "attribution_counts_by_year.csv", "verdicts_milestone4.json"],
    "t-factors": ["factor_attribution_summary.json", "verdicts_milestone4.json"],
}


def number_exhibits(tpl):
    """Numbers the figures and the tables in page order and writes their numbers, anchors and references:
    "Figure {{n:key}}" or "Table {{n:key}}" in a title, {{id:key}} for its anchor, {{ref:key}} for a reference."""
    order, count = [], {"Figure": 0, "Table": 0}
    for kind, key in re.findall(r"(Figure|Table) \{\{n:([\w-]+)\}\}", tpl):
        assert key not in [k for k, _, _ in order], f"exhibit {key} is numbered twice"
        count[kind] += 1
        order.append((key, kind, count[kind]))
    num = {k: (kind, n) for k, kind, n in order}
    tpl = re.sub(r"\{\{n:([\w-]+)\}\}", lambda m: str(num[m.group(1)][1]), tpl)
    tpl = re.sub(r"\{\{id:([\w-]+)\}\}", lambda m: f"{num[m.group(1)][0].lower()}-{num[m.group(1)][1]}", tpl)
    tpl = re.sub(r"\{\{ref:([\w-]+)\}\}", lambda m: f"{num[m.group(1)][0]} {num[m.group(1)][1]}", tpl)
    return tpl, order


def study_map(current=None):
    """The row of puzzle pieces, one for each part, the current part marked, or none on the Summary."""
    cls = "ch-map" if current else "ch-map ch-map--summary"
    pieces = [f'  <a class="ch-map__piece" href="#{view}"' + (' aria-current="step"' if view == current else "")
              + f'><span class="ch-map__n">{k}</span><span class="ch-map__name">{html.escape(name)}</span>'
              f'<span class="ch-map__sub">{html.escape(sub, quote=False)}</span></a>' for k, (view, name, sub) in enumerate(PARTS, 1)]
    return f'<nav class="{cls}" aria-label="The study in three parts">\n' + "\n".join(pieces) + "\n</nav>"


def last_commit(path, _seen={}):
    """The commit that last changed a file of the repository, which its link in the Method view points to. It moves
    only when that file changes, so a rebuild of the page on its own commit is byte-identical. Without git the link
    points to the branch."""
    if path not in _seen:
        try:
            out = subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%H", "--", path],
                                 capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            out = ""
        _seen[path] = out or "main"
    return _seen[path]


def writers():
    """Each committed file and the notebooks that write it, read from the notebooks."""
    found = {}
    for nb in sorted(glob.glob(os.path.join(REPO, "notebooks", "*.ipynb"))):
        cells = json.load(open(nb, encoding="utf-8"))["cells"]
        src = "".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
        for f in re.findall(r'write_results\("([^"]+)"', src):
            found.setdefault(f, []).append(os.path.basename(nb))
    return found


def exhibit_sources(page, order):
    """The Method view's table: each exhibit's number and title, its committed files and the notebooks that write
    them, each linked on GitHub at the commit that last changed it."""
    by = writers()
    rows = []
    for key, kind, n in order:
        title = re.search(rf'<h2[^>]*>{kind} {n} \| (.*?)</h2>', page).group(1)
        files, nbs = [], []
        for f in SOURCES[key]:
            path = f if "/" in f else f"{C.RESULTS_DIR}/{f}"
            assert os.path.exists(os.path.join(REPO, path)), path
            files.append(f'<a href="{REPO_URL}/blob/{last_commit(path)}/{path}">{f}</a>')
            if "/" not in f:
                assert f in by, f"no notebook writes {f}"
                nbs += [nb for nb in by[f] if nb not in nbs]
        nbs.sort()
        links = [f'<a href="{REPO_URL}/blob/{last_commit("notebooks/" + nb)}/notebooks/{nb}">notebook&nbsp;{nb[:2]}</a>' for nb in nbs]
        rows.append([f"{kind} {n} | {title}", ", ".join(files), ", ".join(links)])
    head = ('<p class="cap">The committed tables each exhibit is drawn from, each linked on GitHub at the commit that last '
            'changed it, and the notebooks that write them.</p>')
    return head + '<div class="tw">' + R.table(["Exhibit", "Committed tables", "Written by"], rows, "sources") + "</div>"

ap = argparse.ArgumentParser()
ap.add_argument("--date", help="the build date, YYYY-MM-DD; the day in Amsterdam by default")
args = ap.parse_args()
P, D = R.build(REPO, datetime.date.fromisoformat(args.date) if args.date else None)

tpl = open(os.path.join(SCRIPT_DIR, "template.html"), encoding="utf-8").read()
tpl, order = number_exhibits(tpl)
assert sorted(SOURCES) == sorted(k for k, _, _ in order), "every exhibit has its sources, and only the exhibits"
P = dict(P, map_summary=study_map(), exhibit_sources="@@EXHIBIT_SOURCES@@", **{f"map_{v}": study_map(v) for v, _, _ in PARTS})
missing = sorted(set(re.findall(r"\{\{(\w[\w-]*)\}\}", tpl)) - set(P))
assert not missing, missing
page = re.sub(r"\{\{(\w[\w-]*)\}\}", lambda m: P[m.group(1)], tpl)
assert "{{" not in page
page = page.replace("@@EXHIBIT_SOURCES@@", exhibit_sources(page, order))
for line in re.findall(r'<p class="src">(.*?)</p>', page):
    assert not re.search(r"results/|\.json|\.csv|\.py\b|notebook", line), f"a source line names a file: {line}"
page = page.replace("/*DATA*/null", json.dumps(D, separators=(",", ":")))
open(OUT, "w", encoding="utf-8").write(page)
print("written", os.path.relpath(OUT, REPO), len(page), "bytes; build date", P["build_date"], "; results", C.RESULTS_DIR)

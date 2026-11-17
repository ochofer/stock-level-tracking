"""Builds the dashboard, dashboard/index.html, from the tables committed in results/<vintage>/ and from template.html.

Usage, from the repository root, on any machine with a clone (no extraction is read):
    python3 dashboard/build_dashboard.py [--date YYYY-MM-DD]

Every number on the page is read from a committed file by src/slt/report.py and substituted into the template, so that
the page and the tables cannot differ; the source line under each exhibit names the file. The four charts are drawn in
the browser by Plotly from the data embedded in the page, and every chart has its table under a toggle, written as HTML
so that it reads without the charting library. One table it reads is by month, cap_relaxed.csv, the months in which the
turnover cap gave way, for the text of Figure 3: everything else is summaries over the window and tables by year, and
the page's data holds only the tables of its four charts. The build date is the day the script runs, in Amsterdam,
unless --date gives it, so that a rebuild on a given date from the same files is byte-identical.
"""
import argparse
import datetime
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO, "src"))

from slt import constants as C  # noqa: E402
from slt import report as R  # noqa: E402

OUT = os.path.join(SCRIPT_DIR, "index.html")

ap = argparse.ArgumentParser()
ap.add_argument("--date", help="the build date, YYYY-MM-DD; the day in Amsterdam by default")
args = ap.parse_args()
P, D = R.build(REPO, datetime.date.fromisoformat(args.date) if args.date else None)

tpl = open(os.path.join(SCRIPT_DIR, "template.html"), encoding="utf-8").read()
missing = sorted(set(re.findall(r"\{\{(\w[\w-]*)\}\}", tpl)) - set(P))
assert not missing, missing
page = re.sub(r"\{\{(\w[\w-]*)\}\}", lambda m: P[m.group(1)], tpl)
assert "{{" not in page
page = page.replace("/*DATA*/null", json.dumps(D, separators=(",", ":")))
open(OUT, "w", encoding="utf-8").write(page)
print("written", os.path.relpath(OUT, REPO), len(page), "bytes; build date", P["build_date"], "; results", C.RESULTS_DIR)

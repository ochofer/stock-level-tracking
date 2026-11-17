"""Builds the note, note/stock-level-tracking_note_2026-11.html and its PDF, from the tables committed in results/<vintage>/,
the figures notebook 12 drew in dashboard/figures/ and note/template.html.

Usage, from the repository root, on any machine with a clone (no extraction is read):
    python3 note/build_note.py [--date YYYY-MM-DD] [--no-pdf]

Every number the template names is read from a committed file by src/slt/report.py, as on the dashboard, so the note and
the dashboard give each figure in the same words. The PDF is the HTML printed by Chromium through Playwright on A4, with
the author written into its metadata after printing, because Chromium writes the title and not the author.
"""
import argparse
import datetime
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO, "src"))

from slt import report as R  # noqa: E402

NAME = "stock-level-tracking_note_2026-11"
HTML_OUT = os.path.join(SCRIPT_DIR, NAME + ".html")
PDF_OUT = os.path.join(SCRIPT_DIR, NAME + ".pdf")
FOOTER = ('<div style="width:100%;font:7.5pt Georgia,serif;color:#6b7280;padding:0 20mm;display:flex;justify-content:space-between">'
          '<span>Carlo Hofer &middot; Stock-level index tracking &middot; Note, November 2026</span>'
          '<span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span></div>')

ap = argparse.ArgumentParser()
ap.add_argument("--date", help="the build date, YYYY-MM-DD; the day in Amsterdam by default")
ap.add_argument("--no-pdf", action="store_true", help="write the HTML only")
args = ap.parse_args()
P, _ = R.build(REPO, datetime.date.fromisoformat(args.date) if args.date else None)

tpl = open(os.path.join(SCRIPT_DIR, "template.html"), encoding="utf-8").read()
missing = sorted(set(re.findall(r"\{\{(\w[\w-]*)\}\}", tpl)) - set(P))
assert not missing, missing
page = re.sub(r"\{\{(\w[\w-]*)\}\}", lambda m: P[m.group(1)], tpl)
assert "{{" not in page
for img in re.findall(r'<img src="([^"]+)"', page):
    assert os.path.exists(os.path.join(SCRIPT_DIR, img)), f"{img} is missing: run notebook 12 first"
open(HTML_OUT, "w", encoding="utf-8").write(page)
print("written", os.path.relpath(HTML_OUT, REPO), len(page), "bytes")

if not args.no_pdf:
    from playwright.sync_api import sync_playwright
    from pypdf import PdfReader, PdfWriter
    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page()
        pg.goto("file://" + HTML_OUT)
        pg.emulate_media(media="print")
        pg.wait_for_load_state("networkidle")
        pg.pdf(path=PDF_OUT, format="A4", print_background=True, display_header_footer=True,
               header_template="<span></span>", footer_template=FOOTER,
               margin={"top": "20mm", "bottom": "20mm", "left": "20mm", "right": "20mm"})
        browser.close()
    reader = PdfReader(PDF_OUT)
    writer = PdfWriter(clone_from=reader)
    title = re.search(r"<title>(.*?)</title>", page).group(1)
    writer.add_metadata({"/Title": title, "/Author": "Carlo Hofer", "/Creator": "HTML rendered to PDF with Chromium"})
    with open(PDF_OUT, "wb") as f:
        writer.write(f)
    print("written", os.path.relpath(PDF_OUT, REPO), os.path.getsize(PDF_OUT), "bytes,", len(reader.pages), "pages")

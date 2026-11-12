"""
Builds the notebooks in notebooks/ from this file and from the modules in src/slt/.

A .ipynb is JSON with outputs and execution counts stored beside the code, so a hand-edited notebook
cannot be reviewed as a diff. The notebooks here are build artefacts: this script is the source of
the prose, and src/slt/ is the source of the code. The notebooks import the modules from the
repository checkout and run on the machine that holds the extraction, with the data root set; they
stop with one sentence when it is not.

Run:  python3 build_notebooks.py
Notebooks are written without outputs. A notebook is never committed with outputs.

Carlo Hofer, 2026.
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "notebooks"


def _lines(src: str) -> list:
    """Split text into the list-of-strings form a .ipynb expects (each line keeps its newline)."""
    parts = src.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text.strip("\n"))}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": _lines(text.strip("\n"))}


def notebook(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def write(name: str, cells: list) -> pathlib.Path:
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False) + "\n")
    return path


# The first code cell of every notebook: find the repository, import the modules, find the data root.
SETUP = """
import json
import pathlib
import sys
import time

ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from slt import constants as C
from slt import extraction as X

root = X.data_root()             # stops with one sentence when the data root is not set
manifest = X.check_manifest(root)   # stops when the manifest on disk is not the one the constants fix
print("manifest SHA-256", C.MANIFEST_SHA256[:12], "| files", manifest["counts"]["files"],
      "| rows", f"{manifest['counts']['rows']:,}")
"""

# ---------------------------------------------------------------------------
# 01: the extraction
# ---------------------------------------------------------------------------

NB01 = [
    md("""
# 01. The extraction: what the study reads, checked before anything is read

**What this notebook does.** It checks that the CRSP and Compustat files on this machine are the extraction the study is built on, lists the tables the study reads with their row counts, fetch dates and hashes, and writes the provenance (the record of what a result was built from) that every later result names. It reads no row of any table. Everything it reports comes from the extraction's manifest and from each file's own metadata.

**Why it comes first.** Every number in the later notebooks is a claim about these files. If a file had changed since it was fetched, or the manifest on disk were another one, those numbers would describe an extraction other than the one the constants name. The check takes under a minute and runs before anything else.

**Terms used in this notebook.**

- *extraction*: the copy of CRSP and Compustat tables taken from Wharton Research Data Services, accessed through Queen Mary University of London, on 28 September 2026, with one table, the GICS history, added on 10 November 2026. It stays on the machine that holds it and is never committed.
- *manifest*: the extraction's own record. For every data file it lists the query that produced it, its row and column counts, its size, its SHA-256 hash and the time it was fetched.
- *SHA-256 hash*: a fingerprint of 64 characters computed from a file's bytes. Two files with the same hash are the same file, and a change of one byte changes the hash.
- *data root*: the folder that holds the extraction, named by the environment variable `SLT_DATA_ROOT` on the machine that holds it.
- *vintage*: the version of the data a result was built from, here the extraction's date and its manifest's hash.
- *provenance*: the record of what a result was built from. The provenance record is the file `results/20260928/provenance.json`, which names the vintage, the fetch date of every table used and the versions of the tools.

**What I expect to see, and where it comes from.** From the manifest, read on 10 November 2026: its SHA-256 begins `23e96d18`, and it lists 310 files and 112,161,796 rows. The study reads 9 of its tables, in 45 files with 73,130,868 rows: eight tables fetched on 28 September 2026 and the GICS history fetched on 10 November 2026. Every file's hash on disk equals the manifest's, every file's row count in its own footer equals the manifest's, and every column the later notebooks read is present. This notebook judges no gate and no expectation. A mismatch stops the study before anything is computed. Running time: under a minute, most of it spent hashing the daily stock file, 2.2 GB in 37 yearly files.
"""),
    code(SETUP),
    md("""
## 1. The tables the study reads

The constants name ten tables. One of them, Compustat's current classification, is read only if the GICS history could not be pulled, and it was pulled, so nine are used. Each row below is one table: how many files it is stored in, its rows and columns, and the date it was fetched, in Amsterdam time.
"""),
    code("""
records = X.files_used(manifest)
files = pd.DataFrame(records)
by_table = (files.groupby(["key", "table"], as_index=False)
                 .agg(files=("file", "size"), rows=("rows", "sum"), columns=("columns", "first"), fetch_date=("fetch_date", "max")))
print(by_table.to_string(index=False))
print()
print("tables", len(by_table), "| files", len(files), "| rows", f"{files['rows'].sum():,}")
print("fetch dates:", X.fetch_dates(records))
"""),
    md("""
## 2. The files on disk against the manifest

Two checks per file, each independent of the other. The SHA-256 of the file on disk is computed again and compared with the manifest's. The row count is read from the file's own footer, which the parquet format writes when the file is saved, and compared with the count the pull script wrote into the manifest. A file that fails either stops the study here.
"""),
    code("""
t0 = time.time()
checks = pd.DataFrame(X.verify_files(root, records))
print(f"checked {len(checks)} files in {time.time() - t0:.0f} s")
print("present", int(checks["exists"].sum()), "| hash equal to the manifest's", int(checks["sha256_matches"].sum()),
      "| row count equal to the manifest's", int(checks["rows_match"].sum()))
failed = checks[~(checks["exists"] & checks["sha256_matches"] & checks["rows_match"])]
if len(failed):
    print(failed.to_string(index=False))
    raise SystemExit("At least one file differs from the manifest, so the study stops here.")
"""),
    md("""
## 3. The columns the later notebooks read

Each later notebook reads only the columns it needs, listed in `extraction.COLUMNS`. A column missing from a file's schema would only show up when that notebook runs, so the schemas are checked here, once, before any of them depends on it.
"""),
    code("""
lacking = X.missing_columns(root, records)
print("files lacking a column a later notebook reads:", len(lacking))
if lacking:
    print(lacking)
    raise SystemExit("A column a later notebook reads is missing, so the study stops here.")
"""),
    md("""
## 4. What is committed

Two files go into `results/20260928/`. The first, `extraction_files.csv`, has one row per file used: its key, table, path, the query that produced it, its rows, columns and size, its SHA-256 and its fetch date. It is the manifest's metadata for the files used and holds no value from inside any file. With it, a reader who has access to Wharton Research Data Services can run the same queries and compare the hashes. The second, `provenance.json`, is the provenance record that every later result names.
"""),
    code("""
committed = files[["key", "table", "file", "query", "rows", "columns", "bytes", "sha256", "fetch_date"]]
p1 = X.write_results("extraction_files.csv", frame=committed)
record = X.provenance(manifest, records, written_by="notebooks/01_extraction.ipynb")
p2 = X.write_results("provenance.json", record=record)
print("written:", p1.relative_to(ROOT), "and", p2.relative_to(ROOT))
print(json.dumps({k: record[k] for k in ("vintage", "fetch_dates", "files_used", "rows_used", "versions")}, indent=1))
"""),
    md("""
## What was learned, and what could be wrong

What the checks establish: the files on this machine are, byte for byte, the files the manifest records, the manifest is the one the constants fix, and every column the later notebooks read is present. Every later result names this vintage through the provenance record.

What could still be wrong. A hash shows that a file has not changed since it was fetched. It does not show that the file holds what the vendor intended. Notebooks 02 to 04 test that, against CRSP's own counts and returns. CRSP revises its history once a year, so a reader who runs the same queries later receives a later vintage, and the hashes in `extraction_files.csv` will then differ for reasons that are not errors. And the row counts in the manifest and in the footers are both counts of rows as stored. Neither says anything about a value inside a row.
"""),
]



# The second code cell of notebooks 02 to 05: the modules that build the data layer.
MODULES = """
from slt import layer as L
from slt import returns as R
from slt import sectors as S
from slt import universe as U
from slt import validation as V
"""

# ---------------------------------------------------------------------------
# 02: the universe
# ---------------------------------------------------------------------------

NB02 = [
    md("""
# 02. The universe: the point-in-time S&P 500, month by month

**What this notebook does.** It builds the universe from CRSP's list of index members, month by month, and counts it against CRSP's own count of members on the same day. The first month from which the two counts agree within 10 names opens the usable window, and gate G2 is judged on its length. It then sets the universe beside Compustat's list of index members and counts the members whose share code is not ordinary common stock.

**Why it comes here.** Every later figure is a sum or an average over the universe. A stock wrongly in or out of it changes every figure of that month, and a list read with dates not yet known when the month began would let the study use information an investor could not have had.

**Terms used in this notebook.**

- *universe*: the stocks in CRSP's list of index members on the last trading day of the month before. A stock is in the universe in month t when one of its membership spells covers the last trading day of month t−1.
- *membership spell*: one interval in CRSP's list, from the day a stock entered the index to the day it left, both days included. A stock that left and came back has more than one.
- *PERMNO*: CRSP's permanent number for a security. It does not change when the company changes its name or its ticker.
- *GVKEY*: Compustat's permanent number for a company.
- *link*: CRSP's table of which GVKEY belongs to which PERMNO, and over which dates. Only primary links of types LC and LU are used, with primary flag P or C, each on the dates it was valid.
- *CRSP's count*: the number of index members with a valid price on a day, which CRSP stores on each row of its monthly index file (`totcnt`). Its row for a month carries the count on that month's last trading day.
- *usable window*: the months from the first month in which the universe's count is within 10 names of CRSP's count, to December 2024, CRSP's last month on this subscription.
- *gate*: a test of the data layer, fixed in advance in `src/slt/constants.py`. A gate that is not met stops the study.
- *verdict*: the result of a gate, met or not met, printed in the same words either way.
- *share code*: CRSP's two-digit code for the type of security. Codes 10 and 11 are ordinary common shares. A first digit of 3 marks an ADR, an American depositary receipt: a certificate traded in the United States that stands for shares of a company listed abroad. A second digit of 2 marks a company incorporated outside the United States.

**What I expect to see, and where it comes from.** From structural counts made on 10 and 11 November 2026 (rows, distinct values, earliest and latest dates, read from the files' columns; no row printed): CRSP's list holds 2,064 spells of 1,936 PERMNOs, 119 PERMNOs have more than one spell, the earliest spell starts on 31 December 1925, and 503 spells end on 31 December 2024, the current members. CRSP's monthly index file has 1,189 rows from 31 December 1925, so the universe runs 1,188 months, January 1926 to December 2024.

- *The count.* CRSP builds its count from the same list, so the universe's count is expected within 10 names of CRSP's count in every month, the usable window to open in January 1926 and to run 99 years, and G2, at least 15 years, to be met with room. The same comparison over the usable window is half of expectation E2, judged in notebook 04.
- *Compustat's list.* On this subscription Compustat's history of the index lists the members of September 2026 alone: 503 rows, none with an end date, joined between 31 March 1964 and 21 September 2026. So the cross-check covers names still in the index and says nothing about removals. The share of Compustat's member-months found in the universe is expected near one, below one where a company's GVKEY and the link disagree. Most joins are expected in the same month on both lists, since both date a change by the day it took effect. The two-sided share of disagreement is expected to rise the further back the year, because every name that left the index before September 2026 is in the universe and missing from Compustat's list. None of the three has a verdict.
- *Share codes.* Written before any run: CRSP's total value of the index (`totval`) leaves out ADRs, and the universe filters no share code. So in months in which the list holds ADRs, the universe's summed capitalisation may sit above CRSP's total value. This notebook reports the members outside codes 10 and 11 by year, so the months to look at in notebook 04 are known in advance.

Running time: about a minute.
"""),
    code(SETUP),
    code(MODULES),
    md("""
## 1. CRSP's list of members

The list is read whole: three columns, PERMNO, start and end. Before it is used, five counts say whether it is well formed. Two spells of one PERMNO that overlap, or a spell that ends before it starts, would make membership ambiguous, and each is counted here.
"""),
    code("""
t0 = time.time()
universe = L.load_universe(root)
spells, matrix, ulong = universe["spells"], universe["matrix"], universe["ulong"]
checks = U.spell_checks(spells)
print(json.dumps(checks, indent=1))
print("date values present that did not parse:", universe["unparsed"] or "none")
print(f"months {len(matrix)}, from {matrix.index[0]} to {matrix.index[-1]} | member-months {len(ulong):,} | {time.time() - t0:.0f} s")
"""),
    md("""
## 2. The universe by month

The membership is a table of months by PERMNOs, true where a spell covers the last trading day of the month before. The table below gives the universe's size by decade. Before March 1957 CRSP's list follows the index's predecessor, which held 90 stocks, so the count is expected near 90 there and near 500 from 1957.
"""),
    code("""
size = matrix.sum(axis=1)
by_decade = size.groupby((size.index.year // 10) * 10).agg(["min", "mean", "max"]).round(1)
by_decade.index.name = "decade"
print(by_decade.to_string())
"""),
    md("""
## 3. The count against CRSP's count, and the usable window

The universe of month t is the membership on the last trading day of month t−1, and CRSP's count on its row for month t−1 is the count on that same day, so the two are compared on one day. The committed table gives, by year, the mean difference and the months in which the difference exceeds 10 names. A count of months with any difference at all is printed here and not committed, because in a year with none it would give CRSP's count month by month.
"""),
    code("""
counts = U.counts_against_crsp(matrix, universe["index_file"])
window = U.usable_window(counts)
count_by_year = U.count_check_by_year(counts)
inside = counts[counts.index >= pd.Period(window["first_month"], "M")] if window["first_month"] else counts.iloc[0:0]
print("months with any difference:", int((counts["difference"] != 0).sum()), "of", len(counts), "(printed here, not committed)")
print("months more than", C.USABLE_COUNT_WITHIN, "names apart: whole sample", int((counts["difference"].abs() > C.USABLE_COUNT_WITHIN).sum()),
      "| usable window", int((inside["difference"].abs() > C.USABLE_COUNT_WITHIN).sum()))
print("largest difference in the usable window:", int(inside["difference"].abs().max()) if len(inside) else None)
print(json.dumps(window, indent=1))
print(V.verdict_line("G2", window["g2_met"], f"The usable window runs from {window['first_month']} to {window['last_month']}, "
      f"{window['years']} years, against a level of at least {C.G2_USABLE_YEARS_MIN} years."))
"""),
    md("""
## 4. Compustat's list of members

Compustat keeps its own history of the index's members, by GVKEY (`comp.idxcst_his`, index key `000003`). On this subscription that history lists the members of September 2026 alone, so the cross-check covers names still in the index and says nothing about removals. The differences are reported by year with no verdict, and not reconciled: the universe stays CRSP's list. Three readings are committed.

- *(a) Coverage.* Compustat's members are placed month by month on the universe's rule, each from the month after it joined to December 2024. The share of those member-months found in the universe, through the GVKEY of each universe member on the last trading day of the month before, is reported by year, with the counts of rows, companies, rows whose GVKEY has no primary link, and rows that joined after 31 December 2024.
- *(b) Join dates.* For the same members, the month in which each list dates the join: the same month, within one month, or further apart, with CRSP's date earlier or later. CRSP's date is the start of the membership spell, of a PERMNO linked to the GVKEY, that overlaps Compustat's months the longest. Reading (a) tests only whether Compustat's members are in the universe. Reading (b) tests the dates in both directions.
- *(c) The two-sided share.* The share of the two lists' member-months on which they disagree, by year. It counts every name that left the index before September 2026, which Compustat's history omits, and its column is named for that.

Members of the universe without a GVKEY are counted by year beside them.
"""),
    code("""
links = S.primary_links(X.read_table(root, "crsp_ccm_lnkhist")[0])
linked, more_than_one_link = S.link_gvkeys(ulong, links)
history, bad_history = X.read_table(root, "comp_idxcst_his")
listed = U.index_rows(history)
print("Compustat's rows for the index:", len(listed), "| with an end date:", int(listed["thru"].notna().sum()),
      "| start dates from", listed["from"].min().date(), "to", listed["from"].max().date())
print("member-months with more than one valid primary link:", more_than_one_link)
rows_summary = U.compustat_rows_summary(listed, links)
print(json.dumps(rows_summary))
compustat = U.compustat_members_long(history, universe["ends"])
agreement = U.list_agreement_by_year(linked, compustat)
years = pd.PeriodIndex(linked["month"]).year
base = pd.DataFrame({"universe_months": linked.groupby(years).size(),
                     "universe_months_without_gvkey": linked["gvkey"].isna().groupby(years).sum().astype("int64")})
crosscheck = base.rename_axis("year").reset_index().merge(agreement, on="year", how="left")
shown = crosscheck[(crosscheck["year"] % 10 == 0) | (crosscheck["year"] >= 2015)]
print(shown[["year", "compustat_months", "compustat_months_in_universe", "share_of_compustat_in_universe",
             "two_sided_share_counting_exits_compustat_omits"]].to_string(index=False))
joined = U.join_dates(listed, links, spells)
joins_by_year = U.join_dates_by_year(joined)
join_totals = {k: int(v) for k, v in joins_by_year.drop(columns="year").sum().items()}
print("join dates, all years:", json.dumps(join_totals))
"""),
    md("""
## 5. Share codes

Each member-month takes its share code from CRSP's names record in force on the last trading day of the month before. The universe filters no share code, because CRSP's list is the universe. The count of members outside codes 10 and 11 is reported by year, with the share codes over the whole sample.
"""),
    code("""
names, bad_names = X.read_table(root, "crsp_msenames")
coded, name_overlaps = U.share_codes(ulong, names)
codes_by_year = U.share_codes_by_year(coded)
code_totals = U.share_code_totals(coded)
print(code_totals.to_string(index=False))
print("member-months matched by more than one names record:", name_overlaps)
print(codes_by_year[codes_by_year["months_outside_common_codes"] > 0].to_string(index=False))
"""),
    md("""
## 6. What is committed

Seven files go into `results/20260928/`, each an aggregate with no identifier of a security: the count comparison by year, the usable window with G2's verdict, the comparison with Compustat's list by year (readings a and c), the join dates by the year Compustat dates the join (reading b), the share codes by year and over the whole sample, and a summary of the counts printed in this notebook. Notebooks 03 and 04 read the usable window from its file.
"""),
    code("""
summary = {
    "written_by": "notebooks/02_universe.ipynb", "manifest_sha256": C.MANIFEST_SHA256,
    "membership_rule": C.MEMBERSHIP_RULE, "list": checks,
    "months": int(len(matrix)), "first_month": str(matrix.index[0]), "last_month": str(matrix.index[-1]),
    "member_months": int(len(ulong)), "member_months_with_more_than_one_primary_link": int(more_than_one_link),
    "compustat_rows_for_the_index": int(len(listed)), "compustat_rows_with_an_end_date": int(listed["thru"].notna().sum()),
    "cross_check": C.CROSS_CHECK_REPORT, "compustat_rows": rows_summary, "join_dates_all_years": join_totals,
    "member_months_matched_by_more_than_one_names_record": int(name_overlaps),
    "date_values_not_parsed": {**universe["unparsed"], **bad_history, **bad_names},
}
written = [X.write_results("universe_count_check_by_year.csv", frame=count_by_year),
           X.write_results("usable_window.json", record=window),
           X.write_results("compustat_crosscheck_by_year.csv", frame=crosscheck),
           X.write_results("compustat_join_dates_by_year.csv", frame=joins_by_year),
           X.write_results("share_codes_by_year.csv", frame=codes_by_year),
           X.write_results("share_code_totals.csv", frame=code_totals),
           X.write_results("universe_summary.json", record=summary)]
for path in written:
    print("written:", path.relative_to(ROOT))
"""),
    md("""
## What was learned, and what could be wrong

What the notebook establishes: whether CRSP's list is well formed, how closely the universe built from it agrees with CRSP's own count on the same day, the usable window that agreement opens, and G2's verdict. The comparison with Compustat's list and the share codes are reported without a verdict.

What could still be wrong. The count agreeing does not show that the right stocks are in the universe, only that the number is right: a stock in place of another would leave the count unchanged. Notebook 04 tests membership more finely, because a wrong stock would move the universe's return and its summed capitalisation away from CRSP's. The rule reads membership at the start of the month, so a stock that joins mid-month enters the universe the month after, and one that leaves mid-month stays to the month's end. CRSP's own return counts both from the day of the change, and notebook 04 measures what that costs. And Compustat's history on this subscription lists only the members of September 2026, so it cannot show that the universe's past members are right. Notebook 04 sets the universe's return, each member weighted by its market value, beside the index's own price return over the months to December 2004, when the index still weighted by full capitalisation, which is evidence on past membership independent of CRSP's list.
"""),
]

# ---------------------------------------------------------------------------
# 03: returns, delistings and capitalisation
# ---------------------------------------------------------------------------

NB03 = [
    md("""
# 03. Returns, delistings and market capitalisation

**What this notebook does.** For every universe member in every month it builds the monthly return with any delisting return applied under rule D4, and the market capitalisation (price times shares outstanding) at the start of the month. It counts what rule D4 and the rules on missing values decide, and judges gate G4 on the capitalisation. It then reads the daily stock file once, a year at a time, to judge gate G3 on the stocks that leave the universe and to measure how many member-days have a daily return.

**Why it comes here.** The benchmark of notebook 04 is a weighted sum of these returns with these capitalisations as weights. A delisting left out would drop the loss on the stocks that fail. A capitalisation read at the end of the month would weight each stock by a price not yet known when the month began.

**Terms used in this notebook.**

- *universe*: the stocks in CRSP's list of index members on the last trading day of the month before.
- *member-month*: one stock in the universe in one month.
- *return*: CRSP's monthly return with dividends (`ret`). The *price-only return* (`retx`) leaves dividends out.
- *delisting*: the end of a stock's listing on its exchange. CRSP records its date, a code for its reason (`dlstcd`) and, where it can, the *delisting return* (`dlret`), the return from the last price to the value the holder received.
- *rule D4*: the study's rule for a delisting, after Shumway (1997). CRSP's delisting return where it gives one. Where it is missing, minus 30 per cent for a code from 400 to 599, the codes for a delisting for poor performance, and zero for any other code. CRSP's code 100 marks a security still trading and is not a delisting. In a month with both a return and a delisting return, the two are compounded.
- *market capitalisation*: price times shares outstanding. The capitalisation at the start of month t is the absolute price times shares outstanding at the end of month t−1, in thousands of dollars. CRSP stores the price as negative when it is the midpoint of bid and ask on a day without a trade, hence the absolute value. A price of zero or a missing value gives a missing capitalisation.
- *usable window*: the months from the first in which the universe's count is within 10 names of CRSP's count, to December 2024, as notebook 02 found it.
- *gate*: a test of the data layer, fixed in advance in `src/slt/constants.py`. A gate that is not met stops the study.
- *verdict*: the result of a gate, met or not met, printed in the same words either way.
- *removal*: the end of a membership spell before 31 December 2024. A spell that ends on that date belongs to a current member.
- *trading day*: a day on which the daily stock file has rows.
- *member-day*: a trading day in a month in which the stock is in the universe.

**What I expect to see, and where it comes from.** From the manifest: the monthly stock file has 5,153,763 rows, the delisting file 38,843, and the daily file 37 yearly files, of which the 35 from 1990 to 2024 hold rows, from 2 January 1990 to 31 December 2024. The files for 2025 and 2026 are empty, because CRSP ends on 31 December 2024 on this subscription. From a structural count of 10 November 2026: 9,737 rows of the delisting file carry code 100, and no PERMNO has more than one row there.

- *Rule D4.* A stock usually leaves the index before it fails, so in the universe a delisting is expected to be a merger or an acquisition (codes 200 to 399) with CRSP's delisting return given, and the minus 30 per cent to be used rarely. Every count is printed by year and committed. None has a verdict.
- *Missing values.* A member with no return and no delisting in a month, and a member without a capitalisation, are expected close to zero. Each is counted by year.
- *G4.* CRSP reports price and shares for nearly every listed stock-month, so G4, a capitalisation in at least 90 per cent of a name's membership months for at least 90 per cent of names over the usable window, is expected met with room.
- *G3.* A stock removed from the index usually keeps trading on its exchange, so a price within 10 trading days of the removal is expected for nearly every removal from 2 January 1990, and G3, at least 90 per cent, is expected met with room. Removals before 1990 have no daily file. They are reported on the monthly file with no verdict.
- *Daily coverage.* Member-days with a return are expected close to all member-days in every year, reported with no verdict.

Running time: the monthly part takes about a minute. The daily pass reads the 35 yearly files from 1990 to 2024, 2.2 GB on disk, and prints each year as it finishes.
"""),
    code(SETUP),
    code(MODULES),
    md("""
## 1. The monthly panel

The panel holds one row per member-month: the month's return and price-only return with any delisting applied, the capitalisation from the month before, and how the month's delisting, if any, was treated. It is built in memory from the files on every run and never written. Its number of rows has to equal the number of member-months, because each member-month takes at most one row of each file.
"""),
    code("""
window = X.read_result("usable_window.json")
t0 = time.time()
universe = L.load_universe(root)
data = L.load_panel(root, universe)
panel, monthly = data["panel"], data["monthly"]
repeated = int(monthly.assign(m=pd.PeriodIndex(monthly["date"], freq="M")).duplicated(["permno", "m"]).sum())
print(f"member-months {len(universe['ulong']):,} | panel rows {len(panel):,} | equal: {len(panel) == len(universe['ulong'])}")
print("PERMNO-months with more than one row in the monthly file:", repeated)
print("date values present that did not parse:", {**universe["unparsed"], **data["unparsed"]} or "none")
print(f"usable window from {window['first_month']} | {time.time() - t0:.0f} s")
"""),
    md("""
## 2. Delistings under rule D4

Every delisting month of a member is counted by how its return was set: CRSP's delisting return, minus 30 per cent, or zero. The months with neither a return nor a delisting, and the months without a capitalisation, are counted beside them. Each such member is left out of its month's benchmark return in notebook 04, and the other weights are rescaled.
"""),
    code("""
d4 = R.delisting_by_year(panel)
print(d4.drop(columns="year").sum().to_string())
families = R.delisting_code_families(panel)
print(families.to_string())
print(d4[d4["delisting_months"] > 0].tail(10).to_string(index=False))
"""),
    md("""
## 3. Market capitalisation and gate G4

For each name, the share of its membership months inside the usable window that have a capitalisation. G4 asks that at least 90 per cent of names have one in at least 90 per cent of their months.
"""),
    code("""
g4 = R.cap_coverage(panel, window["first_month"])
print(json.dumps(g4, indent=1))
print(V.verdict_line("G4", g4["g4_met"], f"{g4['names_meeting_month_share']:,} of {g4['names']:,} names ({g4['share_of_names']:.2%}) have a "
      f"capitalisation in at least {C.G4_MONTH_SHARE_MIN:.0%} of their membership months, against a level of {C.G4_NAME_SHARE_MIN:.0%} of names."))
"""),
    md("""
## 4. Removals and the daily file

One pass over the daily file, a year at a time, keeps three things from each year: its trading days, the member-days with a return, and the priced rows of each removed stock within 40 calendar days of its removal, which holds every window of 10 trading days. Nothing else of the file stays in memory, and nothing of it is written.
"""),
    code("""
removed = R.removals(universe["spells"])
in_span = (removed["ending"] >= pd.Timestamp(C.G3_WINDOW[0])) & (removed["ending"] <= pd.Timestamp(C.G3_WINDOW[1]))
print("removals", len(removed), "| from", C.G3_WINDOW[0], "to", C.G3_WINDOW[1], int(in_span.sum()), "| before", int((~in_span).sum()))
t0 = time.time()
daily = L.daily_pass(root, universe["ulong"], removed, progress=True)
calendar = pd.DatetimeIndex(daily["calendar"])
print(f"read in {time.time() - t0:.0f} s | trading days {len(calendar):,}, {calendar[0].date()} to {calendar[-1].date()}"
      f" | priced rows kept near a removal {len(daily['removed_rows']):,}")
print("date values present that did not parse:", daily["unparsed"] or "none")
"""),
    md("""
## 5. Gate G3

For each removal from 2 January 1990, the window runs from 10 trading days before the last trading day on or before the removal date to 10 trading days after it. G3 asks that at least 90 per cent of removals have a price inside their window. Removals before the daily file are reported on the monthly file, as the share with a price at the end of the removal month or of the month before, with no verdict.
"""),
    code("""
windows = R.removal_windows(removed, daily["calendar"])
found = R.priced_near(windows, daily["removed_rows"])
g3 = R.g3_verdict(windows, found)
before = R.removals_before_daily_file(removed, monthly)
print(V.verdict_line("G3", g3["g3_met"], f"{g3['removals_with_price']:,} of {g3['removals']:,} removals from {C.G3_WINDOW[0]} to {C.G3_WINDOW[1]} "
      f"({g3['share']:.2%}) have a price within {C.G3_TRADING_DAYS} trading days, against a level of {C.G3_SHARE_MIN:.0%}."))
print("Before the daily file, with no verdict:", json.dumps(before))
"""),
    md("""
## 6. Daily returns of the members

By year: the member-days expected, which is each month's number of members times its trading days, against the member-days with a return. A member that joins or leaves the exchange inside a month has fewer days than the month, so the share sits a little below one even when nothing is missing.
"""),
    code("""
days = R.trading_days_by_month(daily["calendar"])
coverage = R.daily_coverage_by_year(daily["present"], days, universe["ulong"])
print(coverage.to_string(index=False))
"""),
    md("""
## 7. What is committed

Five files go into `results/20260928/`, each an aggregate with no identifier of a security: rule D4's counts by year, G4's figures and verdict, G3's figures and verdict with the removals before 1990, the daily coverage by year, and a summary of the counts printed in this notebook.
"""),
    code("""
summary = {
    "written_by": "notebooks/03_returns.ipynb", "manifest_sha256": C.MANIFEST_SHA256,
    "member_months": int(len(universe["ulong"])), "panel_rows": int(len(panel)),
    "permno_months_with_more_than_one_monthly_row": repeated,
    "delisting_months_by_code_family": {str(k): int(v) for k, v in families.items()},
    "trading_days": int(len(calendar)), "first_trading_day": str(calendar[0].date()), "last_trading_day": str(calendar[-1].date()),
    "date_values_not_parsed": {**universe["unparsed"], **data["unparsed"], **daily["unparsed"]},
}
written = [X.write_results("delistings_by_year.csv", frame=d4),
           X.write_results("capitalisation_g4.json", record=g4),
           X.write_results("removals_g3.json", record={"in_daily_file": g3, "before_daily_file": before}),
           X.write_results("daily_coverage_by_year.csv", frame=coverage),
           X.write_results("returns_summary.json", record=summary)]
for path in written:
    print("written:", path.relative_to(ROOT))
"""),
    md("""
## What was learned, and what could be wrong

What the notebook establishes: how each delisting month of a member was treated and how many months rule D4's minus 30 per cent decides, how many member-months have no return or no capitalisation, the verdicts of G3 and G4, and the share of member-days with a daily return.

What could still be wrong. Rule D4's minus 30 per cent is an average loss estimated by Shumway (1997) on stocks delisted for performance, and it is applied to every such month with no delisting return, whatever the stock's own loss was. A price within 10 trading days of a removal shows that the removed stock can be valued near the day it leaves. It does not show that the price is the one at which it left. And a capitalisation from shares outstanding counts every share, where the index itself has counted only the shares available to the public since 2005, so the weights here are CRSP's and differ from the index's own.

Shumway, T. (1997). The delisting bias in CRSP data. *Journal of Finance* 52(1), 327 to 340.
"""),
]


# ---------------------------------------------------------------------------
# 04: the reconstruction
# ---------------------------------------------------------------------------

NB04 = [
    md("""
# 04. The reconstruction: the benchmark against CRSP's own figures

**What this notebook does.** It builds the benchmark (the universe with each member weighted by its market value at the start of the month) from the files read in notebooks 02 and 03, and compares the benchmark's return with CRSP's own return of the same list, weighted the same way, with dividends and price-only. That comparison judges gate G1 and expectation E1. It compares the universe's count and summed market value with CRSP's count and total value on the same day, which judges expectation E2. It sets the price-only return beside the index's own price return as S&P publishes it, and does not judge that comparison. And it gathers the data layer's six results in one file.

**Why it comes here.** CRSP computes its return from the same list and the same stock files. If the benchmark built here differs from it by more than two tenths of a percentage point a year, the universe, the returns or the weights are wrong somewhere, and every later result would carry the error.

**Terms used in this notebook.**

- *universe*: the stocks in CRSP's list of index members on the last trading day of the month before.
- *market capitalisation*: price times shares outstanding. The capitalisation at the start of month t is the absolute price times shares outstanding at the end of month t−1.
- *cap-weighted*: weighted by market capitalisation, each stock's weight being its capitalisation divided by the sum over the stocks held. *Value-weighted* is CRSP's word for the same.
- *benchmark*: the cap-weighted universe, with each stock's weight set by its capitalisation at the start of the month, so that nothing unknown when the month begins sets a weight. A member without a capitalisation, or with neither a return nor a delisting in the month, is left out of that month's return and the other weights are rescaled. Both are counted.
- *return with dividends* and *price-only return*: the month's return with dividends reinvested, and without them.
- *CRSP's return*: CRSP's own value-weighted return of the members of its list, computed from its daily data, with dividends (`vwretd`) and price-only (`vwretx`).
- *the index's price return*: S&P's published return of the index without dividends, which CRSP stores beside its own (`sprtrn`). Since 2005 the index weights each stock by its float-adjusted shares, the shares available to the public, which CRSP's share counts are not.
- *CRSP's count* and *CRSP's total value*: the number of index members with a valid price on a day, and the sum of their capitalisations in thousands of dollars, ADRs left out (`totcnt` and `totval`). Each row of CRSP's monthly index file carries them for that month's last trading day.
- *basis point*: one hundredth of a percentage point.
- *mean monthly difference*: the average over the months of the benchmark's return minus CRSP's.
- *tracking error*: the standard deviation of the monthly difference between two returns, times the square root of 12, a yearly figure.
- *usable window*: the months from the first in which the universe's count is within 10 names of CRSP's count, to December 2024, as notebook 02 found it.
- *gate* and *expectation*: tests fixed in advance in `src/slt/constants.py`. A gate that is not met stops the study. An expectation that is not met is reported as a finding.
- *verdict*: the result of a gate or an expectation, met or not met, printed in the same words either way.

**What I expect to see, and where it comes from.** The levels are the constants', committed before any run that judges them.

- *G1*: tracking error against CRSP's value-weighted return with dividends of at most 50 basis points a year over the usable window.
- *E1*: tracking error of at most 20 basis points a year and a mean monthly difference within 2 basis points of zero, with dividends and price-only. CRSP computes its return from the same list and the same file, daily. The differences left are membership changes inside a month, held here to the start of the month, and CRSP's treatment of the day of removal, which concern a small number of names a year, at small weights. So E1, and with it G1, are expected met.
- *E2*: in every month of the usable window the universe's count within 10 names of CRSP's count, and the summed capitalisation within 2 per cent of CRSP's total value in at least 95 per cent of months. Expected met, with one inference written before any run: CRSP's total value leaves out ADRs and the universe filters no share code, so in months with ADR members, which notebook 02 lists by year, the summed capitalisation may sit above CRSP's total value.
- *The index's price return*: until 2005 the index weighted each stock by its full capitalisation, as the benchmark does, so over the months to December 2004 the gap is evidence on membership independent of CRSP's list, removals included, which Compustat's history cannot test. A wider gap is expected from 2005, when the index moved to float-adjusted shares. Neither figure has a verdict.

Running time: about a minute.
"""),
    code(SETUP),
    code(MODULES),
    md("""
## 1. The benchmark

The benchmark is built from the monthly panel of notebook 03, rebuilt here from the files, and kept to the usable window. The members left out of a month's return are counted over the window, with the largest number in any one month.
"""),
    code("""
window = X.read_result("usable_window.json")
first = pd.Period(window["first_month"], "M")
t0 = time.time()
universe = L.load_universe(root)
data = L.load_panel(root, universe)
bench = V.benchmark(data["panel"])
crsp = V.crsp_by_month(universe["index_file"])
b = bench[bench.index >= first]
left_out = {"months": int(len(b)),
            "member_months_without_cap": int(b["members_without_cap"].sum()),
            "most_in_one_month_without_cap": int(b["members_without_cap"].max()),
            "member_months_without_return": int(b["members_without_return"].sum()),
            "most_in_one_month_without_return": int(b["members_without_return"].max())}
print(f"usable window {window['first_month']} to {window['last_month']} | {time.time() - t0:.0f} s")
print(json.dumps(left_out, indent=1))
"""),
    md("""
## 2. Against CRSP's value-weighted return: G1 and E1

Each comparison runs over the months both series have in the usable window. The table printed after the verdicts gives the ten years with the largest tracking error. The full table by year is committed, and no figure of CRSP's by month is. The same two figures over the evaluation window of the later notebooks, July 1999 to December 2024, are printed last and committed in the summary, with no verdict.
"""),
    code("""
with_div = V.compare(b["benchmark_return"], crsp["crsp_return"])
price_only = V.compare(b["benchmark_return_price_only"], crsp["crsp_return_price_only"])
judged = V.judge_g1_e1(with_div, price_only)
for label, c in (("with dividends", with_div), ("price-only", price_only)):
    print(f"{label}: months {c['months']} | tracking error {c['tracking_error'] * 1e4:.1f} basis points a year"
          f" | mean monthly difference {c['mean_difference'] * 1e4:.2f} basis points | correlation {c['correlation']:.6f}")
print(V.verdict_line("G1", judged["G1"]["met"], f"Tracking error {with_div['tracking_error'] * 1e4:.1f} basis points a year against CRSP's "
      f"value-weighted return with dividends, against a level of at most {C.G1_TE_MAX * 1e4:.0f}."))
print(V.verdict_line("E1", judged["E1"]["met"], f"Tracking error {with_div['tracking_error'] * 1e4:.1f} and {price_only['tracking_error'] * 1e4:.1f} "
      f"basis points a year and mean monthly difference {with_div['mean_difference'] * 1e4:.2f} and {price_only['mean_difference'] * 1e4:.2f} "
      f"basis points, with dividends and price-only, against at most {C.E1_TE_MAX * 1e4:.0f} and within {C.E1_MEAN_DIFF_MAX * 1e4:.0f} of zero."))
parts = [V.compare_by_year(b["benchmark_return"], crsp["crsp_return"]).add_suffix("_with_dividends"),
         V.compare_by_year(b["benchmark_return_price_only"], crsp["crsp_return_price_only"])[["tracking_error", "mean_difference"]].add_suffix("_price_only"),
         V.compare_by_year(b["benchmark_return_price_only"], crsp["index_price_return"])[["tracking_error", "mean_difference"]].add_suffix("_against_index_price")]
by_year = parts[0].join(parts[1:]).rename(columns={"months_with_dividends": "months"}).reset_index()
print(by_year.sort_values("tracking_error_with_dividends", ascending=False).head(10).to_string(index=False))
in_eval = (b.index >= pd.Period(C.EVAL_START, "M")) & (b.index <= pd.Period(C.EVAL_END, "M"))
evaluation_window = {"first_month": C.EVAL_START, "last_month": C.EVAL_END,
                     "with_dividends": V.compare(b.loc[in_eval, "benchmark_return"], crsp["crsp_return"]),
                     "price_only": V.compare(b.loc[in_eval, "benchmark_return_price_only"], crsp["crsp_return_price_only"])}
for label in ("with_dividends", "price_only"):
    c = evaluation_window[label]
    print(f"evaluation window, {label}: months {c['months']} | tracking error {c['tracking_error'] * 1e4:.1f} basis points a year"
          f" | mean monthly difference {c['mean_difference'] * 1e4:.2f} basis points, with no verdict")
"""),
    md("""
## 3. Against the index's price return, with no verdict

The benchmark's price-only return beside the index's own price return, over the whole usable window, over the months to December 2004 and over the months from January 2005. Until 2005 the index weighted each stock by its full capitalisation, as the benchmark does, so the gap to December 2004 comes from membership, prices and share counts, and is evidence on membership independent of CRSP's list, removals included. From 2005 the float adjustment adds to it.
"""),
    code("""
last_full = pd.Period(C.INDEX_FULL_CAP_LAST_MONTH, "M")
index_price = {"whole_window": V.compare(b["benchmark_return_price_only"], crsp["index_price_return"]),
               "to_december_2004": V.compare(b.loc[b.index <= last_full, "benchmark_return_price_only"], crsp["index_price_return"]),
               "from_january_2005": V.compare(b.loc[b.index > last_full, "benchmark_return_price_only"], crsp["index_price_return"])}
for label, c in index_price.items():
    print(f"{label}: months {c['months']} | tracking error {c['tracking_error'] * 1e4:.1f} basis points a year"
          f" | mean monthly difference {c['mean_difference'] * 1e4:.2f} basis points | correlation {c['correlation']:.6f}")
"""),
    md("""
## 4. Count and value: E2

The universe at the start of month t, which is the membership and the capitalisation on the last trading day of month t−1, against CRSP's count and total value on its row for month t−1, the same day.
"""),
    code("""
e2 = V.judge_e2(bench, crsp, window["first_month"])
e2_years = V.e2_value_by_year(bench, crsp, window["first_month"])
print(json.dumps(e2, indent=1))
print(V.verdict_line("E2", e2["met"], f"{e2['months_count_outside']} of {e2['months']} months have a count more than {C.E2_COUNT_WITHIN} names "
      f"from CRSP's count, against a level of none, and the summed capitalisation is within {C.E2_VALUE_REL_MAX:.0%} of CRSP's total value "
      f"in {e2['share_value_within']:.2%} of months, against a level of at least {C.E2_VALUE_MONTH_SHARE_MIN:.0%}."))
print(e2_years[e2_years["months_value_outside"] > 0].to_string(index=False))
"""),
    md("""
## 5. The data layer's six verdicts

G2 comes from notebook 02, G3 and G4 from notebook 03, and G1, E1 and E2 from this notebook. If any of the four gates is not met, the study stops at the data layer, which is published alone with the reason.
"""),
    code("""
g3 = X.read_result("removals_g3.json")["in_daily_file"]
g4 = X.read_result("capitalisation_g4.json")
verdicts = {
    "G1": {"met": judged["G1"]["met"], "tracking_error": with_div["tracking_error"], "level": C.G1_TE_MAX},
    "G2": {"met": window["g2_met"], "years": window["years"], "level": C.G2_USABLE_YEARS_MIN},
    "G3": {"met": g3["g3_met"], "share": g3["share"], "removals": g3["removals"], "level": C.G3_SHARE_MIN},
    "G4": {"met": g4["g4_met"], "share_of_names": g4["share_of_names"], "names": g4["names"], "level": C.G4_NAME_SHARE_MIN},
    "E1": judged["E1"],
    "E2": e2,
}
for name, v in verdicts.items():
    print(f"{name}: {'met' if v['met'] else 'not met'}")
gates = all(verdicts[g]["met"] for g in ("G1", "G2", "G3", "G4"))
print(V.verdict_line("The four gates", gates, "The study goes on to the answer key." if gates else "The study stops at the data layer."))
"""),
    md("""
## 6. What is committed

Four files go into `results/20260928/`: the comparisons by year, the value comparison of E2 by year, a summary of every comparison over the usable window and over the evaluation window, and the six verdicts. Nothing is committed by month. The benchmark's return, the universe's count and its summed capitalisation reproduce CRSP's index file so closely, month by month, that a table of them would be that file under other names, so they stay in memory.
"""),
    code("""
summary = {
    "written_by": "notebooks/04_reconstruction.ipynb", "manifest_sha256": C.MANIFEST_SHA256,
    "usable_window": {k: window[k] for k in ("first_month", "last_month", "months")},
    "members_left_out": left_out, "with_dividends": with_div, "price_only": price_only,
    "against_index_price_return": index_price, "e2": e2, "evaluation_window": evaluation_window,
}
written = [X.write_results("reconstruction_by_year.csv", frame=by_year),
           X.write_results("e2_value_by_year.csv", frame=e2_years),
           X.write_results("reconstruction_summary.json", record=summary),
           X.write_results("verdicts_milestone1.json", record=verdicts)]
for path in written:
    print("written:", path.relative_to(ROOT))
"""),
    md("""
## What was learned, and what could be wrong

What the notebook establishes: how closely the benchmark built from the universe reproduces CRSP's own return of the same list, with dividends and price-only, how closely its count and capitalisation match CRSP's on the same day, the verdicts of G1, E1 and E2, and with notebooks 02 and 03 whether the data layer passes its four gates.

What could still be wrong. A close match to CRSP shows that the data layer reproduces CRSP's construction. It does not show that CRSP's list or prices are right, since both sides read the same source. Section 3 is the one comparison with a source of its own: the index S&P publishes, which weighted by full capitalisation until 2005, so its gap to December 2004 tests membership, removals included. From 2005 the index weights by float-adjusted shares and the benchmark by every share outstanding, so the gap grows for a reason other than an error, and section 3 reports it without judging it. And the tracking error is one figure over a long window: a year with a large gap can sit inside a small figure, which is why the table by year is committed beside it.
"""),
]

# ---------------------------------------------------------------------------
# 05: sectors
# ---------------------------------------------------------------------------

NB05 = [
    md("""
# 05. Sectors: GICS through the link, month by month

**What this notebook does.** It gives every universe member in every month its company's GVKEY through the link, and the GICS classification in force on the last trading day of the month before. It counts by year the member-months left Unclassified, and why, judges the coverage condition that sets where the evaluation window starts, and computes by month the weight of each GICS sector in the benchmark (the universe with each member weighted by its market value at the start of the month).

**Why it comes here.** The mandate built later holds each sector within 2 percentage points of its weight in the benchmark, cuts two sectors to half weight and excludes two sub-industries, and the attribution (the split of the portfolio's return minus the benchmark's into parts, sector by sector) reads the same sectors. Each needs a sector for every member-month, read as it stood when the month began.

**Terms used in this notebook.**

- *universe*: the stocks in CRSP's list of index members on the last trading day of the month before.
- *PERMNO*: CRSP's permanent number for a security. *GVKEY*: Compustat's permanent number for a company.
- *link*: CRSP's table of which GVKEY belongs to which PERMNO, and over which dates, each link with a type and a primary flag. For the sector join a member takes the GVKEY of the link of type LC or LU valid on the last trading day of the month before, whatever its primary flag. Where two are valid, the flags are taken in the order P, C, J, N, then the link that started later, and the member-months concerned are counted. The flag says which of a company's securities is its primary one, which matters going from a company to a security and not from a security to its company. A second share class, such as a company's non-voting shares, reaches its company through a link flagged J, and it has its company's sector.
- *GICS*: the Global Industry Classification Standard of S&P and MSCI. It places a company in one of 11 sectors (a two-digit code), and below them in an industry group (four digits), an industry (six) and a sub-industry (eight). Compustat keeps each company's classifications with the dates each was in force.
- *Unclassified*: the twelfth sector, holding every member-month with no GVKEY through a link or no GICS classification in force. It is carried through the mandate and the attribution like the other eleven, so that every sum over sectors still adds up.
- *market capitalisation*: price times shares outstanding, at the end of the month before.
- *benchmark*: the universe weighted by market capitalisation at the start of the month.
- *sector weight*: the sum of the benchmark's weights of the members in a sector. In memory the twelve weights of a month add up to one. The committed table holds the eleven GICS sectors as shares of the classified capitalisation, from July 1999 to December 2024.
- *cell rule*: fixed in `src/slt/constants.py` on 11 November 2026 for every table by month. A cell is committed only if it sums at least three members and no member is more than 80 per cent of it, so that no figure by month is one security's weight under another name. A sector with no member in a month has weight zero and is committed as zero. Unclassified, which holds few members, is never committed by month.
- *mandate*: the full set of constraints the portfolio of the later notebooks obeys.
- *evaluation window*: the months over which the later notebooks run and judge the mandate, from July 1999 to December 2024, 306 months.
- *coverage condition*: fixed in `src/slt/constants.py` on 11 November 2026, before the coverage was seen. In the window's first month and in every later month, at least 95 per cent of the universe's members and at least 95 per cent of its capitalisation have a GICS sector. If July 1999 misses it, the window starts in the first later month of 1999 that meets it. If no month of 1999 meets it, or a later month falls below, the study stops for a decision on the window.

**What I expect to see, and where it comes from.** From the manifest and structural counts of 10 November 2026: Compustat's GICS history has 45,996 rows, pulled on 10 November, and its earliest start date is 30 June 1999. GICS was introduced in 1999. So every member-month before July 1999 is expected Unclassified. From July 1999 the share of member-months with a sector is expected close to one, below one where a member has no GVKEY through a link of type LC or LU or a gap between two of its classifications. Some members are expected to be classified only through a link flagged J, second share classes in the main, and they are counted by year. The member-months with more than one valid link, and with two classifications in force on one day, are expected close to zero, and both are printed. Each month's weights add up to one by construction, which `tests/test_sectors_validation.py` tests on cases worked by hand.

The evaluation window starts in July 1999 for that reason: before it, every member would sit in Unclassified, and the mandate's sector limits, its cut of two sectors to half weight and its two exclusions could not bind. The coverage condition is expected to hold from July 1999, because the GICS history starts with a classification for the companies in the index on 30 June 1999. The eleven GICS sectors are expected to pass the cell rule in every month of the window, and Unclassified to fail it in most. The data layer's gates and its usable window do not depend on sectors and are unaffected.

Running time: about a minute.
"""),
    code(SETUP),
    code(MODULES),
    md("""
## 1. The link and the GICS history

Both tables are read whole. The link keeps the links of type LC or LU with a PERMNO, whatever their primary flag. The GICS history is kept as it is: each row is one classification of one company, with its start and end dates.
"""),
    code("""
t0 = time.time()
universe = L.load_universe(root)
data = L.load_panel(root, universe)
sectors = L.load_sectors(root, data["panel"])
links, history, classified = sectors["links"], sectors["history"], sectors["classified"]
print(f"link rows {sectors['link_rows']:,} | links of types LC and LU with a PERMNO {len(links):,}",
      "| by primary flag", links['linkprim'].value_counts().to_dict())
print("member-months with more than one valid link:", sectors["multiple_gvkeys"])
print(f"GICS rows {len(history):,} | companies {history['gvkey'].nunique():,} | earliest start {history['indfrom'].min().date()}"
      f" | classification types {history['indtype'].value_counts().to_dict()}")
print("member-months with two GICS classifications in force on one day:", sectors["overlapping_gics"])
print("date values present that did not parse:", {**universe["unparsed"], **data["unparsed"], **sectors["unparsed"]} or "none")
print(f"{time.time() - t0:.0f} s")
"""),
    md("""
## 2. Coverage by year, and the Unclassified member-months by cause

By year: the member-months, those with a GVKEY, those with a sector, those with a sector through a link flagged J or N, and the Unclassified ones split by cause. A member-month is Unclassified when its member has no GVKEY through a link, or has one with no classification in force on the day. Of the second kind, those whose company has a classification that starts later are counted: applying a later classification to earlier months would fill them, which the study does not do, because that classification was not known when the month began.
"""),
    code("""
causes = S.later_classification_by_year(classified, history).drop(columns="months_unclassified")
coverage = S.coverage_by_year(classified).merge(causes, on="year")
has_sector = (classified["sector"] != C.UNCLASSIFIED).values
first_sector = classified.loc[has_sector, "month"].min() if has_sector.any() else None
print("first month with a sector:", first_sector)
print(coverage[coverage["year"] >= 1990].to_string(index=False))
"""),
    md("""
## 3. The evaluation window and its coverage condition

By month, the share of the universe's members with a GICS sector and the share of its capitalisation held by members with one, members without a capitalisation left out of the second. The condition is judged from July 1999: the window starts in the first month of 1999 from July that meets both levels, and holds if every later month to December 2024 meets them too. The shares are printed by month for 1999. The committed table gives them by year, the row for 1999 holding July to December.
"""),
    code("""
by_month = S.coverage_by_month(classified)
first_year = pd.Period(C.EVAL_START, "M").year
print(by_month[by_month.index.year == first_year].round(6).to_string())
evaluation = S.evaluation_start(by_month)
evaluation["first_month_with_a_sector"] = str(first_sector)
start = evaluation["start"] or C.EVAL_START
after = S.coverage_by_year_after(by_month, start=start)
print(after.to_string(index=False))
print(json.dumps(evaluation, indent=1))
start_text = (f"The window starts in {evaluation['start']} and runs {evaluation['months']} months to {evaluation['end']}"
              if evaluation["start"] else f"No month of {first_year} from {C.EVAL_START} meets both levels")
print(V.verdict_line("The coverage condition", evaluation["condition_met"],
      f"{start_text}, with {evaluation['months_below_after_start']} later months below a level, against levels of "
      f"{C.EVAL_GICS_MEMBER_SHARE_MIN:.0%} of members and {C.EVAL_GICS_CAP_SHARE_MIN:.0%} of capitalisation."
      + ("" if evaluation["condition_met"] else " The study stops here for a decision on the window.")))
"""),
    md("""
## 4. The benchmark's sector weights, and the cell rule

In memory, the twelve sector weights of each month, Unclassified included, as shares of the universe's capitalisation. Members without a capitalisation have no weight, as in notebook 04. For the committed table, the eleven GICS sectors as shares of the classified capitalisation, from the window's first month to December 2024. Before anything is written, every one of its cells is checked against the cell rule, and the notebook stops if one fails, removing any earlier table so that none is left to commit. Unclassified is checked too, for the record: it is never committed by month.
"""),
    code("""
weights = S.sector_weights(classified)
last = pd.Period(C.EVAL_END, "M")
committed_weights = S.sector_weights(classified, classified_only=True)
committed_weights = committed_weights[(committed_weights["month"] >= pd.Period(start, "M"))
                                      & (committed_weights["month"] <= last)].reset_index(drop=True)
gap = (committed_weights[S.gics_columns()].sum(axis=1) - 1).abs().max()
print(f"months {len(committed_weights)} | largest gap of a month's eleven weights from one: {gap:.2e}")
cells = S.cells_by_month(classified)
rule = S.cell_rule(cells, C.GICS_SECTORS, start, C.EVAL_END)
unclassified_rule = S.cell_rule(cells, [C.UNCLASSIFIED], start, C.EVAL_END)
print("the eleven GICS sectors:", json.dumps(rule))
print("Unclassified, not committed by month:", json.dumps(unclassified_rule))
print(V.verdict_line("The cell rule", rule["rule_met"],
      f"{rule['cells_failing']} of {rule['cells']} cells fail, against at least {C.SMALL_CELL_MIN_MEMBERS} members and no member "
      f"above {C.SMALL_CELL_MAX_MEMBER_SHARE:.0%} of a cell; {rule['empty_cells']} cells are empty and committed as zero."))
if not rule["rule_met"]:
    (ROOT / C.RESULTS_DIR / "sector_weights_monthly.csv").unlink(missing_ok=True)
    raise SystemExit("A cell of the table by month fails the cell rule, so the table is not written and the study stops here.")
names = {"weight_" + code: name for code, name in C.GICS_SECTORS.items()}
names["weight_" + C.UNCLASSIFIED.lower()] = C.UNCLASSIFIED
shown = [m for m in (start, "2007-12", "2024-12") if pd.Period(m, "M") in set(weights["month"])]
table = weights.assign(month=weights["month"].astype(str)).set_index("month").loc[shown, S.sector_columns()].rename(columns=names)
print("In memory, twelve sectors as shares of the universe's capitalisation:")
print(table.T.round(4).to_string())
"""),
    md("""
## 5. What is committed

Four files go into `results/20260928/`, each an aggregate with no identifier of a security: the coverage by year with the Unclassified member-months by cause, the coverage condition's two shares by year from the window's first month, the eleven GICS sector weights by month over the window, every cell of which has passed the cell rule, and a summary of the counts printed in this notebook, the cell rule's figures and the evaluation window's start.
"""),
    code("""
weights_out = committed_weights.assign(month=committed_weights["month"].astype(str))
summary = {
    "written_by": "notebooks/05_sectors.ipynb", "manifest_sha256": C.MANIFEST_SHA256, "gics_timing": C.GICS_TIMING,
    "sector_link_flags": list(C.SECTOR_LINK_FLAGS),
    "link_rows": sectors["link_rows"], "links_kept": int(len(links)),
    "member_months_with_more_than_one_valid_link": int(sectors["multiple_gvkeys"]),
    "gics_rows": int(len(history)), "member_months_with_two_classifications": int(sectors["overlapping_gics"]),
    "evaluation_window": evaluation,
    "cell_rule": {"min_members": C.SMALL_CELL_MIN_MEMBERS, "max_member_share": C.SMALL_CELL_MAX_MEMBER_SHARE,
                  "gics_sectors": rule, "unclassified_not_committed": unclassified_rule},
    "date_values_not_parsed": {**universe["unparsed"], **data["unparsed"], **sectors["unparsed"]},
}
written = [X.write_results("sector_coverage_by_year.csv", frame=coverage),
           X.write_results("evaluation_coverage_by_year.csv", frame=after),
           X.write_results("sector_weights_monthly.csv", frame=weights_out),
           X.write_results("sectors_summary.json", record=summary)]
for path in written:
    print("written:", path.relative_to(ROOT))
"""),
    md("""
## What was learned, and what could be wrong

What the notebook establishes: how many member-months have a sector, by year, how many reach it only through a link flagged J or N, why the others have none, the first month with any sector, whether the coverage condition holds and so where the evaluation window starts, and the eleven GICS sector weights by month, each cell checked against the cell rule. There is no gate or expectation on sectors in the data layer.

What could still be wrong. A sector read through the link depends on the link being right on the day, and a link that points at the wrong company would give a member another company's sector without any count here showing it. Compustat's history records a classification from the date it was assigned, so a company reclassified in a GICS revision, such as the one of 2018 that created Communication Services, changes sector on that date and not before, which is the intended point-in-time reading. And a member with no classification sits in Unclassified with its full weight. Before July 1999 that is every member, which is why the evaluation window starts there.
"""),
]

# ---------------------------------------------------------------------------
# 06: French's answer key
# ---------------------------------------------------------------------------

MODULES_06 = """
from slt import french as FR
from slt import layer as L
from slt import sorts as Q
from slt import validation as V
"""

NB06 = [
    md("""
# 06. French's answer key: momentum and variance sorts of the universe

**What this notebook does.** Each month it ranks the universe's members on two signals known at the start of the month, momentum and the variance of daily returns, cuts them into five groups of equal size, and computes the return of the highest group minus the lowest, each stock in a group weighted by its market value. It sets these spreads beside the same spreads computed by Ken French within the largest stocks of the US market, and judges expectation E3 on the two comparisons. The same sorts in ten groups are reported beside them and not judged.

**Why it comes here.** French's data library publishes these spreads, built from the same CRSP files by a different hand and with different rules for which stocks count as large. If the universe is the set of large US stocks it claims to be, and its returns and weights are right, its spreads should move with French's month by month. This is the data layer's last check from outside CRSP's own index file, and the first that uses daily returns.

**Terms used in this notebook.**

- *universe*: the stocks in CRSP's list of index members on the last trading day of the month before.
- *verdict*: the result of an expectation, a test fixed in advance in `src/slt/constants.py`, met or not met, printed in the same words either way.
- *market capitalisation*: price times shares outstanding, at the end of the month before.
- *value-weighted*: weighted by market capitalisation, each stock's weight being its capitalisation divided by the sum over the stocks held. *Cap-weighted* means the same.
- *benchmark*: the value-weighted universe, as notebook 04 built it, with delisting returns applied.
- *momentum*: the return compounded over the eleven months from month t−12 to month t−2, leaving out month t−1, French's definition. A member enters the momentum sort for month t if it has a price at the end of month t−13, a return for month t−2 and a market capitalisation at the end of t−1, French's conditions. A missing return between t−12 and t−3 is skipped in the compounding and counted.
- *variance*: the variance of a member's daily returns over the 60 trading days ending on the last trading day of month t−1, on at least 20 returns, French's definition and minimum.
- *quintile* and *decile*: the five, or ten, groups of equal numbers of names, to within one, from lowest signal to highest. Ties are broken by CRSP's permanent number for the security, the PERMNO.
- *spread*: the value-weighted return of the highest quintile minus that of the lowest, in the month after the sort. Each group's return is taken over its members with a return, delisting returns applied, as the benchmark's is.
- *French's largest size quintile*: the stocks above the 80th percentile of the market capitalisations of NYSE stocks, in French's 25 portfolios formed on size and on a second signal. French's spread is his highest-signal portfolio minus his lowest within that quintile, value-weighted.
- *NYSE breakpoints*: French's quintiles are cut at percentiles of NYSE stocks alone, and then applied to all US stocks, so his groups are not of equal size.
- *vintage*: the version of CRSP's database a file was built from. French prints it on the first line of each file, and the extraction here holds CRSP to December 2024.
- *provenance*: the record of what a result was built from, here each French file's address, download time and SHA-256 fingerprint.
- *correlation*: the correlation coefficient of the two spreads over the months both exist.
- *standard error*: the standard deviation of the monthly difference between the two spreads divided by the square root of the number of months, the precision of their mean difference. E3 is judged on it. The *Newey-West standard error*, which allows the difference in one month to be correlated with the difference in the next twelve, is printed beside it with no verdict. *Autocorrelation* is that correlation of a series with its own past.
- *basis point*: a hundredth of a percentage point. A basis point a month in the mean difference is about 0.12 percentage points a year.

**What I expect to see, and where it comes from.**

- *E3*, for the momentum sort and for the variance sort: correlation at least 0.80 between the universe's quintile spread and French's, and a mean difference within two standard errors of zero, over the months both exist. The levels date from 6 September 2026 and were fixed before any data was read. French's largest size quintile is set by NYSE breakpoints and holds between about 100 and 530 names a month, fewer than the universe's 500 in almost every month from April 1957 and more than its 90 before, so the two sets overlap without coinciding and a correlation near one is not expected. Below 0.80 the universe is not the large-cap universe it claims to be. The universe goes from about 90 names to about 500 in April 1957, so the momentum comparison is also printed and committed for January 1927 to March 1957 and April 1957 to December 2024, with no verdict. E3's verdict is on the whole window.
- *The months compared*: momentum from January 1927, the first month with a price thirteen months back in the monthly stock file and a value in French's file, to December 2024. Variance from April 1990, the first month with 60 trading days of the daily stock file before it, to December 2024.
- *The members left out*: from structural counts made on 11 November 2026, before this notebook was written, about 2,500 of 440,000 member-months from January 1927 have no price at t−13 and about 100 have no return at t−2, and they are left out of the momentum sort. About 550 are sorted with a skipped month. In the variance sort 76 of 209,020 member-months from April 1990 have fewer than 20 daily returns and are left out.
- *Vintages*: French builds his files on CRSP's database of September 2026. Revisions between that vintage and the extraction's are part of any gap.

Running time: about five minutes, most of it reading the daily stock file. French's four files are downloaded once into `data/french/`, which git ignores.
"""),
    code(SETUP),
    code(MODULES_06),
    md("""
## 1. French's spreads

Each file is downloaded if it is not already in `data/french/`, and its provenance record is kept. From each of the two 25-portfolio files the notebook reads the block of value-weighted monthly returns and the block of the number of firms in each portfolio, cut at December 2024. French's spreads stay in memory: no French series is committed by month.
"""),
    code("""
french, french_firms, records = FR.spreads()
for r in records:
    print(r["filename"], "|", r["vintage_line"], "| SHA-256", r["zip_sha256"][:12], "| downloaded now:", r["fetched_this_run"])
for sort in C.SORTS:
    s = french[sort].dropna()
    print(f"{sort}: {s.index.min()} to {s.index.max()}, {len(s)} months with a value")
print("names in French's largest size quintile, mean by year:")
print(french_firms.loc[[y for y in (1927, 1957, 1958, 1980, 2000, 2024) if y in french_firms.index]].round(0).to_string())
"""),
    md("""
## 2. The momentum sort

The universe and its monthly panel are rebuilt from the files as in notebooks 02 and 03. The signal is computed for every member-month from January 1927 from the monthly stock file, and the member-months that enter the sort are those with a signal and a market capitalisation.
"""),
    code("""
t0 = time.time()
universe = L.load_universe(root)
data = L.load_panel(root, universe)
panel = data["panel"]
wide = Q.monthly_wide(data["monthly"])
signals = {"momentum": Q.momentum(universe["ulong"], wide)}
entered = {"momentum": Q.entering(signals["momentum"], panel)}
counts = {"momentum": Q.counts_by_year(signals["momentum"], entered["momentum"], "momentum")}
c = counts["momentum"]
print(f"{time.time() - t0:.0f} s | member-months {c['member_months'].sum():,} | sorted {c['sorted'].sum():,}")
print(c.drop(columns="year").sum().to_dict())
"""),
    md("""
## 3. The variance sort

The daily stock file is read a year at a time from 1990, for the stocks that are members in any month from April 1990, and held as one matrix of daily returns in memory. Each member's variance for month t uses its own returns over the 60 trading days before the month, whether or not it was a member on those days.
"""),
    code("""
t0 = time.time()
first_v = pd.Period(C.VARIANCE_SORT_START, "M")
ul = universe["ulong"]
permnos = ul.loc[ul["month"] >= first_v, "permno"].unique()
daily, calendar = L.daily_matrix(root, permnos, L.daily_years(), progress=True)
signals["variance"] = Q.variance(ul, daily, universe["ends"])
entered["variance"] = Q.entering(signals["variance"], panel)
counts["variance"] = Q.counts_by_year(signals["variance"], entered["variance"], "variance")
del daily
c = counts["variance"]
print(f"{time.time() - t0:.0f} s | member-months {c['member_months'].sum():,} | sorted {c['sorted'].sum():,}")
print(c.drop(columns="year").sum().to_dict())
"""),
    md("""
## 4. The spreads against French's: E3

For each sort, the quintile spread and the decile spread of the universe by month, compared with French's spread over the months both exist. E3 is judged on the quintiles. The table by year is printed for every tenth year and committed in full.
"""),
    code("""
results, by_year, verdicts = {}, [], {}
for sort in C.SORTS:
    q5 = Q.spread(entered[sort], C.SORT_QUANTILES)
    q10 = Q.spread(entered[sort], C.SORT_QUANTILES_REPORTED)
    r5, r10 = Q.compare(q5, french[sort]), Q.compare(q10, french[sort])
    judged = Q.judge_e3(r5)
    results[sort] = {"quintiles": {**r5, **judged}, "deciles_no_verdict": r10}
    y5, y10 = Q.compare_by_year(q5, french[sort]), Q.compare_by_year(q10, french[sort])
    y = y5.merge(y10[["year", "correlation", "mean_universe_spread", "mean_difference", "standard_error"]],
                 on="year", suffixes=("_quintiles", "_deciles"))
    y = y.rename(columns={"months": "months", "mean_french_spread": "mean_french_spread"})
    names = Q.names_by_quantile(entered[sort], C.SORT_QUANTILES).rename("universe_names_per_quintile_mean")
    y = y.merge(names, left_on="year", right_index=True, how="left")
    y = y.merge(french_firms[sort].rename("french_names_largest_size_quintile_mean"), left_on="year", right_index=True, how="left")
    by_year.append(y.assign(sort=sort))
    print(V.verdict_line(f"E3, {sort}", judged["met"],
          f"Over {r5['months']} months from {r5['first_month']} to {r5['last_month']}, correlation {r5['correlation']:.3f} "
          f"against at least {C.E3_CORRELATION_MIN:.2f}, and a mean difference of {r5['mean_difference'] * 1e4:.1f} basis points "
          f"a month on a standard error of {r5['standard_error'] * 1e4:.1f}, {abs(r5['difference_in_standard_errors']):.2f} standard "
          f"errors against at most {C.E3_MEAN_DIFF_SE:.0f}"))
    print(f"   deciles, no verdict: correlation {r10['correlation']:.3f}, mean difference {r10['mean_difference'] * 1e4:.1f} "
          f"basis points on {r10['standard_error'] * 1e4:.1f} | Newey-West standard error of the quintiles' difference, "
          f"no verdict: {r5['standard_error_newey_west'] * 1e4:.1f}")
    verdicts[f"E3_{sort}"] = results[sort]["quintiles"]
    if sort == "momentum":
        results[sort]["eras_no_verdict"] = {}
        for a, b in C.MOMENTUM_ERAS:
            era = Q.compare(q5.loc[a:b], french[sort].loc[a:b])
            results[sort]["eras_no_verdict"][f"{a}_to_{b}"] = era
            print(f"   {a} to {b}, no verdict: {era['months']} months, correlation {era['correlation']:.3f}, mean difference "
                  f"{era['mean_difference'] * 1e4:.1f} basis points on a standard error of {era['standard_error'] * 1e4:.1f}")
by_year = pd.concat(by_year, ignore_index=True)
cols = ["sort", "year"] + [c for c in by_year.columns if c not in ("sort", "year")]
by_year = by_year[cols]
print(by_year[by_year["year"] % 10 == 0].round(4).to_string(index=False))
"""),
    md("""
## 5. What is committed

Five files go into `results/20260928/`, each an aggregate with no identifier of a security and no series by month: the comparison by year for each sort, the counts of member-months by year and reason, a summary with the two comparisons over the window, the verdicts of milestone 2 (E3 here, E4's first clause added by notebook 07), and the provenance record of French's files.
"""),
    code("""
count_table = pd.concat([counts[s].assign(sort=s) for s in C.SORTS], ignore_index=True)
count_table = count_table[["sort"] + [c for c in count_table.columns if c != "sort"]].fillna(0)
summary = {
    "written_by": "notebooks/06_answer_key.ipynb", "manifest_sha256": C.MANIFEST_SHA256,
    "french_vintages": {r["key"]: r["vintage_line"] for r in records},
    "quantiles_judged": C.SORT_QUANTILES, "quantiles_reported": C.SORT_QUANTILES_REPORTED,
    "standard_error": C.E3_STANDARD_ERROR, "newey_west_lags_reported": C.E3_NEWEY_WEST_LAGS_REPORTED,
    "french_columns": {s: {"top": t, "bottom": b, "file": FR.C.FRENCH_FILES[k]} for s, (k, t, b) in C.E3_FRENCH_SPREADS.items()},
    "comparisons": results,
    "member_months": {s: {k: int(v) for k, v in counts[s].drop(columns=["year"]).sum().items()} for s in C.SORTS},
}
path = ROOT / C.RESULTS_DIR / "verdicts_milestone2.json"
record = json.loads(path.read_text()) if path.exists() else {}
record.update({k: {"met": v["met"], "correlation": v["correlation"], "mean_difference": v["mean_difference"],
                   "standard_error": v["standard_error"], "months": v["months"]} for k, v in verdicts.items()})
written = [X.write_results("answer_key_by_year.csv", frame=by_year),
           X.write_results("answer_key_counts_by_year.csv", frame=count_table),
           X.write_results("answer_key_summary.json", record=summary),
           X.write_results("verdicts_milestone2.json", record=record),
           FR.write_provenance(records)]
for p in written:
    print("written:", p.relative_to(ROOT))
"""),
    md("""
## What was learned, and what could be wrong

What the notebook establishes: whether the universe's momentum and variance spreads move with French's large-stock spreads month by month and agree with them on average, judged as E3, with the deciles and the counts beside them.

What could still be wrong. French's quintiles are cut at NYSE percentiles and the universe's at its own, so a gap in level can come from the different groups as well as from an error. French's files are built on a later vintage of CRSP than the extraction, and revisions are part of any gap. A missing return between t−12 and t−3 is skipped here, where French's rule admits it only as a missing price, and the counts show how often that happens. And a correlation can be high while one spread runs above the other, which is why E3 judges the mean difference as well.
"""),
]

# ---------------------------------------------------------------------------
# 07: the covariance estimators
# ---------------------------------------------------------------------------

MODULES_07 = """
import numpy as np

from slt import estimators as E
from slt import french as FR
from slt import layer as L
from slt import validation as V
"""

NB07 = [
    md("""
# 07. The covariance estimators on the universe

**What this notebook does.** For each of the 306 months of the evaluation window it estimates the covariance matrix, the table of the variances of the members' returns on its diagonal and their covariances off it, from the members' daily returns less the daily risk-free rate, the one-month Treasury bill rate in French's files, over the three years before the month. It does so in four ways, scales each matrix to a month, and records how well conditioned each is. It judges the first clause of expectation E4 on the sample estimator against the Ledoit-Wolf one, and counts the members whose window is not complete and how each is handled.

**Why it comes here.** The mandate of milestone 3 chooses about 500 weights each month by minimising its forecast tracking error, which is computed from one of these matrices. A matrix that understates the risk of some combinations of positions lets the optimiser load on them, so the conditioning is measured before any portfolio is built.

**Terms used in this notebook.**

- *market capitalisation*: price times shares outstanding, at the end of the month before.
- *benchmark*: the universe weighted by market capitalisation at the start of the month.
- *universe*: the stocks in CRSP's list of index members on the last trading day of the month before. A member without a market capitalisation at the start of the month is left out of the benchmark and of the matrices for that month.
- *evaluation window*: July 1999 to December 2024, 306 months, as notebook 05 settled it.
- *excess return*: a return less the risk-free rate.
- *provenance*: the record of what a result was built from, here each French file's address, download time, SHA-256 fingerprint and the line naming the CRSP database it was built from.
- *window*: the trading days of the 36 calendar months before the month, 750 to 759 days.
- *covariance estimator*: a rule that builds a covariance matrix from a window of returns. Each is scaled from a day to a month by multiplying by 21.
- *eigenvector*: a set of weights on the members that makes a portfolio uncorrelated with the portfolios of the other eigenvectors.
- *eigenvalue*: the variance of an eigenvector's portfolio. A covariance matrix has one for each member. A matrix is *positive semi-definite* when none is negative, which every true covariance matrix is, and *singular* when one is zero.
- *condition number*: the largest eigenvalue divided by the smallest. A large one means the matrix is *near-singular*: close to singular, so that solving with it amplifies small errors in its entries.
- *sample estimator*: the covariance of the window's returns, each entry from the data alone. With about 500 members and about 750 days it exists, and its smallest eigenvalues are close to zero.
- *shrinkage*: pulling an estimate towards a simple target, called the *shrinkage target*. The *Ledoit-Wolf estimator* pulls the sample matrix towards a scaled identity matrix, the average variance on the diagonal and zero elsewhere, by a weight from zero to one called the *shrinkage intensity*, chosen from the data to minimise the expected error (Ledoit and Wolf 2004).
- *factor*: a common source of return.
- *loading*: the slope of a member's return on a factor in a regression. A *beta* is the same thing under its usual name.
- *factor model*: each member's return explained by its loadings on French's five factors and his momentum factor. It gives the covariance as the loadings times the factors' covariance times the loadings, plus each member's own variance, called *idiosyncratic*, the variance of what the factors leave.
- *principal component*: one of the eigenvectors of a covariance matrix, ranked by its eigenvalue. The *PCA estimator* takes the benchmark's excess return as the first factor, with each member's beta on it, describes what is left by the first four principal components of the members' residual covariance, and adds each member's own remaining variance.
- *complete member*: a member that lacks a return on at most five days of the window. The sample, Ledoit-Wolf and PCA estimators need a matrix with no gap, so they are estimated on the complete members, on the days on which all of them have a return. A day on which a complete member lacks a return is left out of the window. At most 21 days may be left out: where more would go, the bar for complete falls from five days to four and so on, and the month is counted.
- *model row*: the row of the matrix for a member that is not complete. It comes from a model fitted on the member's own days: its loadings on the six factors and its idiosyncratic variance for the sample, Ledoit-Wolf and factor estimators, and its beta on the benchmark and its residual variance for the PCA estimator. A member with fewer than 60 returns, too few for seven coefficients, takes the market-capitalisation-weighted mean loadings and the median residual variance of the complete members of its GICS sector, when that sector has at least ten of them, and of the whole universe otherwise. The *degrees of freedom* of a regression are its observations less its coefficients.
- *assembled matrix*: the matrix the mandate will use, complete members and model rows together. Each estimator has one construction for every month, positive semi-definite by construction, and its block for the complete members is what the copied function returns for them. The rules are in `src/slt/constants.py`, section 7.
- *expectation*: a test fixed in advance in `src/slt/constants.py`, reported as met or not met in the same words either way. Its result is the *verdict*.

**What I expect to see, and where it comes from.**

- *E4, first clause*: in every month of the evaluation window, the condition number of the sample estimator at least ten times that of the Ledoit-Wolf estimator, on the assembled matrices. About 500 members on about 750 days leave the smallest sample eigenvalues near zero, and shrinkage lifts them. The second clause, on realised tracking error, is judged in milestone 3. The condition numbers of the complete members' blocks are printed beside them with no verdict.
- *The members*: from structural counts made on 11 November 2026, before this notebook was written, about 490 complete members a month and 478 at the least, and about 11 model rows a month and 22 at the most, of which about 3 have fewer than 252 returns and 8 at the most. Fewer than 60 returns: 190 member-months in 134 months.
- *The check*: every assembled matrix is positive semi-definite by construction, so the notebook computes each one's smallest eigenvalue and stops if one is below minus one ten-billionth of the largest. That would be a fault in the code, and nothing from such a run is reported.

Running time: about ten to twenty minutes, most of it the 306 months of four estimators on about 500 members.
"""),
    code(SETUP),
    code(MODULES_07),
    md("""
## 1. The members, their market capitalisations and sectors, and French's daily factors

The universe, its monthly panel and its sectors are rebuilt from the files as in notebooks 02, 03 and 05, for the months from July 1996, the first month of the first window, to December 2024. French's daily factors and risk-free rate are downloaded once into `data/french/`.
"""),
    code("""
t0 = time.time()
universe = L.load_universe(root)
data = L.load_panel(root, universe)
panel = data["panel"]
first_window = pd.Period(C.EVAL_START, "M") - 12 * C.COV_DAILY_WINDOW_YEARS
p = panel[panel["month"] >= first_window]
sectors = L.load_sectors(root, p)
members = sectors["classified"][["month", "permno", "sector"]].merge(p[["month", "permno", "cap"]], on=["month", "permno"])
factors, rf, records = FR.daily_factors()
for r in records:
    print(r["filename"], "|", r["vintage_line"], "| SHA-256", r["zip_sha256"][:12])
print(f"{time.time() - t0:.0f} s | member-months from {first_window}: {len(members):,} | without a capitalisation: {members['cap'].isna().sum():,}")
"""),
    md("""
## 2. The daily excess returns and the benchmark's

The daily stock file is read a year at a time from 1996 for the stocks that are members in any month from July 1996, and held as one matrix in memory. The benchmark's daily excess return is each day's return of that month's members weighted by their market capitalisations at the start of the month, rescaled over the members with a return that day, less the risk-free rate. It is the PCA estimator's first factor.
"""),
    code("""
t0 = time.time()
years = [y for y in L.daily_years() if y >= first_window.year]
daily, calendar = L.daily_matrix(root, members["permno"].unique(), years, progress=True)
rf_d = rf.reindex(daily.index)
excess = daily.sub(rf_d, axis=0)
bench = E.benchmark_daily_excess(daily, rf, members)
del daily
missing_factor_days = int(factors.reindex(excess.index).isna().any(axis=1).sum())
print(f"{time.time() - t0:.0f} s | trading days {len(excess.index):,} | stocks {excess.shape[1]:,} | days without French's factors: {missing_factor_days}")
"""),
    md("""
## 3. The four estimators, month by month

For each month the four assembled matrices are built, checked and measured, and only their scalars are kept: no matrix and no member's row leaves the loop. A year's months are printed as they finish.
"""),
    code("""
t0 = time.time()
factors_d = factors.reindex(excess.index)
rows = []
for month in pd.period_range(C.EVAL_START, C.EVAL_END, freq="M"):
    m = members[(members["month"] == month) & members["cap"].notna()]
    res = E.month_estimates(month, excess, factors_d, bench, m)
    res.pop("matrices")
    E.check_positive_semidefinite(res)
    rows.append(res)
    if month.month == 12:
        print(month.year, f"{time.time() - t0:.0f} s", end=" | ", flush=True)
rows = pd.DataFrame(rows)
print()
print(f"months {len(rows)} | every assembled matrix positive semi-definite within the tolerance")
"""),
    md("""
## 4. Conditioning by year, and E4's first clause

The table by year gives each estimator's condition number (median, least and largest), the complete members' median beside it, the smallest ratio of the sample's condition number to the Ledoit-Wolf one's and the months below ten, the shrinkage intensity, and the counts of complete members, model rows and days left out. Every fifth year is printed, and the table is committed in full.
"""),
    code("""
table = E.by_year(rows)
shown = ["year", "months"] + [f"condition_{n}_median" for n in C.COV_ESTIMATORS] + [
    "ratio_sample_to_ledoit_wolf_least", "months_ratio_below_level", "lw_intensity_mean", "complete_members_mean",
    "model_rows_mean", "model_rows_under_252_mean", "model_rows_under_60_member_months", "days_left_out_most", "months_bar_fell"]
print(table.loc[table["year"] % 5 == 0, shown].round(2).to_string(index=False))
e4 = E.judge_e4_first_clause(rows)
print(V.verdict_line("E4, first clause", e4["met"],
      f"The sample's condition number is at least {C.E4_CONDITION_RATIO_MIN:.0f} times the Ledoit-Wolf one's in "
      f"{e4['months_at_or_above_level']} of {e4['months']} months, the smallest ratio {e4['smallest_ratio']:.1f} in "
      f"{e4['month_of_smallest_ratio']}, the median {e4['median_ratio']:.1f}"))
counts = {"complete_members_mean": float(rows["complete"].mean()), "complete_members_least": int(rows["complete"].min()),
          "model_rows_mean": float(rows["model_rows"].mean()), "model_rows_most": int(rows["model_rows"].max()),
          "model_rows_under_252_mean": float(rows["model_rows_under_252"].mean()),
          "model_rows_under_252_most": int(rows["model_rows_under_252"].max()),
          "model_rows_under_60_member_months": int(rows["model_rows_under_60"].sum()),
          "months_with_a_model_row_under_60": int((rows["model_rows_under_60"] > 0).sum()),
          "days_left_out_most": int(rows["days_left_out"].max()), "months_bar_fell": int(rows["bar_fell"].sum()),
          "days_without_factors": int(rows["days_without_factors"].sum())}
print(json.dumps(counts, indent=1))
"""),
    md("""
## 5. What is committed

Four files go into `results/20260928/`, each an aggregate with no identifier of a security and no series by month: the conditioning and the counts by year, a summary with E4's first clause and the eigenvalue check, the verdicts of milestone 2 with E4's first clause added beside E3, and the provenance record of French's files.
"""),
    code("""
summary = {
    "written_by": "notebooks/07_covariance.ipynb", "manifest_sha256": C.MANIFEST_SHA256,
    "french_vintages": {r["key"]: r["vintage_line"] for r in records},
    "estimators": list(C.COV_ESTIMATORS), "window_years": C.COV_DAILY_WINDOW_YEARS, "scale": C.DAILY_TO_MONTHLY_SCALE,
    "pca_components": C.PCA_N_COMPONENTS, "complete_window_missing_max": C.COMPLETE_WINDOW_MISSING_MAX,
    "complete_days_left_out_max": C.COMPLETE_DAYS_LEFT_OUT_MAX, "model_row_own_min": C.FACTOR_ROW_OWN_MIN,
    "sector_min_complete": C.SECTOR_MIN_COMPLETE, "sample_monthly": C.SAMPLE_MONTHLY_STATUS,
    "e4_first_clause": e4, "read_on": C.E4_READ_ON,
    "eigenvalue_check": {"tolerance": C.PSD_FAULT_TOLERANCE, "passed_every_month": True,
                         "smallest_ratio_by_estimator": {n: float(rows[f"smallest_eigenvalue_ratio_{n}"].min()) for n in C.COV_ESTIMATORS}},
    "counts": counts,
}
path = ROOT / C.RESULTS_DIR / "verdicts_milestone2.json"
record = json.loads(path.read_text()) if path.exists() else {}
record["E4_first_clause"] = {k: e4[k] for k in ("met", "months", "months_at_or_above_level", "smallest_ratio", "month_of_smallest_ratio", "median_ratio")}
written = [X.write_results("covariance_by_year.csv", frame=table),
           X.write_results("covariance_summary.json", record=summary),
           X.write_results("verdicts_milestone2.json", record=record),
           FR.write_provenance(records)]
for p in written:
    print("written:", p.relative_to(ROOT))
"""),
    md("""
## What was learned, and what could be wrong

What the notebook establishes: how close to singular each of the four estimators is, month by month over the evaluation window, whether the sample estimator's condition number is at least ten times the Ledoit-Wolf one's in every month, judged as E4's first clause, and how many members each month needed a model row.

What could still be wrong. A condition number says how much a matrix amplifies errors. Whether its forecasts are right is for the second clause of E4 and for the bias statistic, the ratio of realised to forecast risk, both in milestone 3. A model row rests on the member's own days, and for a recent listing those can be few and unrepresentative. The factor model's rows use French's factors, built from all US stocks, for a universe of large ones. And the window weights every day of three years equally, so a matrix lags a change in how stocks move together.
"""),
]

if __name__ == "__main__":
    for name, cells in (("01_extraction.ipynb", NB01), ("02_universe.ipynb", NB02), ("03_returns.ipynb", NB03),
                        ("04_reconstruction.ipynb", NB04), ("05_sectors.ipynb", NB05), ("06_answer_key.ipynb", NB06),
                        ("07_covariance.ipynb", NB07)):
        print("wrote", write(name, cells).relative_to(ROOT))

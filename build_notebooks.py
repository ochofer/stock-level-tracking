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
- *CRSP's guide*: CRSP's Data Descriptions Guide for CRSPAccess (FIZ) US Stock & Index Databases, last modified 20 November 2025. The descriptions of the fields of CRSP's stock and index files follow it, cited by its printed page. CRSP's list of index members and its link to Compustat are not in it, and are described from the study's design and from counts of the extraction. It names none of the tables of Wharton Research Data Services, and it is not committed.

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

What could still be wrong. A hash shows that a file has not changed since it was fetched. It does not show that the file holds what the vendor intended. Notebooks 02 to 04 test that, against CRSP's own counts and returns. CRSP corrects its history with each release of its data, as errors are identified (CRSP's guide, p. 5), so the same queries run later can return other files, with other hashes in `extraction_files.csv`, for reasons that are not errors. And the row counts in the manifest and in the footers are both counts of rows as stored. Neither says anything about a value inside a row.
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

- *CRSP's guide*: CRSP's Data Descriptions Guide for CRSPAccess (FIZ) US Stock & Index Databases, last modified 20 November 2025. The descriptions of the fields of CRSP's stock and index files follow it, cited by its printed page. CRSP's list of index members and its link to Compustat are not in it, and are described from the study's design and from counts of the extraction. It names none of the tables of Wharton Research Data Services, and it is not committed.
- *universe*: the stocks in CRSP's list of index members on the last trading day of the month before. A stock is in the universe in month t when one of its membership spells covers the last trading day of month t−1.
- *membership spell*: one interval in CRSP's list, from the day a stock entered the index to the day it left, both days included. A stock that left and came back has more than one.
- *PERMNO*: CRSP's permanent number for a security. It does not change when the company changes its name or its ticker.
- *GVKEY*: Compustat's permanent number for a company.
- *link*: CRSP's table of which GVKEY belongs to which PERMNO, and over which dates. Only primary links of types LC and LU are used, with primary flag P or C, each on the dates it was valid.
- *CRSP's count*: the number of securities in the index universe with a valid price on a day (CRSP's guide, p. 43), which CRSP stores on each row of its monthly index file (`totcnt`). Its row for a month carries the count on that month's last trading day.
- *usable window*: the months from the first month in which the universe's count is within 10 names of CRSP's count, to December 2024, CRSP's last month on this subscription.
- *gate*: a test of the data layer, fixed in advance in `src/slt/constants.py`. A gate that is not met stops the study.
- *verdict*: the result of a gate, met or not met, printed in the same words either way.
- *share code*: CRSP's two-digit code for the type of security (CRSP's guide, p. 78). Codes 10 and 11 are ordinary common shares. A first digit of 3 marks an ADR, an American depositary receipt: a certificate traded in the United States that stands for shares of a company listed abroad. A second digit of 2 marks a company incorporated outside the United States.
- *basis point*: one hundredth of a percentage point.

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

The membership is a table of months by PERMNOs, true where a spell covers the last trading day of the month before. The table below gives the universe's size by decade. Before March 1957 the index held 90 stocks (CRSP's guide, pp. 104 and 106), so the count is expected near 90 there and near 500 from 1957.
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

What could still be wrong. The count agreeing does not show that the right stocks are in the universe, only that the number is right: a stock in place of another would leave the count unchanged. Notebook 04 tests membership more finely, because a wrong stock would move the universe's return and its summed capitalisation away from CRSP's. The rule reads membership at the start of the month, so a stock that joins mid-month enters the universe the month after, and one that leaves mid-month stays to the month's end. CRSP's own monthly return reads membership at the end of the month, so it counts a stock that joins mid-month for the whole month and leaves out one that leaves mid-month (CRSP's guide, pp. 104 and 106), and notebook 04 measures how far the two definitions lie apart. And Compustat's history on this subscription lists only the members of September 2026, so it cannot show that the universe's past members are right. Notebook 04 sets the universe's return, each member weighted by its market value, beside the index's own price return, by period. That gap is a coarse check on membership, which would show an error only if the error moved the return by more than 20 to 36 basis points a year, and it does not confirm the list before 1957.
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

- *CRSP's guide*: CRSP's Data Descriptions Guide for CRSPAccess (FIZ) US Stock & Index Databases, last modified 20 November 2025. The descriptions of the fields of CRSP's stock and index files follow it, cited by its printed page. CRSP's list of index members and its link to Compustat are not in it, and are described from the study's design and from counts of the extraction. It names none of the tables of Wharton Research Data Services, and it is not committed.
- *universe*: the stocks in CRSP's list of index members on the last trading day of the month before.
- *member-month*: one stock in the universe in one month.
- *return*: CRSP's monthly return with dividends (`ret`). The *price-only return* (`retx`) leaves dividends out.
- *delisting*: the end of a stock's listing on its exchange. CRSP records its date, a code for its reason (`dlstcd`) and, where it can, the *delisting return* (`dlret`), which compares the value after the delisting, an off-exchange price or the sum of the payments to holders, with the price on the last day of trading (CRSP's guide, p. 30). The guide names the first digit of the code 2 "merger", 3 "exchange", 4 "liquidation" and 5 "delisted by NYSE, NYSE MKT, NASDAQ or Arca" (p. 28).
- *partial-month return*: in the monthly file, where CRSP has no information after the delisting, the value it gives as the delisting return is the return from the last month-end price to the last trading price. CRSP's guide calls it a partial-month return and "not truly" a delisting return, and gives the condition: the amount after delisting (`dlamt`) is not zero and its payment date (`dlpdt`) is on or before the delisting date (pp. 30 and 98).
- *rule D4*: the study's rule for a delisting, after Shumway (1997). CRSP's delisting return where it gives one. Where it is missing, minus 30 per cent for a code from 400 to 599, the codes the study treats as a delisting for poor performance, and zero for any other code. CRSP's code 100 marks a security still trading and is not a delisting. In a month with both a return and a delisting return, the two are compounded.
- *market capitalisation*: price times shares outstanding. The capitalisation at the start of month t is the absolute price times shares outstanding at the end of month t−1, in thousands of dollars, since CRSP records shares outstanding in thousands (CRSP's guide, p. 79). CRSP stores the price as negative when it is the average of bid and ask on a day without a closing price, and as zero when it has neither (p. 70), hence the absolute value. A price of zero or a missing value gives a missing capitalisation.
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
## 7. Partial-month returns among CRSP's delisting returns, added on 19 November 2026

Section 2 counts the delisting months in which CRSP gives the delisting return. Some of those values are partial-month returns, which compare the last trading price with the last month-end price and say nothing of what the holder received after the delisting. This section counts them by the hundreds of the delisting code, with the guide's condition. It was added on 19 November 2026, after every recorded run, changes no return and has no verdict. Rule D4 stays as it was fixed. Four of them have a code from 500 to 599. For those four the return ends at the last trade, so a loss after the delisting is not in it. The study has not measured what its minus 30 per cent would change if applied to them.
"""),
    code("""
partial = R.partial_month_counts(panel, data["delistings"])
print(partial.to_string(index=False))
print(partial.drop(columns="code_family").sum().to_string())
assert int(partial["delisting_return_given"].sum()) == int(d4["delisting_return_given"].sum())
"""),
    md("""
## 8. What is committed

Six files go into `results/20260928/`, each an aggregate with no identifier of a security: rule D4's counts by year, G4's figures and verdict, G3's figures and verdict with the removals before 1990, the daily coverage by year, a summary of the counts printed in this notebook, and, added on 19 November 2026, the count of partial-month returns by the hundreds of the delisting code.
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
           X.write_results("returns_summary.json", record=summary),
           X.write_results("delisting_partial_month.csv", frame=partial)]
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

- *CRSP's guide*: CRSP's Data Descriptions Guide for CRSPAccess (FIZ) US Stock & Index Databases, last modified 20 November 2025. The descriptions of the fields of CRSP's stock and index files follow it, cited by its printed page. CRSP's list of index members and its link to Compustat are not in it, and are described from the study's design and from counts of the extraction. It names none of the tables of Wharton Research Data Services, and it is not committed.
- *universe*: the stocks in CRSP's list of index members on the last trading day of the month before.
- *market capitalisation*: price times shares outstanding. The capitalisation at the start of month t is the absolute price times shares outstanding at the end of month t−1.
- *cap-weighted*: weighted by market capitalisation, each stock's weight being its capitalisation divided by the sum over the stocks held. *Value-weighted* is CRSP's word for the same.
- *benchmark*: the cap-weighted universe, with each stock's weight set by its capitalisation at the start of the month, so that nothing unknown when the month begins sets a weight. A member without a capitalisation, or with neither a return nor a delisting in the month, is left out of that month's return and the other weights are rescaled. Both are counted.
- *return with dividends* and *price-only return*: the month's return with dividends reinvested, and without them.
- *CRSP's return*: the return of CRSP's Value-Weighted Index of the S&P 500 Universe, index 1000500 in CRSP's guide (p. 112), with dividends (`vwretd`) and price-only (`vwretx`). CRSP computes its monthly index returns from monthly security returns (p. 104). A security enters month t if it is a member at the end of month t and has a price at the end of month t and of month t−1, ADRs are left out, and each is weighted by its capitalisation at the end of month t−1 (pp. 104 and 106). A security without a price at the end of the month is out of that month, so no delisting return enters. The guide sets beside this index CRSP's portfolios of the same universe, which select on the previous month's membership and use delisting returns (p. 106).
- *the index's price return*: S&P's published return of the index without dividends (`sprtrn`), the S&P 500 Composite of CRSP's guide, index 1000502 (p. 112). CRSP collects its levels from public sources, the levels exclude dividends, and the return is the level over the level before, minus one (pp. 100 and 104). Since 2005 the index weights each stock by its float-adjusted shares, the shares available to the public. CRSP gives one share count per security and no float factor (pp. 79 and 80), so S&P's float-adjusted weights cannot be rebuilt from it.
- *CRSP's count*: the number of securities in the index universe with a valid price on a day (`totcnt`, CRSP's guide, p. 43). *CRSP's total value*: the total market value of the securities in the index universe that are not ADRs and have a valid price and shares outstanding, in thousands of dollars (`totval`, p. 52). Each row of CRSP's monthly index file carries both for that month's last trading day. *CRSP's used count* (`usdcnt`) is the number of securities the index used in the month, each a member with a valid price at the end of the month and of the month before (pp. 43 and 102).
- *ADR*: an American depositary receipt, a certificate traded in the United States that stands for shares of a company listed abroad, whose share code's first digit is 3 (p. 78).
- *basis point*: one hundredth of a percentage point.
- *mean monthly difference*: the average over the months of the benchmark's return minus CRSP's.
- *tracking error*: the standard deviation of the monthly difference between two returns, times the square root of 12, a yearly figure.
- *usable window*: the months from the first in which the universe's count is within 10 names of CRSP's count, to December 2024, as notebook 02 found it.
- *gate* and *expectation*: tests fixed in advance in `src/slt/constants.py`. A gate that is not met stops the study. An expectation that is not met is reported as a finding.
- *verdict*: the result of a gate or an expectation, met or not met, printed in the same words either way.

**What I expect to see, and where it comes from.** The levels are the constants', committed before any run that judges them.

- *G1*: tracking error against CRSP's value-weighted return with dividends of at most 50 basis points a year over the usable window.
- *E1*: tracking error of at most 20 basis points a year and a mean monthly difference within 2 basis points of zero, with dividends and price-only. CRSP computes its return from the same list and the same file, daily. The differences left are membership changes inside a month, held here to the start of the month, and CRSP's treatment of the day of removal, which concern a small number of names a year, at small weights. So E1, and with it G1, are expected met.

  *Added on 19 November 2026, with the reason above left as it was written before the run.* CRSP's guide computes the monthly index return from monthly security returns, and its index of the S&P 500 universe takes the members at the end of the month that have a price at the end of the month and of the month before, so that no delisting return enters, and leaves ADRs out (pp. 104 and 106). The benchmark takes the members at the start of the month, every share code and the delisting returns. Its tracking error against CRSP's return therefore compares two definitions, and section 6 rebuilds the return on CRSP's.
- *E2*: in every month of the usable window the universe's count within 10 names of CRSP's count, and the summed capitalisation within 2 per cent of CRSP's total value in at least 95 per cent of months. Expected met, with one inference written before any run: CRSP's total value leaves out ADRs and the universe filters no share code, so in months with ADR members, which notebook 02 lists by year, the summed capitalisation may sit above CRSP's total value.
- *The index's price return*: until 2005 the index weighted each stock by its full capitalisation, as the benchmark does, so over the months to December 2004 a member missing from CRSP's list or a removal dated wrongly would widen the gap, which Compustat's history cannot test. A wider gap is expected from 2005, when the index moved to float-adjusted shares. Neither figure has a verdict.

  *Read on 19 November 2026, after the run, with the expectation above left as it was written.* The gap is a coarse check on membership: from 1957 it would show an error only if the error moved the return by more than 20 to 36 basis points a year, and it does not confirm the list before 1957. Pooled over the years, it is 22.2 basis points a year over 2005 to 2024 against 28.3 over the six years before, so no wider gap is found.

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

The benchmark's price-only return beside the index's own price return, over the whole usable window, over the months to December 2004 and over the months from January 2005, each measured over its months in one piece. Until 2005 the index weighted each stock by its full capitalisation, as the benchmark does, so a member missing from CRSP's list or a removal dated wrongly would widen the gap to December 2004. The gap is a coarse check on membership. Pooled over the years of each period, from the table by year that section 7 commits, it lies between 20.6 and 36.0 basis points a year in every period from 1957, so it would show an error only if the error moved the return by more than 20 to 36 basis points a year, and at 145.8 over 1926 to 1956 it does not confirm the list before 1957. A wider gap was expected from 2005, when the index moved to float-adjusted shares. Pooled over the years, it is 22.2 over 2005 to 2024 against 28.3 over the six years before, so no wider gap is found.
"""),
    code("""
last_full = pd.Period(C.INDEX_FULL_CAP_LAST_MONTH, "M")
index_price = {"whole_window": V.compare(b["benchmark_return_price_only"], crsp["index_price_return"]),
               "to_december_2004": V.compare(b.loc[b.index <= last_full, "benchmark_return_price_only"], crsp["index_price_return"]),
               "from_january_2005": V.compare(b.loc[b.index > last_full, "benchmark_return_price_only"], crsp["index_price_return"])}
for label, c in index_price.items():
    print(f"{label}: months {c['months']} | tracking error, measured over the months in one piece, "
          f"{c['tracking_error'] * 1e4:.1f} basis points a year"
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
## 6. The return on CRSP's own definition, added on 19 November 2026

A diagnostic with no verdict, added on 19 November 2026 after every recorded run: the return of the S&P 500 universe built once more on CRSP's definition of its index, as CRSP's guide gives it on pages 104 and 106. In month t it takes the members on the last trading day of month t, with a price at the end of month t and at the end of month t−1, a positive share count at the end of month t−1 and a return in month t, leaves out the ADRs, each weighted by its capitalisation at the end of month t−1, and applies no delisting return. Its tracking error and mean monthly difference against CRSP's return, with dividends and price-only, are printed beside the benchmark's of section 2, and so is the benchmark against it, which measures how far the two definitions lie apart. Beside them, the months in which the number of stocks it uses equals CRSP's used count, and the same count with the ADRs added back.

How the result is read is set in `src/slt/constants.py`, on the tracking error with dividends of the return on CRSP's definition against CRSP's return. The two levels are in a commit of their own, dated 19 November 2026, 10:12, before the commit that holds the result. The repository's history shows that order and no more. At 2.3 basis points a year or less, the benchmark's gap to CRSP's return comes from the difference of definition. At 10.0 or more, the difference of definition does not account for it, and its cause is not known. Between the two, the difference accounts for part of it and the rest has no cause the study can name, and no share is given, because tracking errors do not subtract.
"""),
    code("""
names, bad_names = X.read_table(root, "crsp_msenames")
own = V.crsp_definition(universe["spells"], universe["ends"], data["monthly"], names)
o = own[own.index >= first]
used = crsp["crsp_count_used"].reindex(o.index)
crsp_definition = {"months": int(len(o)),
                   "with_dividends": V.compare(o["return"], crsp["crsp_return"]),
                   "price_only": V.compare(o["return_price_only"], crsp["crsp_return_price_only"]),
                   "benchmark_with_dividends": with_div, "benchmark_price_only": price_only,
                   "benchmark_against_crsp_definition_with_dividends": V.compare(b["benchmark_return"], o["return"]),
                   "benchmark_against_crsp_definition_price_only": V.compare(b["benchmark_return_price_only"], o["return_price_only"]),
                   "months_count_used_equal": int((o["count_used"] == used).sum()),
                   "months_count_used_with_adrs_equal": int(((o["count_used"] + o["adrs_left_out"]) == used).sum()),
                   "months_with_an_adr_left_out": int((o["adrs_left_out"] > 0).sum()),
                   "most_adrs_left_out_in_one_month": int(o["adrs_left_out"].max()),
                   "date_values_not_parsed": bad_names}
for label in ("with_dividends", "price_only"):
    c, b_ = crsp_definition[label], crsp_definition["benchmark_" + label]
    print(f"{label}: on CRSP's definition, months {c['months']} | tracking error {c['tracking_error'] * 1e4:.1f} basis points a year"
          f" | mean monthly difference {c['mean_difference'] * 1e4:.2f} | the benchmark's {b_['tracking_error'] * 1e4:.1f} and {b_['mean_difference'] * 1e4:.2f}, with no verdict")
for label in ("with_dividends", "price_only"):
    c = crsp_definition["benchmark_against_crsp_definition_" + label]
    print(f"{label}: the benchmark against the return on CRSP's definition, tracking error {c['tracking_error'] * 1e4:.1f} basis points a year"
          f" | mean monthly difference {c['mean_difference'] * 1e4:.2f}")
print(json.dumps({k: v for k, v in crsp_definition.items() if k.startswith(("months_", "most_"))}, indent=1))
form = V.crsp_definition_reading(crsp_definition["with_dividends"]["tracking_error"])
crsp_definition["reading"] = {"levels": list(C.CRSP_DEFINITION_READING_LEVELS), "form": form, "says": V.CRSP_DEFINITION_FORMS[form]}
print(f"Reading, by the two levels set in the constants: {form}, {V.CRSP_DEFINITION_FORMS[form]}.")
"""),
    md("""
## 7. What is committed

Five files go into `results/20260928/`: the comparisons by year, the value comparison of E2 by year, a summary of every comparison over the usable window and over the evaluation window, the six verdicts, and, added on 19 November 2026, the summary of the return on CRSP's definition. Nothing is committed by month. The benchmark's return, the universe's count and its summed capitalisation reproduce CRSP's index file so closely, month by month, that a table of them would be that file under other names, so they stay in memory.
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
           X.write_results("verdicts_milestone1.json", record=verdicts),
           X.write_results("reconstruction_crsp_definition.json", record=dict(written_by="notebooks/04_reconstruction.ipynb", **crsp_definition))]
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

- *E3*, for the momentum sort and for the variance sort: correlation at least 0.80 between the universe's quintile spread and French's, and a mean difference within two standard errors of zero, over the months both exist. The levels are in the repository's first commit, of 10 November 2026, and unchanged since. The author's design notes date them to 6 September 2026, before any data was read. French's largest size quintile is set by NYSE breakpoints and holds between about 100 and 530 names a month, fewer than the universe's 500 in almost every month from April 1957 and more than its 90 before, so the two sets overlap without coinciding and a correlation near one is not expected. Below 0.80 the universe is not the large-cap universe it claims to be. The universe goes from about 90 names to about 500 in April 1957, so the momentum comparison is also printed and committed for January 1927 to March 1957 and April 1957 to December 2024, with no verdict. E3's verdict is on the whole window.
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
count_table = count_table[["sort"] + [c for c in count_table.columns if c != "sort"]]
whole = [c for c in count_table.columns if c != "sort"]
count_table[whole] = count_table[whole].astype("Int64")   # whole numbers, and an empty cell where a column belongs to the other sort
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
from slt import universe as U
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
- *Share classes*, a check with no verdict, fixed on 12 November 2026 once the results of section 4 were known: from structural counts, the universe holds one company with two share classes from July 1999 to 2004, none from 2005 to 2013, and up to five from 2014. Where both classes are complete members their daily returns are correlated at about 0.98, which sets the sample matrix's smallest eigenvalue. Section 5 counts them and recomputes the ratio with one class for each company.

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
names, _ = X.read_table(root, "crsp_msenames")
company = U.companies(universe["ulong"][universe["ulong"]["month"] >= first_window], names)
members = members.merge(company[["month", "permno", "permco"]], on=["month", "permno"], how="left")   # held in memory only
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
rows, pairs = [], []
for month in pd.period_range(C.EVAL_START, C.EVAL_END, freq="M"):
    m = members[(members["month"] == month) & members["cap"].notna()]
    res = E.month_estimates(month, excess, factors_d, bench, m)
    res.pop("matrices")
    pairs += [{"month": month, "correlation": r} for r in res.pop("pair_correlations")]
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
## 5. A check on share classes, with no verdict

This check was fixed on 12 November 2026, once the results of section 4 were known, to explain the months in which the ratio reaches E4's level. It carries no verdict, and E4's level and verdict do not move. Some companies have more than one share class in the index, each a member with its own PERMNO, for example a voting and a non-voting class. CRSP gives each company a permanent company number, the PERMCO, which the names file holds for every security over time, read here on the same day as membership. Two classes of one company can move almost together: the difference between them is then a portfolio with almost no variance, and it sets the smallest eigenvalue of the sample matrix. The check counts, by month, the companies with more than one class among the members and the pairs of complete members that are classes of one company, with the correlation of their daily returns over the window. It then keeps one class for each company, the one with the larger market capitalisation at the start of the month, and recomputes the sample and Ledoit-Wolf condition numbers on the complete members of that universe by the same rules. The pairs and their correlations are held in memory: only counts, medians and the months at the level are written, and no weight of these companies, which before 2005 would be one company's weight.
"""),
    code("""
pairs = pd.DataFrame(pairs, columns=["month", "correlation"])
check = E.share_class_check(rows, pairs)
print(f"companies with more than one class among the members: {check['companies_with_more_than_one_class_mean']:.2f} a month on average, "
      f"{check['companies_with_more_than_one_class_most']} at the most | classes left out with one for each company: {check['classes_left_out_mean']:.2f} a month")
median_corr = check["median_correlation_from_that_month"]
print(f"pairs of complete members that are classes of one company, by month: {check['share_class_pairs_by_month']}, "
      f"of which {check['share_class_pairs_from_that_month']} from {check['share_class_pairs_from']}, median correlation "
      + ("none, with no pair" if median_corr is None else f"{median_corr:.3f}"))
print(f"with one class for each company, the ratio is at or above {C.E4_CONDITION_RATIO_MIN:.0f} in {check['months_one_class_at_or_above_level']} of {len(rows)} months "
      f"{check['months_one_class_at_or_above_level_by_year']}, median {check['ratio_one_class_median']:.2f}, least {check['ratio_one_class_least']:.2f}, "
      f"largest {check['ratio_one_class_largest']:.1f}; no verdict")
"""),
    md("""
## 6. What is committed

Four files go into `results/20260928/`, each an aggregate with no identifier of a security and no series by month: the conditioning and the counts by year, with the check on share classes by year, a summary with E4's first clause, the eigenvalue check and the check on share classes over the window, the verdicts of milestone 2 with E4's first clause added beside E3, and the provenance record of French's files.
"""),
    code("""
summary = {
    "written_by": "notebooks/07_covariance.ipynb", "manifest_sha256": C.MANIFEST_SHA256,
    "french_vintages": {r["key"]: r["vintage_line"] for r in records},
    "estimators": list(C.COV_ESTIMATORS), "window_years": C.COV_DAILY_WINDOW_YEARS, "scale": C.DAILY_TO_MONTHLY_SCALE,
    "pca_components": C.PCA_N_COMPONENTS, "complete_window_missing_max": C.COMPLETE_WINDOW_MISSING_MAX,
    "complete_days_left_out_max": C.COMPLETE_DAYS_LEFT_OUT_MAX, "model_row_own_min": C.FACTOR_ROW_OWN_MIN,
    "sector_min_complete": C.SECTOR_MIN_COMPLETE, "sample_monthly": C.SAMPLE_MONTHLY_STATUS,
    "e4_first_clause": e4, "read_on": C.E4_READ_ON, "share_class_check": check,
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

What could still be wrong. A condition number says how much a matrix amplifies errors. Whether its forecasts are right is for the second clause of E4 and for the bias statistic, which compares each month's forecast with what the month realised, both in milestone 3. A model row rests on the member's own days, and for a recent listing those can be few and unrepresentative. The factor model's rows use French's factors, built from all US stocks, for a universe of large ones. And the window weights every day of three years equally, so a matrix lags a change in how stocks move together.
"""),
]

# ---------------------------------------------------------------------------
# 08: the mandate
# ---------------------------------------------------------------------------

MODULES_08 = """
import hashlib

import numpy as np

from slt import evaluation as V
from slt import mandate as M
from slt.source import optimiser as O
"""

NB08 = [
    md("""
# 08. The mandate: the constraints, the copied optimiser and one month worked through

**What this notebook does.** It sets out the four sets of constraints the study runs, written into `src/slt/constants.py` before any path ran. It checks that the optimiser is the source repository's, unchanged, and works one month of the extraction through it under each of the four sets.

**Why it comes here.** Notebook 07 built the month's matrices of variances and covariances. This notebook shows what the optimiser does with one of them before the script runs it 7,344 times, 24 paths over 306 months. Every rule that decides a weight is in this notebook or in the constants, and the script adds nothing to them.

**Terms used in this notebook.**

- *market capitalisation*: price times shares outstanding, at the end of the month before.
- *benchmark*: the universe weighted by market capitalisation at the start of the month.
- *active weight*: a stock's weight in the portfolio minus its weight in the benchmark. The *active return* is the portfolio's return minus the benchmark's.
- *covariance matrix*: the table of the variances of the members' monthly returns on its diagonal and their covariances off it. Notebook 07 built four for each month, each from a *covariance estimator*: a rule that builds the matrix from three years of daily returns.
- *tracking error*: the standard deviation of the active return, here per year. The *forecast tracking error* is the one the covariance matrix implies for the month's active weights, the square root of 12 times a′Σa, with a the active weights and Σ the month's covariance matrix.
- *optimiser*: the program that chooses the weights with the smallest forecast tracking error under the constraints. The problem is a convex quadratic program, with one minimum, solved by the Clarabel solver through cvxpy.
- *basis point*: a hundredth of a percentage point.
- *turnover*: here one-way turnover, half the sum of the absolute changes in weight, the share of the portfolio sold and bought.
- *reference portfolio*: last month's portfolio after last month's returns, with the stocks that left the universe sold, the stocks that entered bought at their benchmark weight, and the others rescaled to sum to one. *Forced turnover* is the turnover from last month's portfolio to the reference.
- *turnover cap*: the limit on the turnover from the reference portfolio to the new weights, 2 per cent a month on the main paths.
- *tilt*: every stock in GICS sector 10 (Energy) or 55 (Utilities) held at no more than half its benchmark weight.
- *exclusion*: weight zero for GICS sub-industries 30203010 (Tobacco) and 20101010 (Aerospace and Defense). A sub-industry is wider than a list of controversial weapons.
- *beta*: the slope of a stock's return on the benchmark's, from the month's covariance matrix. *Beta neutrality* holds the portfolio's beta equal to the benchmark's, one.
- *constraint set*: the constraints of one run, cumulative in four sets. *C0*: long-only, no negative weight. *C1* adds the bound per stock, which holds each stock within 50 basis points of its benchmark weight, the bound per company, which holds the sum over a company's share classes within the same 50 basis points, and the sector band, which holds each sector within 2 percentage points of its benchmark weight. *C2* adds the turnover cap. *C3*, the *mandate*, adds the tilt, the exclusions and beta neutrality. In a set with the tilt the sector band covers the other sectors only, because half of Energy's weight is above 2 percentage points in 268 of the 306 months, and no portfolio could meet both. In the same way, in a set with the tilt or the exclusions the bounds per stock and per company cover the stocks outside the tilted sectors and the excluded sub-industries: a tilted stock is held between zero and half its benchmark weight, and an excluded one at zero.
- *binding*: a constraint is binding when the solution sits on its limit, within one millionth.

**What I expect to see.**

- Under C0, C1 and C2 the portfolio is the benchmark, with a forecast tracking error of zero to within the solver's precision: nothing in those sets pushes it away. The sets are run all the same, because the design names them, and reported by year.
- Under C3 the tilt and the exclusions push the portfolio off the benchmark, and the optimiser spends the bounds on the stocks that best hedge what the tilt removed. A forecast tracking error inside the range E6 sets for the realised one, 50 to 150 basis points a year, is expected. E6 itself is judged in notebook 09.
- No company whose share classes differ in sector or in the bound's scope: the count is printed and expected to be zero.

Running time: about three minutes, most of it reading the extraction.
"""),
    code(SETUP),
    code(MODULES_08),
    md("""
## 1. The copied optimiser

The optimiser is the source repository's `optimiser.py`, copied into `src/slt/source/` with its first line naming the source file and commit and every other line unchanged. The cell recomputes the SHA-256 of each copy without its first line and compares it with the source file's. The study's own module, `src/slt/mandate.py`, builds the same problem from the copy's own pieces and adds the constraints the copy does not take, all of them sums over groups of weights: the bound per company and the sector band. Its tests in `tests/test_mandate.py` solve the source's own constraint sets on made-up months with both and require the same weights within one millionth, a month in which the cap has to be raised included.
"""),
    code("""
for name, (source, digest) in sorted(C.COPIED_FILES.items()):
    path = ROOT / C.COPIED_INTO / name
    if not path.exists():
        print(f"{name}: copied in milestone {C.COPIED_FOR_MILESTONE[name]}")
        continue
    body = path.read_bytes().split(b"\\n", 1)[1]
    ok = hashlib.sha256(body).hexdigest() == digest
    print(f"{name}: {'the source file unchanged' if ok else 'DIFFERS from the source file'} ({digest[:12]})")
"""),
    md("""
## 2. One month worked through

The month is July 2008, when Energy was close to its largest share of the benchmark in the window, so the tilt moves close to the most weight it ever moves. The month's matrices are notebook 07's, the Ledoit-Wolf estimator's is used, and the reference portfolio is the benchmark itself, as on a path's first month, so the turnover cap is left out here: the script applies it from the second month on.
"""),
    code("""
t0 = time.time()
inp = V.inputs(root)
month = pd.Period("2008-07", "M")
m, r = V.month_view(inp["members"], month)
Sigma = V.month_matrices(month, inp, m)["ledoit_wolf"]
print(f"{time.time() - t0:.0f} s | members with a capitalisation: {len(m.b)} | tilted: {int(m.tilt_mask.sum())}, "
      f"{100 * m.b[m.tilt_mask].sum():.1f} per cent of the benchmark | excluded: {int(m.exclude_mask.sum())}, "
      f"{100 * m.b[m.exclude_mask].sum():.1f} per cent")
worked = {}
for cs, constraints in C.CONSTRAINT_SETS.items():
    cons = tuple(c for c in constraints if c != "turnover_cap")
    res = M.solve(Sigma, m, cons, m.b)
    a = res.weights - m.b
    worked[cs] = {"forecast_te_bp": round(res.forecast_te * 1e4, 4), "held": int((res.weights > C.OPT_TOL).sum()),
                  "names_off_benchmark": int((np.abs(a) > C.OPT_TOL).sum()), "binding": res.binding,
                  "company_scope_mismatches": M.company_scope_mismatches(m, constraints)}
    print(f"{cs}: forecast tracking error {res.forecast_te * 1e4:.2f} basis points a year | stocks held {worked[cs]['held']} | "
          f"off the benchmark {worked[cs]['names_off_benchmark']} | binding {res.binding}")
"""),
    md("""
Under the mandate, the sector active weights of the month, in percentage points (printed here and not committed):
"""),
    code("""
res = M.solve(Sigma, m, tuple(c for c in C.CONSTRAINT_SETS["C3"] if c != "turnover_cap"), m.b)
a = pd.Series(res.weights - m.b).groupby(m.sector).sum() * 100
beta = O.benchmark_betas(Sigma, m.b) @ res.weights
print(a.round(2).to_string())
print(f"portfolio beta to the benchmark: {beta:.6f}")
"""),
    md("""
## 3. What is committed

One file goes into `results/20260928/`: the worked month's figures under each set, the forecast tracking error, the counts of stocks held and off the benchmark, the binding counts and the company count. No weight and no name.
"""),
    code("""
record = {"written_by": "notebooks/08_mandate.ipynb", "manifest_sha256": C.MANIFEST_SHA256, "month": str(month),
          "estimator": "ledoit_wolf", "reference": "the benchmark, as on a path's first month; no turnover cap",
          "members": int(len(m.b)), "tilted": int(m.tilt_mask.sum()), "excluded": int(m.exclude_mask.sum()), "sets": worked}
print("written:", X.write_results("mandate_worked_month.json", record=record).relative_to(ROOT))
"""),
    md("""
## What was learned, and what could be wrong

What the notebook establishes: that the optimiser is the source's, that the constraint sets do what the constants say on a month of the extraction, and how far the mandate moves the portfolio from the benchmark in one month.

What could still be wrong. One month says nothing about the others: a constraint that does not bind in July 2008 may bind in another month, and the script records every binding count. The worked month starts from the benchmark, so it shows no turnover cap at work. And the forecast tracking error rests on the covariance matrix; whether it is right is the bias statistic of notebook 09.
"""),
]

# ---------------------------------------------------------------------------
# 09: the rolling evaluation
# ---------------------------------------------------------------------------

MODULES_09 = """
import numpy as np

from slt import evaluation as V

OUT = ROOT / C.OUTPUTS_DIR / "evaluation"
"""

NB09 = [
    md("""
# 09. The rolling evaluation: the 24 paths, the frontier, E4's second clause, E5 and E6

**What this notebook does.** It reads the records the script `scripts/run_evaluation.py` wrote on this machine, one row per path and month, and computes each path's measures over the evaluation window, July 1999 to December 2024, and by year. It judges E4's second clause, E5 and E6, and writes the tables that are committed.

**Why it comes here.** The script runs the paths, which takes hours. This notebook only reads what it wrote, so it can be run again at once. Nothing by security leaves `outputs/`, which git ignores.

**Terms used in this notebook.**

- *market capitalisation*: price times shares outstanding, at the end of the month before.
- *benchmark*: the universe weighted by market capitalisation at the start of the month.
- *active weight*: a stock's weight in the portfolio minus its weight in the benchmark. The *active return* is the portfolio's return minus the benchmark's in the month.
- *covariance matrix*: the table of the variances of the members' monthly returns on its diagonal and their covariances off it.
- *factor*: a common source of return, here French's five factors and his momentum factor.
- *covariance estimator*: a rule that builds the covariance matrix from three years of daily returns. Notebook 07 has four: the sample estimator, the Ledoit-Wolf estimator, the factor estimator, which explains each stock's returns by the factors, and the PCA estimator.
- *turnover*: here one-way turnover, half the sum of the absolute changes in weight, the share of the portfolio sold and bought.
- *turnover cap*: the limit on the turnover the optimiser chooses, measured from the reference portfolio, last month's portfolio after its returns with the stocks that left sold and the entrants bought at their benchmark weight. *Forced turnover* is that move. A cap *binds* when the month's chosen turnover is at the cap, and a month is *relaxed* when the cap had to be raised to the smallest feasible turnover.
- *constraint set*: C0 is long-only. C1 adds bounds per stock, per company and per sector, C2 the turnover cap, and C3, the *mandate*, the *tilt*, which holds every Energy and Utilities stock at no more than half its benchmark weight, the exclusion of two sub-industries and a *beta*, the slope of the portfolio's return on the benchmark's, equal to one. Notebook 08 sets them out.
- *path*: one specification run month by month over the evaluation window: a covariance estimator under a constraint set at a turnover cap. There are 24: the four estimators under the four sets C0 to C3, at the 2 per cent cap where the set has a cap, and the Ledoit-Wolf and factor estimators under the mandate at caps of 0.5, 1 and 4 per cent and with no cap. Every path starts as the benchmark.
- *tracking error*: the standard deviation of the monthly active returns, times the square root of 12. The *realised tracking error* is measured on the returns, and the *forecast tracking error* is the one the month's covariance matrix implied when the weights were chosen.
- *bias statistic*: the standard deviation of each month's active return divided by that month's forecast monthly tracking error. It is one when the forecasts are right on average, above one when they are too low. It is taken over the months whose forecast tracking error is above one millionth a year, and the number of months it rests on is given beside it.
- *circular block bootstrap*: a way to measure the sampling error of a statistic of monthly data. It takes 2,000 new samples of the months, each built from blocks of 12 consecutive months that run past the last month into the first, keeps each month's active return with its forecast, and computes the statistic on each sample. The *95 per cent interval* is the central 95 per cent of those 2,000 statistics. It is not the range of 0.8 to 1.25 that E6 sets for the statistic itself.
- *frontier*: the forecast and realised tracking error of one estimator under the mandate at five caps, from the tightest to none.
- *basis point*: a hundredth of a percentage point.
- *cost*: 10 basis points per unit of total turnover, forced included, and 20 as the sensitivity.
- *Newey-West standard error*: the standard error of a mean that allows each month to be correlated with the next twelve.
- *cell rule*: a cell of the table by month is written when at least three stocks have an active weight above one millionth and no stock's absolute contribution to the active return is more than 80 per cent of the sum of the absolute contributions. A cell that fails is left empty and counted.
- *provenance*: the record of what a result was built from.
- *verdict*: the record of whether an expectation fixed before the run is met.

**What I expect to see, fixed in `src/slt/constants.py` before any path ran.**

- *E4, second clause*: under the mandate at the 2 per cent cap, the sample estimator's realised tracking error is above the Ledoit-Wolf estimator's.
- *E5*: for the Ledoit-Wolf and the factor estimators, the mean forecast tracking error does not rise as the cap loosens from 0.5 per cent to none, and the realised tracking error at the 4 per cent cap is below that at 0.5 per cent.
- *E6*: under the mandate at the 2 per cent cap, the realised tracking error of the Ledoit-Wolf, factor and PCA estimators each lies between 50 and 150 basis points a year, and each one's bias statistic between 0.8 and 1.25.
- Under C0, C1 and C2 the portfolio is the benchmark in every month, to within one millionth of a weight.

Running time: under a minute, once the script has run.
"""),
    code(SETUP),
    code(MODULES_09),
    md("""
## 1. The script's run

The provenance record the script wrote: when it started and finished, the versions of the solver and its libraries, and the threads of linear algebra in each process. Then the records of every path.
"""),
    code("""
prov = json.loads((OUT / "provenance.json").read_text())
print(json.dumps(prov, indent=1))
assert not any(prov["exit_codes"].values()), "a group of the script stopped with an error: run the script again to resume"
records = V.load_records(OUT)
missing = [C.PATH_LABELS[p] for p in C.PATHS if C.PATH_LABELS[p] not in records]
assert not missing, f"no records for {missing}: run the script to its end first"
for lab, rec in records.items():
    assert len(rec) == C.EVAL_MONTHS, f"{lab} has {len(rec)} months"
print(f"{len(records)} paths, {C.EVAL_MONTHS} months each")
"""),
    md("""
## 2. Every path over the window
"""),
    code("""
summaries = {lab: V.summarise(rec) for lab, rec in records.items()}
show = pd.DataFrame({lab: {"realised TE (bp)": s["realised_te"] * 1e4, "forecast TE (bp)": s["forecast_te_mean"] * 1e4,
                           "bias": s["bias"], "bias months": s["bias_months"], "active gross (bp)": s["active_return_gross"] * 1e4,
                           "net at 10 bp (bp)": s["active_return_net_at_10bp"] * 1e4, "turnover / yr": s["turnover_total_per_year"],
                           "cap binds": s["cap_binding_share"], "relaxed": s["relaxed_months"], "held": s["held_mean"]}
                     for lab, s in summaries.items()}).T
print(show.round(3).to_string())
"""),
    md("""
## 3. C0, C1 and C2: the benchmark

Under these sets nothing pushes the portfolio off the benchmark. The months in which it differs from the benchmark by more than one millionth of a weight, and the largest absolute active weight in any month:
"""),
    code("""
for p in C.PATHS:
    if p[1] != C.MANDATE:
        s = summaries[C.PATH_LABELS[p]]
        print(f"{C.PATH_LABELS[p]}: months off the benchmark {s['months_off_benchmark']} | largest absolute active weight {s['largest_abs_active_weight']:.2e}")
"""),
    md("""
## 4. The frontier, E4's second clause, E5 and E6
"""),
    code("""
verdicts = V.judge(summaries)
for est in C.FRONTIER_ESTIMATORS:
    v = verdicts[f"E5_{est}"]
    caps = ["none" if c is None else f"{100 * c:g}%" for c in v["caps"]]
    print(est, "mean forecast TE (bp) by cap:", dict(zip(caps, [round(x * 1e4, 2) for x in v["forecast_te_mean"]])))
    print(f"E5, {est}: {'met' if v['met'] else 'not met'}. The mean forecast tracking error "
          f"{'does not rise' if v['forecast_nonincreasing'] else 'rises'} as the cap loosens, and the realised tracking error is "
          f"{v['realised_te_looser'] * 1e4:.1f} basis points a year at the 4 per cent cap against {v['realised_te_tighter'] * 1e4:.1f} at 0.5 per cent.")
NAMES = {"sample": "sample", "ledoit_wolf": "Ledoit-Wolf", "factor": "factor", "pca": "PCA"}
def the(ests):
    names = [f"the {NAMES[e]}" for e in ests]
    return (", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]) + (" estimators" if len(names) > 1 else " estimator")
def one_line(name):
    v = verdicts[name]
    parts = ([f"holds for {the(v['holds_for'])}"] if v["holds_for"] else []) + ([f"fails for {the(v['fails_for'])}"] if v["fails_for"] else [])
    return f"{name}: {'met' if v['met'] else 'not met'}. It " + " and ".join(parts) + "."
print(one_line("E5"))
v = verdicts["E4_second_clause"]
print(f"E4, second clause: {'met' if v['met'] else 'not met'}. Under the mandate at the 2 per cent cap the sample estimator's realised "
      f"tracking error is {v['sample_realised_te'] * 1e4:.1f} basis points a year against Ledoit-Wolf's {v['ledoit_wolf_realised_te'] * 1e4:.1f}.")
for est in ("ledoit_wolf", "factor", "pca"):
    v = verdicts[f"E6_{est}"]
    print(f"E6, {est}: {'met' if v['met'] else 'not met'}. Realised tracking error {v['realised_te'] * 1e4:.1f} basis points a year "
          f"against 50 to 150, bias statistic {v['bias']:.3f} over {v['bias_months']} months against 0.8 to 1.25.")
print(one_line("E6"))
"""),
    md("""
## 5. The bias statistic with its interval, added on 19 November 2026

Added on 19 November 2026, after every recorded run, with no verdict: E6's verdict is unchanged. Each mandate path's bias statistic is given with a 95 per cent interval from a circular block bootstrap of its months, in the form of the companion study's notebook 12 and with its settings. The interval is computed from the two tables by month that section 6 commits, the active returns and the forecast tracking errors, so it can be computed again from the repository alone. Each bias statistic recomputed from the two tables must equal the one of section 2. The paths whose interval reaches into E6's range, whose upper end is 1.25, are listed from the table below it: two of the twelve, the Ledoit-Wolf estimator at the 0.5 per cent cap and at the 1 per cent cap, on which the excess over 1.25 is not statistically significant at the 5 per cent level. On the other ten it is, each interval lying wholly above 1.25.
"""),
    code("""
table, empty = V.active_returns_table(records)
forecast, forecast_empty = V.forecast_table(records)
assert forecast_empty == {c.replace("active_return_", "forecast_te_"): n for c, n in empty.items()}
bands = V.bias_bands(table, forecast)
for _, r in bands.iterrows():
    s = summaries[r["path"]]
    assert abs(r["bias"] - s["bias"]) < 1e-12 and r["months"] == s["bias_months"], r["path"]
print(bands[["path", "months", "bias", "interval_low", "interval_high"]].round(3).to_string(index=False))
reach = bands.loc[bands["interval_low"] <= C.E6_BIAS_RANGE[1], "path"].tolist()
print(f"interval reaching into the band, whose upper end is {C.E6_BIAS_RANGE[1]}: {', '.join(reach)}")
print(f"interval wholly above it: {len(bands) - len(reach)} of {len(bands)} paths")
"""),
    md("""
## 6. A studentised interval on the bias statistic, added on 24 November 2026

Added on 24 November 2026, after the results were known, with no verdict: E6's verdict is unchanged. Beside the interval of section 5, each mandate path's bias statistic is given with a studentised interval on the log of its variance, at blocks of 3, 6, 12 and 24 months, from the same two tables by month. The log variance of the month's active return over its forecast is a function of two means, those of the ratio and of its square. Its standard error comes from the delta method, i.e., the gradient of the log variance with respect to the two means applied to their long-run covariance, and that covariance is taken from the means of whole blocks: on the sample, its consecutive blocks counted from July 1999, the months after the last whole block in none, and on each of 2,000 resamples, its own blocks, as in the paired test of notebook 13, whose resamples of whole blocks these are. The interval is the estimate plus or minus the 95th percentile of the studentised statistic times its standard error, read back as a bias statistic. The Ledoit-Wolf path at the 2 per cent cap, whose interval of section 5 lies above 1.25 with its lower end nearest it, is printed first.
"""),
    code("""
studentised = V.bias_bands_studentised(table, forecast)
cols = ["path", "block", "whole_blocks", "bias", "critical_value", "interval_low", "interval_high", "percentile_low", "percentile_high"]
lw = C.PATH_LABELS[("ledoit_wolf", C.MANDATE, C.MAIN_CAP)]
print(studentised.loc[studentised["path"] == lw, cols].round(3).to_string(index=False))
print()
print(studentised[cols].round(3).to_string(index=False))
for block, g in studentised.groupby("block"):
    reach = g.loc[g["interval_low"] <= C.E6_BIAS_RANGE[1], "path"].tolist()
    print(f"blocks of {block} months, interval reaching into the band, whose upper end is {C.E6_BIAS_RANGE[1]}: {', '.join(reach) or 'none'}")
"""),
    md("""
## 7. What is committed

Eight files go into `results/20260928/`. The active returns by month of the twelve paths under the mandate, gross of costs, each cell under the cell rule; the measures of every path over the window and by year, with no weight and no name; the verdicts of milestone 3, E7 added by notebook 10; and the provenance of the script's run. Added on 19 November 2026: the forecast tracking error by month of the same twelve paths, each cell empty where the active return's is, and the bias statistics with their intervals. Added on 24 November 2026: the studentised intervals of section 6.
"""),
    code("""
by_year = pd.concat([V.by_year(rec).assign(path=lab) for lab, rec in records.items()], ignore_index=True)
by_year = by_year[["path"] + [c for c in by_year.columns if c != "path"]]
withheld = [C.PATH_LABELS[p] for p in C.PATHS if p[1] not in C.LARGEST_ACTIVE_WEIGHT_SETS]   # one stock's weight under C3
by_year.loc[by_year["path"].isin(withheld), "largest_abs_active_weight"] = float("nan")
committed = {lab: dict(summaries[lab], **({"largest_abs_active_weight": None} if lab in withheld else {})) for lab in records}
summary = V.jsonable({"written_by": "notebooks/09_rolling_evaluation.ipynb", "manifest_sha256": C.MANIFEST_SHA256,
                      "paths": committed, "empty_cells_by_column": empty,
                      "costs": {"base": C.COST_PER_UNIT_TURNOVER, "sensitivity": C.COST_SENSITIVITY, "on": C.COST_BASE}})
path = ROOT / C.RESULTS_DIR / "verdicts_milestone3.json"
earlier = json.loads(path.read_text()) if path.exists() else {}
record = V.jsonable({k: v for k, v in verdicts.items() if k != "E7"})
if "E7" in earlier:
    record["E7"] = earlier["E7"]                     # written by notebook 10
print("empty cells by column:", empty)
for p in [X.write_results("active_returns_monthly.csv", frame=table),
          X.write_results("paths_summary.json", record=summary),
          X.write_results("paths_by_year.csv", frame=by_year),
          X.write_results("verdicts_milestone3.json", record=record),
          X.write_results("provenance_evaluation.json", record=prov),
          X.write_results("forecast_te_monthly.csv", frame=forecast),
          X.write_results("bias_intervals.csv", frame=bands),
          X.write_results("bias_intervals_studentised.csv", frame=studentised)]:
    print("written:", p.relative_to(ROOT))
"""),
    md("""
## What was learned, and what could be wrong

What the notebook establishes: each path's realised and forecast tracking error, the forecasts' calibration, the cost of the turnover each path chose, and the verdicts of E4's second clause, E5 and E6.

What could still be wrong. A realised tracking error over 306 months has its own sampling error, about 4 per cent of its value. Two paths measured on the same months are compared by the paired test of notebook 13, added on 18 November 2026, which allows for what the two series share. A bias statistic near one over the window can hide periods in which the forecasts were too low and others in which they were too high. The costs are a flat rate per unit of turnover, with no market impact. And every path's active return is measured against this study's benchmark, which weights the members by their full market capitalisation. The published index weights them by the shares available to the public, so the two differ.
"""),
]

# ---------------------------------------------------------------------------
# 10: turnover
# ---------------------------------------------------------------------------

NB10 = [
    md("""
# 10. Turnover: membership, reweighting and rebalancing, and E7

**What this notebook does.** It splits each path's monthly *turnover*, the share of the portfolio sold and bought, into three parts, each measured, sums them by year, and judges E7. It then lists the months in which the cap on turnover gave way, with the turnover each of them needed.

**Terms used in this notebook.**

- *one-way turnover*: half the sum of the absolute changes in weight between two portfolios. Every turnover in this notebook is one-way.
- *drifted portfolio*: last month's portfolio after last month's returns. The *total turnover* of a month runs from it to the new weights.
- *market capitalisation*: price times shares outstanding, at the end of the month before.
- *benchmark*: the universe weighted by market capitalisation at the start of the month.
- *reference portfolio*: the drifted portfolio with the stocks that left the universe sold, the entrants bought at their benchmark weight, and the others rescaled to sum to one.
- *membership*, the *forced turnover*: the turnover from the drifted portfolio to the reference portfolio.
- *restored portfolio*: each continuing stock at last month's chosen weight times the ratio of its benchmark weights this month and last, every entrant at its benchmark weight, rescaled to one. It keeps each stock's weight relative to its benchmark weight where the optimiser left it.
- *reweighting*: the turnover from the reference portfolio to the restored portfolio, which following the benchmark's own reweighting takes. It comes from changes in share counts, and from dividends, because the drifted portfolio grows by total returns and the benchmark's weights by prices.
- *rebalancing*: the turnover from the restored portfolio to the new weights, the optimiser's own moves. It is zero for a path that holds the benchmark.
- *overlap*: the three parts less the total, the trades of one step that the next step undoes. By the triangle inequality it is never below zero.
- *chosen turnover*: the turnover from the reference portfolio to the new weights, the quantity the cap bounds.
- *mandate*: constraint set C3, the portfolio the study runs, set out in notebook 08.
- *the cap gives way*: in a month where the other constraints need more chosen turnover than the cap allows, the optimiser takes the least turnover they need, above the cap. The record of such a month marks it as relaxed.
- *cell rule*: a figure by month is committed only if at least three names make it up and no name is more than 80 per cent of it. Here the names are those whose weight moves, and each one's part is half its absolute move.
- *verdict*: the record of whether an expectation fixed before the run is met.

**The benchmark's own figures.** Counting the change from June to July 1999, the benchmark alone turns over 5.82 per cent a year over the 306 months to December 2024: membership 2.84, reweighting 3.73 and an overlap of 0.75. A path that holds the benchmark starts in July 1999 with no turnover in that month, so section 2 shows it at 5.78, 2.80, 3.71 and 0.73. Restoring last month's weights themselves undoes the month's price moves, and on the benchmark alone it would take 30.64 per cent a year, trading that no path does. A middle part defined that way would leave the rest below zero in every month, so the middle part follows the benchmark's own weights. Section 2 computes them by this repository's code, `evaluation.benchmark_turnover`, added on 19 November 2026, and commits them. They describe the benchmark alone and say nothing of the mandate.

**What I expect to see, fixed in `src/slt/constants.py` before any path ran.**

- *E7*: under the mandate at the 2 per cent cap on the Ledoit-Wolf estimator, membership is less than a third of total turnover over the window. The index replaces about four to five per cent of its weight a year, and a cap of 2 per cent a month allows about 24 per cent a year of chosen turnover.
- With no verdict: under C0 and C1, where the portfolio is the benchmark, rebalancing is zero to the solver's precision, and the other figures are close to the benchmark's own.

Running time: about three minutes, most of it reading the extraction, which section 2 needs to follow the benchmark alone and section 4 to rebuild each month's reference portfolio.
"""),
    code(SETUP),
    code(MODULES_09),
    md("""
## 1. The decomposition, per path, over the window and by year

Per year over the window, in per cent: the total, the three parts and the overlap. The table by year is committed in section 3.
"""),
    code("""
records = V.load_records(OUT)
figures = [f"turnover_{k}" for k in C.TURNOVER_FIGURES]
table = pd.concat([V.turnover_by_year(rec).assign(path=lab) for lab, rec in records.items()], ignore_index=True)
table = table[["path"] + [c for c in table.columns if c != "path"]]
years = pd.Series({lab: len(rec) / 12 for lab, rec in records.items()})
window = table.groupby("path", sort=False)[figures].sum().div(years, axis=0)
print((100 * window).round(2).to_string())
low = min(float(rec["turnover_overlap"].min()) for rec in records.values())
print(f"smallest overlap in any month of any path: {low:.2e}")
"""),
    md("""
## 2. The benchmark's own figures, and the paths that hold the benchmark beside them

The benchmark alone, from June 1999 to December 2024, its turnover split as a path's is, with the turnover that restoring last month's weights themselves would take beside it. Its rebalancing must be zero to the precision of the arithmetic. Then the two paths that hold the benchmark, beside its figures.
"""),
    code("""
t0 = time.time()
inp = V.inputs(root)
benchmark = V.benchmark_turnover(inp)
print({k: (round(100 * v, 2) if isinstance(v, float) and k != "largest_rebalancing" else v) for k, v in benchmark.items()}, f"| {time.time() - t0:.0f} s")
assert benchmark["months"] == C.EVAL_MONTHS and benchmark["largest_rebalancing"] < 1e-12
own = {f"turnover_{k}": 100 * benchmark[k] for k in ("total", "membership", "reweighting", "overlap")}
for lab in (V.label("sample", "C0", None), V.label("sample", "C1", None)):
    row = 100 * window.loc[lab]
    print(lab + ": " + " | ".join(f"{k[9:]} {row[k]:.2f}" + (f" (benchmark {own[k]:.2f})" if k in own else "") for k in figures))
"""),
    md("""
## 3. E7
"""),
    code("""
summaries = {lab: V.summarise(rec) for lab, rec in records.items()}
e7 = V.judge(summaries)["E7"]
print(f"E7: {'met' if e7['met'] else 'not met'}. Under the mandate at the 2 per cent cap on the Ledoit-Wolf estimator, membership is "
      f"{100 * e7['membership_share_of_total']:.1f} per cent of total one-way turnover over the window, against less than {100 * C.E7_FORCED_SHARE_MAX:.1f}.")
"""),
    md("""
## 4. The months in which the cap gave way

Every path starts as the benchmark, and the mandate's first month needs more chosen turnover than any cap allows, so on every capped path under the mandate the cap gives way in July 1999. At the cap of 0.5 per cent a month it also gives way in later months. For each such month the chosen turnover is computed again from the recorded weights, from the reference portfolio the script built, and must equal the record's. Each row is then held to the cell rule, and the notebook stops before anything is written if one fails. The rows of each path must equal the count of relaxed months in its records.
"""),
    code("""
t0 = time.time()
relaxed = V.cap_relaxed(inp, OUT, records)
show = relaxed.assign(cap=100 * relaxed["cap"], turnover_chosen=(100 * relaxed["turnover_chosen"]).round(4),
                      largest_share=relaxed["largest_share"].round(3))
print(show.to_string(index=False))
print(f"{len(relaxed)} months on {relaxed['path'].nunique()} paths, in {time.time() - t0:.0f} s")
assert relaxed["cell_ok"].all(), "a month fails the cell rule: nothing is written"
counts = relaxed.groupby("path").size()
for lab, rec in records.items():
    assert int(counts.get(lab, 0)) == int(rec["relaxed"].astype(bool).sum()), lab
"""),
    md("""
## 5. What is committed

Four files: the decomposition by path and year (the total, the three parts and the overlap), the verdicts of milestone 3 with E7 added, the months in which the cap gave way, one row for each, with the path, the month, the cap and the month's chosen turnover, and, added on 19 November 2026, the benchmark's own figures of section 2, one row over the window.
"""),
    code("""
path = ROOT / C.RESULTS_DIR / "verdicts_milestone3.json"
record = json.loads(path.read_text()) if path.exists() else {}
record["E7"] = V.jsonable(e7)
cap_table = relaxed[["path", "month", "cap", "turnover_chosen"]]
for p in [X.write_results("turnover_by_year.csv", frame=table), X.write_results("verdicts_milestone3.json", record=record),
          X.write_results("cap_relaxed.csv", frame=cap_table),
          X.write_results("benchmark_turnover.csv", frame=pd.DataFrame([benchmark]))]:
    print("written:", p.relative_to(ROOT))
"""),
    md("""
## What was learned, and what could be wrong

What the notebook establishes: how much of each path's trading the index's own changes require, how much following the benchmark's own reweighting takes, how much the optimiser chooses, whether the forced part is the small one E7 expects, and in which months the cap gave way to the least turnover the other constraints need.

What could still be wrong. The parts depend on the order of the steps, membership first, then reweighting, then rebalancing: another order would split the same total another way, and the overlap says how much the steps undo of each other. And the forced part counts only the trades the index's changes require on the day the month begins. A portfolio that anticipated a change would trade earlier, and this study does not.
"""),
]


# ---------------------------------------------------------------------------
# 11: the attribution by sector and by factor
# ---------------------------------------------------------------------------

MODULES_11 = """
import numpy as np

from slt import attributions as A
from slt import brinson as B
from slt import evaluation as V
from slt import french as FR
from slt import sorts as Q
from slt.source import attribution as SA

OUT = ROOT / C.OUTPUTS_DIR / "evaluation"
NAMES = {"sample": "sample", "ledoit_wolf": "Ledoit-Wolf", "factor": "factor", "pca": "PCA"}
"""

NB11 = [
    md("""
# 11. The attribution: by sector and by factor, E8 and E9

**What this notebook does.** It explains where the mandate's monthly result against the universe it tracks comes from, in two ways that the terms below set out: by sector, with the method of Brinson and Fachler (1985), and through six common sources of return. It judges E8 and E9 and writes the tables that are committed, by year and over the window, and one by month, added on 24 November 2026.

**Why it comes here.** Notebook 09 measured how far the mandate strayed from the universe. This notebook says what the straying consisted of. It reads the weights the script recorded on this machine, the extraction and French's monthly files, and runs no solver.

**Terms used in this notebook.**

- *market capitalisation*: price times shares outstanding, at the end of the month before.
- *benchmark*: the universe weighted by market capitalisation at the start of the month.
- *active weight*: a weight in the portfolio minus the same weight in the benchmark. The *active return* is the portfolio's return minus the benchmark's, as notebook 09 committed it: each side over the month's names with a return, its weights rescaled to one.
- *mandate*: constraint set C3 of notebook 08, here at the 2 per cent cap on each of the four covariance matrices. The *tilt* is its rule that holds every Energy and Utilities stock at no more than half its benchmark weight.
- *sector*: one of the eleven GICS sectors, or Unclassified, the members with no GICS sector, which holds one to three securities a month.
- *attribution*: a split of the active return into parts that sum to it. Both attributions here use the rescaled weights of the active return, so that they describe the portfolio whose return is committed.
- *allocation*, *selection* and *interaction*: the three parts of the attribution by sector. With w and b a sector's weight in the portfolio and in the benchmark, R_b the benchmark's return, R_b,s and R_p,s the sector's return in the benchmark and in the portfolio: allocation is (w − b)(R_b,s − R_b), what holding the sector at another weight earned; selection is b(R_p,s − R_b,s), what holding other stocks within it earned; interaction is (w − b)(R_p,s − R_b,s), the product of the two differences. Summed over the twelve sectors they equal the active return.
- *held*: a sector is held in a month when one of its stocks with a return has a weight above one millionth. For a sector not held, selection is zero, and interaction is what is left of the sector's part: the sum over its stocks of weight times return, less its weight times the benchmark sector's return.
- *factor*: a common source of return, here French's five factors and momentum, with their monthly returns taken from French's monthly files.
- *loading*: a stock's sensitivity to a factor, from the regression of its daily returns less the daily *risk-free rate*, the one-month Treasury bill rate of French's files, on the six factors over the three years before the month: the factor estimator's loadings of notebook 07, for every path.
- *active exposure*: the loadings weighted by the active weights, one number for each factor. A factor's *contribution* is its active exposure times its return in the month, and the *residual* is the active return less the sum of the contributions. The *factor share* is one minus the residual's variance over the active return's.
- *beta*: the slope of a return on the benchmark's return. The mandate holds its beta at one.
- *tracking error*: the standard deviation of the active return.
- *basis point*: a hundredth of a percentage point.
- *Newey-West standard error*: the standard error of a mean that allows each month to be correlated with the next twelve.
- *verdict*: the record of whether an expectation fixed before the run is met.
- *provenance*: the record of what a result was built from.

**What I expect to see, fixed in `src/slt/constants.py` before this notebook ran on the extraction.**

- *E8*: the three terms sum to the active return within 1e-9 in every month of the four paths. And on each of the Ledoit-Wolf, factor and PCA paths, the mean absolute monthly allocation exceeds the mean absolute monthly selection, and the mean absolute monthly interaction is the smallest of the three, each month's figure being the sum over the twelve sectors. One verdict, met only if it holds on all three. The sample estimator's path is reported with no verdict. The reason: the mandate's active return comes from the tilt, an underweight of two sectors, while the stocks within sectors are chosen to keep the tracking error small.
- *E9*: contributions plus residual equal the active return within 1e-9 in every month. And on each of the four paths the mean active exposure to the market factor is within 0.05 of zero, because the mandate holds its beta at one. One verdict over the four.

Running time: about five minutes, most of it the loadings of 306 months.
"""),
    code(SETUP),
    code(MODULES_11),
    md("""
## 1. The inputs

The extraction as notebook 09's script read it, French's monthly factor returns, and the four paths' records and weights from `outputs/evaluation/`.
"""),
    code("""
t0 = time.time()
inp = V.inputs(root)
factors_m, rf_m, french_records = FR.monthly_factors()
months = pd.period_range(C.EVAL_START, C.EVAL_END, freq="M")
for r_ in french_records:
    print(r_["filename"], r_["zip_sha256"][:12], r_["vintage_line"])
print(f"inputs ready in {time.time() - t0:.0f} s; paths: {[C.PATH_LABELS[p] for p in C.ATTRIBUTION_PATHS]}")
"""),
    md("""
## 2. Both attributions, month by month

Each month the notebook builds the factor estimator's loadings, then for each path splits the active return by sector and computes the inputs of the factor attribution. It checks that the active return it recomputes from the weights equals the one the script recorded.
"""),
    code("""
t0 = time.time()
res = A.run(inp, OUT, months, factors_m, progress=True)
print(f"{time.time() - t0:.0f} s; largest gap between the recomputed and the recorded active return: {res['active_return_gap']:.1e}")
assert res["active_return_gap"] < 1e-12
"""),
    md("""
## 3. The attribution by sector and E8

Per path over the window, in basis points a year with Newey-West standard errors, and the mean absolute monthly figure of each total.
"""),
    code("""
summ = {p[0]: B.summary(res["terms"][p], res["totals"][p], Q.newey_west_se) for p in C.ATTRIBUTION_PATHS}
for est, s in summ.items():
    t = s["total"]
    print(f"{NAMES[est]}: " + " | ".join(f"{k} {1e4 * t[k + '_per_year']:.1f} ({1e4 * t[k + '_per_year_se']:.1f}), mean absolute {1e4 * t[k + '_mean_abs_monthly']:.2f} a month"
                                     for k in C.ATTRIBUTION_TERMS) + f" | sector-months not held {s['sector_months_not_held']}")
print()
print("By sector on the Ledoit-Wolf path, basis points a year:")
lw = summ["ledoit_wolf"]["sectors"]
print(pd.DataFrame({s: {k: round(1e4 * lw[s][k + "_per_year"], 1) for k in C.ATTRIBUTION_TERMS} for s in C.ATTRIBUTION_SECTORS}).T.to_string())
"""),
    code("""
e8 = B.judge_e8(summ)
print(f"E8, identity: {'met' if e8['identity']['met'] else 'not met'}. The largest gap between the sum of the three terms and the active return, "
      f"over every month of the four paths, is {e8['identity']['largest_error']:.1e}, against 1e-9.")
for est, v in e8["second_clause"]["by_estimator"].items():
    label = "no verdict" if est in e8["second_clause"]["no_verdict"] else ("holds" if v["holds"] else "fails")
    print(f"E8, {NAMES[est]} ({label}): mean absolute monthly allocation {1e4 * v['allocation']:.2f}, selection {1e4 * v['selection']:.2f}, "
          f"interaction {1e4 * v['interaction']:.2f} basis points.")
def the(ests):
    names = [f"the {NAMES[e]}" for e in ests]
    return (", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]) + (" estimators" if len(names) > 1 else " estimator")
v = e8["second_clause"]
parts = ([f"holds for {the(v['holds_for'])}"] if v["holds_for"] else []) + ([f"fails for {the(v['fails_for'])}"] if v["fails_for"] else [])
print(f"E8, second clause: {'met' if v['met'] else 'not met'}. It " + " and ".join(parts) + ".")
print(f"E8: {'met' if e8['met'] else 'not met'}.")
"""),
    md("""
## 4. The attribution by factor and E9

The source repository's factor attribution, copied unchanged in `src/slt/source/attribution.py`, on the same active returns.
"""),
    code("""
fsum = {p[0]: SA.summarise(res["factor"][p]) for p in C.ATTRIBUTION_PATHS}
for est, s in fsum.items():
    print(f"{NAMES[est]}: exposures " + ", ".join(f"{n} {s['exposure ' + n]:+.3f}" for n in C.FACTOR_ATTRIBUTION_FACTORS)
          + f" | factor total {1e4 * s['factor total (per year)']:.1f} ({1e4 * s['factor total (standard error)']:.1f}) and residual "
          f"{1e4 * s['residual (per year)']:.1f} ({1e4 * s['residual (standard error)']:.1f}) basis points a year | factor share {s['factor share']:.2f}")
e9 = A.judge_e9(fsum)
print(f"E9, identity: {'met' if e9['identity']['met'] else 'not met'}, largest error {e9['identity']['largest_error']:.1e} against 1e-9.")
v = e9["market_exposure"]
parts = ([f"holds for {the(v['holds_for'])}"] if v["holds_for"] else []) + ([f"fails for {the(v['fails_for'])}"] if v["fails_for"] else [])
print("E9, market exposure: " + ", ".join(f"{NAMES[e]} {x:+.4f}" for e, x in v["by_estimator"].items()) + " against 0.05 either side of zero. It " + " and ".join(parts) + ".")
print(f"E9: {'met' if e9['met'] else 'not met'}.")
"""),
    md("""
## 5. The tilt's cap

The tilt caps every Energy and Utilities stock at half its benchmark weight, and whether the cap binds is a result. For each path this section counts the stock-months of the two sectors among the month's members with a market capitalisation, and how many of them sit at half the benchmark weight within one millionth, below it and above it, from the recorded weights. The benchmark weight here is the capitalisation share among those members. A sector whose stocks all sit at the same fraction of their benchmark weights earns the benchmark sector's return, so the count says why selection in the two sectors is close to zero and how close.
"""),
    code("""
cap = A.tilt_cap(inp, OUT, months)
print(cap.to_string(index=False))
"""),
    md("""
## 6. Two readings added on 19 November 2026

Added on 19 November 2026, after every recorded run, with no verdict: E8 and E9 keep the verdicts of sections 3 and 4. First, the mean absolute monthly allocation three ways for each path, with the month's allocation summed over a group of sectors before its absolute value is taken: all twelve sectors, which is E8's figure, the two tilted sectors, Energy and Utilities, and the other ten. Three figures per path and nothing by sector. Second, each factor's mean contribution, the six-factor total and the residual with Newey-West standard errors at 12 lags beside the plain standard errors of section 4, and the count of the 24 factor contributions that are statistically significant at about the 5 per cent level, more than two Newey-West standard errors from zero, against the count that chance alone would give at the 5 per cent level.
"""),
    code("""
labels = res["labels"]
groups = pd.DataFrame([{"path": labels[p], **B.allocation_by_group(res["terms"][p])} for p in C.ATTRIBUTION_PATHS])
for p in C.ATTRIBUTION_PATHS:
    assert abs(groups.set_index("path").loc[labels[p], "all"] - summ[p[0]]["total"]["allocation_mean_abs_monthly"]) < 1e-14
print((1e4 * groups.set_index("path")).round(2).to_string())
nw = pd.concat([A.factor_newey_west(res["factor"][p], Q.newey_west_se).assign(path=labels[p]) for p in C.ATTRIBUTION_PATHS], ignore_index=True)
nw = nw[["path"] + [c for c in nw.columns if c != "path"]]
show = nw.assign(**{c: 1e4 * nw[c] for c in ("per_year", "standard_error", "newey_west_se")})
print(show.round(2).to_string(index=False))
print(json.dumps(A.factor_beyond_count(nw)))
"""),
    md("""
## 7. Allocation against selection by month, added on 24 November 2026

Added on 24 November 2026, after the results were known, with no verdict: E8 keeps the verdict of section 3. Each path's monthly allocation and selection, each summed over the twelve sectors as E8 reads them, are committed as a table by month, a cell left empty where the path's active return is in notebook 09's table. From that table, for each path, the mean absolute monthly allocation less the mean absolute monthly selection, with a 95 per cent interval from a circular block bootstrap of the months, in blocks of 12, with 2,000 resamples, which keeps each month's two terms together, as section 5 of notebook 09 resamples. An interval that lies wholly above zero says that the difference is statistically significant at the 5 per cent level.
"""),
    code("""
active_committed = pd.read_csv(ROOT / C.RESULTS_DIR / "active_returns_monthly.csv")
monthly, monthly_empty = B.monthly_table(res["totals"], labels, active_committed)
for p in C.ATTRIBUTION_PATHS:
    for t in C.ALLOCATION_SELECTION_TERMS:
        assert abs(monthly[f"{t}_{labels[p]}"].abs().mean() - summ[p[0]]["total"][f"{t}_mean_abs_monthly"]) < 1e-14
print("empty cells by column:", monthly_empty)
difference = B.mean_abs_difference_table(monthly, labels)
show = difference.assign(**{c: 1e4 * difference[c] for c in ("allocation_mean_abs", "selection_mean_abs", "difference", "interval_low", "interval_high")})
print(show[["path", "months", "allocation_mean_abs", "selection_mean_abs", "difference", "interval_low", "interval_high"]].round(2).to_string(index=False))
for _, r in difference.iterrows():
    above = r["interval_low"] > 0
    print(f"{r['path']}: the difference is {'' if above else 'not '}statistically significant at the 5 per cent level, interval "
          f"{1e4 * r['interval_low']:+.2f} to {1e4 * r['interval_high']:+.2f} basis points a month")
"""),
    md("""
## 8. What is committed

Eleven files go into `results/20260928/`, and the record of French's files is brought up to date. All are by year or over the window except one, added on 24 November 2026 with the interval of section 7: each path's allocation and selection by month, each summed over the twelve sectors, a cell left empty where the path's active return is. No weight of any sector and no figure of a sector by month. The two of section 6 were added on 19 November 2026.
"""),
    code("""
labels = res["labels"]
by_year = pd.concat([B.by_year(res["terms"][p]).assign(path=labels[p]) for p in C.ATTRIBUTION_PATHS], ignore_index=True)
by_year = by_year[["path", "year", "sector", "months"] + list(C.ATTRIBUTION_TERMS)]
counts = pd.concat({labels[p]: res["totals"][p][["members_without_return", "sectors_not_held"]] for p in C.ATTRIBUTION_PATHS}, axis=1)
counts.index = pd.PeriodIndex(counts.index, freq="M").year
counts = counts.groupby(level=0).sum()
counts_by_year = pd.DataFrame({"year": counts.index, "name_months_without_return": counts[labels[C.ATTRIBUTION_PATHS[0]]]["members_without_return"].to_numpy()})
for p in C.ATTRIBUTION_PATHS:
    counts_by_year[f"sector_months_not_held_{labels[p]}"] = counts[labels[p]]["sectors_not_held"].to_numpy()
factor_by_year = pd.concat([A.factor_by_year(res["factor"][p]).assign(path=labels[p]) for p in C.ATTRIBUTION_PATHS], ignore_index=True)
factor_by_year = factor_by_year[["path"] + [c for c in factor_by_year.columns if c != "path"]]
brinson_summary = V.jsonable({"written_by": "notebooks/11_attribution.ipynb", "manifest_sha256": C.MANIFEST_SHA256,
                              "weights": C.ATTRIBUTION_WEIGHTS, "linking": C.ATTRIBUTION_LINKING,
                              "paths": {labels[p]: summ[p[0]] for p in C.ATTRIBUTION_PATHS}})
factor_summary = V.jsonable({"written_by": "notebooks/11_attribution.ipynb", "manifest_sha256": C.MANIFEST_SHA256,
                             "loadings": "the factor estimator's of the month, every path", "factor_returns": "French's monthly files",
                             "paths": {labels[p]: fsum[p[0]] for p in C.ATTRIBUTION_PATHS}})
verdicts = V.jsonable({"E8": e8, "E9": e9})
for p_ in [X.write_results("brinson_by_year.csv", frame=by_year),
           X.write_results("brinson_summary.json", record=brinson_summary),
           X.write_results("attribution_counts_by_year.csv", frame=counts_by_year),
           X.write_results("factor_attribution_by_year.csv", frame=factor_by_year),
           X.write_results("factor_attribution_summary.json", record=factor_summary),
           X.write_results("verdicts_milestone4.json", record=verdicts),
           X.write_results("tilt_cap.csv", frame=cap),
           X.write_results("allocation_by_group.csv", frame=groups),
           X.write_results("factor_attribution_newey_west.csv", frame=nw),
           X.write_results("allocation_selection_monthly.csv", frame=monthly),
           X.write_results("allocation_selection_difference.csv", frame=difference),
           FR.write_provenance(french_records)]:
    print("written:", p_.relative_to(ROOT))
"""),
    md("""
## What was learned, and what could be wrong

What the notebook establishes: how much of the mandate's active return came from holding sectors at other weights than the benchmark's, how much from the stocks held within them, and how much the six factors explain, on each of the four covariance matrices.

What could still be wrong. The terms are summed over months arithmetically, with no geometric linking, so a year's terms add up to the sum of its monthly active returns, which differs from the active return compounded over the year. Allocation, selection and interaction depend on the order of the definitions, and another convention moves part of the interaction into one of the other two. The factor attribution reads every path through one set of loadings estimated on daily returns, applied to monthly factor returns, and a loading measured with error moves part of a contribution into the residual. And Unclassified, one to three securities a month, is a sector of its own here only so that every identity holds.
"""),
]


# ---------------------------------------------------------------------------
# 12: the figures
# ---------------------------------------------------------------------------

SETUP_12 = """
import hashlib
import io
import pathlib
import sys

ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
from IPython.display import SVG, display

from slt import constants as C
from slt import figures as F

RES = F.results_dir(ROOT)
FIG = ROOT / "dashboard" / "figures"
print("results:", RES.relative_to(ROOT), "| matplotlib", matplotlib.__version__)


def show(fig):
    # shows a figure as the SVG file it is written to
    buf = io.BytesIO()
    F.save(fig, buf)
    display(SVG(buf.getvalue()))
"""

NB12 = [
    md("""
# 12. The figures

**What this notebook does.** It plots the four figures that the dashboard, the README and the note show, from the tables committed in `results/20260928/` and from nothing else, and writes them to `dashboard/figures/`. It needs no extraction and runs on any machine with a clone of the repository.

**Why it comes here.** Notebooks 09 to 11 wrote every figure the study reports. This notebook only plots four of them, so a reader can check each picture against a table they can open.

**Terms used in this notebook.**

- *market capitalisation*: price times shares outstanding, at the end of the month before.
- *benchmark*: the universe of notebook 02, each stock weighted by its market capitalisation at the start of the month.
- *active weight*: a stock's weight in the portfolio minus its weight in the benchmark. The *active return* is the portfolio's return minus the benchmark's.
- *basis point*: a hundredth of a percentage point. The *tracking error* is the standard deviation of the monthly active return, times the square root of 12, here in basis points a year.
- *factor*: a common source of return, here one of French's five factors or his momentum factor.
- *covariance matrix*: the table of the stocks' variances and of their covariances with each other. A *covariance estimator* is one way of estimating it from three years of daily returns: the sample covariance, the Ledoit-Wolf estimator, the factor estimator, built on the six factors, and the principal-component estimator.
- *forecast tracking error*: the tracking error that the month's covariance matrix implies for the month's active weights. The *bias statistic* is the standard deviation of each month's active return divided by that month's forecast, one when the forecasts are right on average.
- *turnover*: the month's one-way trading as a share of the portfolio. *Chosen turnover* is the part the optimiser chooses and a cap limits, as notebook 10 measured it.
- *mandate*: constraint set C3 of notebook 08. Its *tilt*: every Energy and Utilities stock capped at half its benchmark weight. A *path* is one covariance estimator under one constraint set at one cap on chosen turnover, run month by month from July 1999 to December 2024.
- *attribution*: a split of the active return into parts that sum to it. *Allocation* and *selection* are two of the three parts of the attribution by sector of notebook 11, from holding sectors at other weights than the benchmark's and from holding other stocks within a sector.
- *Newey-West standard error*: the standard error of a mean that allows each month to be correlated with the next twelve.

**What I expect to see.** This notebook judges nothing: every expectation was judged in notebooks 01 to 11. It reproduces committed figures, and each section prints the table behind its figure.

Running time: under a minute.
"""),
    code(SETUP_12),
    md("""
## 1. The inputs

The four committed files the figures read, with the notebook that wrote each and the first twelve characters of its SHA-256.
"""),
    code("""
for name in ["paths_summary.json", "paths_by_year.csv", "brinson_summary.json"]:
    data = (RES / name).read_bytes()
    print(f"{name:24s} {hashlib.sha256(data).hexdigest()[:12]}")
"""),
    md("""
## 2. Figure 1: forecast and realised tracking error under the mandate

For each of the four covariance estimators under the mandate at the 2 per cent cap: the mean forecast tracking error, the realised tracking error and the bias statistic, over the 306 months.
"""),
    code("""
t1 = F.tracking_error_by_estimator(RES)
print(t1.round(2).to_string(index=False))
show(F.draw_tracking_error(t1))
"""),
    md("""
## 3. Figure 2: tracking error by year, the Ledoit-Wolf path

The realised tracking error of each calendar year and the mean of that year's forecasts, on the Ledoit-Wolf estimator under the mandate at the 2 per cent cap. The year 1999 holds six months.
"""),
    code("""
t2 = F.tracking_error_by_year(RES)
print(t2.round(1).to_string(index=False))
show(F.draw_by_year(t2))
"""),
    md("""
## 4. Figure 3: tracking error against chosen turnover at five caps

The Ledoit-Wolf and factor estimators under the mandate at caps of 0.5, 1, 2 and 4 per cent a month and with no cap: chosen turnover in per cent a year, and the realised and mean forecast tracking error in basis points a year. Where two caps give almost the same turnover, one label names both.
"""),
    code("""
t3 = F.frontier(RES)
print(t3.round(1).to_string(index=False))
show(F.draw_frontier(t3))
"""),
    md("""
## 5. Figure 4: allocation and selection by sector, the Ledoit-Wolf path

Each sector's allocation and selection over the window, as annual averages, each with a bar of two Newey-West standard errors on each side. Interaction is printed in the table and not drawn. In Energy and Utilities selection is close to zero because the tilt's cap at half the benchmark weight binds on almost every stock there, as the README says with its count. The last line counts, on each path, the sector terms that are statistically significant at about the 5 per cent level, two standard errors or more from zero, out of 36.
"""),
    code("""
t4 = F.sector_terms(RES)
print(t4.round(2).to_string(index=False))
print("terms two standard errors or more from zero:", F.terms_beyond_two_se(RES))
show(F.draw_sector_terms(t4))
"""),
    md("""
## 6. The files

The four figures are written as SVG files with their text drawn as outlines, so they look the same on every machine. Each is drawn a second time from the same tables and compared byte for byte with the first, so that a rebuild from the committed tables gives the same file.
"""),
    code("""
FIG.mkdir(parents=True, exist_ok=True)
drawings = {"fig1_tracking_error.svg": (F.draw_tracking_error, t1), "fig2_tracking_error_by_year.svg": (F.draw_by_year, t2),
            "fig3_frontier.svg": (F.draw_frontier, t3), "fig4_sector_terms.svg": (F.draw_sector_terms, t4)}
for name, (draw, table) in drawings.items():
    first, second = io.BytesIO(), io.BytesIO()
    F.save(draw(table), first)
    F.save(draw(table), second)
    assert first.getvalue() == second.getvalue(), name
    (FIG / name).write_bytes(first.getvalue())
    print("written:", (FIG / name).relative_to(ROOT), len(first.getvalue()), "bytes")
matplotlib.pyplot.close("all")
"""),
    md("""
## What was learned, and what could be wrong

What the notebook establishes: the four figures are drawn from the committed tables alone, and a second drawing from the same tables gives the same bytes.

What could still be wrong. The figures inherit every limit of the tables they draw, which notebooks 09 to 11 set out. A figure shows four of the many figures the study reports and leaves out the rest, which the README and the dashboard's tables give. And the same tables drawn with another version of matplotlib can give a file that looks the same with other bytes, so the version is printed at the top.
"""),
]


# ---------------------------------------------------------------------------
# 13: the paired tests
# ---------------------------------------------------------------------------

SETUP_13 = """
import pathlib
import sys
import time

ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from slt import constants as C
from slt import extraction as X
from slt import pairs as PR
from slt.source import paired as PT

pd.set_option("display.width", 200)
S = PR.settings()
print(f"set on {C.PAIRED_TEST_DATE}: blocks of {S['block']} months, {S['draws']:,} draws, level {S['level']}, "
      f"seed {C.PAIRED_SEED}, power {C.PAIRED_DETECTABLE_POWER} for the detectable difference")
"""

NB13 = [
    md("""
# 13. The paired tests: two realised tracking errors on the same months

**What this notebook does.** It compares, two at a time, figures that notebook 09 measured on the same 306 months, with a test built for figures that share their months, and writes four tables. It reads only the tables committed in `results/20260928/`, needs no extraction and runs on any machine with a clone of the repository. It was added on 18 November 2026, after every recorded run of the study, and changes no result that notebooks 01 to 11 judged.

**Why it comes here.** Two paths run on the same months move together month by month, so the difference between their figures is far less noisy than either figure, and a test on the paired months measures that difference against its own uncertainty.

**Terms used in this notebook.**

- *market capitalisation*: price times shares outstanding, at the end of the month before.
- *benchmark*: the universe of notebook 02, each stock weighted by its market capitalisation at the start of the month.
- *active return*: the portfolio's return minus the benchmark's in the same month.
- *basis point*: a hundredth of a percentage point.
- *tracking error*: the standard deviation of the monthly active return, times the square root of 12, here in basis points a year. The realised tracking error is the one measured over the months a path has run.
- *covariance matrix*: the table of the stocks' variances and of their covariances with each other.
- *factor*: a common source of return, here one of French's five factors or his momentum factor.
- *covariance estimator*: one way of estimating the covariance matrix from three years of daily returns: the sample covariance, the Ledoit-Wolf estimator, the factor estimator, built on the six factors, and the principal-component estimator.
- *turnover*: the month's one-way trading as a share of the portfolio. Chosen turnover is the part the optimiser chooses, which a cap limits.
- *path*: one covariance estimator under one constraint set at one cap on chosen turnover, run month by month from July 1999 to December 2024. The *mandate* is constraint set C3 of notebook 08.
- *expectation*: a test of a result fixed in advance. Its *verdict* is met or not met.
- *standard error*: the standard deviation an estimate would have across samples of the same length, its uncertainty.
- *paired test*: a test of the difference between two figures measured on the same months, which uses what the two series share month by month. Here it is the variance test of Ledoit and Wolf (2011), copied unchanged from the repository of Optimal versus naive diversification as `src/slt/source/paired.py`.
- *circular block bootstrap*: a way of measuring the uncertainty of a figure from new samples of the same length taken from the months, in blocks of 12 consecutive months, the last month followed by the first, so that the correlation of nearby months is kept. The *seed* is the number that starts the random choice of blocks, so that a rerun gives the same samples.
- *p-value*: from the bootstrap, the probability of a difference at least as far from zero as the one observed if the two variances were equal.
- *95 per cent interval*: the range around the difference of the two tracking errors that the bootstrap gives at the 95 per cent level. A difference whose interval excludes zero is statistically significant at that level.
- *critical value*: the bootstrap's 95th percentile of the difference measured in its own standard errors. The interval is the difference plus and minus the critical value times the standard error.
- *detectable difference*: the difference that the test would find statistically significant at that level about four times in five, the critical value plus 0.84, times the standard error.
- *Holm's correction*: an adjustment of the p-values of a family of comparisons for their number. The smallest p-value is multiplied by the family's size, the next by one less, and each adjusted value is at least the one before it (Holm, 1979).
- *size*: the share of pairs with equal variances that a test rejects at the 5 per cent level, which should be close to 5 per cent.
- *Student-t*: a distribution with fatter tails than the normal one. Its *degrees of freedom* are the parameter that sets how fat, and with 5 the tails are fat.

**What I expect to see, and where it comes from.** The settings were set on 18 November 2026, after every recorded run, and are in `src/slt/constants.py` under `PAIRED_`: blocks of 12 months, 2,000 samples, the level 0.95, the seed 20261009 of Optimal versus naive diversification, and a power of 0.80 for the detectable difference.

- *P1*, the one expectation. In notebook 13 the size check runs before the seven comparisons: on 1,000 simulated pairs of 306 months with equal variances, Student-t shocks with 5 degrees of freedom and a monthly standard deviation of 0.23 per cent, about 80 basis points a year, at a correlation of 0.99 and again at 0.80, the share rejected at 5 per cent is at most 6.5 per cent in each. Below 3.5 per cent at both settings, the text says that the test rejects less often than its level, that an interval that excludes zero stands, and that an interval that includes zero is weaker evidence of no difference than a test of exact size would give.
- *Family A*, with no verdict: the sample, factor and principal-component estimators each against Ledoit-Wolf under the mandate at the 2 per cent cap, with Holm's correction across the three. Its first row bears on the second clause of E4.
- *Family B*, with no verdict: on the Ledoit-Wolf and the factor estimators, the 4 per cent cap and no cap each against the 0.5 per cent cap, with no correction. Its rows bear on the second part of E5.
- *The factor estimator without two months*, added on 19 November 2026, with no verdict: the two rows of family B on the factor estimator again, without July and December 2008, beside the share of the difference between the two variances that those two months carry.
- *The other block lengths*: the seven comparisons again at blocks of 3, 6 and 24 months, with the same seed and 2,000 samples, and Holm's correction across family A at each length. No order of commits is claimed for these lengths. The block is fixed at 12 months, where Ledoit and Wolf (2008, Algorithm 3.1) choose its length from the data. At blocks of 3, 6 and 24 months, added on 19 November 2026, after the study's results were known, with no verdict, the sample estimator's difference from Ledoit-Wolf has intervals of 0.81 to 2.57, 0.84 to 2.55 and 0.61 to 2.77, with Holm p-values of 0.003, 0.0015 and 0.0495, and none of the other six differences is statistically significant at the 5 per cent level at any length, where they could detect differences of 7.1 to 21.1 basis points a year four times in five. The sample estimator's difference is therefore statistically significant at the 5 per cent level at all four lengths, tested on its own. At 24 months its Holm p-value is 0.0495, and 2,000 samples cannot place it on either side of 0.05.

E4 and E5 keep the verdicts of notebook 09, which compare two point estimates as the expectations were fixed before the run. Running time: about two minutes.
"""),
    code(SETUP_13),
    md("""
## 1. The active returns as committed

The table notebook 09 committed, one column for each of the twelve paths under the mandate, read from the repository with its SHA-256, so that this run can be tied to the commit. Every pair tested has a value in every month.
"""),
    code("""
active, sha = PR.read_active(ROOT)
print(f"{PR.ACTIVE_FILE}: {len(active)} months, {active.index[0]} to {active.index[-1]}, {active.shape[1]} paths, SHA-256 {sha}")
assert len(active) == C.EVAL_MONTHS and list(active.columns) == list(C.MONTHLY_ACTIVE_COLUMNS)
pairs = list(C.PAIRED_FAMILY_A) + list(C.PAIRED_FAMILY_B)
for a, b in pairs:
    assert active["active_return_" + C.PATH_LABELS[a]].notna().all() and active["active_return_" + C.PATH_LABELS[b]].notna().all()
print(f"{len(pairs)} comparisons, every month present in each")
"""),
    md("""
## 2. P1, the size of the test, before the seven comparisons

One thousand pairs of 306 months are simulated with equal variances at each of the two correlations, and the test is run on each pair with the settings above. The share of pairs rejected at 5 per cent by the p-value is the verdict, and the share of intervals that exclude zero is reported beside it.
"""),
    code("""
t0 = time.time()
size = PR.size_table()
print(size.T.to_string(header=False))
p1 = (size["P1"] == "met").all()
for _, r in size.iterrows():
    note = ", below 3.5 per cent: the test rejects less often than its level" if r["share rejected"] < C.PAIRED_SIZE_CONSERVATIVE_BELOW else ""
    print(f"correlation {r['correlation']}: {r['rejected by p-value']} of {r['pairs']:,} pairs rejected at 5 per cent, "
          f"{100 * r['share rejected']:.1f} per cent{note} | {100 * r['share of intervals excluding zero']:.1f} per cent of the intervals exclude zero")
print(f"P1: {'met' if p1 else 'not met'}, the share at most {100 * C.PAIRED_SIZE_MAX:.1f} per cent in each case | {time.time() - t0:.0f} s")
"""),
    md("""
## 3. Families A and B, from the committed active returns

Each difference is the first path's realised tracking error minus the second's, in basis points a year. The seed is the same for every comparison, so a rerun gives the same table.
"""),
    code("""
table = PR.paired_table(active)
show = ["comparison", "label", "sd 1", "sd 2", "difference", "standard error", "critical value", "lower", "upper",
        "detectable difference", "correlation", "p-value", "Holm p-value"]
print(table[show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
for _, r in table.iterrows():
    holm = "" if r["comparison"] == "B" else f", {r['Holm p-value']:.3f} with Holm's correction"
    print(f"{r['label']}: {r['difference']:+.2f} bp a year, interval {r['lower']:+.2f} to {r['upper']:+.2f}, p {r['p-value']:.3f}{holm}, "
          f"detectable {r['detectable difference']:.2f}")
"""),
    md("""
## 4. The factor estimator's two rows without July and December 2008

The critical values of the factor estimator's two rows in family B are far above those of the other five. This section shows where that comes from. For each of the two rows it measures the share of the difference between the two variances that July and December 2008 carry, each path's months measured from its own mean over all 306 months, and runs the test again on the other 304 months with the same settings and seed.
"""),
    code("""
left_out = PR.left_out_table(active)
share = "share of the difference between the two variances in those months"
for _, r in left_out.iterrows():
    print(f"{r['label']}: {100 * r[share]:.1f} per cent of the difference between the two variances in {r['months left out']}")
    print(f"  without them, {r['months']} months: {r['difference']:+.2f} bp a year, standard error {r['standard error']:.2f}, "
          f"critical value {r['critical value']:.2f}, interval {r['lower']:+.2f} to {r['upper']:+.2f}, p {r['p-value']:.3f}")
"""),
    md("""
## 5. The seven comparisons at blocks of 3, 6 and 24 months, added on 19 November 2026

The block length is fixed at 12 months. Ledoit and Wolf (2008, Algorithm 3.1) choose it from the data instead, by simulation. This section, added on 19 November 2026, after the study's results were known, with no verdict, runs the seven comparisons again at blocks of 3, 6 and 24 months, with the same seed and 2,000 samples, and Holm's correction across family A at each length. With 2,000 samples a p-value moves in steps of one in 2,001, and the error of the bootstrap's own sampling is printed beside each Holm value of family A.
"""),
    code("""
other = PR.other_blocks_table(active)
print(other[["block"] + show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
for _, r in other[other["comparison"] == "A"].iterrows():
    n = int(r["draws"])
    se = r["Holm p-value"] / r["p-value"] * (r["p-value"] * (1 - r["p-value"]) / n) ** 0.5
    print(f"blocks of {r['block']}: {r['label']}: interval {r['lower']:+.2f} to {r['upper']:+.2f}, "
          f"Holm p {r['Holm p-value']:.4f}, bootstrap error of that value about {se:.4f}")
excl = ((other["lower"] > 0) | (other["upper"] < 0)).groupby(other["block"]).sum()
print("intervals that exclude zero, by block length:", {int(k): int(v) for k, v in excl.items()})
"""),
    md("""
## 6. What is committed

Four tables go into `results/20260928/`: `paired_tests.csv`, the seven comparisons with every column of the test, `paired_size_check.csv`, P1 at its two correlations, `paired_tests_without_two_months.csv`, the two rows of section 4, and, added on 19 November 2026, `paired_tests_other_blocks.csv`, the seven comparisons at each of the three other block lengths of section 5. None is by month, and no table of an earlier notebook changes.
"""),
    code("""
for p_ in [X.write_results("paired_tests.csv", frame=table), X.write_results("paired_size_check.csv", frame=size),
           X.write_results("paired_tests_without_two_months.csv", frame=left_out),
           X.write_results("paired_tests_other_blocks.csv", frame=other)]:
    print("written:", p_.relative_to(ROOT))
print(f"active returns tested: SHA-256 {sha}")
"""),
    md("""
## What was learned, and what could be wrong

What the notebook establishes: the size of the test on series of the study's length, tails and correlation, and, for each pair of paths the write-up compares, the difference of their realised tracking errors with an interval measured on their shared months.

What could still be wrong. The interval is the bootstrap's, and on 306 months a block of 12 months is one of 25, so months of large active returns weigh on the critical value. On the factor estimator two months of 2008, July and December, carry 97 and 99 per cent of the difference between the two variances in the two rows of family B, whose critical values are about 5.9 against 2.2 to 2.4 on the other five, and section 4 shows that without those months the critical value is 2.1 and the change about zero. The test compares variances and says nothing about the mean active return. And the seven comparisons were chosen after the results were known, from the sentences of the write-up that compared two tracking errors, so they are a check of those sentences and not a search among all the pairs.
"""),
]


if __name__ == "__main__":
    for name, cells in (("01_extraction.ipynb", NB01), ("02_universe.ipynb", NB02), ("03_returns.ipynb", NB03),
                        ("04_reconstruction.ipynb", NB04), ("05_sectors.ipynb", NB05), ("06_answer_key.ipynb", NB06),
                        ("07_covariance.ipynb", NB07), ("08_mandate.ipynb", NB08), ("09_rolling_evaluation.ipynb", NB09),
                        ("10_turnover.ipynb", NB10), ("11_attribution.ipynb", NB11),
                        ("12_figures.ipynb", NB12), ("13_paired_tests.ipynb", NB13)):
        print("wrote", write(name, cells).relative_to(ROOT))

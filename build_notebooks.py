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


if __name__ == "__main__":
    for name, cells in (("01_extraction.ipynb", NB01),):
        print("wrote", write(name, cells).relative_to(ROOT))

"""The reconstruction test: could any licensed table be rebuilt, in whole or in part, from what the
repository holds?

Every committed result under results/ is scanned. A file fails when it names a security (an identifier
column), when it carries a series of CRSP's index file or a column that could hold one, when a table by
month carries any column outside the list of what the study computes itself (constants.COMMITTED_MONTHLY),
or when it has more rows than any aggregate of the design needs. Where the data root is present, a
column by month also fails if its values equal a field of CRSP's index file, to the sixth decimal, in more
than 5 per cent of months, because a column can pass by its name and fail by its values. The test runs
before every push with the other tests; it complements the question answered in each commit message and
does not replace it.
"""
import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from slt import constants as C  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Columns that identify a security or a company. A committed table never has one.
IDENTIFIER_COLUMNS = {"permno", "permco", "lpermno", "gvkey", "cusip", "ncusip", "ticker", "tsymbol",
                      "comnam", "conm", "iid", "issuno", "isin", "sedol", "cik"}
# The fields of CRSP's index file. Only summary figures of the comparison with them are committed.
CRSP_INDEX_FIELDS = {"vwretd", "vwretx", "ewretd", "ewretx", "sprtrn", "spindx", "totval", "totcnt",
                     "usdval", "usdcnt"}
# A table by month may hold only what the study computes itself, under these column names, and the
# benchmark's weight in each of the twelve sectors (constants.COMMITTED_MONTHLY).
MONTH_COLUMN = "month"
# Amended 11 November 2026: the eleven GICS sector weights alone (constants.COMMITTED_MONTHLY).
MONTHLY_ALLOWED = {MONTH_COLUMN} | {"weight_" + code for code in C.GICS_SECTORS}
# Twelve sectors over CRSP's 1,189 index months is 14,268 rows in long form, the largest aggregate the
# design names; a table of one row per security and month would run to hundreds of thousands.
ROW_LIMIT = 15000


def committed_results():
    folder = ROOT / "results"
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file())


def header_and_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        return [h.strip().lower() for h in header], sum(1 for _ in reader)


def json_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k).lower()
            yield from json_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from json_keys(v)


def test_no_data_file_type_is_committed():
    allowed = {".csv", ".json", ".md"}
    bad = [p.name for p in committed_results() if p.suffix.lower() not in allowed]
    assert not bad, f"result files of a type that is not reviewed here: {bad}"


def test_no_table_names_a_security_or_carries_a_crsp_index_series():
    for path in committed_results():
        if path.suffix.lower() == ".csv":
            header, _ = header_and_rows(path)
            names = set(header)
        elif path.suffix.lower() == ".json":
            names = set(json_keys(json.loads(path.read_text())))
        else:
            continue
        assert not names & IDENTIFIER_COLUMNS, f"{path.name} names a security: {sorted(names & IDENTIFIER_COLUMNS)}"
        assert not names & CRSP_INDEX_FIELDS, f"{path.name} carries a CRSP index field: {sorted(names & CRSP_INDEX_FIELDS)}"


def test_tables_by_month_hold_only_what_the_study_computes():
    for path in committed_results():
        if path.suffix.lower() != ".csv":
            continue
        header, _ = header_and_rows(path)
        if MONTH_COLUMN in header:
            extra = set(header) - MONTHLY_ALLOWED
            assert not extra, f"{path.name} holds columns by month outside the committed list: {sorted(extra)}"


def test_no_table_is_larger_than_an_aggregate():
    for path in committed_results():
        if path.suffix.lower() == ".csv":
            _, rows = header_and_rows(path)
            assert rows <= ROW_LIMIT, f"{path.name} has {rows} rows, above {ROW_LIMIT}"


def crsp_index_fields():
    """CRSP's monthly index file by month, every numeric field, where the data root is present; else None."""
    import os
    if not os.environ.get(C.DATA_ROOT_ENV, "").strip():
        return None
    import pandas as pd
    import pyarrow.parquet as pq
    from slt import extraction as X
    root = X.data_root()
    path = sorted((root / "raw" / "crsp_msp500").glob("*.parquet"))[0]
    names = [n for n in pq.read_schema(path).names if n.lower() in CRSP_INDEX_FIELDS or n == "caldt"]
    frame, _ = X.read_table(root, "crsp_msp500", columns=names)
    frame.index = pd.PeriodIndex(frame.pop("caldt"), freq="M")
    return frame


def test_no_column_by_month_equals_a_field_of_crsp_index_file():
    import numpy as np
    import pandas as pd
    import pytest
    crsp = crsp_index_fields()
    if crsp is None:
        pytest.skip("the data root is not set on this machine, so the values cannot be compared")
    for path in committed_results():
        if path.suffix.lower() != ".csv":
            continue
        frame = pd.read_csv(path)
        if MONTH_COLUMN not in frame.columns:
            continue
        frame.index = pd.PeriodIndex(frame.pop(MONTH_COLUMN), freq="M")
        for column in frame.columns:
            ours = pd.to_numeric(frame[column], errors="coerce")
            for lag in (0, 1):   # the same month, and CRSP's row for the month before, the same day as the start of the month
                theirs = crsp.reindex(ours.index - lag)
                theirs.index = ours.index
                for field in theirs.columns:
                    both = pd.concat([ours, theirs[field]], axis=1).dropna()
                    if both.empty:
                        continue
                    a, b = both.iloc[:, 0].to_numpy(float), both.iloc[:, 1].to_numpy(float)
                    equal = (np.round(a, C.RECONSTRUCTION_EQUAL_DECIMALS) == np.round(b, C.RECONSTRUCTION_EQUAL_DECIMALS))
                    equal |= np.abs(a - b) <= 10.0 ** -C.RECONSTRUCTION_EQUAL_DECIMALS * np.maximum(np.abs(b), 1.0)
                    equal &= a != 0      # a zero, such as an empty sector, says nothing of CRSP's values
                    share = equal.mean()
                    assert share <= C.RECONSTRUCTION_EQUAL_MONTHS_MAX, (
                        f"{path.name}, column {column}: equal to a field of CRSP's index file in {share:.1%} of months")

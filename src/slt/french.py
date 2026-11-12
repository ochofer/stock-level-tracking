"""
Ken French's data library: download, record, parse.

Adapted from src/bp/french_loader.py of the source repository at commit 5414f14
(constants.SOURCE_ADAPTED_FILES). What differs from the source: the file list is
this study's (constants.FRENCH_FILES); the zips are cached in
constants.FRENCH_CACHE_DIR; rows after CRSP_END are cut, in place of the source's
sample end; one function loads a block and returns its provenance record with it;
three functions are added (spreads, daily_factors, and write_provenance, which keeps
the records of files a notebook did not read). The parsing of the blocks, the
missing codes, the conversion from percent and the provenance record are the
source's, unchanged.

The library serves each dataset as a zip holding one CSV. The CSV is not a single
table. It is a text file with between two and ten blocks, each announced by a title
line ("  Average Value Weighted Returns -- Monthly"), followed by a header row that
starts with a comma, followed by rows that start with a date: eight digits for daily
data (19900102), six for monthly (192607), four for annual (1927). Missing values are
coded -99.99 or -999. Returns are in percent.

This module turns that into pandas DataFrames indexed by a Period, in decimal
returns, with the missing codes replaced by NaN. It also writes a provenance record
for every file it reads: the URL, the download time, the SHA-256 of the zip, and the
CRSP vintage line French prints at the top of each CSV. French revises history, so
the record says which revision a result was computed on.

Nothing here is committed to the repository: the zips live in the cache folder,
which .gitignore excludes.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import constants as C
from . import extraction as X

_DATE_ROW = re.compile(r"^\s*(\d{8}|\d{6}|\d{4})\s*,")   # daily YYYYMMDD, monthly YYYYMM, annual YYYY


def cache_dir() -> Path:
    """The folder that holds the downloaded zips, inside the repository and ignored by git."""
    return X.repo_root() / C.FRENCH_CACHE_DIR


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(key: str, folder: str | Path | None = None, force: bool = False) -> dict:
    """Fetch one French zip by its key in constants.FRENCH_FILES, unless it is already in the cache.

    Returns the provenance record of the file. The record is built from the bytes on disk, so the
    SHA-256 always describes the file actually used.
    """
    from zoneinfo import ZoneInfo
    filename = C.FRENCH_FILES[key]
    url = C.FRENCH_BASE_URL + filename
    folder = cache_dir() if folder is None else Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    if force or not path.exists():
        import requests
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
        fetched_now = True
    else:
        fetched_now = False
    data = path.read_bytes()
    now_utc = datetime.now(timezone.utc)
    return {
        "key": key,
        "filename": filename,
        "url": url,
        "fetched_this_run": fetched_now,
        "download_time_utc": now_utc.isoformat(timespec="seconds"),
        "download_time_local": now_utc.astimezone(ZoneInfo(C.TIMEZONE)).isoformat(timespec="seconds"),
        "zip_sha256": sha256_bytes(data),
        "zip_bytes": len(data),
        "vintage_line": _vintage_line(data),
    }


def _csv_text(zip_bytes: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one CSV in the zip, found {names}")
        return zf.read(names[0]).decode("latin-1")


def _vintage_line(zip_bytes: bytes) -> str:
    """The first line of the CSV, e.g. 'This file was created using the 202609 CRSP database.'"""
    text = _csv_text(zip_bytes)
    for line in text.splitlines():
        if line.strip():
            first = line.strip()
            return first.split(".")[0] + "." if "." in first else first
    return ""


def parse_blocks(text: str) -> list:
    """Split a French CSV into its blocks.

    Each block is returned as {"title": str, "frequency": "daily" | "monthly" | "annual", "frame": DataFrame}
    where the frame is indexed by Period (day, month or year) and holds floats in the file's own units
    (percent). The title is the last non-empty line before the header row that is not itself a data row.
    """
    lines = text.splitlines()
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        if _DATE_ROW.match(lines[i]):
            # Walk back to the header row (starts with a comma) and the title above it.
            header_idx = i - 1
            while header_idx >= 0 and not lines[header_idx].lstrip().startswith(","):
                header_idx -= 1
            if header_idx < 0:
                raise ValueError(f"data row at line {i} has no header row above it")
            title_idx = header_idx - 1
            while title_idx >= 0 and not lines[title_idx].strip():
                title_idx -= 1
            title = lines[title_idx].strip() if title_idx >= 0 else ""
            # The factor files carry no title line, only prose; a sentence is not a title.
            if not title or title.endswith(".") or len(title) > 70:
                title = f"(untitled block {len(blocks)})"
            columns = [c.strip() for c in lines[header_idx].split(",")[1:]]
            rows = []
            j = i
            while j < n and _DATE_ROW.match(lines[j]):
                rows.append(lines[j])
                j += 1
            raw = pd.read_csv(io.StringIO("\n".join(rows)), header=None, index_col=0)
            raw.columns = columns[: raw.shape[1]]
            date_len = len(str(raw.index[0]).strip())
            if date_len == 8:
                idx = pd.PeriodIndex(pd.to_datetime(raw.index.astype(str), format="%Y%m%d"), freq="D")
                freq = "daily"
            elif date_len == 6:
                idx = pd.PeriodIndex([pd.Period(f"{str(d)[:4]}-{str(d)[4:6]}", freq="M") for d in raw.index])
                freq = "monthly"
            else:
                idx = pd.PeriodIndex([pd.Period(str(d), freq="Y") for d in raw.index])
                freq = "annual"
            frame = raw.astype(float)
            frame.index = idx
            frame.index.name = "period"
            blocks.append({"title": title, "frequency": freq, "frame": frame})
            i = j
        else:
            i += 1
    return blocks


def cut_to_sample_end(frame: pd.DataFrame, end: str = C.CRSP_END) -> pd.DataFrame:
    """The frame up to the last date of the extraction: a monthly block to that month, a daily block to that day."""
    if not isinstance(frame.index, pd.PeriodIndex) or len(frame) == 0:
        return frame
    end_d = pd.Period(end, "D")
    freq = frame.index.freqstr or ""
    if freq.startswith("M"):
        limit = end_d.asfreq("M")
    elif freq.startswith("D"):
        limit = end_d
    else:
        limit = end_d.asfreq(frame.index.freq)
    return frame.loc[:limit]


def load_block(key: str, block, folder: str | Path | None = None, as_decimal: bool = True) -> tuple:
    """One block of one file, cut at CRSP_END, missing codes as NaN, in decimal returns unless as_decimal is False.

    block is an integer position among the file's blocks, or a regular expression matched against block
    titles (the first match wins). Returns the frame and the file's provenance record.
    """
    folder = cache_dir() if folder is None else Path(folder)
    record = download(key, folder)
    blocks = parse_blocks(_csv_text((folder / C.FRENCH_FILES[key]).read_bytes()))
    if isinstance(block, int):
        chosen = blocks[block]
    else:
        matches = [b for b in blocks if re.search(block, b["title"], flags=re.I)]
        if not matches:
            raise KeyError(f"no block in {key} matches {block!r}; titles: {[b['title'] for b in blocks]}")
        chosen = matches[0]
    frame = cut_to_sample_end(chosen["frame"].copy())
    for code in C.FRENCH_MISSING_CODES:
        frame = frame.mask(np.isclose(frame, code))
    if as_decimal and C.FRENCH_RETURNS_ARE_PERCENT:
        frame = frame / 100.0
    frame.attrs["title"] = chosen["title"]
    return frame, record


def spreads(folder: str | Path | None = None) -> tuple:
    """French's top-minus-bottom spread within his largest size quintile, one column per sort, monthly, and
    the average number of firms in that quintile by year; with the provenance records of the two files."""
    out, firms, records = {}, {}, []
    for sort, (key, top, bottom) in C.E3_FRENCH_SPREADS.items():
        returns, record = load_block(key, C.FRENCH_BLOCK_RETURNS, folder)
        out[sort] = (returns[top] - returns[bottom]).rename(sort)
        counts, _ = load_block(key, C.FRENCH_BLOCK_FIRMS, folder, as_decimal=False)
        quintile = counts[list(C.E3_FRENCH_LARGEST_QUINTILE[sort])].sum(axis=1)
        firms[sort] = quintile.groupby(quintile.index.year).mean().rename(sort)
        records.append(record)
    return pd.DataFrame(out), pd.DataFrame(firms), records


def daily_factors(folder: str | Path | None = None) -> tuple:
    """The six daily factors of the factor model and the daily risk-free rate, in decimals, indexed by date."""
    f5, r5 = load_block("factors5_daily", 0, folder)
    mom, rm = load_block("momentum_daily", 0, folder)
    mom.columns = [c.strip() for c in mom.columns]
    frame = pd.concat([f5, mom[["Mom"]]], axis=1, join="inner")
    frame.index = frame.index.to_timestamp()
    return frame[list(C.FACTOR_MODEL_FACTORS)], frame["RF"], [r5, rm]


PROVENANCE_NAME = "provenance_french.json"


def write_provenance(records: list) -> Path:
    """The provenance record of French's files, written beside the other results. Records already there for
    other files are kept, so that notebooks 06 and 07 each add theirs; a file's record is replaced by its latest."""
    path = X.repo_root() / C.RESULTS_DIR / PROVENANCE_NAME
    by_key = {}
    if path.exists():
        by_key = {r["key"]: r for r in json.loads(path.read_text()).get("files", [])}
    by_key.update({r["key"]: r for r in records})
    ordered = [by_key[k] for k in sorted(by_key)]
    return X.write_results(PROVENANCE_NAME, record={"source": "Ken French's data library, downloaded at run time and never committed",
                                                    "files": ordered})

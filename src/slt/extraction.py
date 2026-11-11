"""
Reading the extraction in place: the data root, the manifest check, the files used and the provenance record.

The CRSP and Compustat files are licensed and stay on the machine that holds them. This module finds
them through the folder named by the environment variable constants.DATA_ROOT_ENV, checks the
extraction's manifest against the hash fixed in constants.MANIFEST_SHA256 before anything is read,
and reads only the columns a notebook asks for. What it returns for the repository is metadata:
hashes, row and column counts, fetch dates and the queries the manifest records. No row of a licensed
table passes through the functions that write to the repository.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import sys

from . import constants as C

# The columns each notebook reads, by the extraction's key. Notebook 01 checks that every one is in
# the file's schema before any later notebook depends on it.
COLUMNS = {
    "crsp_dsp500list": ("permno", "start", "ending"),
    "crsp_msp500": ("caldt", "vwretd", "vwretx", "sprtrn", "totval", "totcnt"),
    "crsp_msf": ("permno", "date", "prc", "ret", "retx", "shrout"),
    "crsp_msedelist": ("permno", "dlstdt", "dlstcd", "dlret", "dlretx"),
    "crsp_msenames": ("permno", "namedt", "nameendt", "shrcd"),
    "crsp_dsf": ("permno", "date", "prc", "ret"),
    "crsp_ccm_lnkhist": ("gvkey", "lpermno", "linktype", "linkprim", "linkdt", "linkenddt"),
    "comp_idxcst_his": ("gvkey", "gvkeyx", "from", "thru"),
    "comp_co_hgic": ("gvkey", "indtype", "gsector", "ggroup", "gind", "gsubind", "indfrom", "indthru"),
}

# Columns the parquet files hold as text that are dates. They are parsed on reading, and a value that
# is present but does not parse is counted and reported, never dropped without a count.
DATE_COLUMNS = ("start", "ending", "caldt", "date", "dlstdt", "namedt", "nameendt",
                "linkdt", "linkenddt", "from", "thru", "indfrom", "indthru")

MISSING_ROOT = ("The data root is not set: set the environment variable {env} to the folder that holds "
                "the extraction's manifest.json, outside this repository, and start Jupyter again.")
MISSING_MANIFEST = "The folder named by {env} holds no manifest.json: {root}"
WRONG_MANIFEST = ("The manifest on disk has SHA-256 {disk}, and the results are fixed to {fixed}: this is "
                  "not the extraction the constants record, so nothing is read.")
BROKEN_SHA_FILE = "manifest.json does not match its own .sha256 file ({disk} against {recorded})."


def repo_root() -> pathlib.Path:
    """The repository's root folder, wherever the notebook is opened from."""
    return pathlib.Path(__file__).resolve().parents[2]


def data_root(environ=None) -> pathlib.Path:
    """The extraction's folder, from the environment. Stops with one sentence when it is not set."""
    env = os.environ if environ is None else environ
    value = env.get(C.DATA_ROOT_ENV, "").strip()
    if not value:
        raise SystemExit(MISSING_ROOT.format(env=C.DATA_ROOT_ENV))
    root = pathlib.Path(value).expanduser()
    if not (root / C.MANIFEST_FILE).is_file():
        raise SystemExit(MISSING_MANIFEST.format(env=C.DATA_ROOT_ENV, root=root))
    return root


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_manifest(root: pathlib.Path, expected: str = None) -> dict:
    """Read the manifest after checking its hash against the constants and against its own .sha256 file."""
    expected = C.MANIFEST_SHA256 if expected is None else expected
    path = root / C.MANIFEST_FILE
    disk = sha256_of(path)
    sha_file = root / (C.MANIFEST_FILE + ".sha256")
    if sha_file.is_file():
        recorded = sha_file.read_text().split()[0]
        if recorded != disk:
            raise SystemExit(BROKEN_SHA_FILE.format(disk=disk[:12], recorded=recorded[:12]))
    if disk != expected:
        raise SystemExit(WRONG_MANIFEST.format(disk=disk[:12], fixed=expected[:12]))
    return json.loads(path.read_text())


def keys_used() -> list:
    """The extraction's keys the study reads: every table in constants.TABLES except the unused GICS fallback."""
    return [k for k in C.TABLES if k != C.GICS_FALLBACK or C.GICS_SOURCE == C.GICS_FALLBACK]


def amsterdam_date(utc_stamp: str) -> str:
    """The calendar date in Amsterdam of a UTC time stamp as the manifest writes it."""
    from zoneinfo import ZoneInfo
    t = dt.datetime.fromisoformat(utc_stamp)
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(ZoneInfo(C.TIMEZONE)).date().isoformat()


def files_used(manifest: dict, keys=None) -> list:
    """One record per file of the tables used, sorted by path: the manifest's metadata and nothing else."""
    keys = keys_used() if keys is None else list(keys)
    out = []
    for f in manifest["files"]:
        if f["key"] in keys:
            out.append({
                "key": f["key"],
                "table": f["library"] + "." + f["table"],
                "file": f["file"],
                "query": f["sql"],
                "rows": int(f["rows"]),
                "columns": int(f["columns"]),
                "bytes": int(f["bytes"]),
                "sha256": f["sha256"],
                "fetched_utc": f["fetched_utc"],
                "fetch_date": amsterdam_date(f["fetched_utc"]),
            })
    out.sort(key=lambda r: r["file"])
    missing = sorted(set(keys) - {r["key"] for r in out})
    if missing:
        raise SystemExit("The manifest holds no file for: " + ", ".join(missing))
    return out


def fetch_dates(records: list) -> dict:
    """Which tables carry which fetch date, as {date: [keys]}."""
    by_date = {}
    for r in records:
        by_date.setdefault(r["fetch_date"], set()).add(r["key"])
    return {d: sorted(k) for d, k in sorted(by_date.items())}


def verify_files(root: pathlib.Path, records: list) -> list:
    """Each file on disk against the manifest: its SHA-256 and its row count from the parquet footer."""
    import pyarrow.parquet as pq
    out = []
    for r in records:
        path = root / r["file"]
        exists = path.is_file()
        rows = pq.ParquetFile(path).metadata.num_rows if exists else None
        digest = sha256_of(path) if exists else None
        out.append({"file": r["file"], "exists": exists,
                    "sha256_matches": digest == r["sha256"], "rows_match": rows == r["rows"]})
    return out


def missing_columns(root: pathlib.Path, records: list) -> dict:
    """Columns a notebook needs that a file's schema lacks, by file; empty when every one is present."""
    import pyarrow.parquet as pq
    out = {}
    for r in records:
        need = COLUMNS.get(r["key"], ())
        have = set(pq.read_schema(root / r["file"]).names)
        lacking = [c for c in need if c not in have]
        if lacking:
            out[r["file"]] = lacking
    return out


def read_table(root: pathlib.Path, key: str, columns=None, years=None):
    """Read one table's files from the extraction, only the columns asked for, with its dates parsed.

    Returns the frame and a dict of counts of date values that were present and did not parse, by
    column. For the daily file, years limits the yearly files read.
    """
    import pandas as pd
    import pyarrow.parquet as pq
    columns = list(COLUMNS[key] if columns is None else columns)
    folder = root / "raw" / key
    paths = sorted(folder.glob("*.parquet"))
    if years is not None:
        wanted = {str(y) for y in years}
        paths = [p for p in paths if p.stem.rsplit("_", 1)[-1] in wanted]
    if not paths:
        raise SystemExit(f"No file for {key} in {folder}")
    frames = [pq.read_table(p, columns=columns).to_pandas() for p in paths]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return parse_dates(df)


def parse_dates(df):
    """Parse the text date columns; count the values present that do not parse, column by column."""
    import pandas as pd
    unparsed = {}
    for c in df.columns:
        if c in DATE_COLUMNS and df[c].dtype == object:
            parsed = pd.to_datetime(df[c], format="ISO8601", errors="coerce")
            bad = int((df[c].notna() & parsed.isna()).sum())
            if bad:
                unparsed[c] = bad
            df[c] = parsed
    return df, unparsed


def versions() -> dict:
    """The versions of the tools a result was built with."""
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("pandas", "numpy", "pyarrow"):
        mod = sys.modules.get(name)
        if mod is None:
            try:
                mod = __import__(name)
            except ImportError:
                continue
        out[name] = mod.__version__
    return out


def provenance(manifest: dict, records: list, written_by: str) -> dict:
    """The provenance record: what the results were built from, with no row of any licensed table."""
    from zoneinfo import ZoneInfo
    return {
        "source": C.EXTRACTION_SOURCE,
        "extraction_date": C.EXTRACTION_DATE,
        "manifest_sha256": C.MANIFEST_SHA256,
        "manifest_sha256_2026_09_28": C.MANIFEST_SHA256_2026_09_28,
        "manifest_files": int(manifest["counts"]["files"]),
        "manifest_rows": int(manifest["counts"]["rows"]),
        "fetch_dates": fetch_dates(records),
        "tables_used": sorted({r["key"] for r in records}),
        "files_used": len(records),
        "rows_used": sum(r["rows"] for r in records),
        "vintage": "the extraction of " + C.EXTRACTION_DATE + ", with the GICS history added on "
                   + max(fetch_dates(records)) + "; manifest SHA-256 " + C.MANIFEST_SHA256,
        "versions": versions(),
        "written_by": written_by,
        "written_at": dt.datetime.now(ZoneInfo(C.TIMEZONE)).strftime("%Y-%m-%d %H:%M %Z"),
    }


def write_results(name: str, frame=None, record: dict = None) -> pathlib.Path:
    """Write one committed result into the results folder: a table as CSV or a record as JSON."""
    import csv
    folder = repo_root() / C.RESULTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    if frame is not None:
        frame.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")
    else:
        path.write_text(json.dumps(record, indent=1, sort_keys=False) + "\n")
    return path

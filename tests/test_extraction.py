"""The extraction module on a made-up extraction: the manifest check, the files used, the column check,
the reading of a table and the provenance record. No licensed data are involved."""
import json
import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from slt import constants as C  # noqa: E402
from slt import extraction as X  # noqa: E402


def make_extraction(tmp_path, fetched="2026-09-28T09:40:06+00:00"):
    """A folder shaped like the extraction: one made-up file per key used, a manifest and its .sha256."""
    files = []
    for key in X.keys_used():
        cols = X.COLUMNS[key]
        frame = pd.DataFrame({c: (["2001-02-28", "2003-12-31"] if c in X.DATE_COLUMNS else [1.0, 2.0]) for c in cols})
        rel = f"raw/{key}/{key}.parquet"
        (tmp_path / "raw" / key).mkdir(parents=True)
        frame.to_parquet(tmp_path / rel, index=False)
        lib, table = C.TABLES[key].split(".")
        when = "2026-11-10T15:05:50+00:00" if key == C.GICS_SOURCE else fetched
        files.append({"key": key, "library": lib, "table": table, "file": rel, "sql": f"select * from {lib}.{table}",
                      "rows": 2, "columns": len(cols), "bytes": (tmp_path / rel).stat().st_size,
                      "sha256": X.sha256_of(tmp_path / rel), "fetched_utc": when})
    manifest = {"files": files, "counts": {"files": len(files), "rows": 2 * len(files)}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    digest = X.sha256_of(tmp_path / "manifest.json")
    (tmp_path / "manifest.json.sha256").write_text(digest + "  manifest.json\n")
    return digest


def test_data_root_stops_with_one_sentence_when_unset():
    with pytest.raises(SystemExit) as e:
        X.data_root({})
    assert C.DATA_ROOT_ENV in str(e.value) and str(e.value).count(".") >= 1


def test_data_root_stops_when_the_folder_has_no_manifest(tmp_path):
    with pytest.raises(SystemExit):
        X.data_root({C.DATA_ROOT_ENV: str(tmp_path)})


def test_manifest_check_accepts_the_fixed_hash_and_refuses_another(tmp_path):
    digest = make_extraction(tmp_path)
    assert X.check_manifest(tmp_path, expected=digest)["counts"]["files"] == len(X.keys_used())
    with pytest.raises(SystemExit):
        X.check_manifest(tmp_path, expected="0" * 64)


def test_manifest_check_refuses_a_manifest_that_disagrees_with_its_sha256_file(tmp_path):
    digest = make_extraction(tmp_path)
    (tmp_path / "manifest.json.sha256").write_text("f" * 64 + "  manifest.json\n")
    with pytest.raises(SystemExit):
        X.check_manifest(tmp_path, expected=digest)


def test_keys_used_leave_out_the_unused_fallback():
    assert C.GICS_FALLBACK not in X.keys_used()
    assert C.GICS_SOURCE in X.keys_used()
    assert len(X.keys_used()) == len(C.TABLES) - 1


def test_files_used_and_fetch_dates(tmp_path):
    digest = make_extraction(tmp_path)
    records = X.files_used(X.check_manifest(tmp_path, expected=digest))
    assert [r["file"] for r in records] == sorted(r["file"] for r in records)
    dates = X.fetch_dates(records)
    assert dates["2026-11-10"] == [C.GICS_SOURCE]
    assert C.GICS_SOURCE not in dates["2026-09-28"]


def test_amsterdam_date_crosses_midnight():
    assert X.amsterdam_date("2026-11-10T23:30:00+00:00") == "2026-11-11"
    assert X.amsterdam_date("2026-09-28T09:18:09+00:00") == "2026-09-28"


def test_verify_files_and_columns(tmp_path):
    digest = make_extraction(tmp_path)
    records = X.files_used(X.check_manifest(tmp_path, expected=digest))
    checks = X.verify_files(tmp_path, records)
    assert all(c["exists"] and c["sha256_matches"] and c["rows_match"] for c in checks)
    assert X.missing_columns(tmp_path, records) == {}
    records[0]["sha256"] = "0" * 64
    assert not X.verify_files(tmp_path, records[:1])[0]["sha256_matches"]


def test_read_table_parses_dates_and_counts_what_does_not_parse(tmp_path):
    make_extraction(tmp_path)
    frame, unparsed = X.read_table(tmp_path, "crsp_dsp500list")
    assert str(frame["start"].dtype).startswith("datetime64")
    assert unparsed == {}
    odd = pd.DataFrame({"ending": ["2001-02-28", "E", None]})
    parsed, unparsed = X.parse_dates(odd)
    assert unparsed == {"ending": 1}
    assert parsed["ending"].isna().sum() == 2


def test_provenance_holds_no_row(tmp_path):
    digest = make_extraction(tmp_path)
    manifest = X.check_manifest(tmp_path, expected=digest)
    records = X.files_used(manifest)
    record = X.provenance(manifest, records, "test")
    assert record["fetch_dates"]["2026-11-10"] == [C.GICS_SOURCE]
    assert record["rows_used"] == 2 * len(records)
    text = json.dumps(record)
    assert "select" not in text
    assert set(record) >= {"manifest_sha256", "extraction_date", "vintage", "versions", "source"}

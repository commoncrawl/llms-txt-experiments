"""The URL-index funnel that precedes the content analysis.

``scan`` reduces a 6.6M-row CSV to counters. The unit tests run it against a
small CSV written here; the reconciliation logic is exercised in
test_integration.py against the real fixture.
"""
from __future__ import annotations

import json

import pytest

from llmstxt_analysis.aggregate import track_index
from llmstxt_analysis.urlindex import QUERY, scan, write

ROWS = [
    # url, fetch_status, content_mime_detected
    ("https://a.example/llms.txt", 200, "text/markdown"),
    ("https://b.example/llms.txt", 200, "text/plain"),
    ("https://c.example/llms.txt", 200, "text/html"),
    ("https://d.example/llms.txt", 200, "text/x-robots"),
    ("https://e.example/llms.txt", 404, "text/html"),
    ("https://f.example/llms.txt", 404, "text/html"),
    ("https://g.example/llms.txt", 301, "text/html"),
    ("https://h.example/llms.txt", 302, "text/html"),
    ("https://i.example/llms.txt", 503, None),
    ("https://j.example/llms.txt", 403, "text/html"),
    ("https://a.example/llms-full.txt", 200, "text/plain"),
    ("https://b.example/llms-full.txt", 404, "text/html"),
    ("https://c.example/llms-full.txt", 404, "text/html"),
    ("https://d.example/llms-full.txt", 307, "text/html"),
]


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("idx") / "index.csv"
    lines = ["crawl,url,url_host_name,fetch_status,content_mime_detected"]
    for url, status, mime in ROWS:
        host = url.split("/")[2]
        lines.append(f"CC-MAIN-2026-30,{url},{host},{status},{mime or ''}")
    p.write_text("\n".join(lines) + "\n")
    return p


@pytest.fixture(scope="module")
def scanned(csv_path):
    return scan(csv_path)


def test_totals_and_status_counts(scanned):
    assert scanned["total"] == len(ROWS)
    assert scanned["status"][200] == 5
    assert scanned["status"][404] == 4
    assert scanned["status"][301] == 1 and scanned["status"][302] == 1
    assert sum(scanned["status"].values()) == len(ROWS)


def test_mime_is_counted_only_for_200s(scanned):
    m = scanned["mime_200"]
    assert sum(m.values()) == scanned["status"][200]
    assert m["text/markdown"] == 1
    assert m["text/plain"] == 2
    assert m["text/html"] == 1, "404 HTML bodies must not be counted here"


def test_split_by_filename(scanned):
    full = scanned["by_kind"]["llms-full.txt"]
    plain = scanned["by_kind"]["llms.txt"]
    assert plain["total"] + full["total"] == len(ROWS)
    assert full["total"] == 4 and full["s200"] == 1 and full["text200"] == 1
    assert plain["s3012"] == 2, "307 is a redirect but not 301/302"
    assert plain["text200"] == 2


def test_query_is_quoted_not_executed(scanned):
    assert scanned["query"] == QUERY
    assert "url_path = '/llms.txt'" in scanned["query"]
    assert "CC-MAIN-2026-30" in scanned["query"]


def test_missing_csv_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        scan(tmp_path / "nope.csv.zst")


def test_write_round_trips(scanned, tmp_path):
    out = write(scanned, tmp_path / "urlindex.json")
    assert json.loads(out.read_text())["total"] == len(ROWS)


def test_chunked_scan_matches_a_single_pass(csv_path, monkeypatch):
    """Counters must not depend on where the chunk boundaries fall."""
    import llmstxt_analysis.urlindex as U

    monkeypatch.setattr(U, "CHUNK_ROWS", 3)
    chunked = U.scan(csv_path)
    monkeypatch.setattr(U, "CHUNK_ROWS", 10_000)
    whole = U.scan(csv_path)
    assert chunked == whole


def test_track_index_builds_well_formed_tables(scanned, frame):
    out = track_index(scanned, frame, len(frame))
    for key in ("funnel", "status", "mime", "by_kind", "reconcile"):
        t = out[key]
        assert t["rows"], key
        assert all(len(r) == len(t["columns"]) for r in t["rows"]), key
    assert out["pct_200"] == pytest.approx(100 * 5 / len(ROWS), abs=0.01)
    assert out["n_text200"] == 3
    # Percentages of a grouped status table must sum to 100.
    assert sum(r[2] for r in out["status"]["rows"]) == pytest.approx(100.0, abs=0.05)
    assert sum(r[1] for r in out["status"]["rows"]) == len(ROWS)

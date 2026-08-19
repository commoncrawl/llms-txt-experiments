"""The step before the content analysis: what the URL index says.

Every document analysed elsewhere in this project survived a funnel — Common
Crawl attempted millions of `/llms.txt` URLs, most 404ed, a minority returned
200, and only some of those returned a text body. That funnel is the
denominator for everything else, so it belongs in the report.

Input is the CSV the Athena query below produced, already downloaded to
``data/llms-txt-cc-main-2026-30.csv.zst`` (54 MB compressed, 6.6M rows). The
query is *not* re-run: it is quoted verbatim so a reader knows exactly what the
population is.

The CSV is read in chunks and reduced to counters, so memory stays flat rather
than materialising 6.6M rows of URL strings.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

DEFAULT_CSV = "data/llms-txt-cc-main-2026-30.csv.zst"
CHUNK_ROWS = 1_000_000
TEXT_MIMES = frozenset({"text/plain", "text/markdown"})

# The query that produced the CSV. Quoted, never executed here.
QUERY = """select crawl, url, url_host_name, fetch_status, content_mime_detected
  from ccindex
 where (url_path = '/llms.txt'
        or url_path = '/llms-full.txt')
   and url_query is null
   and crawl like 'CC-MAIN-2026-30';"""

# How this population came to exist. CC-MAIN-2026-30 did not happen to contain
# these files: /llms.txt and /llms-full.txt were added to the crawl's seed list on
# purpose, so the funnel below is the result of an experiment rather than an
# incidental by-product. Figures quoted from the crawl experiment, not derived here.
SEEDING = {
    "crawl": "CC-MAIN-2026-30",
    "prior_crawls": ["CC-MAIN-2026-21", "CC-MAIN-2026-25"],
    "sampled_llms_txt": 5_167_831,
    "sampled_llms_full": 1_761_228,
    "known_hosts_llms_txt": 27_394,
    "known_hosts_llms_full": 15_763,
}

STATUS_GROUPS = [
    ("200 OK", lambda s: s == 200),
    ("301/302 redirect", lambda s: s in (301, 302)),
    ("other 3xx", lambda s: 300 <= s < 400 and s not in (301, 302)),
    ("404 Not Found", lambda s: s == 404),
    ("other 4xx", lambda s: 400 <= s < 500 and s != 404),
    ("5xx server error", lambda s: 500 <= s < 600),
]


def _file_kind(url: str) -> str:
    return "llms-full.txt" if url.endswith("/llms-full.txt") else "llms.txt"


def scan(csv_path: str | Path = DEFAULT_CSV) -> dict:
    """Reduce the URL-index CSV to the counters the report needs."""
    path = Path(csv_path)
    if not path.exists():
        raise SystemExit(
            f"{path} not found. It is produced by the Athena query in "
            "llmstxt_analysis/urlindex.py"
        )

    total = 0
    status: Counter = Counter()
    mime_200: Counter = Counter()
    # per (file kind) -> Counter of {"total", "s200", "s404", "s3012", "text200"}
    by_kind: dict[str, Counter] = {"llms.txt": Counter(), "llms-full.txt": Counter()}

    reader = pd.read_csv(
        path,
        usecols=["url", "fetch_status", "content_mime_detected"],
        dtype={"fetch_status": "int32", "content_mime_detected": "string"},
        chunksize=CHUNK_ROWS,
    )
    for chunk in reader:
        total += len(chunk)
        status.update(chunk["fetch_status"].value_counts().to_dict())

        is200 = chunk["fetch_status"] == 200
        mime_200.update(chunk.loc[is200, "content_mime_detected"].fillna("(none)")
                        .value_counts().to_dict())

        kinds = chunk["url"].map(_file_kind)
        istext = is200 & chunk["content_mime_detected"].isin(TEXT_MIMES)
        s3012 = chunk["fetch_status"].isin([301, 302])
        s404 = chunk["fetch_status"] == 404
        for kind in by_kind:
            m = kinds == kind
            c = by_kind[kind]
            c["total"] += int(m.sum())
            c["s200"] += int((m & is200).sum())
            c["s404"] += int((m & s404).sum())
            c["s3012"] += int((m & s3012).sum())
            c["text200"] += int((m & istext).sum())

    return {
        "csv": str(path),
        "query": QUERY,
        "seeding": SEEDING,
        "total": total,
        "n_status_codes": len(status),
        "status": {int(k): int(v) for k, v in status.items()},
        "mime_200": {str(k): int(v) for k, v in mime_200.most_common()},
        "by_kind": {k: dict(v) for k, v in by_kind.items()},
    }


def write(result: dict, out: str | Path) -> Path:
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=1))
    print(f"wrote {p}: {result['total']:,} URL attempts, "
          f"{result['status'].get(200, 0):,} with HTTP 200")
    return p

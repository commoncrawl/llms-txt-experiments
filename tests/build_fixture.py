#!/usr/bin/env python
"""Rebuild the real-data test fixture.

Runs the extractors over one shard and keeps the smallest few documents from
each stratum, so the fixture stays under a megabyte while still exercising
every branch the pipeline can take. Content is copied verbatim.

    uv run tests/build_fixture.py --max-shard-rows 120000

Shards are read from the default Hugging Face cache; nothing is downloaded into
the working tree.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmstxt_analysis import extractors  # noqa: F401
from llmstxt_analysis.corpus import (
    DEFAULT_CONFIG,
    DEFAULT_REPO,
    READ_COLUMNS,
    iter_batches,
    resolve_shards,
)
from llmstxt_analysis.registry import batch_extractors, row_extractors

# stratum -> predicate over a feature dict
STRATA = {
    "wix": lambda f: f["generator_id"] == "wix",
    "yoast": lambda f: f["generator_id"] == "yoast",
    "aioseo": lambda f: f["generator_id"] == "aioseo",
    "rankmath": lambda f: f["generator_id"] == "rankmath",
    "parking": lambda f: f["generator_id"] == "godaddy_parking",
    "docs_platform": lambda f: f["generator_family"] == "docs_platform",
    "unknown": lambda f: f["generator_id"] == "unknown",
    "json": lambda f: f["doc_kind"] == "json",
    "empty": lambda f: f["doc_kind"] == "empty",
    "html": lambda f: f["doc_kind"] == "html",
    "yaml": lambda f: f["doc_kind"] in ("yaml", "yaml_frontmatter"),
    "robots": lambda f: f["doc_kind"] == "robots_txt",
    "plain": lambda f: f["doc_kind"] == "plain",
    "conf4": lambda f: f["conformance_level"] == 4,
    "conf1": lambda f: f["conformance_level"] == 1,
    "forsale": lambda f: f["for_sale"],
    "inject1": lambda f: f["injection_severity"] == 1,
    "inject2": lambda f: f["injection_severity"] == 2,
    "inject3": lambda f: f["injection_severity"] == 3,
    "policy": lambda f: f["has_any_policy"],
    "policy_yaml": lambda f: f["policy_dialect"] in ("yaml", "mixed"),
    "policy_robots": lambda f: f["policy_dialect"] == "robots",
    "training_deny": lambda f: f["training_stance"] == "deny",
    "bots": lambda f: f["n_named_bots"] > 0,
    "spam": lambda f: f["n_spam_categories"] > 0,
    "mcp": lambda f: f["mentions_mcp"],
    "full": lambda f: f["is_full"],
    "glotlid": lambda f: f["lang_source"] == "glotlid",
    "nonlatin": lambda f: f["lang"] in ("jpn", "rus", "zho", "ukr", "ara", "heb", "tha", "kor"),
    "deu": lambda f: f["lang"] == "deu",
    "fra": lambda f: f["lang"] == "fra",
    "nolinks": lambda f: f["n_links"] == 0 and f["doc_kind"] == "markdown",
    "manylinks": lambda f: f["n_links"] > 200,
    "offsite": lambda f: f["offsite_ratio"] > 0.5 and f["n_links"] > 5,
}
PER_STRATUM = 5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DEFAULT_REPO)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out", default="tests/fixtures/train-00000-of-00001.parquet")
    ap.add_argument("--max-shard-rows", type=int, default=120_000)
    args = ap.parse_args()

    shard = resolve_shards(args.dataset, args.config)[0]
    rows = [c() for c in row_extractors()]
    bats = [c() for c in batch_extractors()]
    for e in (*rows, *bats):
        e.setup()

    keep: dict[tuple[int, int], list[str]] = {}
    seen = 0
    for batch in iter_batches(shard, limit_rows=args.max_shard_rows):
        feats = [{} for _ in batch]
        for e in rows:
            for i, r in enumerate(batch):
                feats[i].update(e.extract(r))
        for e in bats:
            for i, d in enumerate(e.extract_batch(batch)):
                feats[i].update(d)
        for name, pred in STRATA.items():
            hits = sorted(
                (i for i, f in enumerate(feats) if pred(f)),
                key=lambda i: len(batch[i].content),
            )
            room = PER_STRATUM - sum(1 for v in keep.values() if name in v)
            for i in hits[: max(0, room)]:
                keep.setdefault((batch[i].rg, batch[i].rg_row), []).append(name)
        seen += len(batch)
        if all(sum(1 for v in keep.values() if s in v) >= PER_STRATUM for s in STRATA):
            break

    missing = [s for s in STRATA if not any(s in v for v in keep.values())]
    if missing:
        print(f"warning: no records found for strata {missing}")

    pf = pq.ParquetFile(shard.path)
    tables = []
    for rg in sorted({rg for rg, _ in keep}):
        idx = sorted(row for r, row in keep if r == rg)
        tables.append(pf.read_row_group(rg, columns=READ_COLUMNS).take(pa.array(idx)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.concat_tables(tables), out, compression="zstd")

    counts = Counter(s for v in keep.values() for s in v)
    print(f"{sum(t.num_rows for t in tables)} records -> {out} "
          f"({out.stat().st_size/1e6:.2f} MB), scanned {seen} rows")
    print(dict(sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

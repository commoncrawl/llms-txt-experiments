# Tests

```bash
uv run pytest tests/ -q
```

No network access and no HF token are required: everything runs against a
committed fixture.

## The fixture is real data

`tests/fixtures/train-00000-of-00001.parquet` holds records taken **verbatim
from the dataset** (`commoncrawl/llms.txt`, config `CC-MAIN-2026-30`, shard 0).
Content, HTTP headers and WARC headers are byte-identical to the corpus —
nothing is synthesised, truncated or hand-edited. `tests/build_fixture.py` runs
the extractors over the shard and keeps the smallest few documents from each of
~34 strata, so that every branch the pipeline can take is represented:

* every `doc_kind`: markdown, plain, empty, json, html, robots_txt, yaml,
  yaml_frontmatter
* every generator: wix, yoast, aioseo, rankmath, godaddy_parking, a docs
  platform, unknown
* every conformance level, 0 through 4
* every policy dialect (prose, YAML, robots.txt), training denials, documents
  naming crawlers, every injection severity, for-sale domains, spam-lexicon
  hits, MCP advertisements
* both LID paths (cld2 and the GlotLID fallback), Latin and non-Latin scripts,
  and `llms-full.txt` as well as `llms.txt`

`test_extractors.py::test_real_records_cover_every_stratum` asserts that
coverage, so a fixture that silently loses a stratum fails the suite rather than
quietly weakening it.

```bash
uv run tests/build_fixture.py          # reads shard 0 from the default HF cache
```

Keep it under a megabyte so it stays committable. Note that tightening a
detector can empty a stratum: when the injection lexicon was made
position-sensitive, the fixture's severity-2 documents turned out to have been
false positives and the coverage test caught it.

## Layout

| File | What it covers |
|---|---|
| `test_record.py` | The derived views every extractor shares — link parsing, prose extraction, the template skeleton, the bounded scan window, URL parsing. Unit tests plus consistency invariants over all 124 real records. |
| `test_lexicon.py` | The anchor invariant: the cheap literal pre-filter must never hide a match the regex would have made. Every lexicon slug has a worked example, and the anchored and unanchored paths are asserted to agree on every real record. |
| `test_extractors.py` | Each extractor in isolation (synthetic documents with known answers), plus feature-level invariants over the real records. |
| `test_integration.py` | The pipeline end to end: `extract` → `derive` → `topics` → `aggregate` → `report` → `spotcheck`, plus the CLI. Includes determinism, locator round-tripping, table well-formedness, and the report's self-containment and theme-awareness contracts. |

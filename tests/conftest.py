"""Shared fixtures.

The corpus fixture is **real data**: 124 records sampled from
``commoncrawl/llms.txt`` (config CC-MAIN-2026-30, shard 0), stratified so that
every document kind, generator, conformance level, policy dialect, injection
severity, spam category and LID path in the corpus is represented. Content is
byte-identical to the dataset — nothing is synthesised or truncated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def shards_root() -> Path:
    assert (FIXTURE_DIR / "train-00000-of-00001.parquet").exists(), (
        "missing test fixture; see tests/README.md for how it is built"
    )
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def records(shards_root):
    """Every fixture record, as ``Record`` objects."""
    from llmstxt_analysis.corpus import find_shards, iter_batches

    shard = find_shards(shards_root)[0]
    out = []
    for batch in iter_batches(shard):
        out.extend(batch)
    return out


@pytest.fixture(scope="session")
def features(records):
    """Feature dicts for every fixture record (the full extractor pipeline)."""
    from llmstxt_analysis import extractors  # noqa: F401
    from llmstxt_analysis.registry import batch_extractors, row_extractors

    rows = [c() for c in row_extractors()]
    bats = [c() for c in batch_extractors()]
    for e in (*rows, *bats):
        e.setup()
    feats = [{} for _ in records]
    for e in rows:
        for i, r in enumerate(records):
            feats[i].update(e.extract(r))
    for e in bats:
        for i, d in enumerate(e.extract_batch(records)):
            feats[i].update(d)
    return feats


@pytest.fixture(scope="session")
def derived_dir(shards_root, tmp_path_factory):
    """Run the real ``extract`` stage over the fixture shard once per session."""
    from llmstxt_analysis.extract import run

    out = tmp_path_factory.mktemp("derived")
    run(shards_root, out, workers=1)
    return out


@pytest.fixture(scope="session")
def frame(derived_dir):
    from llmstxt_analysis.derive import load

    # The fixture is tiny, so the production threshold of 50 would classify
    # nothing as templated; 3 keeps the derived columns exercised.
    return load(derived_dir, template_min_cluster=3, use_cache=False)

"""The extraction pass: shards -> features table (+ topic corpus).

One streaming pass over the corpus, one worker process per shard. Each worker
writes its own parquet part file, so the stage is embarrassingly parallel and
restartable at shard granularity.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from . import extractors  # noqa: F401  (populates the registry)
from .corpus import DEFAULT_CONFIG, Shard, iter_batches, resolve_shards, shard_base_offsets
from .registry import batch_extractors, feature_schema, row_extractors

TOPIC_MIN_CHARS = 200
TOPIC_MAX_CHARS = 2000

TOPIC_SCHEMA = pa.schema(
    [
        pa.field("shard", pa.int32()),
        pa.field("rg", pa.int32()),
        pa.field("rg_row", pa.int32()),
        pa.field("lang", pa.string()),
        pa.field("text", pa.string()),
    ]
)


def _worker(args: tuple[int, str, str, int, int]) -> dict[str, Any]:
    shard_index, shard_path, out_dir, limit_rows, base_offset = args
    shard = Shard(shard_index, Path(shard_path))
    out = Path(out_dir)

    rows = [cls() for cls in row_extractors()]
    batches = [cls() for cls in batch_extractors()]
    for ex in (*rows, *batches):
        ex.setup()

    schema = feature_schema()
    fpath = out / "features" / f"part-{shard_index:05d}.parquet"
    tpath = out / "topic_corpus" / f"part-{shard_index:05d}.parquet"
    fpath.parent.mkdir(parents=True, exist_ok=True)
    tpath.parent.mkdir(parents=True, exist_ok=True)

    fw = pq.ParquetWriter(fpath, schema, compression="zstd")
    tw = pq.ParquetWriter(tpath, TOPIC_SCHEMA, compression="zstd")

    n_rows = 0
    n_topic = 0
    t0 = time.time()
    try:
        for recs in iter_batches(shard, limit_rows=limit_rows, base_offset=base_offset):
            if not recs:
                continue
            feats: list[dict[str, Any]] = [{} for _ in recs]
            for ex in rows:
                for i, rec in enumerate(recs):
                    feats[i].update(ex.extract(rec))
            for bex in batches:
                for i, d in enumerate(bex.extract_batch(recs)):
                    feats[i].update(d)

            fw.write_table(pa.Table.from_pylist(feats, schema=schema))

            tdocs = []
            for rec, f in zip(recs, feats):
                if f.get("generator_id") != "unknown":
                    continue
                prose = rec.prose
                if len(prose) < TOPIC_MIN_CHARS:
                    continue
                tdocs.append(
                    {
                        "shard": rec.shard,
                        "rg": rec.rg,
                        "rg_row": rec.rg_row,
                        "lang": f.get("lang") or "und",
                        "text": prose[:TOPIC_MAX_CHARS],
                    }
                )
            if tdocs:
                tw.write_table(pa.Table.from_pylist(tdocs, schema=TOPIC_SCHEMA))
                n_topic += len(tdocs)

            n_rows += len(recs)
    finally:
        fw.close()
        tw.close()

    return {
        "shard": shard_index,
        "rows": n_rows,
        "topic_docs": n_topic,
        "seconds": round(time.time() - t0, 1),
    }


def run(
    source: str | Path,
    out_dir: str | Path,
    workers: int = 8,
    limit_shards: int = 0,
    limit_rows: int = 0,
    config: str = DEFAULT_CONFIG,
) -> list[dict[str, Any]]:
    shards = resolve_shards(source, config)
    if not shards:
        raise SystemExit(f"no parquet shards found for {source}")
    if limit_shards:
        shards = shards[:limit_shards]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Base offsets come from the *full* shard list so that dataset indices stay
    # correct even when --limit-shards restricts what is processed.
    bases = shard_base_offsets(resolve_shards(source, config))
    jobs = [(s.index, str(s.path), str(out), limit_rows, bases[s.index]) for s in shards]

    print(f"extracting {len(shards)} shard(s) with {workers} worker(s) -> {out}")
    results: list[dict[str, Any]] = []
    if workers <= 1 or len(jobs) == 1:
        for job in jobs:
            r = _worker(job)
            print(f"  shard {r['shard']}: {r['rows']} rows, {r['topic_docs']} topic docs, {r['seconds']}s")
            results.append(r)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(workers, len(jobs))) as pool:
            for r in pool.imap_unordered(_worker, jobs):
                print(f"  shard {r['shard']}: {r['rows']} rows, {r['topic_docs']} topic docs, {r['seconds']}s")
                results.append(r)

    total = sum(r["rows"] for r in results)
    print(f"done: {total} records -> {out/'features'}")
    return sorted(results, key=lambda r: r["shard"])

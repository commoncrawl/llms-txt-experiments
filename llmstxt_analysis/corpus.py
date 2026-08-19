"""Corpus access: shard resolution and row-group streaming.

Shards come from the Hugging Face Hub and live in the **default HF cache**
(`HF_HOME`/`~/.cache/huggingface`), not in this repository. ``resolve_shards``
takes either a repo id or a local directory, so tests can point at the
committed fixture while the real runs stream from the cached dataset.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from .record import Record

DEFAULT_REPO = "commoncrawl/llms.txt"
DEFAULT_CONFIG = "CC-MAIN-2026-30"

READ_COLUMNS = [
    "content",
    "http_headers",
    "WARC-Target-URI",
    "WARC-Date",
    "WARC-IP-Address",
    "WARC-Identified-Payload-Type",
]

_SHARD_RE = re.compile(r"train-(\d+)-of-\d+\.parquet$")


@dataclass(frozen=True)
class Shard:
    index: int
    path: Path

    @property
    def name(self) -> str:
        return self.path.name


def find_shards(root: str | Path) -> list[Shard]:
    """All parquet shards under ``root`` (recursively), ordered by shard index."""
    root = Path(root)
    paths = sorted(root.rglob("*.parquet"))
    shards = []
    for p in paths:
        m = _SHARD_RE.search(p.name)
        shards.append(Shard(int(m.group(1)) if m else len(shards), p))
    return sorted(shards, key=lambda s: s.index)


def hf_shards(repo_id: str = DEFAULT_REPO, config: str = DEFAULT_CONFIG,
              revision: str | None = None, workers: int = 4) -> list[Shard]:
    """Resolve the dataset's parquet shards inside the default HF cache.

    Files already present in the cache are not re-fetched, so this is cheap to
    call repeatedly. Nothing is written into the working tree.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = sorted(
        f for f in api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
        if f.endswith(".parquet") and (not config or f.startswith(f"{config}/"))
    )
    if not files:
        raise SystemExit(f"no parquet files for config {config!r} in {repo_id}")

    shards = []
    for i, name in enumerate(files):
        path = hf_hub_download(repo_id, name, repo_type="dataset", revision=revision)
        m = _SHARD_RE.search(name)
        shards.append(Shard(int(m.group(1)) if m else i, Path(path)))
    return sorted(shards, key=lambda s: s.index)


def resolve_shards(source: str | Path | None = None, config: str = DEFAULT_CONFIG,
                   revision: str | None = None) -> list[Shard]:
    """Shards from a local directory if one is given, otherwise from the Hub.

    ``source`` is treated as a local directory when it exists on disk; anything
    else is taken to be a Hugging Face dataset repo id.
    """
    if source and Path(source).exists():
        return find_shards(source)
    return hf_shards(str(source or DEFAULT_REPO), config, revision)


def cache_root() -> str:
    return os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")


def shard_info(shard: Shard) -> dict:
    pf = pq.ParquetFile(shard.path)
    md = pf.metadata
    return {
        "shard": shard.index,
        "file": shard.name,
        "rows": md.num_rows,
        "row_groups": md.num_row_groups,
        "size_mb": round(shard.path.stat().st_size / 1e6, 1),
    }


def shard_base_offsets(shards: list[Shard]) -> dict[int, int]:
    """Row offset of each shard within the concatenated dataset split.

    Hugging Face concatenates the shards in filename order, so a record's
    dataset index is its shard's base offset plus its row position in that
    shard. Reading parquet footers only — no data is decoded.
    """
    offsets, running = {}, 0
    for s in sorted(shards, key=lambda s: s.index):
        offsets[s.index] = running
        running += pq.ParquetFile(s.path).metadata.num_rows
    return offsets


def iter_batches(shard: Shard, limit_rows: int = 0, base_offset: int = 0) -> Iterator[list[Record]]:
    """Yield one list of ``Record`` per parquet row group."""
    pf = pq.ParquetFile(shard.path)
    seen = 0
    row_in_shard = 0
    for rg in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(rg, columns=READ_COLUMNS)
        cols = {name: tbl.column(name).to_pylist() for name in READ_COLUMNS}
        n = tbl.num_rows
        rg_start = row_in_shard
        row_in_shard += n
        batch = [
            Record(
                shard=shard.index,
                rg=rg,
                rg_row=i,
                dataset_index=base_offset + rg_start + i,
                url=cols["WARC-Target-URI"][i] or "",
                content=cols["content"][i] or "",
                http_headers=cols["http_headers"][i] or "",
                warc_date=cols["WARC-Date"][i] or "",
                warc_ip=cols["WARC-IP-Address"][i] or "",
                payload_type=cols["WARC-Identified-Payload-Type"][i] or "",
            )
            for i in range(n)
        ]
        if limit_rows:
            room = limit_rows - seen
            if room <= 0:
                return
            batch = batch[:room]
        seen += len(batch)
        yield batch
        if limit_rows and seen >= limit_rows:
            return


def fetch_records(source: str | Path, locators: list[tuple[int, int, int]],
                  config: str = DEFAULT_CONFIG) -> dict[tuple[int, int, int], Record]:
    """Read back specific records by ``(shard, rg, rg_row)`` for spot checks."""
    shards = {s.index: s for s in resolve_shards(source, config)}
    by_group: dict[tuple[int, int], list[int]] = {}
    for sh, rg, row in locators:
        by_group.setdefault((sh, rg), []).append(row)

    bases = shard_base_offsets(list(shards.values()))
    out: dict[tuple[int, int, int], Record] = {}
    for (sh, rg), rows in sorted(by_group.items()):
        shard = shards.get(sh)
        if shard is None:
            continue
        pf = pq.ParquetFile(shard.path)
        rg_start = sum(pf.metadata.row_group(i).num_rows for i in range(rg))
        tbl = pf.read_row_group(rg, columns=READ_COLUMNS)
        cols = {name: tbl.column(name) for name in READ_COLUMNS}
        for r in rows:
            if r >= tbl.num_rows:
                continue
            out[(sh, rg, r)] = Record(
                shard=sh,
                rg=rg,
                rg_row=r,
                dataset_index=bases.get(sh, 0) + rg_start + r,
                url=cols["WARC-Target-URI"][r].as_py() or "",
                content=cols["content"][r].as_py() or "",
                http_headers=cols["http_headers"][r].as_py() or "",
                warc_date=cols["WARC-Date"][r].as_py() or "",
                warc_ip=cols["WARC-IP-Address"][r].as_py() or "",
                payload_type=cols["WARC-Identified-Payload-Type"][r].as_py() or "",
            )
    return out

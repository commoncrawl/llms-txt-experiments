"""Core identity, size and HTTP features (every other track joins on these)."""
from __future__ import annotations

import hashlib

from ..record import Record
from ..registry import BOOL, I32, I64, STR, RowExtractor, register


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()


@register
class CoreExtractor(RowExtractor):
    NAME = "core"
    TRACK = "core"
    FIELDS = {
        "shard": I32,
        "rg": I32,
        "rg_row": I32,
        # Position in the HF dataset split, so any example in the report can be
        # re-fetched with load_dataset(...)["train"][dataset_index].
        "dataset_index": I64,
        "url": STR,
        "host": STR,
        "regdomain": STR,
        "tld": STR,
        "is_full": BOOL,
        "scheme": STR,
        "warc_date": STR,
        "payload_type": STR,
        "http_server": STR,
        "http_content_type": STR,
        "n_chars": I64,
        "n_bytes": I64,
        "n_lines": I32,
        "n_words": I32,
        "content_sha1": STR,
        "skeleton_sha1": STR,
    }

    def extract(self, rec: Record) -> dict:
        body = rec.body
        hdrs = rec.headers
        return {
            "shard": rec.shard,
            "rg": rec.rg,
            "rg_row": rec.rg_row,
            "dataset_index": rec.dataset_index,
            "url": rec.url,
            "host": rec.host,
            "regdomain": rec.regdomain,
            "tld": rec.tld,
            "is_full": rec.is_full,
            "scheme": rec._split.scheme,
            "warc_date": rec.warc_date,
            "payload_type": rec.payload_type,
            "http_server": hdrs.get("server", "")[:80],
            "http_content_type": hdrs.get("content-type", "")[:80],
            "n_chars": len(body),
            "n_bytes": len(body.encode("utf-8", "ignore")),
            "n_lines": len(rec.lines),
            "n_words": len(body.split()),
            "content_sha1": _sha1(body),
            "skeleton_sha1": _sha1(rec.skeleton),
        }

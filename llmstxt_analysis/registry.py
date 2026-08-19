"""Extractor registry.

An *extractor* turns records into a fixed set of feature columns. Two flavours:

* ``RowExtractor``  — one record at a time. Use for regex/string work.
* ``BatchExtractor`` — a whole row-group at once. Use when a library is much
  faster in batch (cld2, tiktoken).

Adding an analysis means adding one subclass and importing its module in
``llmstxt_analysis.extractors``. The parquet schema is derived from the
registry at runtime, so nothing else needs to change.
"""
from __future__ import annotations

from typing import Any, ClassVar, Iterable

import pyarrow as pa

# Shorthands for the field type declarations in FIELDS.
STR = pa.string()
I32 = pa.int32()
I64 = pa.int64()
F32 = pa.float32()
BOOL = pa.bool_()
LIST_STR = pa.list_(pa.string())


class Extractor:
    """Base class. Subclasses declare NAME, TRACK and FIELDS."""

    NAME: ClassVar[str] = ""
    TRACK: ClassVar[str] = ""  # A/B/D/E/F/core, for documentation only
    FIELDS: ClassVar[dict[str, pa.DataType]] = {}

    def setup(self) -> None:
        """Called once per worker process, after fork. Load models here."""

    def empty(self) -> dict[str, Any]:
        return {k: None for k in self.FIELDS}


class RowExtractor(Extractor):
    def extract(self, rec) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class BatchExtractor(Extractor):
    def extract_batch(self, recs: list) -> list[dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError


_ROW: list[type[RowExtractor]] = []
_BATCH: list[type[BatchExtractor]] = []


def register(cls):
    """Class decorator adding an extractor to the pipeline."""
    if issubclass(cls, BatchExtractor):
        _BATCH.append(cls)
    elif issubclass(cls, RowExtractor):
        _ROW.append(cls)
    else:  # pragma: no cover
        raise TypeError(f"{cls!r} is neither a RowExtractor nor a BatchExtractor")
    return cls


def row_extractors() -> list[type[RowExtractor]]:
    return list(_ROW)


def batch_extractors() -> list[type[BatchExtractor]]:
    return list(_BATCH)


def all_extractors() -> Iterable[type[Extractor]]:
    return [*_ROW, *_BATCH]


def feature_schema() -> pa.Schema:
    """Parquet schema for the features table, built from the registry."""
    fields: list[pa.Field] = []
    seen: set[str] = set()
    for cls in all_extractors():
        for name, typ in cls.FIELDS.items():
            if name in seen:
                raise ValueError(f"duplicate feature column {name!r} in {cls.NAME}")
            seen.add(name)
            fields.append(pa.field(name, typ))
    return pa.schema(fields)

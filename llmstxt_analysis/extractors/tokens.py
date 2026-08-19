"""Track F — exact token counts.

One tokenizer for the whole corpus (`o200k_base`); model families differ by
roughly +-20%, which is stated as a limitation in the report. Counting is done
in batch because tiktoken releases the GIL and batches amortise the FFI cost
(~28 MB/s per core measured on this corpus).
"""
from __future__ import annotations

from ..registry import I64, BatchExtractor, register

ENCODING = "o200k_base"


@register
class TokenExtractor(BatchExtractor):
    NAME = "tokens"
    TRACK = "F"
    FIELDS = {"n_tokens": I64}

    def setup(self) -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding(ENCODING)

    def extract_batch(self, recs: list) -> list[dict]:
        toks = self._enc.encode_ordinary_batch([r.body for r in recs], num_threads=1)
        return [{"n_tokens": len(t)} for t in toks]

"""Track F — language identification via commonlid (Common Crawl's LID eval kit).

https://github.com/commoncrawl/commonlid-eval#ad-hoc-prediction-no-dataset-no-files

`cld2` is the primary model: a C++ implementation at ~7k docs/s on 4 KB inputs.
Documents where cld2 abstains fall back to `GlotLID` (fastText), which covers
far more languages at some cost in precision on short inputs.

Input is `Record.prose` — the head with link bullets, URLs and markdown syntax
removed — so that link dumps and template boilerplate do not decide the label.
"""
from __future__ import annotations

from ..registry import STR, BatchExtractor, register

MIN_PROSE_CHARS = 20


@register
class LanguageExtractor(BatchExtractor):
    NAME = "language"
    TRACK = "F"
    FIELDS = {
        "lang": STR,
        "lang_cld2": STR,
        "lang_glotlid": STR,
        "lang_source": STR,
    }

    def setup(self) -> None:
        from commonlid import get_model

        self._cld2 = get_model("cld2")
        self._glot = None  # loaded lazily; many batches never need it

    def _glotlid(self):
        if self._glot is None:
            from commonlid import get_model

            self._glot = get_model("GlotLID")
        return self._glot

    def extract_batch(self, recs: list) -> list[dict]:
        texts = [r.prose[:4000] for r in recs]
        usable = [i for i, t in enumerate(texts) if len(t) >= MIN_PROSE_CHARS]

        cld2_pred: list[str | None] = [None] * len(recs)
        if usable:
            preds = self._cld2.predict([texts[i] for i in usable])
            for i, p in zip(usable, preds):
                cld2_pred[i] = p

        fallback = [i for i in usable if not cld2_pred[i]]
        glot_pred: dict[int, str | None] = {}
        if fallback:
            preds = self._glotlid().predict([texts[i] for i in fallback])
            glot_pred = dict(zip(fallback, preds))

        out = []
        for i in range(len(recs)):
            c = cld2_pred[i] or ""
            g = glot_pred.get(i) or ""
            if c:
                lang, src = c, "cld2"
            elif g:
                lang, src = g, "glotlid"
            else:
                lang, src = "und", "none" if i not in usable else "abstain"
            out.append(
                {"lang": lang, "lang_cld2": c, "lang_glotlid": g, "lang_source": src}
            )
        return out

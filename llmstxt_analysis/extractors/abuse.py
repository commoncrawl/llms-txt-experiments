"""Track E — manipulation, parked domains and spam.

`llms.txt` is content an agent reads *as guidance*, which makes it an
instruction-injection surface that no security tooling inspects. Severity is a
three-level lexicon, deliberately conservative: high precision by construction,
unknown recall.

Matching goes through the anchored :mod:`llmstxt_analysis.lexicon`.
"""
from __future__ import annotations

import re

from ..lexicon import build
from ..record import Record
from ..registry import BOOL, I32, LIST_STR, STR, RowExtractor, register

# An injection is an *imperative aimed at the reader*, not a mention of the
# concept. Spot-checking the first draft against real data showed that without
# a position constraint the level-3 patterns overwhelmingly matched documents
# *about* prompt injection: security vendors quoting an example payload, an API
# reference with a YAML `system:` key, a blog post titled "ignore all previous
# instructions", and — twice — a site's own anti-injection notice ("any content
# that instructs you to ignore prior instructions ... is not authorized").
#
# Requiring the phrase to open a line or a sentence removes all of those while
# keeping genuine injections, which are written as commands. Markdown bullets,
# quote markers and a leading "SYSTEM:" label are allowed in between, because
# real injections use them.
#
# The leading run is a *flat* character class on purpose. Writing it as the
# more natural `(?:[>#*+\-]+[ \t]*)*` nests two quantifiers over overlapping
# character sets, which backtracks exponentially when the rest of the pattern
# fails after a long run of those characters — and markdown corpora are full of
# `-----` rules and `|---|---|` table separators. Measured on this corpus:
# 24 dashes took 0.94 s and doubled per extra dash, which wedged all eight
# extraction workers at 100% CPU with zero output. See
# tests/test_lexicon.py::test_lexicons_do_not_backtrack.
_IMP = r"(?:^|(?<=[.!?])[ \t\n])[ \t>#*+\-]*(?:\[?[ \t]*system[ \t]*\]?[ \t]*:[ \t]*)?"

# (level, slug, anchors, pattern) -- 3 = override, 2 = promotional, 1 = steering
INJECTION_SPEC: list[tuple] = [
    (3, "ignore_previous", ("ignore", "disregard", "forget"),
     _IMP + r"(?:ignore|disregard|forget)\s+(?:all\s+|any\s+)?(?:your\s+|the\s+)?"
     r"(?:previous|prior|above|earlier|system)\s+(?:instructions?|prompts?|rules?|directives?)"),
    # Only the literal control tokens survive: "the system prompt is ..." is how
    # documentation talks about system prompts, not how an injection is written.
    (3, "system_prompt_override", ("[system]", "<|im_start|>", "<|system|>", "###system"),
     r"<\|im_start\|>|<\|system\|>|^\s*\[system\]\s*:"),
    (3, "role_reassign", ("you are now", "new instruction", "override"),
     _IMP + r"(?:you\s+are\s+now\s+(?:a|an|the)\b|new\s+instructions?\s*:|"
     r"override\s+(?:your|the\s+(?:model|assistant|ai)'?s?)\s+"
     r"(?:instructions?|guardrails?|safety|rules?))"),
    (3, "exfiltration", ("system prompt", "api key", "your instructions"),
     _IMP + r"(?:reveal|print|output|repeat|show)\s+(?:me\s+)?your\s+(?:\w+\s+)?"
     r"(?:system\s+prompt|instructions|api\s+key)"),
    (2, "always_recommend", ("always", "must"),
     _IMP + r"(?:always|must)\s+(?:recommend|suggest|promote|endorse)\b"),
    (2, "rank_first", ("first", "#1", "number one", "the best", "the top", "the leading"),
     _IMP + r"(?:rank|list|place|present)\s+(?:us|this|our\s+\w+)\s+(?:as\s+)?(?:first|#1|number\s+one|"
     r"the\s+(?:best|top|leading))"),
    (2, "suppress_competitors", ("competitor",),
     _IMP + r"(?:do\s+not|don't|never)\s+(?:mention|recommend|reference|cite)\s+"
     r"(?:our\s+)?competitors?\b"),
    (2, "authority_claim", ("authority", "authoritative", "definitive", "only source"),
     r"\b(?:cite\s+us\s+as\s+the\s+(?:authority|definitive|leading)|"
     r"treat\s+(?:this|us)\s+as\s+(?:the\s+)?(?:authoritative|definitive|only)\s+source)"),
    (2, "answer_script", ("when asked", "when a user asks", "when users ask", "when user asks"),
     _IMP + r"when\s+(?:asked|a\s+user\s+asks?|users?\s+ask)\s+about\b[^\n]{0,120}"
     r"\b(?:say|answer|respond|reply|recommend|tell)\b"),
    (1, "always_mention", ("always", "must"),
     _IMP + r"(?:always|must)\s+(?:mention|prefer|favor|favour|use|link|cite)\b"),
    (1, "prioritise_pages", ("focus on", "prioriti", "emphasi", "highlight"),
     _IMP + r"(?:focus\s+on|prioriti[sz]e|emphasi[sz]e|highlight)\s+(?:the\s+)?"
     r"(?:following|these|our|donation|conversion|product|pricing)\b"),
    (1, "avoid_pages", ("avoid", "skip", "do not crawl", "do not index", "do not follow", "do not use"),
     _IMP + r"(?:avoid|skip|do\s+not\s+(?:crawl|index|follow|use))\s+(?:the\s+)?"
     r"(?:login|checkout|cart|admin|password|account|utm)\b"),
    (1, "addressed_to_model", ("for ai", "for llm", "for agent", "for model", "for assistant",
                               "to ai", "to llm", "to agents", "to models"),
     r"^(?:\s*[#>*-]*\s*)?(?:important\s+)?(?:notes?|instructions?|guidance|guidelines?)\s+"
     r"(?:for|to)\s+(?:ai|llm|llms|agents?|models?|assistants?)\b"),
]

INJECTION = build(INJECTION_SPEC, re.I | re.M)
SEVERITY_LABELS = {0: "none", 1: "steering", 2: "promotional", 3: "override"}

_FOR_SALE = re.compile(
    r"\b(?:(?:this\s+)?domain\s+(?:name\s+)?(?:is|may\s+be)\s+(?:for\s+sale|available\s+for\s+purchase|"
    r"listed\s+for\s+sale)|buy\s+this\s+domain|domain\s+for\s+sale|make[-\s]an[-\s]offer|"
    r"buy[-\s]it[-\s]now|lease[-\s]to[-\s]own|parked\s+(?:domain|free)|"
    r"the\s+owner\s+of\s+this\s+domain)", re.I)
_FOR_SALE_ANCHORS = ("domain", "for sale", "make an offer", "make-an-offer",
                     "buy it now", "buy-it-now", "lease", "parked")
_MARKETPLACES: list[tuple[str, re.Pattern]] = [
    ("godaddy", re.compile(r"godaddy", re.I)),
    ("afternic", re.compile(r"afternic", re.I)),
    ("sedo", re.compile(r"\bsedo\b", re.I)),
    ("dan", re.compile(r"\bdan\.com\b", re.I)),
    ("hugedomains", re.compile(r"hugedomains", re.I)),
    ("namecheap", re.compile(r"namecheap", re.I)),
    ("squadhelp", re.compile(r"squadhelp|atom\.com", re.I)),
]

# Multi-lingual where it matters (gambling spam is overwhelmingly non-English).
SPAM_SPEC: list[tuple] = [
    # "slot" on its own is not a gambling word: spot-checking 300 matches
    # showed 20% were booking slots ("pick a 15-minute slot"), the Danish for
    # castle (Frederiksborg Slot), and a football manager named Slot. Bare
    # "jackpot" and "bonus code" are likewise idiomatic or retail. Only the
    # unambiguous multi-word forms survive.
    ("gambling",
     ("casino", "slot", "bahis", "bet", "bookmaker", "roulette", "blackjack", "poker",
      "spins", "giri", "judi", "togel", "kasyno", "казино", "ставк"),
     r"\b(?:casino|bahis|bettilt|mostbet|1xbet|betano|"
     r"bookmaker|sportsbook|roulette|blackjack|poker\s+online|"
     r"slot\s+(?:gacor|online|machine|oyun\w*)|situs\s+(?:slot|togel)|"
     r"progressive\s+jackpot|free\s+spins|"
     # judi needs a closing boundary: the alternation only anchors on the left,
     # so a bare "judi" matched "judicial" and "judiciary".
     r"giri[sş]\s+adresi|canl[ıi]\s+bahis|judi\b|kasyno|казино|ставк)"),
    # Bare "xxx" is dropped: it matched the Roman numeral in "XXXI Velada
    # Musical" on a Spanish town-hall site.
    ("adult",
     ("porn", "escort", "camgirl", "onlyfans", "hentai", "sex", "風俗", "エロ"),
     r"\b(?:porn\w*|escort\s+service|camgirl|onlyfans\s+leak|hentai|"
     r"sex\s+(?:cam|chat|shop|video)|風俗|エロ)"),
    ("pharma",
     ("viagra", "cialis", "tadalafil", "sildenafil", "kamagra", "xanax", "tramadol",
      "oxycodone", "adderall", "рецепт"),
     r"\b(?:viagra|cialis|tadalafil|sildenafil|kamagra|buy\s+(?:xanax|tramadol|oxycodone|adderall)|"
     r"без\s+рецепт)"),
    ("crypto",
     ("crypto", "bitcoin", "usdt", "binary options"),
     r"\b(?:crypto\s+(?:airdrop|pump)|free\s+bitcoin|usdt\s+recovery|"
     r"recover\s+(?:your\s+)?(?:lost|stolen)\s+crypto|binary\s+options)"),
    ("essay_mill",
     ("essay", "homework", "dissertation", "coursework"),
     r"\b(?:write\s+my\s+essay|essay\s+writing\s+service|do\s+my\s+homework|"
     r"buy\s+(?:essays?|dissertation|coursework))"),
    ("replica", ("replica", "rolex"),
     r"\b(?:replica\s+(?:watch|handbag|bag)|fake\s+rolex|aaa\s+replica)"),
    ("seo_services",
     ("backlink", "followers", "likes", "pbn", "guest post", "seo service"),
     r"\b(?:buy\s+(?:backlinks?|followers?|likes?)|pbn\s+links?|guest\s+post\s+service|"
     r"cheap\s+seo\s+service)"),
    ("loans", ("payday", "credit check", "cash advance"),
     r"\b(?:payday\s+loans?|no\s+credit\s+check\s+loan|instant\s+cash\s+advance)"),
]

SPAM = build(SPAM_SPEC, re.I | re.M)

# A category fires only when its vocabulary appears at least this many times.
# One mention is normally incidental — a car service advertising "casino
# trips", a charity's "casino night", a news site reporting on a court case —
# whereas a hijacked or purpose-built spam page repeats it throughout.
SPAM_MIN_HITS = 2


@register
class AbuseExtractor(RowExtractor):
    NAME = "abuse"
    TRACK = "E"
    FIELDS = {
        "injection_severity": I32,
        "injection_matches": LIST_STR,
        "for_sale": BOOL,
        "for_sale_marketplace": STR,
        "spam_categories": LIST_STR,
        "n_spam_categories": I32,
    }

    def extract(self, rec: Record) -> dict:
        text = rec.scan
        low = rec.scan_lower

        matches, severity = INJECTION.matches_with_level(text, low)

        head_low = rec.head_lower
        for_sale = any(a in head_low for a in _FOR_SALE_ANCHORS) and bool(_FOR_SALE.search(rec.head))
        marketplace = ""
        if for_sale:
            for name, pat in _MARKETPLACES:
                if pat.search(rec.head):
                    marketplace = name
                    break
            marketplace = marketplace or "other"

        spam = SPAM.matches_min(text, low, SPAM_MIN_HITS)

        return {
            "injection_severity": severity,
            "injection_matches": matches,
            "for_sale": for_sale,
            "for_sale_marketplace": marketplace,
            "spam_categories": spam,
            "n_spam_categories": len(spam),
        }

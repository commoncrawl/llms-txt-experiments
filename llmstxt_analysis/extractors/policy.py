"""Track D — AI-usage policy, rights declarations and named-crawler stances.

Three questions:

* In what *dialect* is the policy expressed (prose / YAML / robots.txt)?
* Which *directives* does it contain (no-training, attribution, licence, ...)?
* Which *crawlers* does it name, and does it allow or deny each?

All matching is literal/regex over a bounded window (``Record.scan``), through
the anchored :mod:`llmstxt_analysis.lexicon` so that the common case (no policy
language at all) costs a handful of substring scans.
"""
from __future__ import annotations

import re

from ..lexicon import build
from ..record import Record
from ..registry import BOOL, I32, LIST_STR, STR, RowExtractor, register

# (slug, anchors, pattern)
DIRECTIVE_SPEC: list[tuple] = [
    ("no_training", ("train", "opt-out", "opt out"),
     r"\b(?:no(?:t)?\s+(?:be\s+)?(?:used?|permitted)\s+for\s+(?:ai\s+|model\s+)?train|"
     r"do\s+not\s+train|may\s+not\s+be\s+used\s+to\s+train|"
     r"prohibit(?:ed|s)?\s+(?:from\s+)?(?:ai\s+|model\s+)?train|"
     r"training\s*[:=]\s*(?:deny|disallow|false|no|prohibited)|"
     r"no[-_ ]?ai[-_ ]?training|opt(?:ed)?[-_ ]?out\s+of\s+train)"),
    ("allow_training", ("train",),
     r"\b(?:training\s*[:=]\s*(?:allow|true|yes|permitted)|"
     r"(?:may|can|are\s+welcome\s+to)\s+(?:be\s+)?use[d]?\s+(?:this\s+)?(?:content\s+)?"
     r"(?:for|to)\s+train)"),
    ("disallow_training_header", ("disallow-training", "disallow_training", "disallow training"),
     r"^\s*disallow[-_ ]training\s*:"),
    ("attribution_required", ("attribut",),
     r"\b(?:attribut\w+\s+(?:is\s+)?(?:required|mandatory|expected)|"
     r"must\s+(?:be\s+)?attribut|please\s+attribute|"
     r"require[sd]?\s+attribution|with\s+attribution\s+to)"),
    ("citation_required", ("cite", "citation", "citing", "link back"),
     r"\b(?:(?:please\s+)?cite\s+(?:the\s+)?(?:canonical\s+)?(?:url|source|us|this)|"
     r"citation\s+(?:is\s+)?(?:required|format)|must\s+(?:be\s+)?cite|"
     r"when\s+citing|link\s+back\s+to\s+the\s+(?:original|source))"),
    ("noncommercial_only", ("commercial",),
     r"\b(?:non[-\s]?commercial\s+use\s+only|not\s+for\s+commercial\s+use|"
     r"commercial\s+use\s+(?:is\s+)?(?:prohibited|not\s+permitted|requires))"),
    ("commercial_license", ("licen",),
     r"\b(?:commercial\s+licen[sc]|licen[sc]ing\s+enquir|"
     r"contact\s+us\s+(?:for|to\s+discuss)\s+licen)"),
    ("license_ref", ("cc-by", "cc by", "creativecommons", "cc0", "mit licen", "apache",
                     "gpl", "rsl", "really simple licensing"),
     r"\b(?:CC[ -]BY(?:[- ](?:SA|NC|ND)){0,2}|creativecommons\.org|CC0\b|"
     r"MIT\s+Licen[sc]e|Apache[- ]2\.0|GPL(?:v[23])?\b|"
     r"RSL\b|really\s+simple\s+licensing)"),
    ("copyright_notice", ("©", "(c)", "copyright", "rights reserved"),
     r"(?:©|\(c\)\s*\d{4}|copyright\s+(?:©\s*)?\d{4}|all rights reserved)"),
    ("contact_for_licensing", ("licensing@", "licencing@", "permissions@", "rights@",
                               "copyright@", "legal@"),
     r"\b(?:licen[sc]ing@|permissions@|rights@|copyright@|legal@)"),
    ("respect_robots", ("robots.txt",),
     r"\b(?:see|respect|refer to|per)\s+(?:our\s+)?robots\.txt"),
    ("rate_limit", ("rate limit", "rate-limit", "ratelimit", "crawl-delay", "crawl delay",
                    "requests per"),
     r"\b(?:rate[-\s]?limit\w*|crawl[-\s]?delay|requests?\s+per\s+(?:second|minute))"),
    ("jurisdiction", ("jurisdiction",), r"^\s*jurisdiction\s*:"),
    ("no_derivatives", ("derivative",),
     r"\b(?:no\s+derivative|derivative\s+works?\s+(?:are\s+)?(?:not\s+)?prohibit)"),
    ("no_verbatim", ("verbatim", "quote", "in full"),
     r"\b(?:do\s+not\s+(?:reproduce|quote)\s+(?:verbatim|in\s+full)|"
     r"no\s+verbatim\s+(?:reproduction|reproduc\w+|quot))"),
    ("ai_summary_ok", ("summari",),
     r"\b(?:summari[sz](?:e|ing|ation)\s+(?:is\s+)?(?:allowed|permitted|encouraged|welcome))"),
    ("human_review_notice", ("verify with", "confirm with"),
     r"\b(?:verify\s+with|confirm\s+with)\s+(?:a\s+)?(?:human|our\s+team)"),
    ("paywall_notice", ("paywall", "subscriber", "members only", "members-only"),
     r"\b(?:paywall|subscriber[-\s]only|members?[-\s]only\s+content)"),
    ("medical_legal_disclaimer", ("substitute for", "intended as"),
     r"\b(?:not\s+(?:a\s+substitute\s+for|intended\s+as)\s+(?:professional\s+)?"
     r"(?:medical|legal|financial)\s+(?:advice)?)"),
]

DIRECTIVES = build(DIRECTIVE_SPEC, re.I | re.M)

_POLICY_HEADING = re.compile(
    r"^#{1,4}\s*(?:[\w\s&/-]{0,30})?"
    r"(?:ai[\s-]?(?:usage|polic|guideline|access|permission)|usage\s+(?:polic|guideline|terms|rights)|"
    r"licen[sc]|permissions?|attribution|terms\s+of\s+use|copyright|rights|"
    r"for\s+(?:ai|llm|agent)s?\b|(?:notes?|instructions?)\s+for\s+(?:ai|llm|agent|model)s?)",
    re.I | re.M,
)

_ROBOTS_LINE = re.compile(r"^\s*(?:user-agent|disallow|allow|crawl-delay)\s*:", re.I | re.M)
_YAML_POLICY = re.compile(
    r"^\s*(?:permissions|agents|scope|policy|policies|access|training|usage_policy|ai)\s*:\s*$",
    re.I | re.M,
)

NAMED_BOTS: dict[str, re.Pattern] = {
    name: re.compile(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", re.I)
    for name in [
        "GPTBot", "ChatGPT-User", "OAI-SearchBot",
        "ClaudeBot", "anthropic-ai", "claude-web", "Claude-User", "Claude-SearchBot",
        "CCBot",
        "Google-Extended", "Googlebot", "GoogleOther",
        "PerplexityBot", "Perplexity-User",
        "Applebot-Extended", "Applebot",
        "Bytespider", "Amazonbot", "Meta-ExternalAgent", "FacebookBot", "meta-externalfetcher",
        "YouBot", "Bingbot", "msnbot", "DuckAssistBot",
        "cohere-ai", "Diffbot", "Timpibot", "ImagesiftBot", "AI2Bot", "Omgilibot",
        "PetalBot", "YandexBot", "Bytedance", "MistralAI-User", "GrokBot", "xAI",
        "SemrushBot", "AhrefsBot",
    ]
}
# Lowercase anchors so the (expensive) per-bot regex only runs when the literal
# name is present at all.
_BOT_ANCHORS = {name: name.lower() for name in NAMED_BOTS}

_DENY_MARK = re.compile(
    r"\b(?:false|deny|denied|disallow\w*|block\w*|forbid\w*|prohibit\w*|no\b|"
    r"not\s+(?:allowed|permitted)|excluded?|opt[-\s]?out)\b",
    re.I,
)
_ALLOW_MARK = re.compile(r"\b(?:true|allow\w*|permit\w*|yes\b|welcome|enabled?|grant\w*)\b", re.I)

_CONTACT_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}")


@register
class PolicyExtractor(RowExtractor):
    NAME = "policy"
    TRACK = "D"
    FIELDS = {
        "policy_dialect": STR,
        "policy_directives": LIST_STR,
        "n_policy_directives": I32,
        "has_policy_heading": BOOL,
        "training_stance": STR,
        "named_bots": LIST_STR,
        "bots_allowed": LIST_STR,
        "bots_denied": LIST_STR,
        "n_named_bots": I32,
        "has_contact_email": BOOL,
        "has_any_policy": BOOL,
    }

    def extract(self, rec: Record) -> dict:
        text = rec.scan
        low = rec.scan_lower

        directives = DIRECTIVES.matches(text, low)
        dset = set(directives)

        if "no_training" in dset or "disallow_training_header" in dset:
            stance = "deny"
        elif "noncommercial_only" in dset or "commercial_license" in dset:
            stance = "conditional"
        elif "allow_training" in dset:
            stance = "allow"
        else:
            stance = "none"

        has_heading = ("#" in text) and bool(_POLICY_HEADING.search(text))

        robots_like = ("user-agent" in low or "disallow" in low) and len(_ROBOTS_LINE.findall(text)) >= 2
        yaml_like = bool(_YAML_POLICY.search(text))
        if robots_like and yaml_like:
            dialect = "mixed"
        elif robots_like:
            dialect = "robots"
        elif yaml_like:
            dialect = "yaml"
        elif directives or has_heading:
            dialect = "prose"
        else:
            dialect = "none"

        named, allowed, denied = [], [], []
        if "bot" in low or "crawl" in low or "agent" in low or "gpt" in low:
            for name, pat in NAMED_BOTS.items():
                if _BOT_ANCHORS[name] not in low:
                    continue
                m = pat.search(text)
                if not m:
                    continue
                named.append(name)
                # Verdict from the rest of the line the bot is named on.
                ls = text.rfind("\n", 0, m.start()) + 1
                le = text.find("\n", m.end())
                line = text[ls : le if le != -1 else len(text)]
                rest = line[: m.start() - ls] + " " + line[m.end() - ls :]
                if _DENY_MARK.search(rest):
                    denied.append(name)
                elif _ALLOW_MARK.search(rest):
                    allowed.append(name)

        return {
            "policy_dialect": dialect,
            "policy_directives": directives,
            "n_policy_directives": len(directives),
            "has_policy_heading": has_heading,
            "training_stance": stance,
            "named_bots": named,
            "bots_allowed": allowed,
            "bots_denied": denied,
            "n_named_bots": len(named),
            "has_contact_email": ("@" in rec.head) and bool(_CONTACT_EMAIL.search(rec.head)),
            "has_any_policy": bool(directives) or has_heading or dialect in ("robots", "yaml", "mixed"),
        }

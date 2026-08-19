"""Pull real records back out of the shards for manual inspection."""
from __future__ import annotations

import sys
from pathlib import Path

from .corpus import DEFAULT_CONFIG, fetch_records
from .derive import load

SHOW_COLUMNS = [
    "doc_kind", "conformance_level", "generator_id", "generator_version",
    "lang", "lang_source", "policy_dialect", "training_stance",
    "injection_severity", "for_sale", "n_links", "n_tokens",
]


def run(derived: str | Path, source: str | Path, expr: str, n: int = 10,
        chars: int = 1200, seed: int = 0, out: str | Path | None = None,
        columns: list[str] | None = None, config: str = DEFAULT_CONFIG) -> str:
    # Empties are excluded from the analysis frame but must stay inspectable.
    df = load(derived, drop_empty=False)
    try:
        sub = df.query(expr) if expr else df
    except Exception as exc:  # pragma: no cover - user input
        raise SystemExit(f"bad --filter expression: {exc}")

    total = len(sub)
    if total == 0:
        return f"no records match: {expr}\n"
    sub = sub.sample(min(n, total), random_state=seed)

    locators = list(zip(sub["shard"].astype(int), sub["rg"].astype(int), sub["rg_row"].astype(int)))
    records = fetch_records(source, locators, config)

    cols = columns or SHOW_COLUMNS
    cols = [c for c in cols if c in sub.columns]
    parts = [f"filter: {expr or '(all)'}", f"matches: {total} / {len(df)} ({100*total/len(df):.3f}%)",
             f"showing: {len(sub)}", ""]
    for (_, row), key in zip(sub.iterrows(), locators):
        rec = records.get(key)
        parts.append("=" * 100)
        parts.append(f"URL: {row['url']}")
        parts.append("  " + " | ".join(f"{c}={row[c]!r}" for c in cols))
        for lc in ("conf_flags_str", "policy_directives_str", "named_bots_str",
                   "injection_matches_str", "spam_categories_str"):
            if lc in row and row[lc]:
                parts.append(f"  {lc[:-4]}: {row[lc]}")
        parts.append("-" * 100)
        parts.append(rec.content[:chars] if rec else "<record not found>")
        parts.append("")
    text = "\n".join(parts)

    if out:
        Path(out).write_text(text)
        print(f"wrote {out} ({len(sub)} records)")
    else:
        sys.stdout.write(text)
    return text

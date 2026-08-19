"""Corpus-level derived columns, and the cache that makes them cheap to reuse.

Everything here needs global context (duplicate counts) or a join across
extractor outputs, so it cannot live in a per-record extractor.

Deriving the frame for the full corpus costs about a minute — fine once, but
not on every report iteration. ``load`` therefore memoises the derived frame to
``<derived>/cache/frame-<fingerprint>.parquet``. The fingerprint covers the
feature part files (name, size, mtime), the template threshold, and a version
constant bumped whenever the derivation logic changes, so a stale cache can
never be served.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

# Bump whenever add_derived() changes what it produces.
CACHE_VERSION = 4

# ccTLD -> the language you would expect a site under it to be written in.
# Only used to flag *implausible* content, so English is always acceptable and
# multi-lingual countries list all their languages.
TLD_LANG: dict[str, set[str]] = {
    "de": {"deu"}, "at": {"deu"}, "ch": {"deu", "fra", "ita"},
    "fr": {"fra"}, "be": {"nld", "fra", "deu"}, "nl": {"nld"},
    "es": {"spa", "cat", "eus", "glg"}, "it": {"ita"}, "pt": {"por"}, "br": {"por"},
    "pl": {"pol"}, "cz": {"ces"}, "sk": {"slk"}, "hu": {"hun"}, "ro": {"ron"},
    "gr": {"ell"}, "se": {"swe"}, "no": {"nor", "nno", "nob"}, "dk": {"dan"},
    "fi": {"fin", "swe"}, "ru": {"rus"}, "ua": {"ukr", "rus"}, "tr": {"tur"},
    "jp": {"jpn"}, "cn": {"zho"}, "tw": {"zho"}, "kr": {"kor"},
    "vn": {"vie"}, "th": {"tha"}, "id": {"ind"}, "il": {"heb"},
    "mx": {"spa"}, "ar": {"spa"}, "cl": {"spa"}, "co": {"spa"},
    "lt": {"lit"}, "lv": {"lav"}, "ee": {"est"}, "si": {"slv"}, "hr": {"hrv"},
    "bg": {"bul"}, "rs": {"srp"}, "ir": {"fas"}, "sa": {"ara"}, "eg": {"ara"},
}
LINGUA_FRANCA = {"eng", "und", ""}

DEFAULT_TEMPLATE_MIN_CLUSTER = 50


def load_features(derived: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read the features table produced by ``extract``."""
    path = Path(derived)
    if path.name != "features" and (path / "features").exists():
        path = path / "features"
    tbl = ds.dataset(str(path), format="parquet").to_table(columns=columns)
    return tbl.to_pandas()


def add_derived(df: pd.DataFrame, template_min_cluster: int = DEFAULT_TEMPLATE_MIN_CLUSTER) -> pd.DataFrame:
    """Add corpus-level columns in place and return the frame.

    Callers that loaded only a subset of columns (``topics``) get only the
    derived columns their subset supports; the rest are skipped rather than
    raising, so one column list does not have to satisfy every consumer.
    """
    have = set(df.columns)

    # An empty 200 response is a real thing to count, but it is not a document
    # to analyse: every empty body hashes to the same skeleton, so left in they
    # form a spurious template cluster (the third largest in this corpus) and
    # inflate both the templated share and the unknown-generator share.
    # ``load(drop_empty=True)`` removes them from the analysis frame.
    if "doc_kind" in have:
        df["is_empty"] = df["doc_kind"] == "empty"

    # --- duplication / templates -----------------------------------------
    if "content_sha1" in have:
        df["dup_count"] = df.groupby("content_sha1")["content_sha1"].transform("size")
        df["is_exact_dup"] = df["dup_count"] > 1
    df["template_cluster_size"] = df.groupby("skeleton_sha1")["skeleton_sha1"].transform("size")
    df["is_template"] = (df["template_cluster_size"] >= template_min_cluster) | (
        df["generator_source"] != "none"
    )
    df["is_human_authored"] = (~df["is_template"]) & df["doc_kind"].isin(
        ["markdown", "yaml_frontmatter", "plain"]
    )

    # --- links / abuse ----------------------------------------------------
    if {"offsite_ratio", "n_offsite_domains"} <= have:
        df["offsite_dominant"] = (df["offsite_ratio"] > 0.5) & (df["n_offsite_domains"] >= 5)
    if {"tld", "lang"} <= have:
        expected = [TLD_LANG.get(t) for t in df["tld"]]
        df["lang_mismatch"] = [
            exp is not None and lg not in LINGUA_FRANCA and lg not in exp and not tmpl
            for exp, lg, tmpl in zip(expected, df["lang"], df["is_template"])
        ]
    if "n_spam_categories" in have:
        df["has_spam"] = df["n_spam_categories"] > 0
    if "llm_artefacts" in have:
        df["has_llm_artefacts"] = df["llm_artefacts"].map(
            lambda v: v is not None and len(v) > 0)
    if {"injection_severity", "has_spam", "offsite_dominant", "lang_mismatch"} <= set(df.columns):
        df["is_suspicious"] = (
            (df["injection_severity"] >= 2)
            | df["has_spam"]
            | (df["offsite_dominant"] & df["lang_mismatch"])
        )

    # --- convenience ------------------------------------------------------
    if "is_full" in have:
        df["file_kind"] = df["is_full"].map({True: "llms-full.txt", False: "llms.txt"})
    if "n_links" in have:
        df["has_links"] = df["n_links"] > 0
    for col in ("conf_flags", "policy_directives", "named_bots", "bots_allowed",
                "bots_denied", "spam_categories", "injection_matches", "llm_artefacts"):
        if col in have:
            df[col + "_str"] = df[col].map(lambda v: ",".join(v) if v is not None else "")
    return df


def features_dir(derived: str | Path) -> Path:
    path = Path(derived)
    return path if path.name == "features" else path / "features"


def fingerprint(derived: str | Path, template_min_cluster: int) -> str:
    """Identity of a derived frame: its inputs plus the logic that made it."""
    parts = sorted(features_dir(derived).glob("*.parquet"))
    h = hashlib.sha1()
    h.update(f"v{CACHE_VERSION}|t{template_min_cluster}|".encode())
    for p in parts:
        st = p.stat()
        h.update(f"{p.name}:{st.st_size}:{st.st_mtime_ns}|".encode())
    return h.hexdigest()[:16]


def cache_path(derived: str | Path, template_min_cluster: int) -> Path:
    base = Path(derived)
    base = base.parent if base.name == "features" else base
    return base / "cache" / f"frame-{fingerprint(derived, template_min_cluster)}.parquet"


def cache_info(derived: str | Path, template_min_cluster: int = DEFAULT_TEMPLATE_MIN_CLUSTER) -> dict:
    p = cache_path(derived, template_min_cluster)
    other = sorted(q for q in p.parent.glob("frame-*.parquet") if q != p) if p.parent.exists() else []
    return {
        "path": str(p),
        "fresh": p.exists(),
        "size_mb": round(p.stat().st_size / 1e6, 1) if p.exists() else 0.0,
        "stale_entries": [q.name for q in other],
    }


def clear_cache(derived: str | Path) -> int:
    base = Path(derived)
    base = base.parent if base.name == "features" else base
    removed = 0
    for q in (base / "cache").glob("frame-*.parquet"):
        q.unlink()
        removed += 1
    return removed


def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The analysis frame: responses that actually contain a document."""
    return df[~df["is_empty"]] if "is_empty" in df.columns else df


def load(derived: str | Path, template_min_cluster: int = DEFAULT_TEMPLATE_MIN_CLUSTER,
         use_cache: bool = True, verbose: bool = False,
         drop_empty: bool = True) -> pd.DataFrame:
    """Features plus derived columns, memoised to parquet.

    Empty 200 responses are excluded by default: they are counted in the
    corpus overview but analysing them skews every distribution. Pass
    ``drop_empty=False`` to get them back — ``spotcheck`` does, so they stay
    inspectable.

    Set ``use_cache=False`` to force recomputation (the cache is keyed on the
    inputs, so this is only needed when debugging the derivation itself). The
    cache always holds the *complete* frame; the filter is applied on return,
    so one cache entry serves both callers.
    """
    if not use_cache:
        df = add_derived(load_features(derived), template_min_cluster)
        return drop_empty_rows(df) if drop_empty else df

    cache = cache_path(derived, template_min_cluster)
    if cache.exists():
        t0 = time.perf_counter()
        df = pd.read_parquet(cache)
        if verbose:
            print(f"derived frame from cache in {time.perf_counter()-t0:.1f}s ({len(df):,} rows)")
        return drop_empty_rows(df) if drop_empty else df

    t0 = time.perf_counter()
    df = add_derived(load_features(derived), template_min_cluster)
    cache.parent.mkdir(parents=True, exist_ok=True)
    # Drop superseded entries: only one input state is ever current.
    for q in cache.parent.glob("frame-*.parquet"):
        q.unlink()
    tmp = cache.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(cache)
    (cache.parent / "cache.json").write_text(json.dumps({
        "version": CACHE_VERSION,
        "template_min_cluster": template_min_cluster,
        "rows": len(df),
        "columns": list(df.columns),
        "built_seconds": round(time.perf_counter() - t0, 1),
    }, indent=1))
    if verbose:
        print(f"derived frame built and cached in {time.perf_counter()-t0:.1f}s ({len(df):,} rows)")
    return drop_empty_rows(df) if drop_empty else df

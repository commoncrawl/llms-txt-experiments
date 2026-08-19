"""Standalone chart files for the blog post.

``charts.py`` draws SVG *fragments* for the report: they carry no XML namespace,
they inherit ``.ch-*`` styling from the report's stylesheet, and their fills are
``var(--series-N)`` so the report can restyle a plot without redrawing it. All
three of those make a fragment unusable as a ``.svg`` file on its own — a
standalone file is parsed as XML and rejected without ``xmlns``, unstyled text
falls back to black serif, unstyled ``<line>`` has ``stroke:none`` and vanishes,
and an unresolved ``var()`` fill paints black. librsvg in particular does not
implement CSS custom properties at all, so a ``var()`` fill rasterises black.

This module bakes the report's resolved values into each file, so the plots stay
on the Common Crawl palette while standing alone.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from . import charts

# The literal values the report's tokens resolve to. Kept in step with the
# ``:root`` block in ``report.py`` by hand; there are eight of them and they are
# the style guide's named accents, so they change about as often as the logo.
TOKENS = {
    "--series-1": "#2e5f8a",   # sapphire / CC blue
    "--series-2": "#846730",   # topaz
    "--series-3": "#2b674f",   # emerald
    "--series-4": "#5b437f",   # amethyst
    "--grid": "#e2e8f0",
    "--axis": "#cbd5e1",
    "--cc-text": "#152a47",
    "--cc-text-subtle": "#475569",
    "--cc-text-muted": "#94a3b8",
}

# No @font-face here: inlining the two vendored woff2 files as data URIs would
# add ~120 KB to every figure, and both families degrade to a sane system font.
_BODY = "'Libre Franklin','Segoe UI',system-ui,-apple-system,sans-serif"
_MONO = "'IBM Plex Mono',ui-monospace,'SFMono-Regular',Menlo,monospace"

STYLE = f"""<style>
text{{font-family:{_BODY}}}
.ch-title{{fill:{TOKENS['--cc-text']};font-size:13px;font-weight:700}}
.ch-label{{fill:{TOKENS['--cc-text-subtle']};font-size:12px}}
.ch-value{{fill:{TOKENS['--cc-text-muted']};font-size:11px;dominant-baseline:auto;
  font-family:{_MONO};font-variant-numeric:tabular-nums}}
.ch-tick{{fill:{TOKENS['--cc-text-muted']};font-size:10px;font-family:{_MONO}}}
.ch-axis{{stroke:{TOKENS['--axis']};stroke-width:1}}
.ch-grid{{stroke:{TOKENS['--grid']};stroke-width:1}}
</style>"""

_VIEWBOX = re.compile(r'viewBox="0 0 ([\d.]+) ([\d.]+)"')
_VAR = re.compile(r"var\((--[a-z0-9-]+)\)")


def resolve_vars(svg: str) -> str:
    """Substitute every ``var(--token)`` for its literal value."""
    return _VAR.sub(lambda m: TOKENS.get(m.group(1), "#000"), svg)


def standalone(fragment: str, *, background: str = "#fff") -> str:
    """Turn a ``charts`` fragment into a complete, self-contained SVG document.

    The backdrop matters: the labels are muted greys chosen for a white card, and
    a transparent figure dropped on a dark page background is unreadable.
    """
    if not fragment.startswith("<svg"):
        raise ValueError("expected an SVG fragment from llmstxt_analysis.charts")
    end = fragment.index(">") + 1
    head, body = fragment[:end], fragment[end:]

    m = _VIEWBOX.search(head)
    if not m:
        raise ValueError("fragment has no viewBox to size the document from")
    width, height = m.group(1), m.group(2)

    # A fixed width, not 100%: standalone files are sized by their own attributes.
    head = head.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    head = head.replace('width="100%"', f'width="{width}"')
    backdrop = f'<rect width="100%" height="100%" fill="{background}"/>'
    return resolve_vars(f"{head}{STYLE}{backdrop}{body}") + "\n"


# Titles live in the markdown heading above each figure, not inside the image:
# the style guide's .cc-chart component puts them there, so every chart is drawn
# with title="".
CHARTS: dict[str, tuple[tuple[str, ...], dict]] = {
    "funnel-status": (("index", "status_chart"), {"label_width": 200}),
    "funnel-mime": (("index", "mime_chart"), {"label_width": 200}),
    "generators": (("B", "generator_chart"), {"label_width": 170}),
    "conformance": (("A", "conformance_chart"), {"label_width": 230}),
    "policy-dialects": (("D", "dialect_chart"), {"label_width": 170}),
}


def _dig(stats: dict, path: tuple[str, ...]):
    node = stats
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def to_png(svg_path: Path, *, scale: int = 2) -> Path | None:
    """Rasterise with rsvg-convert, or return None if it isn't installed."""
    if not shutil.which("rsvg-convert"):
        return None
    svg = svg_path.read_text()
    m = _VIEWBOX.search(svg)
    width = int(float(m.group(1))) if m else charts.WIDTH
    png_path = svg_path.with_suffix(".png")
    subprocess.run(
        ["rsvg-convert", "-w", str(width * scale), "-o", str(png_path), str(svg_path)],
        check=True, capture_output=True,
    )
    return png_path


def write(stats: dict, outdir: str | Path, *, png: bool = True) -> list[Path]:
    """Write one standalone SVG (and optionally PNG) per chart in ``CHARTS``."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    raster = png and shutil.which("rsvg-convert") is not None
    if png and not raster:
        print("warning: rsvg-convert not found; writing SVG only")

    for name, (path, kw) in CHARTS.items():
        spec = _dig(stats, path)
        if not spec or not spec.get("labels"):
            print(f"skip {name}: stats has no {'.'.join(path)}")
            continue
        svg_path = out / f"{name}.svg"
        svg_path.write_text(standalone(charts.from_series(spec, title="", **kw)))
        written.append(svg_path)
        print(f"wrote {svg_path}  ({len(spec['labels'])} bars)")
        if raster:
            written.append(to_png(svg_path))
    return written

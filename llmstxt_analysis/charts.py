"""Inline SVG charts for the standalone report.

Hand-rolled rather than matplotlib so that every colour is a CSS custom
property from the Common Crawl palette, restyleable without regenerating the
figures — which a rasterised or colour-baked chart cannot offer. No external
requests, no scripts.

Design rules applied (see the dataviz method): one axis, single-hue marks for a
single series, categorical slots in fixed order for multi-series, 4px rounded
data-ends anchored to the baseline, a 2px surface gap between adjacent bars,
recessive axes, selective direct labels, and a table beside every chart.
"""
from __future__ import annotations

from html import escape

SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)"]

# Intrinsic drawing width. The rendered SVG is capped at this in CSS rather
# than stretched to the container: an SVG scaled up by its viewBox scales its
# text too, which is how a 12px axis label ends up rendering at 19px on a wide
# screen. Drawing at the width the chart is actually displayed keeps every
# label at the size the stylesheet says.
WIDTH = 900


def _fmt(v: float) -> str:
    if v >= 1_000_000_000:
        return f"{v/1e9:.1f}B"
    if v >= 1_000_000:
        return f"{v/1e6:.1f}M"
    if v >= 10_000:
        return f"{v/1e3:.0f}k"
    if isinstance(v, float) and not v.is_integer():
        return f"{v:.2f}"
    return f"{int(v):,}"


def _bar_path(x: float, y: float, w: float, h: float, r: float = 4.0) -> str:
    """Bar anchored at ``x`` with only the data end (right) rounded."""
    r = max(0.0, min(r, w, h / 2))
    if r <= 0.5:
        return f"M{x},{y}h{w}v{h}h{-w}z"
    return (
        f"M{x},{y}h{w - r}a{r},{r} 0 0 1 {r},{r}"
        f"v{h - 2*r}a{r},{r} 0 0 1 {-r},{r}h{-(w - r)}z"
    )


def hbar(
    labels: list[str],
    values: list[float],
    *,
    unit: str = "",
    label_width: int = 190,
    bar_h: int = 19,
    gap: int = 8,
    title: str = "",
    color: str = SERIES[0],
) -> str:
    """Horizontal bars, one series, direct-labelled."""
    if not labels:
        return ""
    n = len(labels)
    pad_t, pad_b, pad_r = (26 if title else 8), 6, 66
    height = pad_t + n * (bar_h + gap) - gap + pad_b
    width = WIDTH
    plot_w = width - label_width - pad_r
    vmax = max(values) or 1

    out = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" width="100%" '
        f'height="{height}" role="img" preserveAspectRatio="xMinYMin meet" '
        f'aria-label="{escape(title or "bar chart")}">'
    ]
    if title:
        out.append(f'<text class="ch-title" x="0" y="14">{escape(title)}</text>')
    out.append(
        f'<line class="ch-axis" x1="{label_width}" y1="{pad_t - 4}" '
        f'x2="{label_width}" y2="{height - pad_b}"/>'
    )
    for i, (lab, val) in enumerate(zip(labels, values)):
        y = pad_t + i * (bar_h + gap)
        w = max(1.0, plot_w * (val / vmax))
        out.append(
            f'<text class="ch-label" x="{label_width - 8}" y="{y + bar_h*0.75:.1f}" '
            f'text-anchor="end">{escape(str(lab)[:34])}</text>'
        )
        out.append(
            f'<path d="{_bar_path(label_width + 1, y, w, bar_h - 2)}" fill="{color}">'
            f"<title>{escape(str(lab))}: {_fmt(val)}{escape(unit)}</title></path>"
        )
        out.append(
            f'<text class="ch-value" x="{label_width + w + 7:.1f}" '
            f'y="{y + bar_h*0.75:.1f}">{_fmt(val)}{escape(unit)}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def grouped_hbar(
    labels: list[str],
    series: list[tuple[str, list[float]]],
    *,
    unit: str = "",
    label_width: int = 190,
    bar_h: int = 15,
    title: str = "",
) -> str:
    """Two or three series per category. Legend is mandatory above the plot."""
    if not labels or not series:
        return ""
    n, k = len(labels), len(series)
    group_gap, inner_gap = 12, 2
    pad_t, pad_b, pad_r = (46 if title else 30), 6, 66
    group_h = k * bar_h + (k - 1) * inner_gap
    height = pad_t + n * (group_h + group_gap) - group_gap + pad_b
    width = WIDTH
    plot_w = width - label_width - pad_r
    vmax = max(max(v) for _, v in series) or 1

    out = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" width="100%" '
        f'height="{height}" role="img" preserveAspectRatio="xMinYMin meet" '
        f'aria-label="{escape(title or "grouped bar chart")}">'
    ]
    if title:
        out.append(f'<text class="ch-title" x="0" y="14">{escape(title)}</text>')
    lx = 0
    for i, (name, _) in enumerate(series):
        ly = pad_t - 12
        out.append(f'<rect x="{lx}" y="{ly-8}" width="10" height="10" rx="2" fill="{SERIES[i]}"/>')
        out.append(f'<text class="ch-label" x="{lx+15}" y="{ly}" text-anchor="start">{escape(name)}</text>')
        lx += 22 + 7 * len(name)
    for i, lab in enumerate(labels):
        gy = pad_t + i * (group_h + group_gap)
        out.append(
            f'<text class="ch-label" x="{label_width - 8}" y="{gy + group_h/2 + 4:.1f}" '
            f'text-anchor="end">{escape(str(lab)[:34])}</text>'
        )
        for j, (name, vals) in enumerate(series):
            val = vals[i]
            y = gy + j * (bar_h + inner_gap)
            w = max(1.0, plot_w * (val / vmax))
            out.append(
                f'<path d="{_bar_path(label_width + 1, y, w, bar_h - 2)}" fill="{SERIES[j]}">'
                f"<title>{escape(name)} — {escape(str(lab))}: {_fmt(val)}{escape(unit)}</title></path>"
            )
            out.append(
                f'<text class="ch-value" x="{label_width + w + 7:.1f}" '
                f'y="{y + bar_h*0.78:.1f}">{_fmt(val)}{escape(unit)}</text>'
            )
    out.append(
        f'<line class="ch-axis" x1="{label_width}" y1="{pad_t - 6}" '
        f'x2="{label_width}" y2="{height - pad_b}"/>'
    )
    out.append("</svg>")
    return "".join(out)


def loghist(edges: list[float], counts: list[int], *, title: str = "",
            xlabel: str = "", color: str = SERIES[0]) -> str:
    """Vertical histogram over log-spaced bins (documents per size bucket)."""
    if not edges or not counts:
        return ""
    width, height = WIDTH, 260
    pad_l, pad_r, pad_t, pad_b = 44, 12, (28 if title else 10), 38
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n = len(counts)
    bw = plot_w / n
    cmax = max(counts) or 1

    out = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" width="100%" '
        f'height="{height}" role="img" preserveAspectRatio="xMinYMin meet" '
        f'aria-label="{escape(title or "histogram")}">'
    ]
    if title:
        out.append(f'<text class="ch-title" x="0" y="14">{escape(title)}</text>')
    for frac in (0.25, 0.5, 0.75, 1.0):
        y = pad_t + plot_h * (1 - frac)
        out.append(f'<line class="ch-grid" x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}"/>')
        out.append(f'<text class="ch-tick" x="{pad_l-6}" y="{y+3:.1f}" text-anchor="end">'
                   f"{_fmt(cmax*frac)}</text>")
    for i, c in enumerate(counts):
        h = plot_h * (c / cmax)
        x = pad_l + i * bw
        y = pad_t + plot_h - h
        lo, hi = edges[i], edges[i + 1]
        out.append(
            f'<rect x="{x+1:.1f}" y="{y:.1f}" width="{max(bw-2,1):.1f}" height="{max(h,0.5):.1f}" '
            f'rx="2" fill="{color}"><title>{_fmt(lo)}–{_fmt(hi)}: {_fmt(c)} documents</title></rect>'
        )
    out.append(f'<line class="ch-axis" x1="{pad_l}" y1="{pad_t+plot_h}" x2="{width-pad_r}" '
               f'y2="{pad_t+plot_h}"/>')
    step = max(1, n // 6)
    for i in range(0, n + 1, step):
        x = pad_l + i * bw
        out.append(f'<text class="ch-tick" x="{x:.1f}" y="{pad_t+plot_h+15}" text-anchor="middle">'
                   f"{_fmt(edges[min(i, len(edges)-1)])}</text>")
    if xlabel:
        out.append(f'<text class="ch-tick" x="{pad_l+plot_w/2:.1f}" y="{height-6}" '
                   f'text-anchor="middle">{escape(xlabel)}</text>')
    out.append("</svg>")
    return "".join(out)


def from_series(spec: dict, **kw) -> str:
    """Render a ``{"labels": [...], "values": [...]}`` block."""
    if not spec or not spec.get("labels"):
        return ""
    return hbar(spec["labels"], spec["values"], **kw)

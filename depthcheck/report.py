"""One-page, self-contained HTML diagnostic report.

The report is written for a reviewer: verdict first, then the three numbers,
then the criterion that produced the verdict, then provenance. It contains no
JavaScript, no external assets and no timestamps, so two runs of the same
command produce byte-identical files.
"""

from __future__ import annotations

import html
import math
from typing import Dict, Optional

from .decompose import AMBIGUOUS, FAIL, PASS, OPERATIONAL_TEXT, PLAN_TEXT
from .decompose import Decomposition

_COLORS = {
    PASS: ("#1b7f4d", "#e8f5ee", "#cfe8da"),
    FAIL: ("#a8201a", "#fdecea", "#f6d5d2"),
    AMBIGUOUS: ("#8a5a00", "#fff4e0", "#f3e0bd"),
}


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _num(x: float, digits: int = 4) -> str:
    if x is None or not math.isfinite(float(x)):
        return "NA"
    return f"{float(x):.{digits}f}"


def _svg_bars(dec: Decomposition, width: int = 860, height: int = 230) -> str:
    rows = [
        ("observed &#916;F", dec.observed, dec.observed_ci, "#1f4e79"),
        ("implied K&#183;&#916;E[1/L]", dec.implied, dec.implied_ci, "#b5651d"),
        ("residual", dec.residual, dec.residual_ci, "#3d3d3d"),
    ]
    values = [dec.observed, dec.implied, dec.residual, 0.0]
    for _, _, ci, _ in rows:
        if math.isfinite(ci[0]) and math.isfinite(ci[1]):
            values.extend(ci)
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        lo, hi = lo - 0.05, hi + 0.05
    pad = 0.12 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    left, right, top = 190, 30, 30
    plot_w = width - left - right
    row_h = (height - top - 30) / len(rows)

    def x_of(v: float) -> float:
        return left + (v - lo) / (hi - lo) * plot_w

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="observed, implied and residual Fano components">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
    ]
    band_lo, band_hi = dec.band
    bx0, bx1 = sorted((x_of(band_lo), x_of(band_hi)))
    parts.append(
        f'<rect x="{bx0:.2f}" y="{top - 12}" width="{max(bx1 - bx0, 1.0):.2f}" '
        f'height="{height - top - 6}" fill="#f0f0f0"/>'
    )
    parts.append(
        f'<text x="{((bx0 + bx1) / 2):.1f}" y="{top - 16}" font-size="11" '
        f'fill="#777" text-anchor="middle">depth-implied band</text>'
    )
    parts.append(
        f'<line x1="{x_of(0):.2f}" y1="{top - 12}" x2="{x_of(0):.2f}" '
        f'y2="{height - 18}" stroke="#bbbbbb" stroke-width="1" stroke-dasharray="3,3"/>'
    )
    for i, (label, value, ci, color) in enumerate(rows):
        y = top + row_h * (i + 0.5)
        parts.append(
            f'<text x="{left - 14}" y="{y + 4:.1f}" font-size="13" fill="#333" '
            f'text-anchor="end">{label}</text>'
        )
        if math.isfinite(ci[0]) and math.isfinite(ci[1]):
            x0, x1 = sorted((x_of(ci[0]), x_of(ci[1])))
            parts.append(
                f'<line x1="{x0:.2f}" y1="{y:.1f}" x2="{x1:.2f}" y2="{y:.1f}" '
                f'stroke="{color}" stroke-width="2" opacity="0.45"/>'
            )
            for xx in (x0, x1):
                parts.append(
                    f'<line x1="{xx:.2f}" y1="{y - 6:.1f}" x2="{xx:.2f}" '
                    f'y2="{y + 6:.1f}" stroke="{color}" stroke-width="2" opacity="0.45"/>'
                )
        parts.append(
            f'<circle cx="{x_of(value):.2f}" cy="{y:.1f}" r="5.5" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{x_of(value) + 10:.2f}" y="{y - 9:.1f}" font-size="12" '
            f'fill="{color}">{value:+.3f}</text>'
        )
    for t in range(5):
        v = lo + (hi - lo) * t / 4
        x = x_of(v)
        parts.append(
            f'<text x="{x:.1f}" y="{height - 4}" font-size="11" fill="#777" '
            f'text-anchor="middle">{v:+.2f}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _kv_rows(items) -> str:
    out = []
    for key, value in items:
        out.append(
            f"<tr><th>{_esc(key)}</th><td>{_esc(value)}</td></tr>"
        )
    return "\n".join(out)


def write_report(
    decomposition: Decomposition,
    path: str,
    meta: Optional[Dict[str, object]] = None,
) -> str:
    """Write the HTML report and return the path.

    Parameters
    ----------
    decomposition : Decomposition
    path : str
        Output ``.html`` path.
    meta : dict, optional
        Provenance entries (input file, group labels, cell type, seed, version).
        Values are stringified; no timestamps are added by the writer.
    """
    meta = dict(meta or {})
    ink, bg, border = _COLORS[decomposition.verdict]
    title = "depthcheck report"
    subtitle = meta.get("unit", "")
    if subtitle:
        title = f"depthcheck report &middot; {_esc(subtitle)}"

    ctx = decomposition.context
    scalar_rows = [
        ("observed &#916;F (median over genes)", _num(decomposition.observed)),
        ("implied K&#183;&#916;E[1/L]", _num(decomposition.implied)),
        ("residual", _num(decomposition.residual)),
        (
            "implied 95% CI",
            f"{_num(decomposition.implied_ci[0])} to {_num(decomposition.implied_ci[1])}",
        ),
        (
            "residual 95% CI",
            f"{_num(decomposition.residual_ci[0])} to {_num(decomposition.residual_ci[1])}",
        ),
        ("depth-implied band", f"{_num(decomposition.band[0])} to {_num(decomposition.band[1])}"),
        (
            "p (label permutation)",
            "NA" if not math.isfinite(decomposition.p_exact)
            else f"{decomposition.p_exact:.4g} ({_esc(decomposition.permutation_method)}, "
                 f"n={decomposition.n_assignments})",
        ),
    ]
    if "H_abs" in ctx:
        scalar_rows.append(("|H| = K&#183;|&#916;E[1/L]|", _num(ctx["H_abs"])))
    if "n_genes" in ctx:
        scalar_rows.append(("genes in contrast", _esc(ctx["n_genes"])))
    if "delta_strat" in ctx and ctx["delta_strat"] is not None:
        scalar_rows.append(
            (
                "depth-balanced &#916;F (sensitivity)",
                f"{_num(ctx['delta_strat'])} (coverage {_num(ctx.get('strat_coverage', float('nan')), 2)})",
            )
        )
    if "j_gap" in ctx and ctx["j_gap"] is not None:
        scalar_rows.append(("J<sub>A</sub> &#8722; J<sub>B</sub>", _num(ctx["j_gap"])))

    prov = [(k, v) for k, v in meta.items() if k != "unit"]

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{_esc(meta.get('unit', 'depthcheck'))}</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin: 0; padding: 32px 20px 56px; background: #fbfbfa; color: #1d1d1b;
         font: 15px/1.6 -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }}
  .wrap {{ max-width: 900px; margin: 0 auto; }}
  .card {{ background: #fff; border: 1px solid #e4e2dd; border-radius: 10px;
           padding: 24px 26px; margin-bottom: 18px; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; font-weight: 650; letter-spacing: -0.01em; }}
  h2 {{ font-size: 15px; margin: 0 0 10px; font-weight: 650; color: #3a3a37; }}
  .sub {{ color: #6b6a66; font-size: 13px; margin: 0; }}
  .badge {{ display: inline-block; padding: 6px 14px; border-radius: 999px;
            background: {bg}; color: {ink}; border: 1px solid {border};
            font-weight: 700; letter-spacing: 0.06em; font-size: 13px; }}
  .reading {{ margin: 14px 0 0; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ text-align: left; padding: 7px 10px; border-bottom: 1px solid #efeeea;
            vertical-align: top; }}
  th {{ width: 42%; color: #55534f; font-weight: 550; }}
  td {{ font-variant-numeric: tabular-nums; }}
  .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
           font-size: 12.5px; }}
  .crit {{ background: #f7f6f2; border-left: 3px solid #c9c5ba; padding: 12px 16px;
           font-size: 13px; color: #43423e; }}
  .crit p {{ margin: 6px 0; }}
  footer {{ color: #7a7873; font-size: 12px; text-align: center; margin-top: 8px; }}
  .warn {{ color: #8a5a00; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>{title}</h1>
    <p class="sub">{_esc(meta.get('subtitle', 'Depth decomposition of a normalized Fano contrast'))}</p>
    <p style="margin:18px 0 0"><span class="badge">{decomposition.verdict}</span></p>
    <p class="reading">{_esc(decomposition.reading)}</p>
  </div>

  <div class="card">
    <h2>Components (Fano units, K = 1e4)</h2>
    <table>
      <tr><th>observed &#916;F (median over genes)</th><td>{_num(decomposition.observed)}</td></tr>
      <tr><th>implied K&#183;&#916;E[1/L]</th><td>{_num(decomposition.implied)}</td></tr>
      <tr><th>residual</th><td>{_num(decomposition.residual)}</td></tr>
      <tr><th>implied 95% CI</th><td>{_num(decomposition.implied_ci[0])} to {_num(decomposition.implied_ci[1])}</td></tr>
      <tr><th>residual 95% CI</th><td>{_num(decomposition.residual_ci[0])} to {_num(decomposition.residual_ci[1])}</td></tr>
      <tr><th>depth-implied band</th><td>{_num(decomposition.band[0])} to {_num(decomposition.band[1])}</td></tr>
      {''.join(f'<tr><th>{k}</th><td>{v}</td></tr>' for k, v in scalar_rows[6:])}
    </table>
  </div>

  <div class="card">
    <h2>Where the observation sits</h2>
    {_svg_bars(decomposition)}
    <p class="sub">Bars show point estimates; thin lines are 95% CIs. The grey area is the
    interval that the depth term alone predicts, widened by its own uncertainty and by 0.02.</p>
  </div>

  <div class="card">
    <h2>Verdict rule (fixed before any result)</h2>
    <div class="crit">
      <p>{_esc(PLAN_TEXT)}</p>
      <p>{_esc(OPERATIONAL_TEXT)}</p>
    </div>
  </div>

  <div class="card">
    <h2>Provenance</h2>
    <table>
      {_kv_rows(prov)}
    </table>
    <p class="sub">The depth term is computable from per-cell total counts alone. It is a
    <em>non-identifiable</em> mixture of capture efficiency and cellular RNA content; this
    report does not attribute it to either.</p>
  </div>

  <footer>generated by depthcheck &middot; verdict rule {_esc(decomposition.as_dict()['criteria_version'])}</footer>
</div>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(html_text)
    return path


def write_summary(rows, path: str, meta: Optional[Dict[str, object]] = None,
                  n_ok: Optional[int] = None) -> str:
    """Multi-unit summary page: one row per (unit, contrast).

    ``rows`` are the dictionaries produced by the CLI; only scalar fields are
    used, so the page stays readable for a 66-unit run.
    """
    meta = dict(meta or {})
    body = []
    for r in rows:
        verdict = str(r.get("verdict", "NA"))
        ink, bg, border = _COLORS.get(verdict, ("#55534f", "#f2f1ed", "#ddd9cf"))
        body.append(
            "<tr>"
            f"<td>{_esc(r.get('unit', ''))}</td>"
            f"<td>{_esc(r.get('group_a', ''))} vs {_esc(r.get('group_b', ''))}</td>"
            f"<td>{_esc(r.get('n_cells_a', ''))} / {_esc(r.get('n_cells_b', ''))}</td>"
            f"<td>{_esc(r.get('n_genes', ''))}</td>"
            f"<td>{_num(r.get('observed_dF', float('nan')))}</td>"
            f"<td>{_num(r.get('implied_dF', float('nan')))}</td>"
            f"<td>{_num(r.get('residual', float('nan')))}</td>"
            f'<td><span style="padding:2px 8px;border-radius:999px;background:{bg};'
            f'color:{ink};border:1px solid {border};font-weight:650">{_esc(verdict)}</span></td>'
            "</tr>"
        )
    done = n_ok if n_ok is not None else sum(1 for r in rows if r.get("status") == "ok")
    html_text = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>depthcheck summary</title>
<style>
  body {{ margin: 0; padding: 32px 20px 56px; background: #fbfbfa; color: #1d1d1b;
         font: 14px/1.6 -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }}
  .wrap {{ max-width: 1000px; margin: 0 auto; }}
  .card {{ background:#fff; border:1px solid #e4e2dd; border-radius:10px; padding:22px 24px; }}
  h1 {{ font-size: 19px; margin: 0 0 6px; }}
  p.sub {{ color:#6b6a66; font-size:13px; margin:0 0 16px; }}
  table {{ border-collapse: collapse; width:100%; font-variant-numeric: tabular-nums; }}
  th, td {{ padding:6px 9px; border-bottom:1px solid #efeeea; text-align:left; }}
  th {{ font-size:12px; color:#55534f; font-weight:600; background:#faf9f6; }}
  td:nth-child(n+3) {{ text-align:right; }}
</style></head>
<body><div class="wrap"><div class="card">
<h1>depthcheck &middot; {_esc(meta.get('version', ''))}</h1>
<p class="sub">{_esc(meta.get('input', ''))} &middot; group column
{_esc(meta.get('group_col', ''))} &middot; {done} of {len(rows)} units estimable &middot;
verdict rule {_esc(meta.get('criteria_version', ''))}</p>
<table>
<tr><th>unit</th><th>contrast</th><th>cells A / B</th><th>genes</th>
<th>observed</th><th>implied</th><th>residual</th><th>verdict</th></tr>
{''.join(body)}
</table>
<p class="sub" style="margin-top:14px">Implied is K&middot;&#916;E[1/L], computed from per-cell
total counts alone. A unit marked NA was not estimable; the reason is in
depthcheck_result.tsv.</p>
</div></div></body></html>
"""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(html_text)
    return path

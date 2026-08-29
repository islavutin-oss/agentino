"""Create PDF documents from markdown/text content."""

import inspect
import json
import os as _os
import re
import tempfile
from datetime import datetime

from agentino.core.tool import tool

from ._pdf import render_pdf
from .storage import get_default_store

# ── Chart/datatable block rendering (for PDF export) ───────────────

_TH = 'style="background:#f3f4f6;padding:6px 10px;border:1px solid #e5e7eb;text-align:left;font-weight:600;color:#374151"'
_TD = 'style="padding:6px 10px;border:1px solid #e5e7eb;color:#1f2937"'
_TDN = 'style="padding:6px 10px;border:1px solid #e5e7eb;color:#1f2937;text-align:right"'

_NUM_RE = re.compile(r"^[+\-]?[€$£]?[\d,]+\.?\d*%?$")


def _is_num(v) -> bool:
    return bool(_NUM_RE.match(str(v).strip()))


def _datatable_to_html(match) -> str:
    try:
        d = json.loads(match.group(1))
        cols, rows = d.get("columns", []), d.get("rows", [])
        if not cols:
            return ""
        title = d.get("title", "")
        out = (
            f'<p style="font-weight:600;font-size:11pt;margin:10px 0 4px">{title}</p>'
            if title
            else ""
        )
        out += '<table style="width:100%;border-collapse:collapse;font-size:9pt;margin:8px 0">\n<thead><tr>'
        out += "".join(f"<th {_TH}>{c}</th>" for c in cols)
        out += "</tr></thead>\n<tbody>"
        for i, row in enumerate(rows):
            bg = ' style="background:#f9fafb"' if i % 2 else ""
            out += f"<tr{bg}>"
            for j, cell in enumerate(row):
                out += f"<td {_TDN if j > 0 and _is_num(cell) else _TD}>{cell}</td>"
            out += "</tr>\n"
        out += "</tbody></table>"
        return out
    except Exception:
        return ""


def _chart_to_html(match) -> str:
    """Render a chart block as horizontal bars (works reliably in PDF)."""
    try:
        d = json.loads(match.group(1))
        items, x_key, y_key = d.get("data", []), d.get("xKey", ""), d.get("yKey", "")
        y2_key = d.get("y2Key", "")
        color, y_fmt = d.get("color", "#7C3AED"), d.get("yFormat", "")
        y2_color = d.get("y2Color", "#9CA3AF")
        if not items or not x_key or not y_key:
            return ""
        title = d.get("title", "")
        values = [i.get(y_key, 0) for i in items] + (
            [i.get(y2_key, 0) for i in items] if y2_key else []
        )
        mx = max(values, default=1) or 1

        def fmt(v):
            try:
                n = float(v)
            except (TypeError, ValueError):
                return str(v)
            if y_fmt == "currency":
                return f"€{n:,.0f}"
            if y_fmt == "percent":
                return f"{n:.1f}%"
            return f"{n:,.0f}"

        out = (
            f'<p style="font-weight:600;font-size:11pt;margin:10px 0 4px">{title}</p>'
            if title
            else ""
        )
        out += '<table style="width:100%;border-collapse:collapse;font-size:9pt;margin:8px 0">'
        for it in items:
            xv = str(it.get(x_key, ""))
            yv = it.get(y_key, 0)
            try:
                pct = max(2, int((float(yv) / mx) * 100))
            except (TypeError, ValueError):
                pct = 2
            bar_rows = f'<div style="background:{color};height:14px;width:{pct}%;border-radius:3px;opacity:0.9"></div>'
            label_right = f"<span>{fmt(yv)}</span>"
            if y2_key:
                y2v = it.get(y2_key, 0)
                try:
                    pct2 = max(2, int((float(y2v) / mx) * 100))
                except (TypeError, ValueError):
                    pct2 = 2
                bar_rows += (
                    f'<div style="background:{y2_color};height:14px;width:{pct2}%;'
                    f'border-radius:3px;opacity:0.9;margin-top:3px"></div>'
                )
                label_right = f'<span>{fmt(yv)}</span><br><span style="color:#9CA3AF;font-size:7.5pt">prev: {fmt(y2v)}</span>'
            out += (
                f'<tr><td style="width:70px;padding:4px 6px;color:#6b7280;white-space:nowrap">{xv}</td>'
                f'<td style="padding:4px 2px">{bar_rows}</td>'
                f'<td style="width:100px;padding:4px 6px;text-align:right;color:#374151;white-space:nowrap;font-weight:600">{label_right}</td></tr>'
            )
        out += "</table>"
        return out
    except Exception:
        return ""


def _render_blocks(content: str) -> str:
    """Convert ```chart / ```datatable fenced blocks into HTML for PDF rendering."""
    content = re.sub(r"```datatable\s*\n(.*?)\n```", _datatable_to_html, content, flags=re.DOTALL)
    content = re.sub(r"```chart\s*\n(.*?)\n```", _chart_to_html, content, flags=re.DOTALL)
    return content


# ── Tool ─────────────────────────────────────────────────────────


@tool
async def create_pdf(title: str, content: str) -> str:
    """Create a PDF document and attach it for download. Supports markdown formatting AND structured blocks (```chart, ```datatable) that render as real charts/tables in the PDF.

    Args:
        title: Document title
        content: The markdown/text content to render. Include ```chart or ```datatable blocks verbatim.
    """
    rendered_content = _render_blocks(content)

    now = datetime.now().strftime("%Y-%m-%d")
    subtitle = f"Generated {now}"

    safe = title.lower().replace(" ", "-").replace("/", "-")[:40]
    final_filename = f"{safe}-{now}.pdf"

    # Renderers write to a path, so go via a temp file, then read and upload.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        tmp_path = tf.name
    try:
        try:
            outcome = render_pdf(rendered_content, f"{title}\n{subtitle}", tmp_path)
            if inspect.isawaitable(outcome):
                await outcome
        except Exception as exc:
            return f"PDF generation failed: {exc}"
        if not _os.path.exists(tmp_path) or _os.path.getsize(tmp_path) == 0:
            return "PDF generation failed: the renderer produced no output"
        with open(tmp_path, "rb") as f:
            content_bytes = f.read()
    finally:
        try:
            _os.unlink(tmp_path)
        except OSError:
            pass

    # Indexable preview from the source markdown (better than parsing PDF text).
    extracted = re.sub(r"```[a-z]*\n.*?```", "", content, flags=re.DOTALL)[:50_000]

    try:
        stored = get_default_store().save(
            content_bytes=content_bytes,
            filename=final_filename,
            mime="application/pdf",
            title=title,
            description=subtitle,
            extracted_text=extracted,
        )
    except Exception as e:
        return f"PDF generation failed: {e}"

    return f"PDF created. Download: [{stored.filename}]({stored.url})"

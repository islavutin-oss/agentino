"""How `create_pdf` turns rendered HTML into a PDF.

Agentino ships a plain renderer built on reportlab so the tool works out of
the box. A host with its own house style — logo, brand colour, footer —
registers a renderer instead:

    from agentino.tools.std import set_pdf_renderer

    set_pdf_renderer(my_render)

The renderer takes ``(html, title, out_path)`` and writes a PDF to
``out_path``. Registration replaces the default for the whole process.
"""

from __future__ import annotations

import html as _html
import re
from collections.abc import Callable

PdfRenderer = Callable[[str, str, str], None]

_renderer: PdfRenderer | None = None


def set_pdf_renderer(renderer: PdfRenderer | None) -> None:
    """Register the renderer `create_pdf` should use, or None to restore
    the built-in one."""
    global _renderer
    _renderer = renderer


def _strip_tags(fragment: str) -> str:
    return _html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def _default_renderer(html_body: str, title: str, out_path: str) -> None:
    """A plain, unbranded PDF: headings, paragraphs and tables.

    Requires reportlab, which comes with the ``docgen`` extra.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise RuntimeError(
            "create_pdf needs reportlab: pip install 'agentino-framework[docgen]', "
            "or register your own renderer with set_pdf_renderer()"
        ) from exc

    styles = getSampleStyleSheet()
    story: list = [Paragraph(_html.escape(title), styles["Title"]), Spacer(1, 12)]

    # Split the body into tables and everything else, in order.
    for chunk in re.split(r"(<table.*?</table>)", html_body, flags=re.S):
        if not chunk.strip():
            continue
        if chunk.lstrip().startswith("<table"):
            rows = [
                [_strip_tags(cell) for cell in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.S)]
                for row in re.findall(r"<tr[^>]*>(.*?)</tr>", chunk, re.S)
            ]
            rows = [r for r in rows if r]
            if not rows:
                continue
            table = Table(rows, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story += [table, Spacer(1, 10)]
            continue
        for para in re.split(r"</(?:p|h[1-6]|div)>", chunk):
            text = _strip_tags(para)
            if not text:
                continue
            heading = re.search(r"<h([1-6])", para)
            style = (
                styles[f"Heading{min(int(heading.group(1)), 4)}"] if heading else styles["BodyText"]
            )
            story += [Paragraph(_html.escape(text), style), Spacer(1, 6)]

    SimpleDocTemplate(out_path, pagesize=A4, title=title).build(story)


def render_pdf(html_body: str, title: str, out_path: str):
    """Render to `out_path` using the registered renderer, or the default.

    A registered renderer may be async; the return value is passed straight
    back so the caller can await it when it is awaitable.
    """
    return (_renderer or _default_renderer)(html_body, title, out_path)

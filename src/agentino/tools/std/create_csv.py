"""Create CSV files for download."""

import csv
import io

from agentino.core.tool import tool

from .storage import get_default_store


@tool
async def create_csv(
    title: str, columns: list[str], rows: list[list[str]], filename: str = ""
) -> str:
    """Create a CSV file and attach it for download. Use when the user asks to export data as CSV.

    Args:
        title: Report title (used in filename if filename not provided)
        columns: Column headers
        rows: Data rows (list of lists)
        filename: Optional output filename (without extension). Defaults to title-based name.
    """
    if not filename:
        safe = title.lower().replace(" ", "-").replace("/", "-")[:40]
        filename = f"{safe}.csv"
    if not filename.endswith(".csv"):
        filename += ".csv"

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    writer.writerows(rows)
    content_bytes = buf.getvalue().encode("utf-8")

    # Plain-text content goes straight into extracted_text — F12 search will
    # match exactly what the user can read.
    text_preview = buf.getvalue()[:50_000]  # cap at 50KB of indexable text

    try:
        stored = get_default_store().save(
            content_bytes=content_bytes,
            filename=filename,
            mime="text/csv",
            title=title,
            extracted_text=text_preview,
        )
    except Exception as e:
        return f"CSV generation failed: {e}"

    row_count = len(rows)
    return (
        f"CSV created ({row_count} rows, {len(columns)} columns). "
        f"Download: [{stored.filename}]({stored.url})"
    )

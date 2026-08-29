"""agentino.tools.std — standard tool catalog for agentino agents.

Categories (re-exported via this module for convenient import):

  - File IO:        read_file, list_files
  - Web:            web_search, fetch_web_data, read_rss
  - Doc generation (Tier 1, "std-create"):
                    create_pdf, create_csv,
                    create_spreadsheet, create_document, create_presentation
  - Memory:         remember, forget, read_memory, update_memory
  - Weather:        get_weather, get_weather_forecast
  - Translation:    translate_text

Usage:

    from agentino.tools.std import read_file, web_search, create_pdf

## Tiered Office tooling

  Tier 1 — **std-create**  (this module, auto-loaded)
    create_document    → DOCX one-shot
    create_spreadsheet → XLSX one-shot
    create_presentation→ PPTX one-shot
    Use when: agent has data in memory, needs a polished file out fast.
    Token cost: low — model sends a structured payload, library does layout.

  Tier 2 — **skill-{docx,xlsx,pptx}**  (per-tenant, opt-in)
    Lives in `<tenant>/skills/<name>/` (e.g.
    `acme/.../tenants/acme/agents/finance/skills/xlsx_brand/`),
    declared via `skills:` in that tenant's agents.yml. Bundle =
    `SKILL.md` + `tools/` + `templates/`. Multi-tool workflow for
    opening existing files, filling templates, editing specific
    paragraphs/cells/slides, branded deliverables.

    Skills aren't centralized — each tenant ships only what its agents
    need. If 2+ tenants need the same skill, lift to a peer
    `agentic/skills/` dir at that point.

  Picking between them in SOUL.md:
    "Use create_* for ad-hoc files generated from chat data. Use
    skill-{docx,xlsx,pptx} when filling a template, editing an existing
    file, or producing a branded deliverable."

  Compare empirically via `runners.yml` `type: ab` and the dashboard's
  /workspace/runs/compare view.

"""

from ._file_storage import (
    FileMetadata,
    LocalFileStorage,
    get_file_storage,
    set_file_storage_provider,
)
from ._pdf import set_pdf_renderer
from ._web_search import web_search
from .create_csv import create_csv
from .create_document import create_document
from .create_pdf import create_pdf
from .create_presentation import create_presentation
from .create_spreadsheet import create_spreadsheet
from .fetch_web_data import fetch_web_data
from .forget import forget
from .get_weather import get_weather
from .get_weather_forecast import get_weather_forecast
from .list_files import list_files
from .read_file import read_file
from .read_memory import read_memory
from .read_rss import read_rss
from .remember import remember
from .translate_text import translate_text
from .update_memory import update_memory

# Tier 1 — std-create office tools. Listed explicitly here so a single
# `tools: std_create` allowlist (or `from agentino.tools.std import
# STD_CREATE_TOOLS`) gives consumers the whole tier without naming each.
STD_CREATE_OFFICE_TOOLS = (
    "create_document",
    "create_spreadsheet",
    "create_presentation",
)


__all__ = [
    # Storage seam — a host application can replace the default.
    "FileMetadata",
    "LocalFileStorage",
    "get_file_storage",
    "set_file_storage_provider",
    "set_pdf_renderer",
    "read_file",
    "list_files",
    "web_search",
    "read_rss",
    "fetch_web_data",
    "create_pdf",
    "create_csv",
    "create_spreadsheet",
    "create_document",
    "create_presentation",
    "remember",
    "forget",
    "read_memory",
    "update_memory",
    "get_weather",
    "get_weather_forecast",
    "translate_text",
]

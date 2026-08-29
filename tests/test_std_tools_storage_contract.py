"""Storage invariants for the std file-producing tools.

These guards used to live in the workspace repo, which is where the tools
used to live. They belong next to the code they constrain.
"""

from pathlib import Path

import pytest

import agentino.tools.std as std

STD_DIR = Path(std.__file__).parent

CREATE_TOOLS = [
    "create_csv",
    "create_spreadsheet",
    "create_document",
    "create_pdf",
    "create_presentation",
]
READ_TOOLS = ["read_file", "list_files", "storage"]


def _source(tool: str) -> str:
    path = STD_DIR / f"{tool}.py"
    assert path.is_file(), f"{tool}.py is missing from agentino.tools.std"
    return path.read_text()


@pytest.mark.parametrize("tool", CREATE_TOOLS + READ_TOOLS)
def test_tool_does_not_hardcode_a_storage_path(tool):
    """Writes and reads go through the storage facade, never a fixed dir."""
    assert "/tmp/workspace-files" not in _source(tool)


@pytest.mark.parametrize("tool", CREATE_TOOLS)
def test_create_tool_uses_the_storage_facade(tool):
    """A file-producing tool must call get_default_store() rather than
    rolling its own write, so the backend stays swappable."""
    assert "get_default_store" in _source(tool)

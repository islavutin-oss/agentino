"""Built-in tools that ship with Agentino.

Available by default for all agents unless explicitly overridden.
Mirrors the core toolset from frameworks like OpenClaw.
"""

import os
import re
import subprocess
from pathlib import Path

from agentino.core.tool import tool
from agentino.reliability.errors import (
    error_blocked,
    error_internal,
    error_not_found,
    error_permission,
    error_timeout,
    error_unavailable,
    error_validation,
)


def _safe_path(path: str) -> str:
    """Resolve a path and verify it stays within the working directory.

    When AGENTINO_PROJECT_DIR is set (agent mode), resolves relative paths
    against it and restricts access to that directory tree.
    Without it (direct usage / tests), allows any path.

    Raises ValueError if the resolved path escapes the allowed directory.
    """
    jail = os.environ.get("AGENTINO_PROJECT_DIR")
    if jail:
        jail_real = os.path.realpath(jail)
        # Resolve relative paths against AGENTINO_PROJECT_DIR, not cwd
        expanded = os.path.expanduser(path)
        if not os.path.isabs(expanded):
            expanded = os.path.join(jail_real, expanded)
        resolved = os.path.realpath(expanded)
        # Use Path.relative_to for robust containment check (no prefix tricks)
        try:
            Path(resolved).relative_to(jail_real)
        except ValueError:
            raise ValueError(f"Path '{path}' is outside the working directory")
        return resolved
    return os.path.realpath(os.path.expanduser(path))


@tool(is_read_only=True)
def read_file(path: str) -> str:
    """Read a file and return its contents with line numbers."""
    try:
        path = _safe_path(path)
        with open(path) as f:
            lines = f.readlines()
        # Record read state for staleness detection on subsequent edits
        _record_read(path)
        if len(lines) > 500:
            numbered = [f"{i + 1:4d} | {line}" for i, line in enumerate(lines[:500])]
            return "".join(numbered) + f"\n... ({len(lines)} total lines, truncated at 500)"
        numbered = [f"{i + 1:4d} | {line}" for i, line in enumerate(lines)]
        return "".join(numbered)
    except FileNotFoundError:
        return error_not_found("file", path)
    except PermissionError:
        return error_permission("read", f"file {path}")
    except Exception as e:
        return error_internal("reading file", e)


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed."""
    try:
        path = _safe_path(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"Written {len(content)} bytes to {path}"
    except PermissionError:
        return error_permission("write", f"file {path}")
    except Exception as e:
        return error_internal("writing file", e)


@tool(is_read_only=True)
def list_files(directory: str) -> str:
    """List files in a directory. Skips hidden dirs, __pycache__, node_modules."""
    try:
        directory = _safe_path(directory)
        entries = sorted(os.listdir(directory))
        result = []
        skip = {"__pycache__", "node_modules", ".venv", "venv", ".git", ".tox"}
        for entry in entries:
            if entry.startswith(".") or entry in skip:
                continue
            full = os.path.join(directory, entry)
            if os.path.isdir(full):
                result.append(f"  {entry}/")
            else:
                size = os.path.getsize(full)
                result.append(f"  {entry} ({size} bytes)")
        return "\n".join(result) if result else "(empty directory)"
    except PermissionError:
        return error_permission("list", "directory")
    except FileNotFoundError:
        return error_not_found("directory", directory)
    except Exception as e:
        return error_internal("listing directory", e)


# NOT a security boundary. It catches a handful of shapes that are almost
# always a mistake — `rm -rf /`, mkfs, dd to a device, a fork bomb — and
# nothing else. `rm -rf /*`, `find / -delete` and `curl … | sh` all pass
# straight through, and any blocklist over a shell language always will.
#
# The boundary is whether an agent gets this tool at all. An agent handling
# untrusted input should not have it; use a gate, a PreToolUse hook, or simply
# leave `shell` out of its tool list.
_SHELL_BLOCKLIST = re.compile(
    r"(?:^|\s*(?:;|&&|\|\|)\s*)"  # start of command or chained
    r"(?:rm\s+-[^\s]*r[^\s]*\s+/(?:\s|$)"  # rm -rf /
    r"|mkfs\b"
    r"|dd\s+.*of=/dev/"
    r"|:\(\)\{.*\};"  # fork bomb
    r")",
    re.IGNORECASE,
)


@tool
def shell(command: str) -> str:
    """Run a shell command and return stdout + stderr."""
    try:
        # A tripwire for accidents, not a defence against intent — see the
        # note on _SHELL_BLOCKLIST.
        if _SHELL_BLOCKLIST.search(command):
            return error_blocked("command blocked — contains potentially destructive pattern")
        cwd = os.environ.get("AGENTINO_PROJECT_DIR") or None
        # Timeout configurable via AGENTINO_SHELL_TIMEOUT (seconds), default: no limit
        timeout_str = os.environ.get("AGENTINO_SHELL_TIMEOUT", "")
        timeout = int(timeout_str) if timeout_str else None
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += result.stderr
        if result.returncode != 0:
            output += f"\n(exit code: {result.returncode})"
        return output.strip() if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return error_timeout("shell command", timeout)
    except Exception as e:
        return error_internal("shell command", e)


_SEARCH_SKIP = {"__pycache__", "node_modules", ".venv", "venv", ".git", ".tox"}


@tool
def search_files(directory: str, pattern: str) -> str:
    """Search for files matching a pattern (glob). Example: pattern='**/*.py'"""
    import glob

    try:
        directory = _safe_path(directory)
        raw = sorted(glob.glob(os.path.join(directory, pattern), recursive=True))
        # Filter out noise directories
        matches = [m for m in raw if not any(skip in m.split(os.sep) for skip in _SEARCH_SKIP)]
        if not matches:
            return f"No files matching '{pattern}' in {directory}"
        # Show relative paths
        result = []
        for m in matches[:100]:
            rel = os.path.relpath(m, directory)
            result.append(f"  {rel}")
        out = "\n".join(result)
        if len(matches) > 100:
            out += f"\n  ... ({len(matches)} total, showing first 100)"
        return out
    except Exception as e:
        return error_internal("searching files", e)


@tool
def grep(path: str, pattern: str) -> str:
    """Search file contents for a regex pattern. path can be a file or directory."""
    try:
        path = _safe_path(path)
        result = subprocess.run(
            [
                "grep",
                "-rn",
                "--include=*.py",
                "--include=*.go",
                "--include=*.rs",
                "--include=*.js",
                "--include=*.ts",
                "--include=*.tsx",
                "--include=*.jsx",
                "--include=*.java",
                "--include=*.kt",
                "--include=*.c",
                "--include=*.h",
                "--include=*.cpp",
                "--include=*.yml",
                "--include=*.yaml",
                "--include=*.json",
                "--include=*.toml",
                "--include=*.md",
                "--include=*.txt",
                "--include=*.proto",
                "--include=*.sql",
                "--include=*.sh",
                "--include=*.dockerfile",
                "--include=Dockerfile",
                "--include=Makefile",
                "--include=*.cfg",
                "--include=*.ini",
                "-I",
                pattern,
                path,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = result.stdout.strip()
        lines = output.split("\n") if output else []
        if len(lines) > 50:
            return "\n".join(lines[:50]) + f"\n... ({len(lines)} total matches, showing first 50)"
        return output if output else f"No matches for '{pattern}'"
    except subprocess.TimeoutExpired:
        return error_timeout("grep search", 15)
    except Exception as e:
        return error_internal("grep search", e)


@tool
def web_search(query: str) -> str:
    """Search the web. Returns top results with titles, URLs, and descriptions.

    Uses Brave Search API if BRAVE_API_KEY is set, otherwise falls back to DuckDuckGo.
    """
    api_key = os.environ.get("BRAVE_API_KEY") or os.environ.get("OPENCLAW_BRAVE_API_KEY")
    if api_key:
        result = _web_search_brave(query, api_key)
        if not result.startswith("Error"):
            return result
        import logging

        logging.getLogger(__name__).warning(
            f"Brave search failed, falling back to DuckDuckGo: {result}"
        )
        return _web_search_ddg(query)
    return _web_search_ddg(query)


def _web_search_brave(query: str, api_key: str) -> str:
    """Brave Search API."""
    try:
        import httpx

        resp = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": 8},
            headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("web", {}).get("results", [])[:8]:
            title = item.get("title", "")
            url = item.get("url", "")
            desc = item.get("description", "")
            if title and url:
                results.append(f"- {title}\n  {url}\n  {desc}")
        return "\n\n".join(results) if results else f"No results for '{query}'"
    except Exception as e:
        return error_internal("Brave search", e)


def _web_search_ddg(query: str) -> str:
    """DuckDuckGo fallback (no API key needed)."""
    try:
        import re

        import httpx

        resp = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (compatible; Agentino/1.0)"},
            timeout=10,
            follow_redirects=True,
        )
        resp.raise_for_status()
        results = []
        for match in re.finditer(
            r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
            r'class="result__snippet"[^>]*>(.*?)</(?:td|div)',
            resp.text,
            re.DOTALL,
        ):
            url = match.group(1)
            title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            snippet = re.sub(r"<[^>]+>", "", match.group(3)).strip()
            if title and url:
                results.append(f"- {title}\n  {url}\n  {snippet}")
            if len(results) >= 8:
                break
        return "\n\n".join(results) if results else f"No results for '{query}'"
    except Exception as e:
        return error_internal("DuckDuckGo search", e)


@tool
def web_fetch(url: str) -> str:
    """Fetch a URL and return its text content (HTML tags stripped). Max 10KB."""
    try:
        import re

        import httpx

        resp = httpx.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Agentino/1.0)"},
            timeout=15,
            follow_redirects=True,
        )
        resp.raise_for_status()
        text = resp.text
        # Strip HTML tags
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 10000:
            text = text[:10000] + "\n... (truncated at 10KB)"
        return text if text else "(empty page)"
    except ImportError:
        return error_unavailable("web fetch", "httpx not installed (pip install httpx)")
    except Exception as e:
        return error_internal("web fetch", e)


# ---------------------------------------------------------------------------
# File state cache — tracks read timestamps for staleness detection
# ---------------------------------------------------------------------------
_file_read_state: dict[str, tuple[float, str]] = {}  # path → (mtime, content_hash)


def _record_read(path: str) -> None:
    """Record file state after reading (for staleness check on edit)."""
    try:
        import hashlib

        mtime = os.path.getmtime(path)
        with open(path, "rb") as f:
            h = hashlib.md5(f.read(8192)).hexdigest()
        _file_read_state[os.path.realpath(path)] = (mtime, h)
    except Exception:
        pass


def _check_stale(path: str) -> str | None:
    """Check if file changed since last read. Returns error msg or None."""
    rpath = os.path.realpath(path)
    if rpath not in _file_read_state:
        return None  # never read — skip check
    old_mtime, old_hash = _file_read_state[rpath]
    try:
        import hashlib

        cur_mtime = os.path.getmtime(path)
        if cur_mtime == old_mtime:
            return None
        # mtime changed — verify content actually differs
        with open(path, "rb") as f:
            cur_hash = hashlib.md5(f.read(8192)).hexdigest()
        if cur_hash == old_hash:
            return None  # content same despite mtime change
        return error_blocked(
            f"{path} has been modified since you last read it. Read it again before editing."
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Quote normalization — model outputs straight quotes, files may have curly
# ---------------------------------------------------------------------------
_CURLY_QUOTES = {
    "\u2018": "'",  # left single curly → straight
    "\u2019": "'",  # right single curly → straight
    "\u201c": '"',  # left double curly → straight
    "\u201d": '"',  # right double curly → straight
}


def _normalize_quotes(s: str) -> str:
    """Normalize curly quotes to straight quotes."""
    for curly, straight in _CURLY_QUOTES.items():
        s = s.replace(curly, straight)
    return s


def _find_actual_string(content: str, search: str) -> tuple[str | None, int]:
    """Find string in content with quote normalization fallback.
    Returns (actual_string, count) where actual_string preserves original quotes.
    """
    # Exact match first
    count = content.count(search)
    if count > 0:
        return search, count

    # Try with normalized quotes
    norm_search = _normalize_quotes(search)
    norm_content = _normalize_quotes(content)
    count = norm_content.count(norm_search)
    if count > 0:
        # Find actual string in original content at normalized position
        idx = norm_content.index(norm_search)
        actual = content[idx : idx + len(search)]
        return actual, count

    # Try with trailing whitespace stripped per line
    stripped_search = "\n".join(line.rstrip() for line in search.split("\n"))
    stripped_content = "\n".join(line.rstrip() for line in content.split("\n"))
    count = stripped_content.count(stripped_search)
    if count > 0:
        idx = stripped_content.index(stripped_search)
        # Map back to original content position
        actual = content[idx : idx + len(stripped_search)]
        return actual, count

    return None, 0


@tool
def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Replace an exact string in a file. Use for surgical edits instead of rewriting entire files.

    old_string must match exactly (including whitespace/indentation).
    Fails if old_string is not found or matches multiple locations.
    Uses quote normalization (curly↔straight) and trailing whitespace tolerance.
    """
    try:
        path = _safe_path(path)

        # Encoding detection
        encoding = "utf-8"
        try:
            with open(path, "rb") as f:
                raw = f.read(4)
                if raw[:2] == b"\xff\xfe":
                    encoding = "utf-16-le"
                elif raw[:2] == b"\xfe\xff":
                    encoding = "utf-16-be"
        except Exception:
            pass

        with open(path, encoding=encoding) as f:
            content = f.read()

        # Staleness check
        stale_msg = _check_stale(path)
        if stale_msg:
            return stale_msg

        # Find match with normalization fallback
        actual, count = _find_actual_string(content, old_string)

        if actual is None or count == 0:
            # Provide helpful context for debugging
            lines = content.split("\n")
            search_first_line = old_string.split("\n")[0][:60]
            for i, line in enumerate(lines):
                if search_first_line[:20] in line or _normalize_quotes(
                    search_first_line[:20]
                ) in _normalize_quotes(line):
                    ctx = "\n".join(lines[max(0, i - 1) : i + 3])
                    return error_not_found(
                        "text",
                        f"old_string in {path}",
                        detail=f"Closest match near line {i + 1}:\n{ctx}",
                    )
            return error_not_found("text", f"old_string in {path}")

        if count > 1:
            return error_validation(
                "old_string",
                f"matches {count} locations in {path}. Provide more context to make it unique.",
            )

        new_content = content.replace(actual, new_string, 1)
        with open(path, "w", encoding=encoding) as f:
            f.write(new_content)

        # Update read state cache
        _record_read(path)

        old_lines = old_string.count("\n") + 1
        new_lines = new_string.count("\n") + 1
        return f"Edited {path}: replaced {old_lines} lines with {new_lines} lines"
    except FileNotFoundError:
        return error_not_found("file", path)
    except PermissionError:
        return error_permission("edit", f"file {path}")
    except Exception as e:
        return error_internal("editing file", e)


# ---------------------------------------------------------------------------
# Stage verdict — structured stage completion signal
# ---------------------------------------------------------------------------


@tool
def stage_verdict(
    result: str,
    summary: str = "",
    details: str = "",
) -> str:
    """Signal stage completion with a structured verdict.
    result: ACCEPT, FAIL, or REJECT.
    summary: one-line description of what was done.
    details: comma-separated specifics (files changed, tests failed, etc.)."""
    result = result.strip().upper()
    if result not in ("ACCEPT", "FAIL", "REJECT"):
        result = "ACCEPT"
    return f"VERDICT:{result}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BUILTIN_TOOLS = [
    read_file,
    write_file,
    edit_file,
    list_files,
    shell,
    search_files,
    grep,
    web_search,
    web_fetch,
    stage_verdict,
]

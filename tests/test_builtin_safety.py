"""Tests for built-in tool safety boundaries.

Validates observable behavior of built-in tools around:
- write_file: parent directory creation, overwrite behavior
- read_file: missing files, large file truncation
- edit_file: non-unique match rejection, missing file
- shell: timeout enforcement, exit code reporting
- search_files / grep: result limits
"""

import os

from agentino.builtin_tools import (
    edit_file,
    grep,
    list_files,
    read_file,
    search_files,
    shell,
    write_file,
)

# ---------------------------------------------------------------------------
# 1. write_file safety
# ---------------------------------------------------------------------------


class TestWriteFile:
    def test_creates_parent_dirs(self, tmp_path):
        target = str(tmp_path / "deep" / "nested" / "file.txt")
        result = write_file.fn(target, "hello")
        assert "Written" in result
        assert os.path.exists(target)
        with open(target) as f:
            assert f.read() == "hello"

    def test_overwrites_existing(self, tmp_path):
        target = str(tmp_path / "file.txt")
        write_file.fn(target, "first")
        write_file.fn(target, "second")
        with open(target) as f:
            assert f.read() == "second"

    def test_reports_byte_count(self, tmp_path):
        target = str(tmp_path / "file.txt")
        result = write_file.fn(target, "12345")
        assert "5 bytes" in result


# ---------------------------------------------------------------------------
# 2. read_file safety
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_missing_file_returns_error(self):
        result = read_file.fn("/nonexistent/path/file.txt")
        assert "Error" in result
        assert "not found" in result

    def test_reads_with_line_numbers(self, tmp_path):
        target = str(tmp_path / "file.txt")
        with open(target, "w") as f:
            f.write("line1\nline2\nline3\n")
        result = read_file.fn(target)
        assert "1" in result
        assert "line1" in result
        assert "line3" in result

    def test_large_file_truncation(self, tmp_path):
        target = str(tmp_path / "big.txt")
        with open(target, "w") as f:
            for i in range(600):
                f.write(f"line {i}\n")
        result = read_file.fn(target)
        assert "truncated" in result.lower() or "total lines" in result.lower()
        # Should only show first 500 lines
        assert "line 499" in result
        assert "line 500" not in result


# ---------------------------------------------------------------------------
# 3. edit_file safety
# ---------------------------------------------------------------------------


class TestEditFile:
    def test_rejects_nonunique_match(self, tmp_path):
        target = str(tmp_path / "file.txt")
        with open(target, "w") as f:
            f.write("foo\nbar\nfoo\n")
        result = edit_file.fn(target, "foo", "baz")
        assert "Error" in result
        assert "2 locations" in result or "matches" in result

    def test_rejects_missing_string(self, tmp_path):
        target = str(tmp_path / "file.txt")
        with open(target, "w") as f:
            f.write("hello world\n")
        result = edit_file.fn(target, "nonexistent", "replacement")
        assert "Error" in result
        assert "not found" in result

    def test_missing_file_returns_error(self):
        result = edit_file.fn("/nonexistent/file.txt", "a", "b")
        assert "Error" in result
        assert "not found" in result

    def test_successful_edit(self, tmp_path):
        target = str(tmp_path / "file.txt")
        with open(target, "w") as f:
            f.write("hello world\n")
        result = edit_file.fn(target, "hello", "goodbye")
        assert "Edited" in result
        with open(target) as f:
            assert "goodbye world" in f.read()


# ---------------------------------------------------------------------------
# 4. shell safety
# ---------------------------------------------------------------------------


class TestShell:
    def test_captures_stdout(self):
        result = shell.fn("echo hello")
        assert "hello" in result

    def test_captures_stderr(self):
        result = shell.fn("echo err >&2")
        assert "err" in result

    def test_reports_exit_code(self):
        result = shell.fn("exit 42")
        assert "exit code: 42" in result

    def test_timeout_returns_error(self):
        import os

        os.environ["AGENTINO_SHELL_TIMEOUT"] = "1"
        try:
            result = shell.fn("sleep 300")
            assert "timed out" in result.lower() or "timeout" in result.lower()
        finally:
            os.environ.pop("AGENTINO_SHELL_TIMEOUT", None)

    def test_empty_output(self):
        result = shell.fn("true")
        assert result == "(no output)"


# ---------------------------------------------------------------------------
# 5. search_files / grep result limits
# ---------------------------------------------------------------------------


class TestSearchLimits:
    def test_search_files_caps_at_100(self, tmp_path):
        # Create 110 .py files
        for i in range(110):
            (tmp_path / f"file_{i:03d}.py").write_text("pass")
        result = search_files.fn(str(tmp_path), "*.py")
        assert "110 total" in result or "showing first 100" in result

    def test_grep_caps_at_50(self, tmp_path):
        # Create a file with 60 matching lines
        target = tmp_path / "big.py"
        target.write_text("\n".join(f"MATCH_LINE_{i}" for i in range(60)))
        result = grep.fn(str(target), "MATCH_LINE")
        assert "60 total" in result or "showing first 50" in result

    def test_search_no_matches(self, tmp_path):
        result = search_files.fn(str(tmp_path), "*.xyz")
        assert "No files" in result

    def test_grep_no_matches(self, tmp_path):
        target = tmp_path / "file.py"
        target.write_text("nothing here")
        result = grep.fn(str(target), "IMPOSSIBLE_PATTERN_12345")
        assert "No matches" in result


# ---------------------------------------------------------------------------
# 6. list_files skips hidden/special dirs
# ---------------------------------------------------------------------------


class TestListFiles:
    def test_skips_hidden_and_special(self, tmp_path):
        (tmp_path / ".hidden").mkdir()
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "visible.py").write_text("pass")
        result = list_files.fn(str(tmp_path))
        assert "visible.py" in result
        assert "__pycache__" not in result
        assert ".hidden" not in result
        assert "node_modules" not in result

    def test_empty_dir(self, tmp_path):
        result = list_files.fn(str(tmp_path))
        assert "empty" in result.lower()

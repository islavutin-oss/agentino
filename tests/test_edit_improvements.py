"""Tests for edit_file improvements: quote normalization, staleness detection, encoding."""

import os
import tempfile
import time

from agentino.builtin_tools import (
    _check_stale,
    _find_actual_string,
    _normalize_quotes,
    _record_read,
    edit_file,
    read_file,
)

# ---------------------------------------------------------------------------
# Quote normalization
# ---------------------------------------------------------------------------


class TestQuoteNormalization:
    def test_curly_single_quotes(self):
        assert _normalize_quotes("\u2018hello\u2019") == "'hello'"

    def test_curly_double_quotes(self):
        assert _normalize_quotes("\u201chello\u201d") == '"hello"'

    def test_mixed_quotes(self):
        assert _normalize_quotes("\u201cit\u2019s\u201d") == '"it\'s"'

    def test_straight_unchanged(self):
        assert _normalize_quotes("'hello' \"world\"") == "'hello' \"world\""

    def test_no_quotes(self):
        assert _normalize_quotes("no quotes here") == "no quotes here"


class TestFindActualString:
    def test_exact_match(self):
        actual, count = _find_actual_string("hello world", "hello")
        assert actual == "hello"
        assert count == 1

    def test_curly_quote_match(self):
        content = "She said \u201chello\u201d"
        search = 'She said "hello"'
        actual, count = _find_actual_string(content, search)
        assert actual is not None
        assert count == 1

    def test_no_match(self):
        actual, count = _find_actual_string("hello world", "goodbye")
        assert actual is None
        assert count == 0

    def test_multiple_matches(self):
        content = "foo bar foo"
        actual, count = _find_actual_string(content, "foo")
        assert count == 2

    def test_trailing_whitespace_tolerance(self):
        content = "hello   \nworld  "
        search = "hello\nworld"
        actual, count = _find_actual_string(content, search)
        assert actual is not None
        assert count == 1


# ---------------------------------------------------------------------------
# Staleness detection
# ---------------------------------------------------------------------------


class TestStalenessDetection:
    def test_no_stale_after_read(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("original content")
            path = f.name
        try:
            _record_read(path)
            assert _check_stale(path) is None
        finally:
            os.unlink(path)

    def test_stale_after_modify(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("original content")
            path = f.name
        try:
            _record_read(path)
            time.sleep(0.1)
            with open(path, "w") as f:
                f.write("modified content")
            result = _check_stale(path)
            assert result is not None
            assert "modified" in result
        finally:
            os.unlink(path)

    def test_no_stale_for_unread_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("content")
            path = f.name
        try:
            # Don't record read
            assert _check_stale(path) is None
        finally:
            os.unlink(path)

    def test_mtime_change_same_content(self):
        """mtime changed but content identical — not stale."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("same content")
            path = f.name
        try:
            _record_read(path)
            time.sleep(0.1)
            # Rewrite same content (changes mtime)
            with open(path, "w") as f:
                f.write("same content")
            assert _check_stale(path) is None
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# edit_file integration
# ---------------------------------------------------------------------------


class TestEditFileIntegration:
    def test_basic_edit(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("x = 1\ny = 2\n")
            path = f.name
        try:
            result = edit_file.fn(path, "x = 1", "x = 42")
            assert "replaced" in result.lower() or "edited" in result.lower()
            with open(path) as f:
                assert f.read() == "x = 42\ny = 2\n"
        finally:
            os.unlink(path)

    def test_edit_with_curly_quotes(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("She said \u201chello\u201d\n")
            path = f.name
        try:
            result = edit_file.fn(path, 'She said "hello"', 'She said "hi"')
            assert "error" not in result.lower()
        finally:
            os.unlink(path)

    def test_edit_not_found(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("x = 1\n")
            path = f.name
        try:
            result = edit_file.fn(path, "y = 2", "y = 3")
            assert "error" in result.lower()
        finally:
            os.unlink(path)

    def test_edit_multiple_matches(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("x = 1\nx = 1\n")
            path = f.name
        try:
            result = edit_file.fn(path, "x = 1", "x = 2")
            assert "matches 2" in result.lower()
        finally:
            os.unlink(path)

    def test_edit_stale_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("x = 1\n")
            path = f.name
        try:
            # Read file to establish baseline
            read_file.fn(path)
            time.sleep(0.1)
            # External modification
            with open(path, "w") as f:
                f.write("x = 999\n")
            # Edit should detect staleness
            result = edit_file.fn(path, "x = 1", "x = 2")
            assert "modified" in result.lower()
        finally:
            os.unlink(path)

    def test_edit_utf16_file(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"\xff\xfe")  # UTF-16 LE BOM
            f.write("hello world".encode("utf-16-le"))
            path = f.name
        try:
            result = edit_file.fn(path, "hello", "goodbye")
            assert "error" not in result.lower()
        finally:
            os.unlink(path)

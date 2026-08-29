"""Tests for agentino.sanitize — input cleanup for LLM-corrupted args."""

from agentino.safety.sanitize import clean_path, normalize_text, sanitize_tool_args


def test_clean_path_normal():
    path, corrupted = clean_path("inbox/message.txt")
    assert path == "inbox/message.txt"
    assert not corrupted


def test_clean_path_json_brackets():
    path, corrupted = clean_path("inbox/msg.txt|{extra}")
    assert path == "inbox/msg.txt"
    assert corrupted


def test_clean_path_unicode():
    path, corrupted = clean_path("inbox/msg\u200b.txt")
    assert path == "inbox/msg.txt"
    assert corrupted


def test_clean_path_pipe():
    path, corrupted = clean_path("outbox/reply.txt|grep foo")
    assert path == "outbox/reply.txt"
    assert corrupted


def test_clean_path_total_garbage():
    path, corrupted = clean_path("{{{garbage}}}")
    # When everything is garbage, returns original
    assert path == "{{{garbage}}}"
    assert corrupted


def test_normalize_text_zero_width():
    text = "hel\u200blo\u200dworld"
    result = normalize_text(text)
    assert result == "helloworld"


def test_normalize_text_clean():
    text = "normal text"
    assert normalize_text(text) == "normal text"


def test_sanitize_tool_args_cleans_paths():
    args = {"path": "inbox/msg.txt|extra", "content": "hello"}
    result = sanitize_tool_args(args, path_params=["path"])
    assert result["path"] == "inbox/msg.txt"
    assert result["content"] == "hello"  # non-path arg unchanged


def test_sanitize_tool_args_no_params():
    args = {"path": "inbox/msg.txt|extra"}
    result = sanitize_tool_args(args)
    assert result["path"] == "inbox/msg.txt|extra"  # unchanged


def test_sanitize_tool_args_missing_param():
    args = {"content": "hello"}
    result = sanitize_tool_args(args, path_params=["path"])
    assert result == {"content": "hello"}

# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from 1.0
onward.

## [1.1.1] — 2026-08-29

### Fixed

- `serve()` binds `127.0.0.1` rather than every interface. The endpoint runs an
  agent for whoever reaches it and has no authentication of its own, so the old
  default published that to the network. `WhatsAppChannel` likewise.
- Session and memory identifiers are reduced to a single safe path segment
  before they become a filename. `session_id="../../../tmp/x"` wrote
  `/tmp/x.jsonl`, outside the session directory — and `session_id` comes from
  HTTP bodies and chat ids.
- A base URL that is neither Anthropic nor Codex is treated as an
  OpenAI-compatible `/chat/completions` endpoint. It previously defaulted to
  Codex, so pointing Agentino at vLLM, Ollama or api.openai.com sent a wire
  format those servers do not speak.
- `POST /chat` answers 400 on a malformed body and 404 on an unknown agent,
  instead of 500 with the exception text.

## [1.1.0] — 2026-08-29

### Added

- Apache-2.0 licence, `NOTICE`, contribution guide, security policy, code of
  conduct, and issue and pull-request templates.
- `agentino.tools.std._llm_env`, one vendor-neutral resolver for the LLM
  endpoint used by the standard tools. Configure with `AGENTINO_BASE_URL` /
  `AGENTINO_API_KEY` or the conventional `OPENAI_*` pair.
- `__all__` on the public packages, so the supported surface is explicit.
- `gpt-5.4-codex` in the model registry; it is now the default Codex model.
- `agentino.tools.std.set_pdf_renderer`, plus a plain reportlab renderer, so
  `create_pdf` works in a standalone install. A host with its own house style
  registers a renderer instead; async renderers are awaited.
- Storage-contract tests for the file-producing standard tools.
- `agentino.tools.std.set_file_storage_provider`, so a host application can
  supply its own file storage, plus a local-filesystem default
  (`AGENTINO_FILES_DIR`, falling back to `./.agentino/files`) so the
  file-producing tools work with no configuration.

- `py.typed`, so a type checker no longer ignores every annotation in the
  package.
- Tests that check the documentation against the package: every documented
  import resolves, every file path named in prose exists, and a list of names
  that were promised but never existed cannot reappear.

### Changed

- Published on PyPI as `agentino-framework`: `pip install agentino-framework`.
  The `agentino` name there belongs to an unrelated project, so the
  distribution carries a second name; the import name is unaffected and
  remains `agentino`. Releases also attach wheels and sdists to the GitHub
  release, and are uploaded by Trusted Publishing rather than a stored token.
- Licence changed from MIT to Apache-2.0. The repository previously declared
  MIT in metadata with no licence file present.
- `print_summary` now reports how many stages passed and prints `FAILED`
  rather than `COMPLETE` when any stage failed.
- CI runs `ruff check` and `ruff format --check`, covers Python 3.10 through
  3.13, and builds the distribution with a `twine` metadata check.

### Fixed

- `builtin_tools` raised `NameError` instead of a readable message when
  `httpx` was missing: the `ImportError` handler called `error_unavailable`
  without importing it.
- Task notifications dropped the notification body, showing only the
  description.
- `fetch_web_data` and `translate_text` read different environment variables
  for the same endpoint and defaulted to a hardcoded private host.
- Four standard tools imported modules belonging to the workspace runtime
  built *on top of* this framework — `protocols` in three, and
  `helpers.documents.branded_pdf` in `create_pdf`. A standalone install
  had no such module, so `read_file`, `list_files` and file creation raised
  ImportError. The dependency is now inverted: the host registers storage.
- A tenant id of `..` traversed out of the storage root. Path segments that
  are only dots are replaced, not just character-substituted.

- `examples/hello.py` and `examples/booking_bot.py` called `agent.run(...)`
  without awaiting it, so both printed a coroutine object and never reached a
  model. The smallest example in the repository did not work.
- `discover_tools_from_dir` is exported publicly but was annotated `Path` and
  called `.is_dir()` directly, so the natural call with a string raised
  `AttributeError`. Every caller inside the package passed a `Path`, which is
  why it went unnoticed.
- The transport package documented `pip install agentino[telegram]`, `[slack]`
  and `[serve]`. None of those extras existed, so each of those commands
  failed.
- The README declared MIT while `LICENSE`, `NOTICE` and `pyproject` all
  declare Apache-2.0 — the one statement that would have mattered to someone
  deciding whether they could use this.
- The README promised `Runner.run_sync`. There is no synchronous wrapper
  anywhere; the core is async throughout.
- `LLMClient`'s docstring listed `OPENAI_BASE_URL` as a fallback for the
  endpoint. It is not read at all, so setting it silently did nothing.
- `SECURITY.md` described the project as pre-1.0 at version 1.1.0, in the
  section a reader consults to learn which releases are patched.
- The README's layout omitted `tools/`, the built-in tools package.

### Removed

- `browse_web` and `crawl_page`. Both imported `helpers.crawler`, a module
  belonging to a host application that has never existed in this package, so
  they returned an error string on every call and no test covered them. The
  `browser` extra goes with them.
- An internal cross-repository planning document.
- CI exclusions for two test files whose 30 tests pass.

## [1.1.0] and earlier

Released before this changelog was kept. See the git history.

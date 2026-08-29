# Contributing to Agentino

Thanks for taking the time. This is a small project, so the process is short.

## Before you start

For anything beyond a bug fix or a typo, **open an issue first**. It is much
less painful to disagree about an approach in an issue than in a finished
pull request.

## Getting set up

```bash
git clone https://github.com/islavutin-oss/agentino.git
cd agentino
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Run the tests and the linter:

```bash
pytest tests/ -q
ruff check .
ruff format --check .
```

CI runs exactly these, on Python 3.10 through 3.13.

## What a good pull request looks like

- **One change per pull request.** Unrelated fixes bundled together are hard
  to review and harder to revert.
- **Tests come with the change.** A bug fix should include a test that fails
  before it and passes after.
- **The suite is green.** Not "green except for a flaky one" — if something
  is flaky, that is worth an issue of its own.
- **Commit messages say what changed and why.** The subject line is
  imperative and under about 72 characters; the body explains the reasoning
  a reviewer cannot get from the diff.

We use [Conventional Commits](https://www.conventionalcommits.org) prefixes:
`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `ci:`.

## Style

The linter settles formatting arguments, so there is nothing to debate:
`ruff format` owns line width, `ruff check` owns the rest. Type hints on
public functions. Comments should explain why something is the way it is —
the code already says what it does.

## Licensing

Contributions are accepted under the [Apache License 2.0](LICENSE), the same
licence the project ships under. There is no CLA.

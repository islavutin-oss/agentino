# Security Policy

## Supported versions

Security fixes land on `main` and in the next release. Only the latest
release is patched; there are no backports to earlier ones.

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Use GitHub's private vulnerability reporting: go to the
[Security tab](https://github.com/islavutin-oss/agentino/security/advisories/new)
and open a draft advisory. That thread is visible only to you and the
maintainer.

Please include what you can:

- what an attacker can do, and what access they need to start
- a minimal reproduction
- the version or commit you tested

You should get a first reply within a week. If a report is valid you will be
credited in the advisory unless you would rather not be.

## Scope

In scope: anything that lets untrusted input escape its boundary — tool
arguments reaching the shell or filesystem outside the configured root, one
tenant reading another's data, secrets reaching logs or model prompts,
sandbox and gate bypasses.

Also out of scope: what the `shell` tool runs. It executes the command it is
given, which is its purpose. Its blocklist stops a few shapes that are almost
always accidents and is not a sandbox — the boundary is whether an agent has
the tool at all. Giving `shell` to an agent that reads untrusted input and
reporting that it ran something unwelcome is not a vulnerability in agentino.

Out of scope: what an LLM chooses to say. Agentino gives you gates, hooks and
sanitizers to constrain tool use; it cannot make a model's output safe, and
prompt injection that only produces bad text is not treated as a
vulnerability here.

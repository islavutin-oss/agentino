"""Context compaction — summarize old messages when context fills up.

Auto-compact: triggered when tokens exceed threshold.
Reactive compact: forced on context overflow (max_output_tokens).
Post-compact: re-injects recently edited file paths.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from agentino.core.message import Message

if TYPE_CHECKING:
    from agentino.core.llm import LLMClient

# Import estimate_tokens from resilience (single source of truth)
# Can't import at module level due to circular import — use lazy import in compact_history


async def compact_history(
    messages: list[Message],
    llm: LLMClient,
    max_tokens: int = 128_000,
    keep_recent: int = 6,
    threshold: float = 0.7,
    reactive: bool = False,
) -> list[Message]:
    """Summarize old messages when estimated tokens exceed threshold.

    Args:
        reactive: If True, force compaction regardless of threshold.
    """
    from agentino.reliability.resilience import estimate_tokens

    estimated = estimate_tokens(messages)
    if not reactive and estimated < max_tokens * threshold:
        return messages

    system_msgs = [m for m in messages if m.role == "system"]
    non_system = [m for m in messages if m.role != "system"]

    # Skip already-compacted summaries
    already_compacted = []
    to_process = []
    for m in non_system:
        if m.content and m.content.startswith("[Previous conversation summary]"):
            already_compacted.append(m)
        else:
            to_process.append(m)

    actual_keep = max(2, keep_recent // 2) if reactive else keep_recent
    if len(to_process) <= actual_keep:
        return messages

    old = to_process[:-actual_keep]
    recent = to_process[-actual_keep:]

    # Truncate oversized messages
    _MAX_SINGLE = 15_000
    for i, m in enumerate(old):
        if m.content and len(m.content) > _MAX_SINGLE:
            head = _MAX_SINGLE * 2 // 3
            tail = _MAX_SINGLE - head
            old[i] = Message(
                role=m.role,
                content=m.content[:head]
                + f"\n[...truncated {len(m.content) - _MAX_SINGLE} chars...]\n"
                + m.content[-tail:],
                tool_calls=m.tool_calls,
                tool_call_id=m.tool_call_id,
                name=m.name,
            )

    # Build summary input
    old_parts = [m.content for m in already_compacted if m.content]
    for m in old:
        if m.content:
            if m.role == "tool":
                old_parts.append(f"tool_result: {m.content[:2000]}")
            else:
                old_parts.append(f"{m.role}: {m.content}")
    old_text = "\n".join(old_parts)

    _MAX_INPUT = 60_000
    if len(old_text) > _MAX_INPUT:
        head = _MAX_INPUT * 2 // 3
        tail = _MAX_INPUT - head
        old_text = (
            old_text[:head]
            + f"\n\n[...{len(old_text) - _MAX_INPUT} chars omitted...]\n\n"
            + old_text[-tail:]
        )

    try:
        resp = await llm.chat(
            [
                Message(
                    "system",
                    "Summarize this conversation concisely. Preserve:\n- Key decisions and conclusions\n- File paths that were read/written\n- Tool results that affect next steps\n- Current task state and what remains to be done\nBe brief but complete.",
                ),
                Message("user", old_text),
            ]
        )
        summary_text = resp.message.content
    except Exception:
        summary_text = f"[Previous conversation — summarization failed]\n{old_text[-2000:]}"

    summary_msg = Message(role="system", content=f"[Previous conversation summary]\n{summary_text}")

    # Post-compact: re-inject recently referenced files
    file_hints = _extract_recent_files(old)
    if file_hints:
        restore_msg = Message(
            role="system",
            content=f"[Files from previous context — re-read if needed]\n{file_hints}",
        )
        return system_msgs + [summary_msg, restore_msg] + recent

    return system_msgs + [summary_msg] + recent


def _extract_recent_files(messages: list[Message], max_files: int = 5) -> str:
    """Extract recently referenced file paths for post-compact restoration."""
    files: dict[str, str] = {}
    for m in reversed(messages):
        if len(files) >= max_files:
            break
        content = m.content or ""
        if m.name in ("read_file", "read"):
            match = re.search(r'path[=:]\s*["\']?([^\s"\']+)', content)
            if match and match.group(1) not in files:
                files[match.group(1)] = "read"
        elif m.name in ("edit_file", "write_file", "write"):
            match = re.search(r'path[=:]\s*["\']?([^\s"\']+)', content)
            if match and match.group(1) not in files:
                files[match.group(1)] = "edited"
        elif m.role == "tool" and m.name:
            for match in re.findall(
                r"(?:Edited|Written|Read|Deleted)\s+(\S+\.(?:py|ts|js|yml|json|md))", content
            ):
                if match not in files and len(files) < max_files:
                    files[match] = "referenced"
    if not files:
        return ""
    return "\n".join(f"  - {path} ({action})" for path, action in files.items())

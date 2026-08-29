"""Error handling utilities — LLM API error classification and tool error formatting.

Ported from Claude Code's error handling patterns.

LLM Error Recovery:
    from agentino.reliability.errors import classify_error, get_retry_delay, ErrorClass

    err_class = classify_error(exception)
    if err_class == ErrorClass.RATE_LIMIT:
        delay = get_retry_delay(attempt=2)
        await asyncio.sleep(delay)
    elif err_class == ErrorClass.CONTEXT_OVERFLOW:
        messages = await compact_history(messages, llm, reactive=True)

Tool Error Formatting:
    from agentino.reliability.errors import error_not_found, error_timeout, format_error

    return error_not_found("file", path="/missing.txt")
    return error_timeout("shell command", timeout_secs=30.0)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

# =============================================================================
# LLM API Error Classification
# =============================================================================


class ErrorClass(Enum):
    """Classification of API/network errors."""

    RATE_LIMIT = "rate_limit"  # 429 — retry with backoff
    AUTH_FAILURE = "auth_failure"  # 401/403 — check credentials
    CONTEXT_OVERFLOW = "context_overflow"  # 400 prompt_too_long — compact
    SERVER_ERROR = "server_error"  # 500/502/503/529 — retry
    SSL_ERROR = "ssl_error"  # TLS/certificate issues
    CONNECTION_ERROR = "connection"  # Network unreachable, timeout
    INVALID_REQUEST = "invalid_request"  # 400 bad request (not overflow)
    UNKNOWN = "unknown"


def classify_error(error: Exception) -> ErrorClass:
    """Classify an exception into an error category for recovery decisions."""
    err_str = str(error).lower()

    # HTTP status-based classification
    try:
        import httpx

        if isinstance(error, httpx.HTTPStatusError):
            status = error.response.status_code
            if status == 429:
                return ErrorClass.RATE_LIMIT
            if status in (401, 403):
                return ErrorClass.AUTH_FAILURE
            if status in (500, 502, 503, 529):
                return ErrorClass.SERVER_ERROR
            if status == 400:
                body = error.response.text
                if "prompt_too_long" in body or "context_length" in body:
                    return ErrorClass.CONTEXT_OVERFLOW
                return ErrorClass.INVALID_REQUEST
    except ImportError:
        pass

    # String-based classification (for wrapped errors)
    if "429" in err_str or "rate limit" in err_str:
        return ErrorClass.RATE_LIMIT
    if "401" in err_str or "403" in err_str or "unauthorized" in err_str:
        return ErrorClass.AUTH_FAILURE
    if "prompt_too_long" in err_str or "context_length" in err_str:
        return ErrorClass.CONTEXT_OVERFLOW
    if any(code in err_str for code in ("500", "502", "503", "529")):
        return ErrorClass.SERVER_ERROR

    # SSL/TLS chain walking
    if _is_ssl_error(error):
        return ErrorClass.SSL_ERROR

    # Connection errors
    if any(kw in err_str for kw in ("connection", "timeout", "unreachable", "refused")):
        return ErrorClass.CONNECTION_ERROR

    return ErrorClass.UNKNOWN


def _is_ssl_error(error: Exception) -> bool:
    """Walk the exception cause chain to detect SSL/TLS errors."""
    current: BaseException | None = error
    seen: set[int] = set()

    while current and id(current) not in seen:
        seen.add(id(current))
        err_str = str(current).lower()
        err_type = type(current).__name__.lower()

        if "ssl" in err_type or "ssl" in err_str:
            return True
        if any(
            kw in err_str
            for kw in (
                "certificate",
                "cert_expired",
                "hostname_mismatch",
                "self_signed",
                "unable to get local issuer",
            )
        ):
            return True

        current = current.__cause__ or current.__context__

    return False


def get_ssl_hint(error: Exception) -> str | None:
    """Get actionable hint for SSL errors."""
    if not _is_ssl_error(error):
        return None

    err_str = str(error).lower()
    if "self_signed" in err_str or "self signed" in err_str:
        return "Self-signed certificate detected. Set SSL_CERT_FILE or REQUESTS_CA_BUNDLE to your CA bundle."
    if "expired" in err_str:
        return "Certificate has expired. Check your system clock and CA certificates."
    if "hostname" in err_str:
        return "Hostname mismatch. If behind a proxy, set HTTPS_PROXY and ensure proxy certificate is trusted."
    return "SSL/TLS error. If behind a corporate proxy, set NODE_EXTRA_CA_CERTS or SSL_CERT_FILE to your CA bundle."


def get_retry_delay(attempt: int, base: float = 1.0, max_delay: float = 60.0) -> float:
    """Exponential backoff delay for retries."""
    import random

    delay = min(base * (2**attempt), max_delay)
    # Add jitter (±25%)
    jitter = delay * 0.25 * (2 * random.random() - 1)
    return max(0.1, delay + jitter)


def get_overflow_tokens(error: Exception) -> int | None:
    """Extract token overflow amount from context_length errors."""
    err_str = str(error)
    # Common patterns: "maximum context length is 128000 tokens, however you requested 140000"
    match = re.search(r"requested\s+(\d+)\s*tokens.*maximum.*?(\d+)", err_str)
    if match:
        requested = int(match.group(1))
        maximum = int(match.group(2))
        return requested - maximum

    match = re.search(r"maximum.*?(\d+).*requested\s+(\d+)", err_str)
    if match:
        maximum = int(match.group(1))
        requested = int(match.group(2))
        return requested - maximum

    return None


# =============================================================================
# Tool Error Formatting
# =============================================================================

# Standard error categories for tools
ErrorCategory = Literal[
    "timeout",  # Operation exceeded time limit
    "invalid_args",  # Wrong arguments provided
    "validation",  # Arguments failed validation
    "not_found",  # Resource doesn't exist
    "permission",  # Access denied
    "blocked",  # Operation blocked (safety/security)
    "unavailable",  # Service/resource temporarily unavailable
    "internal",  # Unexpected error
]


@dataclass(frozen=True)
class ToolError:
    """Structured error from tool execution.

    Categories:
        - timeout: Operation exceeded time limit
        - invalid_args: Wrong arguments provided
        - validation: Arguments failed validation
        - not_found: Resource doesn't exist
        - permission: Access denied
        - blocked: Operation blocked (safety/security)
        - unavailable: Service/resource temporarily unavailable
        - internal: Unexpected error
    """

    category: ErrorCategory | str
    tool_name: str
    detail: str

    def format(self) -> str:
        """Format as a string for the LLM to interpret."""
        return f"Error: [{self.category}] {self.tool_name}: {self.detail}"


def format_error(category: ErrorCategory | str, detail: str, tool_name: str = "") -> str:
    """Format an error message consistently.

    Args:
        category: Error category (timeout, not_found, permission, etc.)
        detail: Human-readable error details
        tool_name: Optional tool name (empty string if not tool-specific)

    Returns:
        Formatted error string for LLM consumption
    """
    if tool_name:
        return f"Error: [{category}] {tool_name}: {detail}"
    return f"Error: [{category}] {detail}"


# Common error helpers


def error_not_found(resource: str, path: str = "", tool_name: str = "") -> str:
    """Resource not found error."""
    detail = f"{resource} not found"
    if path:
        detail += f": {path}"
    return format_error("not_found", detail, tool_name)


def error_permission(action: str, resource: str = "", tool_name: str = "") -> str:
    """Permission denied error."""
    detail = "Permission denied"
    if action:
        detail += f" to {action}"
    if resource:
        detail += f" {resource}"
    return format_error("permission", detail, tool_name)


def error_timeout(
    operation: str = "", timeout_secs: float | None = None, tool_name: str = ""
) -> str:
    """Operation timeout error."""
    detail = "Operation timed out"
    if operation:
        detail = f"{operation} timed out"
    if timeout_secs:
        detail += f" after {timeout_secs}s"
    detail += " — do NOT retry the same call, try a different approach"
    return format_error("timeout", detail, tool_name)


def error_invalid_args(param: str = "", reason: str = "", tool_name: str = "") -> str:
    """Invalid arguments error."""
    detail = "Invalid arguments"
    if param:
        detail = f"Invalid argument for '{param}'"
    if reason:
        detail += f": {reason}"
    return format_error("invalid_args", detail, tool_name)


def error_validation(field: str = "", reason: str = "", tool_name: str = "") -> str:
    """Validation failed error."""
    detail = "Validation failed"
    if field:
        detail = f"Validation failed for '{field}'"
    if reason:
        detail += f": {reason}"
    return format_error("validation", detail, tool_name)


def error_blocked(reason: str, tool_name: str = "") -> str:
    """Operation blocked (safety/security)."""
    return format_error("blocked", reason, tool_name)


def error_unavailable(service: str = "", reason: str = "", tool_name: str = "") -> str:
    """Service/resource unavailable error."""
    detail = "Service unavailable"
    if service:
        detail = f"{service} unavailable"
    if reason:
        detail += f": {reason}"
    return format_error("unavailable", detail, tool_name)


def error_internal(operation: str = "", error: Exception | str = "", tool_name: str = "") -> str:
    """Internal/unexpected error."""
    detail = "Internal error"
    if operation:
        detail = f"Internal error during {operation}"
    if error:
        error_msg = str(error) if isinstance(error, Exception) else error
        detail += f": {error_msg}"
    return format_error("internal", detail, tool_name)


def error_duplicate(tool_call_desc: str = "") -> str:
    """Duplicate tool call detected."""
    detail = "Duplicate tool call detected, breaking loop"
    if tool_call_desc:
        detail = f"Duplicate {tool_call_desc} call detected, breaking loop"
    return format_error("blocked", detail)


def error_unknown_tool(tool_name: str) -> str:
    """Unknown tool error."""
    return format_error("not_found", f"Unknown tool '{tool_name}'")

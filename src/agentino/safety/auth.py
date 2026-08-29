"""OAuth authentication — login with OpenAI (Codex) or Anthropic subscription.

Supports:
- OpenAI Codex: OAuth PKCE flow via browser → ChatGPT subscription
- Anthropic: setup-token from `claude setup-token` CLI

Token storage: ~/.agentino/auth.json
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import secrets
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUTH_DIR = Path.home() / ".agentino"
AUTH_FILE = AUTH_DIR / "auth.json"
CODEX_AUTH_FILE = Path.home() / ".codex" / "auth.json"

# OpenAI OAuth endpoints (same as Codex CLI uses)
OPENAI_AUTH_URL = "https://auth.openai.com/oauth/authorize"
OPENAI_TOKEN_URL = "https://auth.openai.com/oauth/token"
OPENAI_CLIENT_ID = "app_live_cFBK3GmMRanLz7FA5hITmlCT"  # Codex CLI public client ID
OPENAI_SCOPES = "openid profile email offline_access"
CALLBACK_PORT = 14551  # local callback port
CALLBACK_URL = f"http://127.0.0.1:{CALLBACK_PORT}/auth/callback"


def decode_jwt_claims(token: str) -> dict[str, Any] | None:
    """Decode JWT payload claims without signature verification.

    Returns None if the token is not a valid JWT structure.
    """
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        payload = parts[1] + "=" * (4 - len(parts[1]) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Auth store
# ---------------------------------------------------------------------------


@dataclass
class AuthCredentials:
    """Stored authentication credentials."""

    provider: str  # "openai" or "anthropic"
    access_token: str
    refresh_token: str = ""
    expires_at: float = 0.0  # epoch seconds
    email: str = ""
    account_id: str = ""

    @property
    def is_expired(self) -> bool:
        if self.expires_at == 0:
            return False  # no expiry (API keys, setup-tokens)
        return time.time() > self.expires_at - 60  # 60s buffer

    def to_dict(self) -> dict[str, Any]:
        """Serialize credentials to dict for storage."""
        d: dict[str, Any] = {
            "provider": self.provider,
            "access_token": self.access_token,
        }
        if self.refresh_token:
            d["refresh_token"] = self.refresh_token
        if self.expires_at:
            d["expires_at"] = self.expires_at
        if self.email:
            d["email"] = self.email
        if self.account_id:
            d["account_id"] = self.account_id
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuthCredentials:
        """Deserialize credentials from stored dict."""
        return cls(
            provider=data.get("provider", "openai"),
            access_token=data.get("access_token", ""),
            refresh_token=data.get("refresh_token", ""),
            expires_at=data.get("expires_at", 0.0),
            email=data.get("email", ""),
            account_id=data.get("account_id", ""),
        )


def save_credentials(creds: AuthCredentials) -> None:
    """Save credentials to ~/.agentino/auth.json."""
    AUTH_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing
    store = _load_store()
    store[creds.provider] = creds.to_dict()

    with open(AUTH_FILE, "w") as f:
        json.dump(store, f, indent=2)
    os.chmod(AUTH_FILE, 0o600)  # owner-only read/write


def load_credentials(provider: str = "openai") -> AuthCredentials | None:
    """Load credentials for a provider from ~/.agentino/auth.json."""
    store = _load_store()
    data = store.get(provider)
    if not data:
        return None
    return AuthCredentials.from_dict(data)


def _load_store() -> dict[str, Any]:
    if AUTH_FILE.exists():
        try:
            with open(AUTH_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def load_codex_credentials() -> AuthCredentials | None:
    """Load OpenAI credentials from ~/.codex/auth.json (Codex CLI).

    Allows reusing existing Codex CLI authentication without separate login.
    """
    if not CODEX_AUTH_FILE.exists():
        return None
    try:
        with open(CODEX_AUTH_FILE) as f:
            data = json.load(f)
        tokens = data.get("tokens", {})
        access_token = tokens.get("access_token", "")
        if not access_token:
            return None

        # Parse JWTs to extract expiry, email, account_id
        expires_at = 0.0
        email = ""
        account_id = tokens.get("account_id", "")
        claims = decode_jwt_claims(access_token)
        if claims:
            expires_at = float(claims.get("exp", 0))
            email = claims.get("email", "")
            if not account_id:
                account_id = claims.get("sub", "")
        # id_token often has email when access_token doesn't
        if not email:
            id_token = tokens.get("id_token", "")
            if id_token:
                id_claims = decode_jwt_claims(id_token)
                if id_claims:
                    email = id_claims.get("email", "")

        return AuthCredentials(
            provider="openai",
            access_token=access_token,
            refresh_token=tokens.get("refresh_token", ""),
            expires_at=expires_at,
            email=email,
            account_id=account_id,
        )
    except (json.JSONDecodeError, OSError, KeyError):
        return None


def get_api_key(provider: str = "openai") -> str | None:
    """Get a valid API key/token, refreshing if needed.

    Priority:
    1. Environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, ANTHROPIC_SETUP_TOKEN)
    2. Stored OAuth credentials (auto-refreshed)
    """
    # Check env vars first
    if provider == "openai":
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            return env_key
    elif provider == "anthropic":
        for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_SETUP_TOKEN"):
            env_key = os.getenv(var)
            if env_key:
                return env_key

    # Load stored credentials (agentino's own)
    creds = load_credentials(provider)

    # Refresh if expired
    if creds and creds.is_expired and creds.refresh_token:
        refreshed = refresh_openai_token(creds)
        if refreshed and not refreshed.is_expired:
            save_credentials(refreshed)
            creds = refreshed
        else:
            creds = None  # refresh failed — fall through to codex

    # Fall back to Codex CLI credentials for OpenAI
    if not creds and provider == "openai":
        creds = load_codex_credentials()
        # Refresh codex token if needed too
        if creds and creds.is_expired and creds.refresh_token:
            refreshed = refresh_openai_token(creds)
            if refreshed and not refreshed.is_expired:
                save_credentials(refreshed)
                creds = refreshed

    if not creds:
        return None

    return creds.access_token if creds else None


# ---------------------------------------------------------------------------
# OpenAI OAuth PKCE flow
# ---------------------------------------------------------------------------


def login_openai() -> AuthCredentials:
    """Run the OpenAI OAuth PKCE login flow. Opens browser.

    Returns credentials on success, raises on failure.
    """
    import webbrowser

    # Generate PKCE challenge
    code_verifier = secrets.token_urlsafe(64)
    code_challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    state = secrets.token_urlsafe(32)

    # Build auth URL
    params = {
        "response_type": "code",
        "client_id": OPENAI_CLIENT_ID,
        "redirect_uri": CALLBACK_URL,
        "scope": OPENAI_SCOPES,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    auth_url = f"{OPENAI_AUTH_URL}?{urllib.parse.urlencode(params)}"

    # Start local callback server
    auth_code: dict[str, str] = {}

    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        """HTTP request handler for OAuth callback."""

        def do_GET(self) -> None:
            """Handle OAuth callback GET request."""
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            if "code" in query:
                auth_code["code"] = query["code"][0]
                auth_code["state"] = query.get("state", [""])[0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<h2>Login successful! You can close this tab.</h2>")
            else:
                self.send_response(400)
                self.end_headers()
                error = query.get("error", ["unknown"])[0]
                auth_code["error"] = error
                self.wfile.write(f"Error: {error}".encode())

        def log_message(self, format: str, *args: Any) -> None:
            """Suppress default HTTP request logging."""
            pass

    server = http.server.HTTPServer(("127.0.0.1", CALLBACK_PORT), CallbackHandler)
    server.timeout = 120  # 2 min timeout

    print("\nOpening browser for OpenAI login...")
    print(f"If the browser doesn't open, visit:\n{auth_url}\n")

    webbrowser.open(auth_url)

    # Wait for callback
    while "code" not in auth_code and "error" not in auth_code:
        server.handle_request()

    server.server_close()

    if "error" in auth_code:
        raise RuntimeError(f"OAuth login failed: {auth_code['error']}")

    if auth_code.get("state") != state:
        raise RuntimeError("OAuth state mismatch — possible CSRF attack")

    # Exchange code for tokens
    return _exchange_code(auth_code["code"], code_verifier)


def _exchange_code(code: str, code_verifier: str) -> AuthCredentials:
    """Exchange authorization code for access + refresh tokens."""
    resp = httpx.post(
        OPENAI_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": CALLBACK_URL,
            "client_id": OPENAI_CLIENT_ID,
            "code_verifier": code_verifier,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()

    expires_in = data.get("expires_in", 3600)
    expires_at = time.time() + expires_in

    # Try to extract account ID from the access token (JWT)
    account_id = ""
    email = ""
    claims = decode_jwt_claims(data["access_token"])
    if claims:
        account_id = claims.get("sub", "")
        email = claims.get("email", "")

    creds = AuthCredentials(
        provider="openai",
        access_token=data["access_token"],
        refresh_token=data.get("refresh_token", ""),
        expires_at=expires_at,
        email=email,
        account_id=account_id,
    )
    save_credentials(creds)
    return creds


def refresh_openai_token(creds: AuthCredentials) -> AuthCredentials | None:
    """Refresh an expired OpenAI OAuth token."""
    if not creds.refresh_token:
        return None

    try:
        resp = httpx.post(
            OPENAI_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": creds.refresh_token,
                "client_id": OPENAI_CLIENT_ID,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()

        expires_in = data.get("expires_in", 3600)

        return AuthCredentials(
            provider="openai",
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", creds.refresh_token),
            expires_at=time.time() + expires_in,
            email=creds.email,
            account_id=creds.account_id,
        )
    except Exception:
        # Return old creds as fallback (may still work briefly)
        return creds


# ---------------------------------------------------------------------------
# Anthropic setup-token
# ---------------------------------------------------------------------------


def save_anthropic_token(token: str) -> AuthCredentials:
    """Save an Anthropic setup-token (from `claude setup-token`)."""
    creds = AuthCredentials(
        provider="anthropic",
        access_token=token,
    )
    save_credentials(creds)
    return creds

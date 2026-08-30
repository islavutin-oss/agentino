"""Tests for auth — credential storage, resolution, token detection."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from agentino.safety.auth import (
    AuthCredentials,
    get_api_key,
    load_credentials,
    save_anthropic_token,
    save_credentials,
)


class TestAuthCredentials:
    def test_to_dict(self):
        creds = AuthCredentials(
            provider="openai",
            access_token="sk-test",
            refresh_token="rt-test",
            expires_at=1700000000.0,
            email="user@test.com",
            account_id="acct_123",
        )
        d = creds.to_dict()
        assert d["provider"] == "openai"
        assert d["access_token"] == "sk-test"
        assert d["refresh_token"] == "rt-test"
        assert d["email"] == "user@test.com"

    def test_from_dict(self):
        data = {
            "provider": "openai",
            "access_token": "sk-test",
            "refresh_token": "rt-test",
            "expires_at": 1700000000.0,
        }
        creds = AuthCredentials.from_dict(data)
        assert creds.provider == "openai"
        assert creds.access_token == "sk-test"

    def test_is_expired(self):
        import time

        creds = AuthCredentials(provider="openai", access_token="x", expires_at=time.time() - 100)
        assert creds.is_expired is True

    def test_is_not_expired(self):
        import time

        creds = AuthCredentials(provider="openai", access_token="x", expires_at=time.time() + 3600)
        assert creds.is_expired is False

    def test_no_expiry_not_expired(self):
        creds = AuthCredentials(provider="openai", access_token="x", expires_at=0)
        assert creds.is_expired is False

    def test_optional_fields_omitted(self):
        creds = AuthCredentials(provider="openai", access_token="x")
        d = creds.to_dict()
        assert "refresh_token" not in d
        assert "email" not in d


class TestCredentialStorage:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth_file = Path(tmp) / "auth.json"
            with (
                patch("agentino.safety.auth.AUTH_FILE", auth_file),
                patch("agentino.safety.auth.AUTH_DIR", Path(tmp)),
            ):
                creds = AuthCredentials(provider="openai", access_token="sk-test123")
                save_credentials(creds)

                loaded = load_credentials("openai")
                assert loaded is not None
                assert loaded.access_token == "sk-test123"

    def test_load_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth_file = Path(tmp) / "auth.json"
            with patch("agentino.safety.auth.AUTH_FILE", auth_file):
                assert load_credentials("openai") is None

    def test_multiple_providers(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth_file = Path(tmp) / "auth.json"
            with (
                patch("agentino.safety.auth.AUTH_FILE", auth_file),
                patch("agentino.safety.auth.AUTH_DIR", Path(tmp)),
            ):
                save_credentials(AuthCredentials(provider="openai", access_token="oai-key"))
                save_credentials(AuthCredentials(provider="anthropic", access_token="ant-key"))

                oai = load_credentials("openai")
                ant = load_credentials("anthropic")
                assert oai.access_token == "oai-key"
                assert ant.access_token == "ant-key"

    def test_save_anthropic_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth_file = Path(tmp) / "auth.json"
            with (
                patch("agentino.safety.auth.AUTH_FILE", auth_file),
                patch("agentino.safety.auth.AUTH_DIR", Path(tmp)),
            ):
                save_anthropic_token("sk-ant-oat-test123")
                loaded = load_credentials("anthropic")
                assert loaded.access_token == "sk-ant-oat-test123"


class TestGetApiKey:
    def test_env_var_takes_priority(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            key = get_api_key("openai")
            assert key == "env-key"

    def test_falls_back_to_stored(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth_file = Path(tmp) / "auth.json"
            with (
                patch("agentino.safety.auth.AUTH_FILE", auth_file),
                patch("agentino.safety.auth.AUTH_DIR", Path(tmp)),
                patch.dict("os.environ", {}, clear=True),
            ):
                # Remove env vars that might interfere
                import os

                for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_SETUP_TOKEN"):
                    os.environ.pop(var, None)

                save_credentials(AuthCredentials(provider="openai", access_token="stored-key"))
                key = get_api_key("openai")
                assert key == "stored-key"

    def test_returns_none_when_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth_file = Path(tmp) / "nonexistent" / "auth.json"
            codex_file = Path(tmp) / "nonexistent" / "codex_auth.json"
            with (
                patch("agentino.safety.auth.AUTH_FILE", auth_file),
                patch("agentino.safety.auth.CODEX_AUTH_FILE", codex_file),
                patch.dict("os.environ", {}, clear=True),
            ):
                import os

                for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_SETUP_TOKEN"):
                    os.environ.pop(var, None)
                key = get_api_key("openai")
                assert key is None

    def test_anthropic_env_vars(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "ant-key"}):
            key = get_api_key("anthropic")
            assert key == "ant-key"

        with patch.dict("os.environ", {"ANTHROPIC_SETUP_TOKEN": "sk-ant-oat-x"}, clear=False):
            import os

            os.environ.pop("ANTHROPIC_API_KEY", None)
            key = get_api_key("anthropic")
            assert key == "sk-ant-oat-x"
